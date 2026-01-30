import hydra
import hashlib
import json
import asyncio
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import time
from pathlib import Path
import warnings

from translator import Translator
from evaluator import Evaluator
import litellm
from utils import load_data, save_results, identify_columns
from observability import init_langfuse
from prompt_manager import PromptManager
from dataset_loader import load_dataset_as_dataframe, sync_local_dataset
from langfuse import Langfuse

from system_prompts.translator.current import (
    SINGLE_TRANSLATE_PROMPT,
    BATCH_TRANSLATE_PROMPT,
)
from system_prompts.evaluator.current import (
    EVALUATION_RUBRIC,
    BATCH_JUDGE_PROMPT,
    SINGLE_JUDGE_PROMPT,
)

# Initialize Langfuse tracing
init_langfuse()


def _hash_prompt(prompt: str) -> str:
    """Generate a short hash of prompt content for cache versioning."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:8]


# Suppress Pydantic warnings
warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings.*")

# Define current directory
current_dir = Path(__file__).parent


def load_input_data(input_arg: str, base_dir: Path) -> tuple[pd.DataFrame, str, str]:
    """
    Load data from file or Langfuse.
    Returns: (DataFrame, dataset_name, source_lang_hint)
    """
    if not input_arg:
        raise ValueError("No input provided (file path or dataset name).")

    # Method 1: Check if file exists
    input_path = Path(input_arg)
    if input_path.exists() or (base_dir / input_arg).exists():
        path_ref = input_path if input_path.exists() else (base_dir / input_arg)
        print(f"Loading local file: {path_ref}")

        # Load local data
        local_df = load_data(path_ref)

        # Derive dataset name from filename
        dataset_name = path_ref.stem.replace("_", "-").lower()

        # NEW: Sync with Langfuse (upload if new/larger) and get back DF with linking IDs
        df = sync_local_dataset(local_df, dataset_name)

        print(f"Dataset '{dataset_name}' ready with {len(df)} items.")
    else:
        # Method 2: Assume Langfuse Dataset
        print(f"Loading Langfuse dataset: {input_arg}")
        try:
            df = load_dataset_as_dataframe(input_arg)
            dataset_name = input_arg
        except Exception as e:
            raise ValueError(
                f"Could not load input '{input_arg}' as file or dataset. Error: {e}"
            )

    # Heuristic for source lang
    input_col, _, _ = identify_columns(df)
    source_lang = "Sanskrit"  # Default
    if input_col:
        # Simple heuristic based on column name
        if "pali" in input_col.lower():
            source_lang = "Pali"
        elif "viet" in input_col.lower():
            source_lang = "Vietnamese"

    return df, dataset_name, source_lang


def safe_create_score(langfuse_client, retry_count=3, **kwargs):
    """Safely create a score in Langfuse with retries for rate limits."""
    delay = 1
    for attempt in range(retry_count + 1):
        try:
            langfuse_client.create_score(**kwargs)
            return
        except Exception as e:
            # Check for rate limit indicators in the exception message
            err_msg = str(e).lower()
            if "429" in err_msg or "rate limit" in err_msg:
                if attempt < retry_count:
                    # Exponential backoff
                    time.sleep(delay)
                    delay *= 2
                    continue

            # If not a rate limit or retries exhausted, log and break
            print(f"Error creating score: {e}")
            break


def safe_flush(langfuse_client, retry_count=3):
    """Safely flush Langfuse with retries for timeouts."""
    delay = 1
    for attempt in range(retry_count + 1):
        try:
            langfuse_client.flush()
            return
        except Exception as e:
            # Check for timeout
            if "timeout" in str(e).lower():
                if attempt < retry_count:
                    print(f"Langfuse flush timed out, retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2
                    continue

            # If not a timeout or retries exhausted
            print(f"Error flushing Langfuse: {e}")
            break


async def run_benchmark(
    dataset_name: str,
    source_lang: str,
    df: pd.DataFrame,
    translator_prompts: dict,
    evaluator: Evaluator,
    output_prefix: str,
    base_dir: Path,
    models_config: list,
    default_translator_config: dict,
    session_id: str = None,
    batch_size: int = 10,
):
    print(f"\n{'#' * 60}")
    print(f"# Running Dataset: {dataset_name} ({len(df)} items)")
    print(f"# Source Language: {source_lang}")
    print(f"{'#' * 60}")

    # Identify columns dynamically
    input_col, ref_cols, _ = identify_columns(df)
    if not input_col:
        print("Error: Could not identify input column.")
        return None, []

    print(f"Input Column: {input_col}")
    print(f"Reference Columns: {ref_cols}")

    dataset_item_ids = (
        df["langfuse_item_id"].tolist() if "langfuse_item_id" in df.columns else None
    )

    # If using local file with IDs, we can reconstruct the expected Item IDs
    if not dataset_item_ids and "id" in df.columns:
        dataset_item_ids = [f"{dataset_name}-{row['id']}" for _, row in df.iterrows()]
        print(f"Reconstructed {len(dataset_item_ids)} Item IDs for linking.")

    # Prepare references for metric calculation
    transposed_references = []
    for col in ref_cols:
        transposed_references.append(df[col].fillna("").astype(str).tolist())

    benchmark_results = []
    results_df = df.copy()

    timestamp = time.strftime("%Y%m%d-%H%M")

    for model_info in models_config:
        model_id = model_info["id"]
        model_name = model_info["name"]

        experiment_name = f"{dataset_name} {model_name} {timestamp}"

        print(f"\n{'=' * 50}")
        print(f"Model: {model_name} -> Experiment: {experiment_name}")
        print(f"{'=' * 50}")

        model_temp = model_info.get(
            "temperature",
            default_translator_config.get("temperature", 0.3),
        )
        api_base = model_info.get("api_base")
        api_key = model_info.get("api_key")

        # Extract extra params (everything except known keys)
        known_keys = {"id", "name", "temperature", "api_base", "api_key"}
        extra_params = {k: v for k, v in model_info.items() if k not in known_keys}

        translator = Translator(
            model_name=model_id,
            temperature=model_temp,
            api_base=api_base,
            api_key=api_key,
            extra_params=extra_params,
            single_prompt_template=translator_prompts["single"],
            batch_prompt_template=translator_prompts["batch"],
        )

        # 1. Translate (Async)
        print("Translating...")
        try:
            translations, elapsed_time, trace_ids = await translator.batch_translate(
                df[input_col].tolist(),
                source_lang=source_lang,
                session_id=session_id,
                dataset_item_ids=dataset_item_ids,
                batch_size=batch_size,
                dataset_name=dataset_name,
                experiment_name=experiment_name,
            )
        except Exception as e:
            print(f"Translation failed: {e}")
            translations = [""] * len(df)
            elapsed_time = 0.0
            trace_ids = []

        col_trans = f"trans_{model_name.replace(' ', '_')}"
        results_df[col_trans] = translations

        # 2. Metrics
        print("Calculating automated metrics...")
        metrics = evaluator.calculate_metrics(transposed_references, translations)
        item_metrics = evaluator.calculate_item_metrics(
            transposed_references, translations
        )

        # 3. LLM Judge (Async)
        print("Running LLM Judge...")
        col_judge = f"judge_{model_name.replace(' ', '_')}"

        primary_ref = ref_cols[0] if ref_cols else None

        # Determine valid items that actually need judging
        valid_indices = []
        valid_sources = []
        valid_refs = []
        valid_cands = []

        for i, t in enumerate(translations):
            if t and t.strip():
                valid_indices.append(i)
                valid_sources.append(str(df.iloc[i][input_col]))
                valid_refs.append(str(df.iloc[i][primary_ref]) if primary_ref else "")
                valid_cands.append(t)

        batch_results = []
        if valid_cands:
            batch_results = await evaluator.batch_llm_judge(
                valid_sources,
                valid_refs,
                valid_cands,
                source_lang,
                batch_size=30,  # Could also use config batch_size
                model_id=model_id,
                session_id=session_id,
            )

        # Reconstruct judgements list matching the dataframe
        judgements = ['{"accuracy": 0, "fluency": 0}'] * len(df)

        # We need to map the batch_results (which are only for valid items) back to the full DF
        # batch_results is parallel to valid_indices
        for i, idx in enumerate(valid_indices):
            if i < len(batch_results):
                judgements[idx] = batch_results[i]

        results_df[col_judge] = judgements

        # Calculate Averages for Summary
        total_acc = 0
        total_flu = 0
        count = 0
        for j_str in judgements:
            try:
                j = json.loads(j_str) if isinstance(j_str, str) else j_str
                acc = float(j.get("accuracy", 0))
                flu = float(j.get("fluency", 0))
                if acc > 0:
                    total_acc += acc
                    total_flu += flu
                    count += 1
            except:
                pass

        avg_acc = total_acc / count if count else 0
        avg_flu = total_flu / count if count else 0

        benchmark_results.append(
            {
                "Dataset": dataset_name,
                "Model": model_name,
                "BLEU ↑": metrics.get("BLEU", 0),
                "BERTScore ↑": metrics.get("BERTScore_F1", 0),
                "LLM Judge Accuracy (1-5) ↑": avg_acc,
                "LLM Judge Fluency (1-5) ↑": avg_flu,
                "Time (s) ↓": elapsed_time,
            }
        )

        # Langfuse Scoring
        if dataset_item_ids:
            print("Scoring items in Langfuse...")
            # We can process this in parallel eventually, but scoring is sensitive to rates
            # Let's keep it simple for now, maybe use async later if slow.
            try:
                lf = Langfuse()
                dataset = lf.get_dataset(dataset_name)

                scored_count = 0
                if dataset and hasattr(dataset, "items"):
                    item_lookup = {item.id: item for item in dataset.items}

                    for i, item_id in enumerate(dataset_item_ids):
                        if not item_id or item_id not in item_lookup:
                            continue

                        item = item_lookup[item_id]

                        # Parse judgement
                        try:
                            j_str = judgements[i]
                            j = json.loads(j_str) if isinstance(j_str, str) else j_str
                            acc_val = float(j.get("accuracy", 0))
                            flu_val = float(j.get("fluency", 0))

                            trace_id = None

                            # Using trace linking
                            with item.run(
                                run_name=experiment_name,
                                run_metadata={
                                    "model": model_name,
                                    "source_lang": source_lang,
                                },
                            ) as run_span:
                                run_span.update(
                                    input=str(df.iloc[i][input_col]),
                                    output=translations[i],
                                )
                                trace_id = run_span.trace_id

                            if trace_id:
                                safe_create_score(
                                    lf,
                                    trace_id=trace_id,
                                    name="judge-accuracy",
                                    value=acc_val,
                                )
                                safe_create_score(
                                    lf,
                                    trace_id=trace_id,
                                    name="judge-fluency",
                                    value=flu_val,
                                )

                                if i < len(item_metrics["BLEU"]):
                                    safe_create_score(
                                        lf,
                                        trace_id=trace_id,
                                        name="bleu",
                                        value=item_metrics["BLEU"][i],
                                    )
                                if i < len(item_metrics["BERTScore_F1"]):
                                    safe_create_score(
                                        lf,
                                        trace_id=trace_id,
                                        name="bertscore",
                                        value=item_metrics["BERTScore_F1"][i],
                                    )

                                scored_count += 1

                        except Exception as e:
                            print(f"Error scoring item {i}: {e}")

                    safe_flush(lf)
                    print(
                        f"Scored {scored_count} items in experiment '{experiment_name}'"
                    )
            except Exception as e:
                print(f"Langfuse scoring failed: {e}")

    return results_df, benchmark_results


@hydra.main(version_base=None, config_path="../", config_name="config")
def main(cfg: DictConfig):
    # Wrapper to run async main
    asyncio.run(async_main(cfg))


async def async_main(cfg: DictConfig):
    base_dir = current_dir.parent

    # Resolve Input
    input_arg = cfg.input
    if not input_arg:
        print("Error: Input argument required (file path or dataset name).")
        return

    try:
        df, dataset_name, source_lang = load_input_data(input_arg, base_dir)
    except Exception as e:
        print(e)
        return

    if cfg.limit:
        df = df.head(cfg.limit)
        print(f"Limiting to first {cfg.limit} rows.")

    # Init Prompt Manager
    prompt_manager = PromptManager()
    tr_prompts = {
        "single": prompt_manager.get_prompt(
            "translator-single", fallback=SINGLE_TRANSLATE_PROMPT
        ),
        "batch": prompt_manager.get_prompt(
            "translator-batch", fallback=BATCH_TRANSLATE_PROMPT
        ),
    }

    # Init Cache
    if cfg.clear_cache:
        # Clear existing disk cache
        import shutil

        cleared_any = False

        # Check both home dir and project root for .litellm_cache
        cache_locations = [
            Path.home() / ".litellm_cache",
            base_dir / ".litellm_cache",
        ]

        for cache_dir in cache_locations:
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                print(f"Cleared LiteLLM cache at {cache_dir}")
                cleared_any = True

        # Also clear legacy cache folder in project root
        legacy_cache_dir = base_dir / "cache"
        if legacy_cache_dir.exists():
            shutil.rmtree(legacy_cache_dir)
            print(f"Cleared legacy cache at {legacy_cache_dir}")
            cleared_any = True

        if not cleared_any:
            print("No cache to clear")

    if not cfg.no_cache:
        litellm.cache = litellm.Cache(type="disk")
        print("LiteLLM Disk Cache enabled")

    # Init Evaluator
    evaluator = Evaluator(
        judge_model=cfg.judge.model,
        temperature=cfg.judge.temperature,
        rubric=prompt_manager.get_prompt(
            "evaluator-rubric", fallback=EVALUATION_RUBRIC
        ),
        single_judge_prompt_template=prompt_manager.get_prompt(
            "evaluator-single", fallback=SINGLE_JUDGE_PROMPT
        ),
        batch_judge_prompt_template=prompt_manager.get_prompt(
            "evaluator-batch", fallback=BATCH_JUDGE_PROMPT
        ),
    )

    # OUTPUT Setup
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M")
    out_csv = results_dir / f"results_{dataset_name}_{timestamp}.csv"

    # Run
    models_config_list = OmegaConf.to_container(cfg.translator.models, resolve=True)
    translator_defaults = OmegaConf.to_container(cfg.translator.defaults, resolve=True)

    res_df, stats = await run_benchmark(
        dataset_name=dataset_name,
        source_lang=source_lang,
        df=df,
        translator_prompts=tr_prompts,
        evaluator=evaluator,
        output_prefix=dataset_name,
        base_dir=base_dir,
        models_config=models_config_list,
        default_translator_config=translator_defaults,
        batch_size=cfg.batch_size,
    )

    if res_df is not None:
        save_results(res_df, out_csv)

        # Print Stats
        stats_df = pd.DataFrame(stats)
        print("\nBenchmark Summary:")
        print(stats_df.to_markdown(index=False, floatfmt=".2f"))


if __name__ == "__main__":
    main()
