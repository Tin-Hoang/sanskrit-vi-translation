import argparse
import hashlib
import json
import pandas as pd
import time
import uuid
from pathlib import Path
import warnings
import os

from translator import Translator
from evaluator import Evaluator
from cache import BenchmarkCache
from utils import load_data, save_results, identify_columns
from observability import init_langfuse
from prompt_manager import PromptManager
from dataset_loader import load_dataset_as_dataframe
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

# Judge model (constant)
JUDGE_MODEL = "gemini/gemini-3-flash-preview"

MODELS_CONFIG = [
    {"id": "groq/openai/gpt-oss-120b", "name": "GPT-OSS-120b"},
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark LLM translation of Buddhist texts"
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Input CSV file path OR Langfuse Dataset name. If not provided, defaults to legacy behavior or error.",
    )
    # Legacy arguments for backward compat (optional, or just remove if we want hard break)
    parser.add_argument("--task", help="Deprecated: Task name", default=None)

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear existing cache",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for translation (default: 10)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of items to process (for testing)",
    )
    return parser.parse_args()


def load_input_data(input_arg: str, base_dir: Path) -> tuple[pd.DataFrame, str, str]:
    """
    Load data from file or Langfuse.
    Returns: (DataFrame, dataset_name, source_lang_hint)
    """
    if not input_arg:
        # Fallback for now if user runs without args (legacy default?)
        # For this refactor, let's enforce input or providing a default if None
        # But let's check if it looks like a path
        pass

    # Method 1: Check if file exists
    input_path = Path(input_arg)
    if input_path.exists() or (base_dir / input_arg).exists():
        path_ref = input_path if input_path.exists() else (base_dir / input_arg)
        print(f"Loading local file: {path_ref}")
        df = load_data(path_ref)
        # Derive dataset name from filename
        dataset_name = path_ref.stem.replace("_", "-").lower()
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
        if "pali" in input_col.lower():
            source_lang = "Pali"
        elif "viet" in input_col.lower():
            source_lang = "Vietnamese"

    return df, dataset_name, source_lang


def run_benchmark(
    dataset_name: str,
    source_lang: str,
    df: pd.DataFrame,
    translator_prompts: dict,
    evaluator: Evaluator,
    output_prefix: str,
    base_dir: Path,
    cache: BenchmarkCache = None,
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
    # (Assuming they were uploaded with standard naming: dataset_name-id)
    if not dataset_item_ids and "id" in df.columns:
        dataset_item_ids = [f"{dataset_name}-{row['id']}" for _, row in df.iterrows()]
        print(f"Reconstructed {len(dataset_item_ids)} Item IDs for linking.")

    # Prepare references for metric calculation
    transposed_references = []
    for col in ref_cols:
        transposed_references.append(df[col].fillna("").astype(str).tolist())

    benchmark_results = []
    results_df = df.copy()

    # Generate an experiment name for this run
    # This aligns trace linking to a specific "run" in Langfuse
    timestamp = time.strftime("%Y%m%d-%H%M")

    for model_info in MODELS_CONFIG:
        model_id = model_info["id"]
        model_name = model_info["name"]

        experiment_name = f"{dataset_name} {model_name} {timestamp}"

        print(f"\n{'=' * 50}")
        print(f"Model: {model_name} -> Experiment: {experiment_name}")
        print(f"{'=' * 50}")

        translator = Translator(
            model_name=model_id,
            single_prompt_template=translator_prompts["single"],
            batch_prompt_template=translator_prompts["batch"],
        )

        # 1. Translate
        print(f"Translating...")
        try:
            translations, elapsed_time, trace_ids = translator.batch_translate(
                df[input_col].tolist(),
                source_lang=source_lang,
                cache=cache,
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

        # 2. Evaluate (Qualitative - Judge)
        # Note: We do judge BEFORE metrics or parallel?
        # Actually in the original code, metrics were first.
        # But we need to use the automated metrics too.

        # Calculate Metrics (Fast)
        print("Calculating automated metrics...")
        try:
            metrics = evaluator.calculate_metrics(transposed_references, translations)
        except Exception:
            metrics = {"BLEU": 0.0, "BERTScore_F1": 0.0}

        # LLM Judge
        print("Running LLM Judge...")
        col_judge = f"judge_{model_name.replace(' ', '_')}"

        # Prepare valid items for judge
        primary_ref = ref_cols[0] if ref_cols else None

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
            batch_results = evaluator.batch_llm_judge(
                valid_sources,
                valid_refs,
                valid_cands,
                source_lang,
                batch_size=30,
                cache=cache,
                model_id=model_id,
                session_id=session_id,
            )

        # Reconstruct judgements
        judgements = ['{"accuracy": 0, "fluency": 0}'] * len(df)
        for i, idx in enumerate(valid_indices):
            if i < len(batch_results):
                judgements[idx] = batch_results[i]

        results_df[col_judge] = judgements

        # Calculate Judge Averages
        total_acc = 0
        total_flu = 0
        count = 0
        for j_str in judgements:
            try:
                j = (
                    json.loads(j_str.replace("```json", "").replace("```", "").strip())
                    if isinstance(j_str, str)
                    else j_str
                )
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
                "Experiment": experiment_name,
                "Model": model_name,
                "BLEU": metrics.get("BLEU", 0),
                "BERTScore": metrics.get("BERTScore_F1", 0),
                "Judge Accuracy": avg_acc,
                "Judge Fluency": avg_flu,
                "Time (s)": elapsed_time,
            }
        )

        # Upload Scores to Langfuse linked to Items
        if dataset_item_ids:
            print("Uploading scores to Langfuse Items...")
            try:
                lf = Langfuse()
                dataset = lf.get_dataset(dataset_name)

                # Iterate and score
                for i, item_id in enumerate(dataset_item_ids):
                    if not item_id:
                        continue

                    # Get judgement
                    j_str = judgements[i]
                    try:
                        j = (
                            json.loads(
                                j_str.replace("```json", "").replace("```", "").strip()
                            )
                            if isinstance(j_str, str)
                            else j_str
                        )

                        # Score on the ITEM, linked to the run (experiment_name)
                        # We use dataset.get_item(id).score(...) pattern if available?
                        # Actually client.score(..., trace_id=...) is common.
                        # But SDK documentation says:
                        # dataset_item.link(trace_or_observation=..., run_name=...)
                        # To attach score to the experiment run involving this item:
                        # lf.score(name="...", value=..., trace_id=trace_id_of_generation)
                        # IF we have the trace_id. We DO have `trace_ids` from batch_translate!

                        t_id = trace_ids[i] if i < len(trace_ids) else None
                        if t_id:
                            lf.score(
                                trace_id=t_id,
                                name="judge-accuracy",
                                value=float(j.get("accuracy", 0)),
                                comment=j.get("explanation"),
                            )
                            lf.score(
                                trace_id=t_id,
                                name="judge-fluency",
                                value=float(j.get("fluency", 0)),
                            )
                    except:
                        pass
                lf.flush()
                print("Scores pushed.")
            except Exception as e:
                print(f"Score upload failed: {e}")

    return results_df, benchmark_results


def main():
    args = parse_args()
    base_dir = current_dir.parent

    # Resolve Input
    input_arg = args.input
    if not input_arg:
        if args.task:
            # Legacy mapping
            task_map = {
                "sanskrit-vi": "data/sanskrit_vi_heart_sutra.csv",
                "pali-vi": "data/pali_vi_dhammapada.csv",
            }
            input_arg = task_map.get(args.task)
            if not input_arg:
                print("Error: Please provide input file/dataset or valid --task")
                return
        else:
            print("Error: Input argument required (file path or dataset name).")
            return

    try:
        df, dataset_name, source_lang = load_input_data(input_arg, base_dir)
    except Exception as e:
        print(e)
        return

    if args.limit:
        df = df.head(args.limit)
        print(f"Limiting to first {args.limit} rows.")

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
    cache = None
    if not args.no_cache:
        cache_dir = base_dir / "cache"
        cache = BenchmarkCache(
            cache_dir,
            dataset_name,  # Use dataset name as task key
            translator_prompt_hash=_hash_prompt(tr_prompts["batch"]),
        )
        if args.clear_cache:
            cache.clear()

    # Init Evaluator
    evaluator = Evaluator(
        judge_model=JUDGE_MODEL,
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
    res_df, stats = run_benchmark(
        dataset_name=dataset_name,
        source_lang=source_lang,
        df=df,
        translator_prompts=tr_prompts,
        evaluator=evaluator,
        output_prefix=dataset_name,
        base_dir=base_dir,
        cache=cache,
        batch_size=args.batch_size,
    )

    if res_df is not None:
        save_results(res_df, out_csv)

        # Print Stats
        stats_df = pd.DataFrame(stats)
        print("\nBenchmark Summary:")
        print(stats_df.to_markdown(index=False, floatfmt=".2f"))


if __name__ == "__main__":
    main()
