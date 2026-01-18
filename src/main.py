import argparse
import json
import pandas as pd
import time
from pathlib import Path
import warnings

from translator import Translator
from evaluator import Evaluator
from utils import load_data, save_results

# Suppress Pydantic serializer warnings from litellm/pydantic interaction
warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings.*")

# Define current directory
current_dir = Path(__file__).parent

# Task configurations for different benchmarks
TASK_CONFIG = {
    "sanskrit-vi": {
        "name": "Sanskrit-Vietnamese Heart Sutra",
        "data_file": "sanskrit_vi_heart_sutra.csv",
        "source_column": "sanskrit_text",
        "source_lang": "Sanskrit",
        "output_prefix": "sanskrit",
    },
    "pali-vi": {
        "name": "Pali-Vietnamese (Dhammapada)",
        "data_file": "pali_vi_dhammapada.csv",
        "source_column": "pali_text",
        "source_lang": "Pali",
        "output_prefix": "pali",
    },
    "compare": {
        "name": "Pali vs Sanskrit Comparison",
        "data_file": "dhammapada_udanavarga_parallel.csv",
        "source_columns": ["pali_text", "sanskrit_text"],
        "source_langs": ["Pali", "Sanskrit"],
        "output_prefix": "comparison",
        "compare_mode": True,
    },
}

# Define models to benchmark
MODELS_CONFIG = [
    {"id": "groq/llama-3.3-70b-versatile", "name": "Llama-3.3-70b"},
    {"id": "groq/openai/gpt-oss-120b", "name": "GPT-OSS-120b"},
    {"id": "groq/moonshotai/kimi-k2-instruct-0905", "name": "Kimi-k2"},
    {"id": "groq/qwen/qwen3-32b", "name": "Qwen3-32b"},
    {"id": "gemini/gemini-3-flash-preview", "name": "Gemini-3-Flash"},  # Judge model
]

# Judge model (constant for fair comparison)
# Using Gemini 3 Flash Preview for powerful evaluation with generous free tier
JUDGE_MODEL = "gemini/gemini-3-flash-preview"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark LLM translation of Buddhist texts"
    )
    parser.add_argument(
        "--task",
        choices=list(TASK_CONFIG.keys()),
        default="sanskrit-vi",
        help="Benchmark task to run (default: sanskrit-vi)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="Custom data file path (overrides task default)",
    )
    return parser.parse_args()


def run_single_benchmark(
    task_name: str,
    source_column: str,
    source_lang: str,
    df: pd.DataFrame,
    evaluator: Evaluator,
    output_prefix: str,
    base_dir: Path,
) -> tuple[pd.DataFrame, list[dict]]:
    """Run benchmark for a single source language.

    Returns:
        Tuple of (results_df with translations, benchmark_results list)
    """
    print(f"\n{'#' * 60}")
    print(f"# Running: {task_name} ({source_lang})")
    print(f"{'#' * 60}")

    # Identify reference columns
    ref_cols = [c for c in df.columns if c.startswith("ref_")]
    if not ref_cols and "vietnamese_reference" in df.columns:
        ref_cols = ["vietnamese_reference"]

    print(f"Using reference columns: {ref_cols}")
    print(f"Source column: {source_column}")

    # Prepare references for metric calculation
    transposed_references = []
    for col in ref_cols:
        transposed_references.append(df[col].fillna("").astype(str).tolist())

    benchmark_results = []
    results_df = df.copy()

    for model_info in MODELS_CONFIG:
        model_id = model_info["id"]
        model_name = model_info["name"]

        print(f"\n{'=' * 50}")
        print(f"Processing Model: {model_name} ({model_id})")
        print(f"{'=' * 50}")

        translator = Translator(model_name=model_id)

        # 1. Translate
        print(f"Translating {len(df)} sentences from {source_lang}...")
        try:
            translations, elapsed_time = translator.batch_translate(
                df[source_column].tolist(),
                source_lang=source_lang,
            )
        except Exception as e:
            print(f"Failed to translate with {model_name}: {e}")
            translations = [""] * len(df)
            elapsed_time = 0.0

        print(f"Translation completed in {elapsed_time:.2f}s (inference time)")

        # Store translations with source lang prefix for comparison mode
        col_trans = f"trans_{source_lang}_{model_name}"
        results_df[col_trans] = translations

        # 2. Evaluate (Quantitative)
        print("Calculating metrics...")
        candidates = results_df[col_trans].tolist()
        try:
            metrics = evaluator.calculate_metrics(transposed_references, candidates)
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            metrics = {"BLEU": 0.0, "BERTScore_F1": 0.0}

        print(f"Metrics: {metrics}")

        # 3. Evaluate (Qualitative - Judge) using batched evaluation
        print("Running LLM Judge (batched)...")
        col_judge = f"judge_{source_lang}_{model_name}"

        primary_ref_col = ref_cols[0] if ref_cols else None

        # Prepare data for batch evaluation
        sources_list = df[source_column].tolist()
        refs_list = [
            str(row[primary_ref_col]) if primary_ref_col else ""
            for _, row in df.iterrows()
        ]
        candidates_list = results_df[col_trans].tolist()

        # Handle empty translations before batching
        valid_indices = []
        valid_sources = []
        valid_refs = []
        valid_candidates = []

        for i, cand in enumerate(candidates_list):
            if cand and cand.strip():
                valid_indices.append(i)
                valid_sources.append(sources_list[i])
                valid_refs.append(refs_list[i])
                valid_candidates.append(cand)

        # Batch evaluate valid translations
        if valid_candidates:
            batch_results = evaluator.batch_llm_judge(
                valid_sources,
                valid_refs,
                valid_candidates,
                source_lang=source_lang,
                batch_size=30,  # 30 items per API call
            )
        else:
            batch_results = []

        # Reconstruct full judgements list
        judgements = [
            '{"accuracy": 0, "fluency": 0, "explanation": "Translation failed"}'
        ] * len(df)
        for i, idx in enumerate(valid_indices):
            if i < len(batch_results):
                judgements[idx] = batch_results[i]

        # Calculate averages
        total_accuracy = 0
        total_fluency = 0
        valid_judgements = 0

        for judge_result in judgements:
            try:
                if isinstance(judge_result, str):
                    clean_json = (
                        judge_result.replace("```json", "").replace("```", "").strip()
                    )
                    j = json.loads(clean_json)
                else:
                    j = judge_result

                acc = float(j.get("accuracy", 0))
                flu = float(j.get("fluency", 0))
                if acc > 0 or flu > 0:  # Only count valid judgements
                    total_accuracy += acc
                    total_fluency += flu
                    valid_judgements += 1
            except Exception:
                pass

        print("Judge completed.")
        results_df[col_judge] = judgements

        avg_accuracy = total_accuracy / valid_judgements if valid_judgements > 0 else 0
        avg_fluency = total_fluency / valid_judgements if valid_judgements > 0 else 0

        benchmark_results.append(
            {
                "Source Lang": source_lang,
                "Model": model_name,
                "BLEU": metrics.get("BLEU", 0),
                "BERTScore": metrics.get("BERTScore_F1", 0),
                "Judge Accuracy": avg_accuracy,
                "Judge Fluency": avg_fluency,
                "Time (s)": elapsed_time,
            }
        )

    return results_df, benchmark_results


def generate_report(
    benchmark_results: list[dict],
    task_name: str,
    data_path: Path,
    sample_count: int,
    report_path: Path,
):
    """Generate markdown benchmark report."""
    benchmark_df = pd.DataFrame(benchmark_results)

    column_mapping = {
        "Source Lang": "Source",
        "Model": "Model",
        "BLEU": "BLEU ↑",
        "BERTScore": "BERTScore ↑",
        "Judge Accuracy": "LLM Judge Accuracy (1-5) ↑",
        "Judge Fluency": "LLM Judge Fluency (1-5) ↑",
        "Time (s)": "Time (s) ↓",
    }
    benchmark_df = benchmark_df.rename(columns=column_mapping)

    markdown_table = benchmark_df.to_markdown(index=False, floatfmt=".2f")

    report_content = f"""# {task_name} Benchmark Results

**Date**: {time.strftime("%Y-%m-%d %H:%M:%S")}
**Judge Model**: {JUDGE_MODEL}
**Dataset**: {data_path.name} ({sample_count} samples)

## Performance Summary

{markdown_table}

*Evaluation powered by LLM Judge (Gemini 2.0 Flash) using a 5-point rubric for Accuracy and Fluency.*
"""

    with open(report_path, "w") as f:
        f.write(report_content)

    print(f"\nSummary:\n{markdown_table}")


def main():
    args = parse_args()
    task_key = args.task
    task_cfg = TASK_CONFIG[task_key]

    base_dir = current_dir.parent

    # Determine data path
    if args.data:
        data_path = args.data
    else:
        data_path = base_dir / "data" / task_cfg["data_file"]

    # Output paths (results folder)
    output_prefix = task_cfg["output_prefix"]
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    output_csv_path = results_dir / f"results_{output_prefix}_benchmark.csv"
    report_md_path = results_dir / f"BENCHMARK_REPORT_{output_prefix.upper()}.md"

    print(f"Task: {task_cfg['name']}")
    print(f"Loading data from {data_path}...")

    try:
        df = load_data(data_path)
    except Exception as e:
        print(f"Error: {e}")
        return

    evaluator = Evaluator(judge_model=JUDGE_MODEL)
    print(f"Using Judge Model: {JUDGE_MODEL}")

    all_benchmark_results = []

    # Check if this is comparison mode
    if task_cfg.get("compare_mode"):
        # Run benchmark for each source language
        source_columns = task_cfg["source_columns"]
        source_langs = task_cfg["source_langs"]
        results_df = df.copy()

        for src_col, src_lang in zip(source_columns, source_langs):
            results_df, benchmark_results = run_single_benchmark(
                task_name=task_cfg["name"],
                source_column=src_col,
                source_lang=src_lang,
                df=results_df,
                evaluator=evaluator,
                output_prefix=output_prefix,
                base_dir=base_dir,
            )
            all_benchmark_results.extend(benchmark_results)
    else:
        # Single source language benchmark
        results_df, benchmark_results = run_single_benchmark(
            task_name=task_cfg["name"],
            source_column=task_cfg["source_column"],
            source_lang=task_cfg["source_lang"],
            df=df,
            evaluator=evaluator,
            output_prefix=output_prefix,
            base_dir=base_dir,
        )
        all_benchmark_results.extend(benchmark_results)

    # Save detailed CSV
    save_results(results_df, output_csv_path)

    # Generate Report
    generate_report(
        all_benchmark_results,
        task_cfg["name"],
        data_path,
        len(df),
        report_md_path,
    )

    print(f"\n{'=' * 50}")
    print("BENCHMARK COMPLETED")
    print(f"Detailed results: {output_csv_path}")
    print(f"Summary report: {report_md_path}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
