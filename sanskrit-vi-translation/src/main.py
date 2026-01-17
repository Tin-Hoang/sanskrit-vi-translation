import sys
import pandas as pd
import time
from pathlib import Path
import warnings
import litellm

# Suppress Pydantic serializer warnings from litellm/pydantic interaction
warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings.*")

# Add current directory to path to allow imports from same directory
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from translator import Translator
from evaluator import Evaluator
from utils import load_data, save_results


def main():
    # Define paths
    base_dir = current_dir.parent
    # Prioritize extended dataset
    data_path = base_dir / "data" / "sanskrit_vi_heart_sutra.csv"
    if not data_path.exists():
        print(
            f"Extended dataset not found at {data_path}, falling back to parallel.csv"
        )
        data_path = base_dir / "data" / "heart_sutra_sanskrit_vi_parallel.csv"

    output_csv_path = base_dir / "results_benchmark.csv"
    report_md_path = base_dir / "BENCHMARK_REPORT.md"

    print(f"Loading data from {data_path}...")
    try:
        df = load_data(data_path)
    except Exception as e:
        print(f"Error: {e}")
        return

    # Identify reference columns
    ref_cols = [c for c in df.columns if c.startswith("ref_")]
    if not ref_cols and "vietnamese_reference" in df.columns:
        ref_cols = ["vietnamese_reference"]

    print(f"Using reference columns: {ref_cols}")

    # Prepare references for metric calculation
    # list of lists where outer list is per-reference-source, inner is per-sample
    transposed_references = []
    for col in ref_cols:
        transposed_references.append(df[col].fillna("").astype(str).tolist())

    # Define models to benchmark
    # Using 'groq/' prefix for litellm to use Groq provider.
    # The suffix is the model ID passed to Groq.
    models_config = [
        {"id": "groq/llama-3.3-70b-versatile", "name": "Llama-3.3-70b"},
        {"id": "groq/openai/gpt-oss-120b", "name": "GPT-OSS-120b"},
        {"id": "groq/moonshotai/kimi-k2-instruct-0905", "name": "Kimi-k2"},
        {"id": "groq/qwen/qwen3-32b", "name": "Qwen3-32b"},
    ]

    # Judge model (constant for fair comparison)
    judge_model = "groq/llama-3.3-70b-versatile"
    evaluator = Evaluator(judge_model=judge_model)
    print(f"Using Judge Model: {judge_model}")

    benchmark_results = []

    # Initialize separate columns for each model if not present
    # We will just append new columns

    for model_info in models_config:
        model_id = model_info["id"]
        model_name = model_info["name"]

        print(f"\n{'=' * 50}")
        print(f"Processing Model: {model_name} ({model_id})")
        print(f"{'=' * 50}")

        translator = Translator(model_name=model_id)

        # 1. Translate
        print(f"Translating {len(df)} sentences...")
        start_time = time.time()
        try:
            translations = translator.batch_translate(df["sanskrit_text"].tolist())
        except Exception as e:
            print(f"Failed to translate with {model_name}: {e}")
            # Fill with empty strings on failure
            translations = [""] * len(df)

        elapsed_time = time.time() - start_time
        print(f"Translation completed in {elapsed_time:.2f}s")

        # Store translations
        col_trans = f"trans_{model_name}"
        df[col_trans] = translations

        # 2. Evaluate (Quantitative)
        print("Calculating metrics...")
        candidates = df[col_trans].tolist()
        try:
            metrics = evaluator.calculate_metrics(transposed_references, candidates)
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            metrics = {"BLEU": 0.0, "BERTScore_F1": 0.0}

        print(f"Metrics: {metrics}")

        # 3. Evaluate (Qualitative - Judge)
        print("Running LLM Judge...")
        col_judge = f"judge_{model_name}"
        judgements = []

        # Use first reference for judge
        primary_ref_col = ref_cols[0] if ref_cols else None

        total_rows = len(df)
        total_accuracy = 0
        total_fluency = 0
        valid_judgements = 0

        for idx, row in df.iterrows():
            print(f"Judging {idx + 1}/{total_rows}...", end="\r")

            primary_ref_text = str(row[primary_ref_col]) if primary_ref_col else ""
            candidate_text = row[col_trans]

            # Skip judging if empty translation (failed)
            if not candidate_text.strip():
                judge_result = (
                    '{"accuracy": 0, "fluency": 0, "explanation": "Translation failed"}'
                )
            else:
                judge_result = evaluator.llm_judge(
                    row["sanskrit_text"],
                    primary_ref_text,
                    candidate_text,
                )

            judgements.append(judge_result)

            # Simple parsing for averaging (assuming json string returned)
            try:
                import json

                # Handle cases where judge returns python dict string or actual json
                if isinstance(judge_result, str):
                    clean_json = (
                        judge_result.replace("```json", "").replace("```", "").strip()
                    )
                    j = json.loads(clean_json)
                else:
                    j = judge_result

                acc = float(j.get("accuracy", 0))
                flu = float(j.get("fluency", 0))
                total_accuracy += acc
                total_fluency += flu
                valid_judgements += 1
            except:
                pass

            # Rate limit mitigation mostly for Judge
            time.sleep(3)

        print("\nJudge completed.")
        df[col_judge] = judgements

        avg_accuracy = total_accuracy / valid_judgements if valid_judgements > 0 else 0
        avg_fluency = total_fluency / valid_judgements if valid_judgements > 0 else 0

        # Collect consolidated stats
        benchmark_results.append(
            {
                "Model": model_name,
                "BLEU": metrics.get("BLEU", 0),
                "BERTScore": metrics.get("BERTScore_F1", 0),
                "Judge Accuracy": avg_accuracy,
                "Judge Fluency": avg_fluency,
                "Time (s)": elapsed_time,
            }
        )

    # Save detailed CSV
    save_results(df, output_csv_path)

    # Generate Report
    benchmark_df = pd.DataFrame(benchmark_results)

    # Rename columns with arrows and specific names
    column_mapping = {
        "Model": "Model",
        "BLEU": "BLEU ↑",
        "BERTScore": "BERTScore ↑",
        "Judge Accuracy": "LLM Judge Accuracy (1-5) ↑",
        "Judge Fluency": "LLM Judge Fluency (1-5) ↑",
        "Time (s)": "Time (s) ↓",
    }
    benchmark_df = benchmark_df.rename(columns=column_mapping)

    markdown_table = benchmark_df.to_markdown(index=False, floatfmt=".2f")

    report_content = f"""# Sanskrit-Vietnamese Translation Benchmark Results

**Date**: {time.strftime("%Y-%m-%d %H:%M:%S")}
**Judge Model**: {judge_model}
**Dataset**: {data_path.name} ({len(df)} samples)

## Performance Summary

{markdown_table}

*Evaluation powered by LLM Judge (Llama-3.3-70b) using a 5-point rubric for Accuracy and Fluency.*
"""

    with open(report_md_path, "w") as f:
        f.write(report_content)

    print(f"\n{'=' * 50}")
    print("BENCHMARK COMPLETED")
    print(f"Detailed results: {output_csv_path}")
    print(f"Summary report: {report_md_path}")
    print(f"{'=' * 50}")
    print("\nSummary:")
    print(markdown_table)


if __name__ == "__main__":
    main()
