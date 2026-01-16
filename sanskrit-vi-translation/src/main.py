import os
import sys
import pandas as pd
import json
import time
from pathlib import Path
import warnings

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

    output_path = base_dir / "results_extended.csv"

    print(f"Loading data from {data_path}...")
    try:
        df = load_data(data_path)
    except Exception as e:
        print(f"Error: {e}")
        return

    # Initialize models
    # Note: Ensure GROQ_API_KEY is set in .env
    translator_model = "groq/llama-3.3-70b-versatile"
    judge_model = "groq/llama-3.3-70b-versatile"

    translator = Translator(model_name=translator_model)
    evaluator = Evaluator(judge_model=judge_model)

    print(f"Translating {len(df)} sentences with {translator_model}...")

    # Translate
    translations = translator.batch_translate(df["sanskrit_text"].tolist())
    df["candidate_translation"] = translations

    # Identify reference columns
    ref_cols = [c for c in df.columns if c.startswith("ref_")]
    if not ref_cols and "vietnamese_reference" in df.columns:
        ref_cols = ["vietnamese_reference"]

    print(f"Using reference columns: {ref_cols}")

    # Quantitative Evaluation
    print("Calculating quantitative metrics...")

    # Prepare references in [ [ref1_0, ref1_1], [ref2_0, ref2_1] ] format for multi-ref BLEU
    # We transpose the data: list of lists where outer list is per-reference-source, inner is per-sample
    transposed_references = []
    for col in ref_cols:
        # Fill NaNs with empty string
        transposed_references.append(df[col].fillna("").astype(str).tolist())

    candidates = df["candidate_translation"].tolist()

    metrics = evaluator.calculate_metrics(transposed_references, candidates)
    print(f"Metrics: {metrics}")

    # Qualitative Evaluation (LLM Judge)
    print("Running LLM-as-a-judge...")
    judgements = []

    # Use the first reference column as the primary reference for the judge
    primary_ref_col = ref_cols[0] if ref_cols else None

    total_rows = len(df)
    for idx, row in df.iterrows():
        print(f"Judging {idx + 1}/{total_rows}...", end="\r")

        primary_ref_text = str(row[primary_ref_col]) if primary_ref_col else ""

        judge_result = evaluator.llm_judge(
            row["sanskrit_text"],
            primary_ref_text,
            row["candidate_translation"],
        )
        judgements.append(judge_result)
        time.sleep(2)  # Avoid Rate Limit (30 RPM)
    print("\nLLM Judge completed.")

    df["llm_judgement"] = judgements

    # Save results
    save_results(df, output_path)
    print("Done!")


if __name__ == "__main__":
    main()
