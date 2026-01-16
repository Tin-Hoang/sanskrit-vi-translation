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
    data_path = base_dir / "data" / "sanskrit_vi_parallel.csv"
    output_path = base_dir / "results.csv"

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

    # Quantitative Evaluation
    print("Calculating quantitative metrics...")
    references = df["vietnamese_reference"].tolist()
    candidates = df["candidate_translation"].tolist()

    metrics = evaluator.calculate_metrics(references, candidates)
    print(f"Metrics: {metrics}")

    # Qualitative Evaluation (LLM Judge)
    print("Running LLM-as-a-judge...")
    judgements = []

    # Process incrementally to avoid long waits or potential rate limits if we add more calls later
    total_rows = len(df)
    for idx, row in df.iterrows():
        print(f"Judging {idx + 1}/{total_rows}...", end="\r")
        judge_result = evaluator.llm_judge(
            row["sanskrit_text"],
            row["vietnamese_reference"],
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
