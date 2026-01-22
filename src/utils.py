import pandas as pd
from pathlib import Path
from typing import Optional


def load_data(file_path: Path) -> pd.DataFrame:
    """Loads the dataset from a CSV file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found at {file_path}")

    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")


def identify_columns(df: pd.DataFrame) -> tuple[Optional[str], list[str], list[str]]:
    """
    Identify input, expected_output, and metadata columns based on heuristics.
    Returns: (input_col, ref_cols, metadata_cols)
    """
    input_col = None
    # Priority for input columns
    input_candidates = ["pali_text", "sanskrit_text", "text", "input", "source_text"]

    for candidate in input_candidates:
        if candidate in df.columns:
            input_col = candidate
            break

    if not input_col:
        # Fallback: take the second column (assuming first is ID) or first if no ID
        if "id" in df.columns and len(df.columns) > 1:
            input_col = df.columns[1]
        else:
            input_col = df.columns[0]

    # expected_output columns (references)
    ref_cols = [
        c for c in df.columns if c.startswith("ref_") or "vietnamese" in c.lower()
    ]

    # Metadata: everything else except id and input
    exclude = {input_col, "id"} | set(ref_cols)
    metadata_cols = [c for c in df.columns if c not in exclude]

    return input_col, ref_cols, metadata_cols


def save_results(df: pd.DataFrame, file_path: Path):
    """Saves the results to a CSV file."""
    try:
        df.to_csv(file_path, index=False)
        print(f"Results saved to {file_path}")
    except Exception as e:
        print(f"Error saving results: {e}")
