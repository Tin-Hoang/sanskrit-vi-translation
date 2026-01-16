import pandas as pd
from pathlib import Path


def load_data(file_path: Path) -> pd.DataFrame:
    """Loads the dataset from a CSV file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found at {file_path}")

    try:
        df = pd.read_csv(file_path)
        # Ensure required columns exist
        required_columns = ["sanskrit_text", "vietnamese_reference"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        return df
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")


def save_results(df: pd.DataFrame, file_path: Path):
    """Saves the results to a CSV file."""
    try:
        df.to_csv(file_path, index=False)
        print(f"Results saved to {file_path}")
    except Exception as e:
        print(f"Error saving results: {e}")
