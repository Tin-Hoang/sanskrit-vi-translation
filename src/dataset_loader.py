import os
import pandas as pd
from langfuse import Langfuse
from typing import Optional


def load_dataset_as_dataframe(dataset_name: str) -> pd.DataFrame:
    """
    Fetch a dataset from Langfuse and convert it to a pandas DataFrame
    compatible with the existing benchmark pipeline.

    This function:
    1. Fetches all items from the specified Langfuse dataset
    2. Maps 'input' to 'pali_text' (or generic source column based on content)
    3. Expands 'expected_output' dictionary into separate reference columns
    4. Extracts 'metadata' fields back to columns
    5. Adds 'langfuse_item_id' for trace linking

    Args:
        dataset_name: Name of the Langfuse dataset to load

    Returns:
        pd.DataFrame: DataFrame with source, references, and metadata
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        raise ValueError("Langfuse credentials not found in environment variables")

    print(f"Fetching dataset '{dataset_name}' from Langfuse...")
    langfuse = Langfuse(public_key=public_key, secret_key=secret_key, host=host)

    try:
        dataset = langfuse.get_dataset(dataset_name)
    except Exception as e:
        raise ValueError(f"Failed to fetch dataset '{dataset_name}': {e}")

    items = []
    for item in dataset.items:
        # Base row data from input
        row = {
            "pali_text": item.input
        }  # Defaulting to pali_text for this specific dataset

        # Add Langfuse Item ID for linking
        row["langfuse_item_id"] = item.id

        # Expand expected_output (references)
        if isinstance(item.expected_output, dict):
            for key, value in item.expected_output.items():
                row[key] = value
        elif item.expected_output:
            # Fallback for simple string expected output
            row["expected_output"] = item.expected_output

        # Add metadata
        if item.metadata:
            for key, value in item.metadata.items():
                if key not in row:  # Don't overwrite existing
                    row[key] = value

        items.append(row)

    if not items:
        raise ValueError(f"Dataset '{dataset_name}' is empty")

    df = pd.DataFrame(items)
    print(f"Loaded {len(df)} items from dataset '{dataset_name}'")
    return df
