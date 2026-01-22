import os
import pandas as pd
from langfuse import Langfuse

from src.utils import identify_columns


def get_langfuse_client():
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        raise ValueError("Langfuse credentials not found in environment variables")

    return Langfuse(public_key=public_key, secret_key=secret_key, host=host)


def upload_dataset(dataset_name: str, df: pd.DataFrame) -> None:
    """
    Uploads a pandas DataFrame to Langfuse as a dataset.
    Items are keyed by 'id' if available, otherwise auto-generated.
    """
    langfuse = get_langfuse_client()
    print(f"Uploading dataset '{dataset_name}' to Langfuse ({len(df)} items)...")

    # Define Schema
    # Input: Simple string (the source text)
    input_schema = {"type": "string"}

    # Expected Output: Dictionary of strings (references)
    expected_output_schema = {
        "type": "object",
        "additionalProperties": {"type": "string"},
    }

    # Ensure dataset exists with Schema
    import time

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # First try to create it (this fails if it exists usually, or it might just return it)
            # Langfuse SDK `create_dataset` usually doesn't update if it exists.
            langfuse.create_dataset(
                name=dataset_name,
                input_schema=input_schema,
                expected_output_schema=expected_output_schema,
            )
            break
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                sleep_time = (attempt + 1) * 2
                print(f"Rate limit hit creating dataset. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                print(f"Dataset creation note: {e}")
                # If it exists, we assume it's fine.
                # Ideally we would check if schema matches, but SDK might not expose that easily.
                break

    # Validation step: Check if dataset is strictly created
    # Langfuse might be async or dataset creation might need a moment?
    # But usually API is synchronous. Let's verify we have it.
    try:
        remote_ds = langfuse.get_dataset(dataset_name)
        # Verify schema if possible?
        # print(f"Remote Schema: {remote_ds.input_schema}")
    except Exception:
        raise ValueError(
            f"Dataset '{dataset_name}' count not be retrieved after creation attempt. Aborting upload."
        )

    # Identify columns
    input_col, ref_cols, metadata_cols = identify_columns(df)

    if not input_col:
        raise ValueError("Could not identify input column for upload.")

    # Iterate and create items
    for _, row in df.iterrows():
        input_data = row[input_col]

        # Expected Output (References)
        expected_output = {}
        for ref in ref_cols:
            if pd.notna(row[ref]):
                expected_output[ref] = str(row[ref])

        # If only one reference and generic name, maybe simplify?
        # But keeping as dict is robust for the benchmark reader.

        # Metadata
        metadata = {}
        for meta in metadata_cols:
            if pd.notna(row[meta]):
                metadata[meta] = row[meta]

        # Retry item creation
        for item_attempt in range(3):
            try:
                langfuse.create_dataset_item(
                    dataset_name=dataset_name,
                    input=input_data,
                    expected_output=expected_output,
                    metadata=metadata,
                    # id=str(row["id"]) if "id" in df.columns else None, # Disable explicit ID to fix 404
                )
                break  # Success
            except Exception as e:
                is_rate_limit = "429" in str(e) or "rate limit" in str(e).lower()
                is_not_found = "404" in str(e)

                if (is_rate_limit or is_not_found) and item_attempt < 2:
                    wait = (item_attempt + 1) * 2
                    # print(f"Retry item {row.get('id')} due to error: {e}. Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    if item_attempt == 2:
                        print(f"Failed to create item {row.get('id', '?')}: {e}")
                        print("Likely Schema Validation Failure or duplicate ID.")

    print(f"Successfully uploaded {len(df)} items to '{dataset_name}'.")


def sync_local_dataset(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Checks if Langfuse dataset exists.
    If not, uploads.
    If exists but has fewer items, uploads (upserts) the local items.

    Returns:
        pd.DataFrame: A dataframe containing Langfuse Item IDs (fetched after upload/check).
    """
    langfuse = get_langfuse_client()

    should_upload = False
    try:
        remote_dataset = langfuse.get_dataset(dataset_name)
        remote_count = len(remote_dataset.items)
        local_count = len(df)

        print(f"Dataset '{dataset_name}': Local={local_count}, Remote={remote_count}")

        if local_count > remote_count:
            print(f"Local has more items. Syncing to Langfuse...")
            should_upload = True
        else:
            print("Remote dataset is up-to-date.")

    except Exception:
        print(f"Dataset '{dataset_name}' not found on Langfuse. Creating...")
        should_upload = True

    if should_upload:
        upload_dataset(dataset_name, df)

    # Always reload from Langfuse to get the canonical Item IDs for tracing
    return load_dataset_as_dataframe(dataset_name)


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
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    langfuse = get_langfuse_client()

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
