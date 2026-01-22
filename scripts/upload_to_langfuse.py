"""
Script to upload local CSV datasets to Langfuse Datasets.

Usage:
    python scripts/upload_to_langfuse.py

Requirements:
    - LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST in .env
"""

import os
import time
from pathlib import Path
from typing import Optional
import pandas as pd
from langfuse import Langfuse
from dotenv import load_dotenv

load_dotenv()


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


def upload_dataset(client: Langfuse, csv_path: Path):
    clean_name = csv_path.stem.replace("_", "-").replace(" ", "-").lower()
    dataset_name = clean_name

    print(f"\nProcessing {csv_path.name} -> Dataset: {dataset_name}")
    try:
        df = pd.read_csv(csv_path, skipinitialspace=True)
        # Robustly strip whitespace from column names
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"Skipping {csv_path.name}: Failed to read CSV - {e}")
        return

    if df.empty:
        print(f"Skipping {csv_path.name}: Empty file")
        return

    input_col, ref_cols, meta_cols = identify_columns(df)

    if not input_col:
        print(f"Skipping {csv_path.name}: Could not identify input column")
        return

    print(f"  Input: {input_col}")
    print(f"  Refs: {ref_cols}")

    # Check if dataset exists and is full
    try:
        existing_dataset = client.get_dataset(dataset_name)
        # Using items count if available in the object or fetch items
        # NOTE: get_dataset might return object with limited items?
        # But let's check basic existence first.
        # Actually Langfuse SDK get_dataset returns a wrapper.
        # To be safe against partial uploads, we check item count.
        # This might require fetching items, but usually it's metadata.
        # Let's assume if it exists we check length of items list if loaded,
        # or we just assume if it exists we rely on user intent "ignore already existing".
        # But to be robust against the failed one:
        if existing_dataset.items and len(existing_dataset.items) >= len(df):
            print(
                f"  ⚠️  Dataset '{dataset_name}' already exists with {len(existing_dataset.items)} items. Skipping."
            )
            return
        else:
            print(
                f"  Note: Dataset exists but might be partial ({len(existing_dataset.items) if existing_dataset.items else 0}/{len(df)}). merging/upserting..."
            )
    except Exception:
        pass  # Dataset does not exist, proceed to create

    try:
        client.create_dataset(
            name=dataset_name,
            description=f"Imported from {csv_path.name}",
            metadata={"source_file": str(csv_path), "type": "benchmark"},
        )
    except Exception:
        pass  # Dataset might exist

    success_count = 0
    for _, row in df.iterrows():
        expected_output = {col: str(row[col]) for col in ref_cols}
        metadata = {col: str(row[col]) for col in meta_cols}
        if "id" in df.columns:
            metadata["original_id"] = str(row["id"])

        # Retry logic for rate limits
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                # Use explicit ID for upsert behavior if available
                if "id" in df.columns:
                    item_id = f"{dataset_name}-{row['id']}"
                else:
                    item_id = None

                client.create_dataset_item(
                    dataset_name=dataset_name,
                    input=str(row[input_col]),
                    expected_output=expected_output,
                    metadata=metadata,
                    id=item_id,
                )
                success_count += 1

                # print(f"  Uploaded item {row.get('id', success_count)}", end="\r")
                time.sleep(0.1)
                break
            except Exception as e:
                # Basic rate limit check
                if "429" in str(e) and attempt < max_retries:
                    sleep_time = (attempt + 1) * 1.0
                    time.sleep(sleep_time)
                    continue
                else:
                    print(f"  Failed item: {e}")
                    pass  # Fail silently for existing items or permanent errors
                    break

    print(f"  ✅ Uploaded {success_count} items (processed)")


def main():
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        print("❌ Error: LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY not set.")
        return

    print(f"Connecting to Langfuse at {host}...")
    langfuse = Langfuse(public_key=public_key, secret_key=secret_key, host=host)

    data_dir = Path("data")
    if not data_dir.exists():
        print(f"❌ Error: Data directory not found at {data_dir}")
        return

    # Find all CSV files
    csv_files = sorted(list(data_dir.glob("*.csv")))

    # Filter out raw/crawled data
    csv_files = [
        f
        for f in csv_files
        if "raw" not in f.name.lower() and "crawled" not in f.name.lower()
    ]

    print(f"Found {len(csv_files)} CSV files in {data_dir} (excluding raw/crawled)")

    for csv_file in csv_files:
        upload_dataset(langfuse, csv_file)

    # Flush to ensure all events are sent
    langfuse.flush()


if __name__ == "__main__":
    main()
