"""
Script to upload local CSV datasets to Langfuse Datasets.

Usage:
    python scripts/upload_all_datasets_to_langfuse.py

Requirements:
    - LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST in .env
"""

import os
import time
import argparse
import logging
from pathlib import Path
from typing import Optional
import pandas as pd
from langfuse import Langfuse

try:
    from langfuse.api.core.api_error import ApiError
except ImportError:
    # Fallback if path changes in future versions
    class ApiError(Exception):
        pass


from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


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


def force_delete_dataset(client: Langfuse, dataset_name: str):
    """
    Force delete a dataset using the underlying HTTP client since the SDK
    does not explicitly expose a delete method for datasets.
    """
    print(f"  Force enabled: Deleting existing dataset '{dataset_name}'...")
    try:
        # Try finding the delete method on the API resource first (future proofing)
        if hasattr(client.api.datasets, "delete"):
            client.api.datasets.delete(name=dataset_name)
            time.sleep(1)
            return

        # Fallback: Use raw HTTP request via the internal client wrapper
        # This accesses private attributes, but is necessary if SDK doesn't expose it.
        # URL structure: /api/public/datasets/{dataset_name}
        # Encoding the dataset name is important
        import urllib.parse

        encoded_name = urllib.parse.quote(dataset_name)

        # Access the httpx client from the wrapper
        # client._client_wrapper.httpx_client is a standard httpx.Client
        if hasattr(client, "_client_wrapper") and hasattr(
            client._client_wrapper, "httpx_client"
        ):
            httpx_client = client._client_wrapper.httpx_client
            # The base_url is already configured in the client
            response = httpx_client.request(
                "DELETE", f"/api/public/datasets/{encoded_name}"
            )

            if response.status_code in [200, 204]:
                print(f"  Successfully deleted dataset '{dataset_name}'")
            elif response.status_code == 404:
                print(f"  Dataset '{dataset_name}' not found (already deleted)")
            else:
                print(
                    f"  Warning: Failed to delete dataset via HTTP: {response.status_code} - {response.text}"
                )
        else:
            print(
                "  Warning: Could not access underlying HTTP client to delete dataset."
            )

        time.sleep(1)  # Ensure propagation
    except Exception as e:
        print(f"  Warning: Failed to delete dataset '{dataset_name}': {e}")


def upload_dataset(client: Langfuse, csv_path: Path, force: bool = False):
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

    # Check existence to decide on force delete
    dataset_exists = False
    try:
        existing_dataset = client.get_dataset(dataset_name)
        if existing_dataset:
            dataset_exists = True
    except Exception:
        pass

    if force and dataset_exists:
        force_delete_dataset(client, dataset_name)
        dataset_exists = False  # Reset status

    if not force and dataset_exists:
        try:
            if existing_dataset.items and len(existing_dataset.items) >= len(df):
                print(
                    f"  ⚠️  Dataset '{dataset_name}' already exists with {len(existing_dataset.items)} items. Skipping (use --force to overwrite)."
                )
                return
            else:
                print(
                    f"  Note: Dataset exists but might be partial ({len(existing_dataset.items) if existing_dataset.items else 0}/{len(df)}). Merging/Upserting..."
                )
        except Exception:
            pass

    # Create dataset if not exists (or recreated)
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

        # Retry logic for rate limits with exponential backoff
        max_retries = 5
        base_delay = 1.0  # seconds

        for attempt in range(max_retries + 1):
            try:
                # Use explicit ID for upsert behavior if available
                if "id" in df.columns:
                    item_id = f"{dataset_name}-{row['id']}"
                else:
                    item_id = None

                # Format input as a dictionary matching the prompt template variables
                # Prompts use: {{text}}, {{source_lang}}, {{target_lang}}
                raw_text = str(row[input_col])

                # Detect source language from input column name
                source_lang = "Sanskrit"  # Default
                if "pali" in input_col.lower():
                    source_lang = "Pali"
                elif "viet" in input_col.lower():
                    source_lang = "Vietnamese"

                item_input = {
                    "text": raw_text,
                    "source_lang": source_lang,
                    "target_lang": "Vietnamese",
                }

                client.create_dataset_item(
                    dataset_name=dataset_name,
                    input=item_input,
                    expected_output=expected_output,
                    metadata=metadata,
                    id=item_id,
                )
                success_count += 1

                # Small delay to be nice to API
                time.sleep(0.1)
                break
            except Exception as e:
                # Check for rate limit (429)
                is_rate_limit = False
                if "429" in str(e):
                    is_rate_limit = True
                if hasattr(e, "status_code") and e.status_code == 429:
                    is_rate_limit = True

                if is_rate_limit and attempt < max_retries:
                    # Exponential backoff with jitter: 2, 4, 8, 16, 32, 64
                    sleep_time = 2.0 * (2**attempt)
                    print(
                        f"  Rate limit hit. Retrying in {sleep_time}s... (Attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(sleep_time)
                    continue
                else:
                    if attempt == max_retries:
                        print(f"  Failed item after {max_retries} retries: {e}")
                    else:
                        # Non-retriable error
                        print(f"  Failed item: {e}")
                    break

    print(f"  ✅ Uploaded {success_count} items (processed)")


def main():
    parser = argparse.ArgumentParser(description="Upload datasets to Langfuse")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force delete existing datasets before uploading",
    )
    args = parser.parse_args()

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
    if args.force:
        print("⚠️  Force mode enabled: Existing datasets will be deleted!")

    for csv_file in csv_files:
        upload_dataset(langfuse, csv_file, force=args.force)

    # Flush to ensure all events are sent
    langfuse.flush()


if __name__ == "__main__":
    main()
