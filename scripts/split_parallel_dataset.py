import pandas as pd
from pathlib import Path


def split_parallel_dataset():
    data_dir = Path("data")
    input_file = data_dir / "dhammapada_udanavarga_parallel.csv"

    if not input_file.exists():
        print(f"Error: {input_file} not found.")
        return

    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)

    # Common columns
    common_cols = ["id", "ref_viet_thichminhchau", "ref_viet_modern", "theme", "source"]

    # 1. Pali Dataset
    pali_cols = ["pali_text"] + common_cols
    df_pali = df[pali_cols].copy()
    output_pali = data_dir / "pali_vi_dhammapada.csv"
    df_pali.to_csv(output_pali, index=False)
    print(f"Created {output_pali} ({len(df_pali)} rows)")

    # 2. Sanskrit Dataset
    sanskrit_cols = ["sanskrit_text"] + common_cols
    df_sanskrit = df[sanskrit_cols].copy()
    output_sanskrit = data_dir / "sanskrit_vi_udanavarga.csv"
    df_sanskrit.to_csv(output_sanskrit, index=False)
    print(f"Created {output_sanskrit} ({len(df_sanskrit)} rows)")


if __name__ == "__main__":
    split_parallel_dataset()
