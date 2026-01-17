import pandas as pd
import re
from pathlib import Path


def clean_text(text):
    # Remove numbering like (1), (2) but keep the text
    # Or maybe remove the whole line if it's just a number?
    # The user example: "nguồn mạch của Tuệ giác vô thượng (1), Người Tỉnh Thức"
    # We want to remove "(1)".
    text = re.sub(r"\(\d+\)", "", text)

    # Remove trailing/leading whitespace
    text = text.strip()
    return text


def is_noise(text):
    # Filter out obvious noise
    noise_patterns = [
        r"^last updated:",
        r"\(Traduction",
        r"\(cung cấp bởi",
        r"budsas\.net",
        r"^This document is written",
        r"^HTML",
        r"^BODY",
    ]
    for p in noise_patterns:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False


def process_data():
    base_dir = Path("data")
    raw_path = base_dir / "crawled_raw.csv"
    output_path = base_dir / "vietnamese_candidates.csv"

    if not raw_path.exists():
        print("Raw file not found.")
        return

    df = pd.read_csv(raw_path)

    # Segment boundaries (Heuristics based on observed file)
    versions = {
        "han_viet": {"start": "Quán Tự Tại Bồ Tát hành thâm Bát", "limit": 25},
        "modern": {"start": "Ngài Bồ Tát Quán Tự Tại khi", "limit": 25},
        "poetic": {
            "start": "Khi quán chiếu thâm sâu",
            "limit": 50,
        },  # Poetic uses short lines
        "scholarly": {
            "start": "nguồn mạch của Tuệ giác vô thượng (1), Người Tỉnh Thức",
            "limit": 40,
        },
    }

    all_lines = [
        str(x).strip()
        for x in df["raw_text"]
        if not is_noise(str(x)) and isinstance(x, str)
    ]

    extracted_versions = {}

    for v_name, indicators in versions.items():
        start_idx = -1
        for i, line in enumerate(all_lines):
            # Check partial match for robustness
            if indicators["start"] in line:
                start_idx = i
                break

        if start_idx != -1:
            # Extract and clean chunk
            raw_chunk = all_lines[start_idx : start_idx + indicators["limit"]]
            # Clean each line further if needed
            cleaned_chunk = [clean_text(l) for l in raw_chunk]
            extracted_versions[v_name] = cleaned_chunk
            print(f"Found version '{v_name}' starting at index {start_idx}")
        else:
            print(f"Warning: Could not find version '{v_name}'")

    # Save segmented data
    # Find max length to align in CSV columns (padding with empty strings)
    max_len = max((len(v) for v in extracted_versions.values()), default=0)

    rows = []
    for i in range(max_len):
        row = {}
        for v_name in versions.keys():
            content = extracted_versions.get(v_name, [])
            row[v_name] = content[i] if i < len(content) else ""
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)
    print(f"Saved segmented versions to {output_path}")
    print(out_df.head())


if __name__ == "__main__":
    process_data()
