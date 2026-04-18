"""Code-based grader: utils module (identify_columns, load_data)."""
import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import identify_columns, load_data


class TestIdentifyColumns:
    def test_detects_pali_text(self):
        df = pd.DataFrame({"pali_text": ["a"], "ref_vi": ["b"]})
        input_col, ref_cols, _ = identify_columns(df)
        assert input_col == "pali_text"

    def test_detects_sanskrit_text(self):
        df = pd.DataFrame({"sanskrit_text": ["a"], "ref_vi": ["b"]})
        input_col, ref_cols, _ = identify_columns(df)
        assert input_col == "sanskrit_text"

    def test_pali_takes_priority_over_sanskrit(self):
        df = pd.DataFrame({"sanskrit_text": ["a"], "pali_text": ["b"]})
        input_col, _, _ = identify_columns(df)
        assert input_col == "pali_text"

    def test_detects_ref_columns_by_prefix(self):
        df = pd.DataFrame({"pali_text": ["a"], "ref_thich": ["b"], "ref_minh": ["c"]})
        _, ref_cols, _ = identify_columns(df)
        assert set(ref_cols) == {"ref_thich", "ref_minh"}

    def test_detects_vietnamese_ref_columns(self):
        df = pd.DataFrame({"pali_text": ["a"], "vietnamese_ref": ["b"]})
        _, ref_cols, _ = identify_columns(df)
        assert "vietnamese_ref" in ref_cols

    def test_fallback_to_second_column_when_id_present(self):
        df = pd.DataFrame({"id": [1], "content": ["text"], "ref_a": ["ref"]})
        input_col, _, _ = identify_columns(df)
        assert input_col == "content"

    def test_metadata_excludes_input_id_and_refs(self):
        df = pd.DataFrame({
            "pali_text": ["a"],
            "id": [1],
            "ref_vi": ["b"],
            "notes": ["c"],
        })
        _, _, metadata_cols = identify_columns(df)
        assert "notes" in metadata_cols
        assert "pali_text" not in metadata_cols
        assert "id" not in metadata_cols
        assert "ref_vi" not in metadata_cols


class TestLoadData:
    def test_loads_valid_csv(self, tmp_path):
        csv = tmp_path / "test.csv"
        csv.write_text("pali_text,ref_vi\nhello,world\n")
        df = load_data(csv)
        assert len(df) == 1
        assert "pali_text" in df.columns

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_data(tmp_path / "nonexistent.csv")
