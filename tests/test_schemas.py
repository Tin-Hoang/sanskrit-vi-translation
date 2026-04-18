"""Code-based grader: Pydantic schema validation."""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pydantic import ValidationError
from schemas import (
    TranslationItem,
    BatchTranslationResult,
    JudgementItem,
    BatchJudgementResult,
)


class TestTranslationItem:
    def test_minimal_valid(self):
        item = TranslationItem(translation="Xin chào")
        assert item.translation == "Xin chào"
        assert item.original_text is None
        assert item.thinking_process is None

    def test_full_fields(self):
        item = TranslationItem(
            translation="Xin chào",
            original_text="Hello",
            thinking_process="I thought about this...",
        )
        assert item.original_text == "Hello"

    def test_missing_translation_raises(self):
        with pytest.raises(ValidationError):
            TranslationItem()


class TestBatchTranslationResult:
    def test_single_item(self):
        result = BatchTranslationResult(
            translations=[TranslationItem(translation="Xin chào")]
        )
        assert len(result.translations) == 1

    def test_empty_list_allowed(self):
        result = BatchTranslationResult(translations=[])
        assert result.translations == []

    def test_from_dict(self):
        data = {"translations": [{"translation": "A"}, {"translation": "B"}]}
        result = BatchTranslationResult.model_validate(data)
        assert len(result.translations) == 2


class TestJudgementItem:
    def test_valid_scores(self):
        item = JudgementItem(accuracy=4.5, fluency=3.0, explanation="Good")
        assert item.accuracy == 4.5

    def test_score_boundary_min(self):
        item = JudgementItem(accuracy=1.0, fluency=1.0, explanation="Bad")
        assert item.accuracy == 1.0

    def test_score_boundary_max(self):
        item = JudgementItem(accuracy=5.0, fluency=5.0, explanation="Perfect")
        assert item.fluency == 5.0

    def test_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            JudgementItem(accuracy=6.0, fluency=3.0, explanation="Invalid")

    def test_below_range_raises(self):
        with pytest.raises(ValidationError):
            JudgementItem(accuracy=0.0, fluency=3.0, explanation="Invalid")


class TestBatchJudgementResult:
    def test_from_dict(self):
        data = {
            "evaluations": [
                {"accuracy": 4.0, "fluency": 5.0, "explanation": "Excellent"},
                {"accuracy": 2.0, "fluency": 3.0, "explanation": "Needs work"},
            ]
        }
        result = BatchJudgementResult.model_validate(data)
        assert len(result.evaluations) == 2
        assert result.evaluations[1].accuracy == 2.0
