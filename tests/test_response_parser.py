"""Code-based grader: response_parser module."""
import json
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from response_parser import (
    clean_json_response,
    safe_json_parse,
    validate_pydantic,
    fix_double_encoded_list,
)
from schemas import BatchTranslationResult, BatchJudgementResult


class TestCleanJsonResponse:
    def test_strips_json_code_block(self):
        raw = '```json\n{"key": "value"}\n```'
        assert clean_json_response(raw) == '{"key": "value"}'

    def test_strips_plain_code_block(self):
        raw = '```\n{"key": "value"}\n```'
        assert clean_json_response(raw) == '{"key": "value"}'

    def test_passthrough_clean_json(self):
        raw = '{"key": "value"}'
        assert clean_json_response(raw) == '{"key": "value"}'

    def test_strips_leading_trailing_whitespace(self):
        raw = '  \n{"key": "value"}\n  '
        assert clean_json_response(raw) == '{"key": "value"}'

    def test_strips_json_block_without_trailing_backticks(self):
        raw = '```json\n{"key": "value"}'
        result = clean_json_response(raw)
        assert '```' not in result
        assert '"key"' in result


class TestSafeJsonParse:
    def test_parses_plain_json(self):
        result = safe_json_parse('{"a": 1}')
        assert result == {"a": 1}

    def test_parses_markdown_wrapped_json(self):
        result = safe_json_parse('```json\n{"a": 1}\n```')
        assert result == {"a": 1}

    def test_raises_on_invalid_json(self):
        with pytest.raises(json.JSONDecodeError):
            safe_json_parse("not json at all")


class TestFixDoubleEncodedList:
    def test_fixes_string_items_in_list(self):
        data = {"translations": ['{"translation": "hello"}', '{"translation": "world"}']}
        result = fix_double_encoded_list(data, "translations")
        assert result["translations"] == [
            {"translation": "hello"},
            {"translation": "world"},
        ]

    def test_passthrough_already_dict_items(self):
        data = {"translations": [{"translation": "hello"}]}
        result = fix_double_encoded_list(data, "translations")
        assert result["translations"] == [{"translation": "hello"}]

    def test_passthrough_missing_key(self):
        data = {"other": [1, 2, 3]}
        result = fix_double_encoded_list(data, "translations")
        assert result == {"other": [1, 2, 3]}

    def test_wraps_invalid_json_strings(self):
        data = {"translations": ["not-json"]}
        result = fix_double_encoded_list(data, "translations")
        assert result["translations"] == [{"value": "not-json"}]


class TestValidatePydantic:
    def test_validates_batch_translation_result(self):
        payload = json.dumps(
            {"translations": [{"translation": "Xin chào"}, {"translation": "Tạm biệt"}]}
        )
        result = validate_pydantic(payload, BatchTranslationResult)
        assert len(result.translations) == 2
        assert result.translations[0].translation == "Xin chào"

    def test_validates_batch_judgement_result(self):
        payload = json.dumps(
            {
                "evaluations": [
                    {"accuracy": 4.0, "fluency": 5.0, "explanation": "Good translation"}
                ]
            }
        )
        result = validate_pydantic(payload, BatchJudgementResult)
        assert result.evaluations[0].accuracy == 4.0
        assert result.evaluations[0].fluency == 5.0

    def test_raises_on_schema_mismatch(self):
        from pydantic import ValidationError
        payload = json.dumps({"translations": [{"wrong_field": "oops"}]})
        with pytest.raises(ValidationError):
            validate_pydantic(payload, BatchTranslationResult)

    def test_validates_markdown_wrapped_payload(self):
        payload = '```json\n{"translations": [{"translation": "Hòa bình"}]}\n```'
        result = validate_pydantic(payload, BatchTranslationResult)
        assert result.translations[0].translation == "Hòa bình"
