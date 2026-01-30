"""
Response parsing utilities for LLM JSON responses.

Provides:
- JSON cleanup (markdown code block removal)
- Safe JSON parsing with error handling
- Pydantic model validation
"""

import json
from typing import Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def clean_json_response(content: str) -> str:
    """
    Strip markdown code blocks from LLM JSON responses.

    Many LLMs wrap JSON in ```json ... ``` blocks. This function
    removes those wrappers to get the raw JSON string.

    Args:
        content: Raw LLM response content

    Returns:
        Cleaned JSON string without markdown formatting
    """
    content = content.strip()

    if content.startswith("```json"):
        content = content[7:]  # Remove ```json
        if content.endswith("```"):
            content = content[:-3]
    elif content.startswith("```"):
        content = content[3:]  # Remove ```
        if content.endswith("```"):
            content = content[:-3]

    return content.strip()


def safe_json_parse(content: str) -> dict:
    """
    Parse JSON with automatic markdown cleanup.

    Args:
        content: Raw or markdown-wrapped JSON string

    Returns:
        Parsed dictionary

    Raises:
        json.JSONDecodeError: If parsing fails after cleanup
    """
    cleaned = clean_json_response(content)
    return json.loads(cleaned)


def validate_pydantic(content: str, model_class: Type[T]) -> T:
    """
    Parse and validate JSON response against a Pydantic model.

    Args:
        content: Raw or markdown-wrapped JSON string
        model_class: Pydantic model class to validate against

    Returns:
        Validated Pydantic model instance

    Raises:
        ValidationError: If the data doesn't match the schema
        json.JSONDecodeError: If JSON parsing fails
    """
    data = safe_json_parse(content)
    return model_class.model_validate(data)


def fix_double_encoded_list(data: dict, key: str) -> dict:
    """
    Fix double-encoded JSON strings in a list field.

    Some LLMs return lists where items are JSON strings instead of objects.
    This function detects and fixes that pattern.

    Args:
        data: Dictionary containing the list
        key: Key of the list field to fix

    Returns:
        Dictionary with fixed list (items as dicts, not strings)
    """
    if key not in data or not isinstance(data[key], list):
        return data

    fixed_list = []
    for item in data[key]:
        if isinstance(item, str):
            try:
                fixed_list.append(json.loads(item))
            except json.JSONDecodeError:
                # If it's not valid JSON, wrap it as a simple value
                fixed_list.append({"value": item})
        else:
            fixed_list.append(item)

    data[key] = fixed_list
    return data
