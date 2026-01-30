"""
Translator module for LLM-based translation.

Extends BaseLLMClient to provide translation-specific functionality
while inheriting shared LLM logic (retry, timing, batch processing).
"""

import json
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv

from llm_client import BaseLLMClient, llm_retry
from response_parser import validate_pydantic, fix_double_encoded_list, safe_json_parse
from schemas import BatchTranslationResult
from prompt_manager import render_prompt

load_dotenv()

# Optional Langfuse Import
try:
    from langfuse import observe, Langfuse
except ImportError:
    Langfuse = None

    def observe(**kwargs):
        def decorator(func):
            return func

        return decorator


class Translator(BaseLLMClient):
    """
    LLM-based translator for batch translation tasks.

    Inherits retry logic and batch processing from BaseLLMClient.
    Implements translation-specific prompt building and response parsing.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.3,
        single_prompt_template: Optional[str] = None,
        batch_prompt_template: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            api_base=api_base,
            api_key=api_key,
            extra_params=extra_params,
        )
        self._load_prompt_templates(single_prompt_template, batch_prompt_template)

    def _load_prompt_templates(
        self,
        single_prompt_template: Optional[str],
        batch_prompt_template: Optional[str],
    ) -> None:
        """Load default templates if not provided."""
        if not single_prompt_template or not batch_prompt_template:
            from system_prompts.translator.current import (
                SINGLE_TRANSLATE_PROMPT,
                BATCH_TRANSLATE_PROMPT,
            )

            self.single_prompt_template = (
                single_prompt_template or SINGLE_TRANSLATE_PROMPT
            )
            self.batch_prompt_template = batch_prompt_template or BATCH_TRANSLATE_PROMPT
        else:
            self.single_prompt_template = single_prompt_template
            self.batch_prompt_template = batch_prompt_template

    def _build_prompt(self, batch_items: List[str], source_lang: str, **kwargs) -> str:
        """
        Build translation prompt from batch items.

        Args:
            batch_items: List of texts to translate
            source_lang: Source language name

        Returns:
            Formatted prompt string
        """
        items_text = ""
        for i, text in enumerate(batch_items):
            items_text += f"\n--- Item {i + 1} ---\nText: {text}\n"

        return render_prompt(
            self.batch_prompt_template,
            source_lang=source_lang,
            items_text=items_text,
        )

    def _parse_response(self, content: str, batch_size: int) -> List[str]:
        """
        Parse BatchTranslationResult into list of translation strings.

        Args:
            content: Cleaned JSON response content
            batch_size: Expected number of translations

        Returns:
            List of translation strings
        """
        try:
            # Parse JSON and fix double-encoded strings if present
            data = safe_json_parse(content)
            data = fix_double_encoded_list(data, "translations")

            result = BatchTranslationResult.model_validate(data)
            translations = [item.translation for item in result.translations]

            # Pad if shorter than expected
            if len(translations) < batch_size:
                translations += [""] * (batch_size - len(translations))

            return translations[:batch_size]

        except Exception as e:
            print(f"Schema Validation Failed: {e}")
            print(f"Raw Output: {content[:200]}...")
            return [""] * batch_size

    @observe(as_type="generation")
    async def batch_translate(
        self,
        texts: List[str],
        source_lang: str = "Sanskrit",
        batch_size: int = 10,
        session_id: Optional[str] = None,
        dataset_item_ids: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ) -> tuple[List[str], float, List[Optional[str]]]:
        """
        Async batch translation with parallelism.

        Args:
            texts: List of texts to translate
            source_lang: Source language (default: Sanskrit)
            batch_size: Items per batch
            session_id: Optional session ID for tracking
            dataset_item_ids: Optional list of dataset item IDs
            dataset_name: Optional dataset name
            experiment_name: Optional experiment name

        Returns:
            Tuple of (translations, total_time, trace_ids)
        """
        # Get parent trace ID if available
        lf = Langfuse() if Langfuse else None
        current_trace_id = lf.get_current_trace_id() if lf else None

        translations, total_time = await self.process_batches(
            items=texts,
            batch_size=batch_size,
            session_id=session_id,
            trace_id=current_trace_id,
            default_result="",
            source_lang=source_lang,
        )

        # All items share the parent trace
        trace_ids = [current_trace_id] * len(texts)

        return translations, total_time, trace_ids
