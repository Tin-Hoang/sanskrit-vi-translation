import asyncio
from typing import List, Optional
import litellm
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from dotenv import load_dotenv
from litellm.exceptions import RateLimitError
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


class Translator:
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.3,
        single_prompt_template: Optional[str] = None,
        batch_prompt_template: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.api_base = api_base
        self.api_key = api_key

        # Load default templates if not provided
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

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
    )
    async def _process_batch(
        self,
        batch_texts: List[str],
        source_lang: str,
        session_id: Optional[str],
        trace_id: Optional[str],
    ) -> tuple[List[str], float]:
        """Process a single batch with retries and structured output."""

        # Prepare batch prompt
        items_text = ""
        for i, text in enumerate(batch_texts):
            items_text += f"\n--- Item {i + 1} ---\nText: {text}\n"

        prompt = render_prompt(
            self.batch_prompt_template, source_lang=source_lang, items_text=items_text
        )

        model_params = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            # Use json_object instead of strict schema class to be safer with Groq/OpenSource models
            "response_format": {"type": "json_object"},
            "metadata": {
                "session_id": session_id,
                "tags": ["translation", self.model_name, source_lang],
                "trace_id": trace_id,
            },
        }

        if self.api_base:
            model_params["api_base"] = self.api_base
        if self.api_key:
            model_params["api_key"] = self.api_key

        start_time = time.time()

        # Async LiteLLM Call
        response = await litellm.acompletion(**model_params)

        end_time = time.time()
        inference_time = end_time - start_time

        # Parse Pydantic Result
        try:
            # LiteLLM/Instructor integration usually returns the model instance directly
            # if we use certain wrappers, but standard litellm.acompletion returns a response object.
            # Using validation via json.loads is standard for raw structured outputs.
            content = response.choices[0].message.content

            # Additional cleanup for frequent JSON errors
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "")

            import json

            data = json.loads(content)

            # Fix double-encoded JSON strings in 'translations' list
            if "translations" in data and isinstance(data["translations"], list):
                fixed_list = []
                for item in data["translations"]:
                    if isinstance(item, str):
                        try:
                            fixed_list.append(json.loads(item))
                        except:
                            fixed_list.append({"translation": item})  # Fallback
                    else:
                        fixed_list.append(item)
                data["translations"] = fixed_list

            result = BatchTranslationResult.model_validate(data)

            # Map back to list of strings
            translations = [item.translation for item in result.translations]

            # Pad if shorter (shouldn't happen with strict schemas but safety first)
            if len(translations) < len(batch_texts):
                translations += [""] * (len(batch_texts) - len(translations))

            return translations[: len(batch_texts)], inference_time

        except Exception as e:
            print(f"Schema Validation Failed: {e}")
            print(f"Raw Output: {response.choices[0].message.content[:200]}...")
            # Fallback empty
            return [""] * len(batch_texts), inference_time

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
        """

        # Get parent trace ID if available
        lf = Langfuse() if Langfuse else None
        current_trace_id = lf.get_current_trace_id() if lf else None

        tasks = []
        batch_map = []  # Track which indices belong to which batch

        # 1. Create Tasks
        for i in range(0, len(texts), batch_size):
            batch_end = min(i + batch_size, len(texts))
            batch_texts = texts[i:batch_end]
            batch_map.append((i, batch_end))

            tasks.append(
                self._process_batch(
                    batch_texts=batch_texts,
                    source_lang=source_lang,
                    session_id=session_id,
                    trace_id=current_trace_id,
                )
            )

        print(f"Dispatching {len(tasks)} async batches for {len(texts)} items...")

        # 2. Run in Parallel
        # Use return_exceptions=True to prevent one batch failure from crashing everything
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 3. Aggregate Results
        all_translations = [""] * len(texts)
        total_time = 0.0
        all_trace_ids = [current_trace_id] * len(texts)  # Inherit parent trace

        for idx, result in enumerate(results):
            start_idx, end_idx = batch_map[idx]

            if isinstance(result, Exception):
                print(f"Batch {idx} Failed: {result}")
                # Empty strings already set as default
            else:
                translations, batch_time = result
                total_time += batch_time

                # Place in correct slots
                for k, trans in enumerate(translations):
                    if start_idx + k < len(all_translations):
                        all_translations[start_idx + k] = trans

        return all_translations, total_time, all_trace_ids
