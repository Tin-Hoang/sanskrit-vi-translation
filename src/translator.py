import json
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from cache import BenchmarkCache

from cache import BenchmarkCache
import litellm
from litellm.exceptions import RateLimitError, BadRequestError
import time
from dotenv import load_dotenv

load_dotenv()


class Translator:
    def __init__(
        self,
        model_name: str = "groq/llama-3.3-70b-versatile",
        single_prompt_template: Optional[str] = None,
        batch_prompt_template: Optional[str] = None,
    ):
        self.model_name = model_name
        # Use provided templates or load from current (fallback logic to be safe)
        if not single_prompt_template or not batch_prompt_template:
            from system_prompts.translator.current import (
                SINGLE_TRANSLATE_PROMPT,
                BATCH_TRANSLATE_PROMPT,
            )

        self.single_prompt_template = single_prompt_template or SINGLE_TRANSLATE_PROMPT
        self.batch_prompt_template = batch_prompt_template or BATCH_TRANSLATE_PROMPT

    def translate(
        self,
        text: str,
        source_lang: str = "Sanskrit",
        target_lang: str = "Vietnamese",
        session_id: Optional[str] = None,
    ) -> str:
        """Single text translation (kept for backwards compatibility)."""
        prompt = self.single_prompt_template.format(
            source_lang=source_lang,
            target_lang=target_lang,
            text=text,
        )
        try:
            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                metadata={
                    "session_id": session_id,
                    "tags": ["translation", self.model_name, source_lang],
                },
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error translating text: {e}")
            return ""

    def batch_translate(
        self,
        texts: List[str],
        source_lang: str = "Sanskrit",
        batch_size: int = 10,
        cache: Optional["BenchmarkCache"] = None,
        session_id: Optional[str] = None,
    ) -> tuple[List[str], float]:
        """
        Translate multiple texts using batched API calls to reduce RPM usage.

        Args:
            texts: List of source texts to translate
            source_lang: Source language name
            batch_size: Number of texts to translate per API call (default: 10)
            cache: Optional BenchmarkCache instance for caching translations

        Returns:
            Tuple of (translations list, total inference time in seconds)
        """
        all_translations: List[Optional[str]] = [None] * len(texts)
        total_inference_time = 0.0
        cached_time = 0.0
        total = len(texts)
        cache_hits = 0

        # First pass: check cache for existing translations
        texts_to_translate: List[tuple[int, str]] = []  # (original_index, text)
        for idx, text in enumerate(texts):
            if cache:
                cached = cache.get_translation(self.model_name, text)
                if cached is not None:
                    translation, time_seconds = cached
                    all_translations[idx] = translation
                    cached_time += time_seconds
                    cache_hits += 1
                    continue
            texts_to_translate.append((idx, text))

        if cache_hits > 0:
            print(
                f"  Cache: {cache_hits}/{total} translations cached, {len(texts_to_translate)} to translate"
            )

        if not texts_to_translate:
            # All translations were cached - return accumulated cached time
            return [t if t is not None else "" for t in all_translations], cached_time

        # Second pass: batch translate remaining texts
        for batch_start in range(0, len(texts_to_translate), batch_size):
            batch_end = min(batch_start + batch_size, len(texts_to_translate))
            batch_items = texts_to_translate[batch_start:batch_end]
            batch_texts = [item[1] for item in batch_items]

            progress_start = batch_start + cache_hits + 1
            progress_end = batch_end + cache_hits
            print(
                f"  Translating batch {progress_start}-{progress_end} of {total}...",
                end="\r",
            )

            # Build batch prompt with indexed items
            items_text = ""
            for i, text in enumerate(batch_texts):
                items_text += f"\n--- Item {i + 1} ---\nText: {text}\n"

            prompt = self.batch_prompt_template.format(
                source_lang=source_lang,
                items_text=items_text,
            )
            # Retry logic for rate limits: 60s, 120s, 180s
            retry_delays = [60, 120, 180]
            max_retries = len(retry_delays)
            response = None

            use_json_mode = True
            for attempt in range(max_retries + 1):
                try:
                    start_time = time.time()
                    completion_kwargs = {
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3,
                        "metadata": {
                            "session_id": session_id,
                            "tags": ["translation", self.model_name, source_lang],
                        },
                    }
                    if use_json_mode:
                        completion_kwargs["response_format"] = {"type": "json_object"}
                    response = litellm.completion(**completion_kwargs)
                    end_time = time.time()
                    total_inference_time += end_time - start_time
                    break  # Success, exit retry loop
                except BadRequestError as e:
                    # Handle JSON validation failures (e.g., Kimi-k2 on Groq)
                    if "json_validate_failed" in str(e) and use_json_mode:
                        print(
                            f"\n  JSON mode failed for {self.model_name}, retrying without JSON mode..."
                        )
                        use_json_mode = False
                        continue  # Retry without JSON mode
                    else:
                        print(f"\nFailed to translate with {self.model_name}: {e}")
                        raise
                except RateLimitError as e:
                    if attempt < max_retries:
                        wait_time = retry_delays[attempt]
                        print(
                            f"\n  Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}..."
                        )
                        time.sleep(wait_time)
                    else:
                        print(f"\n  Rate limit exceeded after {max_retries} retries.")
                        raise

            if response is None:
                raise RuntimeError("No response received after retries")

            try:
                result_text = response.choices[0].message.content
                clean_json = (
                    result_text.replace("```json", "").replace("```", "").strip()
                )

                # Try to parse as JSON first
                try:
                    parsed = json.loads(clean_json)
                    if isinstance(parsed, list):
                        translations = parsed
                    else:
                        translations = parsed.get("translations", [])
                except json.JSONDecodeError:
                    # If that fails, try to extract JSON from the response
                    import re

                    json_match = re.search(
                        r'\{[\s\S]*"translations"[\s\S]*\}', clean_json
                    )
                    if json_match:
                        try:
                            parsed = json.loads(json_match.group())
                            translations = parsed.get("translations", [])
                        except json.JSONDecodeError:
                            translations = []
                    else:
                        translations = []

                # Extract translations and map back to original indices
                # Calculate time per item for this batch
                batch_time = end_time - start_time
                time_per_item = batch_time / len(batch_texts) if batch_texts else 0.0

                for i, item in enumerate(translations):
                    if i >= len(batch_items):
                        break
                    original_idx = batch_items[i][0]
                    original_text = batch_items[i][1]

                    if isinstance(item, dict):
                        translation = item.get("translation", "")
                    elif isinstance(item, str):
                        # Try to parse as JSON in case model returned stringified objects
                        try:
                            parsed_item = json.loads(item)
                            if isinstance(parsed_item, dict):
                                translation = parsed_item.get("translation", item)
                            else:
                                translation = item
                        except (json.JSONDecodeError, TypeError):
                            translation = item
                    else:
                        translation = ""

                    all_translations[original_idx] = translation

                    # Cache the result with proportional time
                    if cache and translation:
                        cache.set_translation(
                            self.model_name, original_text, translation, time_per_item
                        )

                # Handle case where fewer translations returned than expected
                for i in range(len(translations), len(batch_items)):
                    original_idx = batch_items[i][0]
                    all_translations[original_idx] = ""

            except Exception as e:
                print(f"\nError in batch translate: {e}")
                # Fill with empty translations for this batch
                for original_idx, _ in batch_items:
                    if all_translations[original_idx] is None:
                        all_translations[original_idx] = ""

        print()  # New line after progress
        # Return total time: fresh API calls + cached original times
        return [
            t if t is not None else "" for t in all_translations
        ], total_inference_time + cached_time
