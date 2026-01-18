import json
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from cache import BenchmarkCache
import litellm
import time
from dotenv import load_dotenv

from system_prompts.translator.current import (
    SINGLE_TRANSLATE_PROMPT,
    BATCH_TRANSLATE_PROMPT,
)

load_dotenv()


class Translator:
    def __init__(self, model_name: str = "groq/llama-3.3-70b-versatile"):
        self.model_name = model_name

    def translate(
        self, text: str, source_lang: str = "Sanskrit", target_lang: str = "Vietnamese"
    ) -> str:
        """Single text translation (kept for backwards compatibility)."""
        prompt = SINGLE_TRANSLATE_PROMPT.format(
            source_lang=source_lang,
            target_lang=target_lang,
            text=text,
        )
        try:
            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
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
        total = len(texts)
        cache_hits = 0

        # First pass: check cache for existing translations
        texts_to_translate: List[tuple[int, str]] = []  # (original_index, text)
        for idx, text in enumerate(texts):
            if cache:
                cached = cache.get_translation(self.model_name, text)
                if cached is not None:
                    all_translations[idx] = cached
                    cache_hits += 1
                    continue
            texts_to_translate.append((idx, text))

        if cache_hits > 0:
            print(
                f"  Cache: {cache_hits}/{total} translations cached, {len(texts_to_translate)} to translate"
            )

        if not texts_to_translate:
            # All translations were cached
            return [t if t is not None else "" for t in all_translations], 0.0

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

            prompt = BATCH_TRANSLATE_PROMPT.format(
                source_lang=source_lang,
                items_text=items_text,
            )
            try:
                start_time = time.time()
                response = litellm.completion(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                )
                end_time = time.time()
                total_inference_time += end_time - start_time

                result_text = response.choices[0].message.content
                clean_json = (
                    result_text.replace("```json", "").replace("```", "").strip()
                )
                parsed = json.loads(clean_json)
                translations = parsed.get("translations", [])

                # Extract translations and map back to original indices
                for i, item in enumerate(translations):
                    if i >= len(batch_items):
                        break
                    original_idx = batch_items[i][0]
                    original_text = batch_items[i][1]

                    if isinstance(item, dict):
                        translation = item.get("translation", "")
                    elif isinstance(item, str):
                        translation = item
                    else:
                        translation = ""

                    all_translations[original_idx] = translation

                    # Cache the result
                    if cache and translation:
                        cache.set_translation(
                            self.model_name, original_text, translation
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
        return [
            t if t is not None else "" for t in all_translations
        ], total_inference_time
