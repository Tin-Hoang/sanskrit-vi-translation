import json
from typing import List
import litellm
import time
from dotenv import load_dotenv

load_dotenv()


class Translator:
    def __init__(self, model_name: str = "groq/llama-3.3-70b-versatile"):
        self.model_name = model_name

    def translate(
        self, text: str, source_lang: str = "Sanskrit", target_lang: str = "Vietnamese"
    ) -> str:
        """Single text translation (kept for backwards compatibility)."""
        prompt = f"""
Translate the following {source_lang} text into {target_lang}.
Provide ONLY the translation, no extra commentary.

Text: {text}
Translation:
"""
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
        self, texts: List[str], source_lang: str = "Sanskrit", batch_size: int = 10
    ) -> tuple[List[str], float]:
        """
        Translate multiple texts using batched API calls to reduce RPM usage.

        Args:
            texts: List of source texts to translate
            source_lang: Source language name
            batch_size: Number of texts to translate per API call (default: 10)

        Returns:
            Tuple of (translations list, total inference time in seconds)
        """
        all_translations = []
        total_inference_time = 0.0
        total = len(texts)

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_texts = texts[batch_start:batch_end]

            print(
                f"  Translating batch {batch_start + 1}-{batch_end} of {total}...",
                end="\r",
            )

            # Build batch prompt with indexed items
            items_text = ""
            for i, text in enumerate(batch_texts):
                items_text += f"\n--- Item {i + 1} ---\nText: {text}\n"

            prompt = f"""Translate each of the following {source_lang} texts into Vietnamese.
Provide ONLY the translations in JSON format.

{items_text}

Return a JSON object with a "translations" array containing each translation in ORDER:
{{
  "translations": [
    "{{"item": 1, "translation": "<Vietnamese translation of item 1>"}}",
    "{{"item": 2, "translation": "<Vietnamese translation of item 2>"}}",
    ...
  ]
}}
"""
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

                # Extract translations in order
                for item in translations:
                    if isinstance(item, dict):
                        all_translations.append(item.get("translation", ""))
                    elif isinstance(item, str):
                        all_translations.append(item)

                # Handle case where fewer translations returned than expected
                while len(all_translations) < batch_end:
                    all_translations.append("")

            except Exception as e:
                print(f"\nError in batch translate: {e}")
                # Fill with empty translations for this batch
                for _ in range(batch_end - batch_start):
                    all_translations.append("")

        print()  # New line after progress
        return all_translations, total_inference_time
