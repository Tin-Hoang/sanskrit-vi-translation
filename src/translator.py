import os
from typing import List, Dict, Optional
import litellm
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


class Translator:
    def __init__(self, model_name: str = "groq/llama-3.3-70b-versatile"):
        self.model_name = model_name

    def translate(
        self, text: str, source_lang: str = "Sanskrit", target_lang: str = "Vietnamese"
    ) -> str:
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

    def batch_translate(self, texts: List[str]) -> List[str]:
        translations = []
        for text in texts:
            translations.append(self.translate(text))
            import time

            time.sleep(3)  # Rate limit: 30 RPM = 1 per 2s. Using 3s to be safe.
        return translations
