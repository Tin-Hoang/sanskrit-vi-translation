"""
Translator prompts v1 - Initial version (2026-01-18)

Contains prompts for single and batch translation of Buddhist texts.
"""

# Single text translation prompt
# Single text translation prompt
SINGLE_TRANSLATE_PROMPT = """Translate the following {{source_lang}} text into {{target_lang}}.
Provide ONLY the translation, no extra commentary.

Text: {{text}}
Translation:
"""

# Batch translation prompt with JSON output
BATCH_TRANSLATE_PROMPT = """Translate each of the following {{source_lang}} texts into Vietnamese.
Provide ONLY the translations in JSON format.

{{items_text}}

Return a JSON object with a "translations" array containing each translation in ORDER:
{
  "translations": [
    {"item": 1, "translation": "<Vietnamese translation of item 1>"},
    {"item": 2, "translation": "<Vietnamese translation of item 2>"},
    ...
  ]
}
"""
