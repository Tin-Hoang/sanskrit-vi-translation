"""
Evaluator prompts v1 - Initial version (2026-01-18)

Contains prompts for LLM-as-judge evaluation of translations.
"""

# Shared rubric for evaluation
EVALUATION_RUBRIC = """Rubric:
1: Completely incorrect, meaningless, or hallucinated.
2: Major errors, missing key information, or very awkward phrasing.
3: Generally preserves meaning but has noticeable errors or unnatural phrasing.
4: Accurate and readable, but with minor imperfections or slight awkwardness.
5: Perfect translation, professionally accurate and native-level fluency."""

# Batch evaluation prompt
BATCH_JUDGE_PROMPT = """You are a professional translator and evaluator.
Rate EACH of the following Vietnamese translations of {{source_lang}} texts on a scale from 1 to 5 for:
1. Accuracy (Meaning preservation)
2. Fluency (Natural Vietnamese)

{{rubric}}

{{items_text}}

Provide the output as a JSON object with an "evaluations" array containing one object per item, in the SAME ORDER as the items above:
{
  "evaluations": [
    {"item": 1, "accuracy": <number>, "fluency": <number>, "explanation": "<short explanation>"},
    {"item": 2, "accuracy": <number>, "fluency": <number>, "explanation": "<short explanation>"},
    ...
  ]
}
"""

# Single item evaluation prompt
SINGLE_JUDGE_PROMPT = """You are a professional translator and evaluator.
Rate the following Vietnamese translation of a {{source_lang}} text on a scale from 1 to 5 for:
1. Accuracy (Meaning preservation)
2. Fluency (Natural Vietnamese)

{{rubric}}

Source ({{source_lang}}): {{text}}
Reference (Vietnamese): {{reference}}
Candidate (Vietnamese): {{candidate}}

Provide the output in the following JSON format ONLY:
{
  "accuracy": <number>,
  "fluency": <number>,
  "explanation": "<short explanation>"
}
"""
