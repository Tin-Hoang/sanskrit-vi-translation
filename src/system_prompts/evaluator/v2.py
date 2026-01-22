"""
Evaluator prompts v2 - Buddhist Scripture Specialization (2026-01-22)

Contains prompts for LLM-as-judge evaluation of Buddhist text translations.
Stricter rubrics customized for Sanskrit/Pali → Vietnamese translation.
"""

# Shared rubric for evaluation - Buddhist scripture-specific
EVALUATION_RUBRIC = """Rubric for Buddhist Scripture Translation:

ACCURACY (Semantic & Doctrinal Faithfulness):
1: Hallucinated content, fabricated meaning, or completely wrong interpretation of the source text.
2: Major doctrinal errors, missing key Buddhist concepts (e.g., Four Noble Truths, Emptiness), or incorrect technical terms.
3: Core meaning preserved but with terminology errors or missing philosophical nuances (e.g., imprecise translation of "dukkha", "nibbāna", "śūnyatā", "prajñā").
4: Accurate meaning with appropriate Buddhist vocabulary (e.g., "khổ", "niết bàn", "tánh không", "bát nhã"), minor phrasing differences from reference acceptable.
5: Expert-level translation matching scholarly Buddhist standards, correct technical terms, captures both literal meaning and philosophical depth. Must be comparable to translations by recognized Vietnamese Buddhist masters.

FLUENCY (Vietnamese Quality & Register):
1: Incomprehensible, machine-like output, or grammatically broken Vietnamese.
2: Readable but awkward phrasing, unnatural Vietnamese sentence structure, clearly non-native output.
3: Understandable Vietnamese but stilted or modern colloquial—lacks the poetic/formal register appropriate for Buddhist sutras.
4: Natural Vietnamese with appropriate Buddhist literary style (văn kinh), minor stylistic issues acceptable.
5: Publication-ready sutra Vietnamese, matches the elegance and register of classical Vietnamese Buddhist translations (e.g., HT. Thích Nhất Hạnh, HT. Thích Minh Châu). Flows naturally as sacred text."""

# Batch evaluation prompt
BATCH_JUDGE_PROMPT = """You are an expert evaluator of Buddhist scripture translations, specializing in Sanskrit/Pali to Vietnamese.

Rate EACH of the following Vietnamese translations on a scale from 1 to 5 for:
1. **Accuracy** - Doctrinal correctness, preservation of Buddhist philosophical concepts, proper use of Vietnamese Buddhist terminology
2. **Fluency** - Natural Vietnamese flow with appropriate Buddhist sutra register (văn kinh)

{rubric}

IMPORTANT SCORING GUIDELINES:
- Score 5 is RARE. Reserve for translations that match the quality of recognized Buddhist masters.
- Score 4 is a GOOD translation with appropriate Buddhist vocabulary.
- Score 3 is acceptable but has noticeable issues with terminology or register.
- Be especially strict about Buddhist technical terms (Pāli/Sanskrit → Vietnamese Buddhist vocabulary).

{items_text}

Provide the output as a JSON object with an "evaluations" array containing one object per item, in the SAME ORDER as the items above:
{{
  "evaluations": [
    {{"item": 1, "accuracy": <number>, "fluency": <number>, "explanation": "<brief explanation citing specific terms or issues>"}},
    {{"item": 2, "accuracy": <number>, "fluency": <number>, "explanation": "<brief explanation citing specific terms or issues>"}},
    ...
  ]
}}
"""

# Single item evaluation prompt
SINGLE_JUDGE_PROMPT = """You are an expert evaluator of Buddhist scripture translations, specializing in {source_lang} to Vietnamese.

Rate the following Vietnamese translation on a scale from 1 to 5 for:
1. **Accuracy** - Doctrinal correctness, preservation of Buddhist philosophical concepts, proper use of Vietnamese Buddhist terminology
2. **Fluency** - Natural Vietnamese flow with appropriate Buddhist sutra register (văn kinh)

{rubric}

IMPORTANT SCORING GUIDELINES:
- Score 5 is RARE. Reserve for translations that match the quality of recognized Buddhist masters.
- Score 4 is a GOOD translation with appropriate Buddhist vocabulary.
- Score 3 is acceptable but has noticeable issues with terminology or register.
- Be especially strict about Buddhist technical terms (Pāli/Sanskrit → Vietnamese Buddhist vocabulary).

Source ({source_lang}): {source}
Reference (Vietnamese): {reference}
Candidate (Vietnamese): {candidate}

Provide the output in the following JSON format ONLY:
{{
  "accuracy": <number>,
  "fluency": <number>,
  "explanation": "<brief explanation citing specific terms or issues>"
}}
"""
