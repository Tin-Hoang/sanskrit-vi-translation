from typing import List, Dict
import sacrebleu
from bert_score import score
import torch
import litellm
from dotenv import load_dotenv

load_dotenv()


class Evaluator:
    def __init__(self, judge_model: str = "groq/llama-3.3-70b-versatile"):
        self.judge_model = judge_model

    def calculate_metrics(
        self, references: List[List[str]], candidates: List[str]
    ) -> Dict[str, float]:
        # BLEU Score
        # references should be list of lists: [[ref1_all], [ref2_all], ...]
        bleu = sacrebleu.corpus_bleu(candidates, references)

        # BERTScore
        # checking if references is a list of lists or list of strings
        # For simplicity in this benchmark, we use the first reference list for BERTScore
        # or we could flatten/maximize, but let's stick to Ref 0 (Primary) for finding semantic similarity.
        primary_ref = references[0] if isinstance(references[0], list) else references

        # Suppress some warnings for cleaner output if desired, or handle device checks
        P, R, F1 = score(candidates, primary_ref, lang="vi", verbose=False)

        return {"BLEU": bleu.score, "BERTScore_F1": F1.mean().item()}

    def llm_judge(self, source: str, reference: str, candidate: str) -> Dict[str, any]:
        prompt = f"""
You are a professional translator and evaluator.
Rate the following Vietnamese translation of a Sanskrit text on a scale from 1 to 5 for:
1. Accuracy (Meaning preservation)
2. Fluency (Natural Vietnamese)

Rubric:
1: Completely incorrect, meaningless, or hallucinated.
2: Major errors, missing key information, or very awkward phrasing.
3: Generally preserves meaning but has noticeable errors or unnatural phrasing.
4: Accurate and readable, but with minor imperfections or slight awkwardness.
5: Perfect translation, professionally accurate and native-level fluency.

Source (Sanskrit): {source}
Reference (Vietnamese): {reference}
Candidate (Vietnamese): {candidate}

Provide the output in the following JSON format ONLY:
{{
  "accuracy": <number>,
  "fluency": <number>,
  "explanation": "<short explanation>"
}}
"""
        try:
            response = litellm.completion(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            return response.choices[0].message.content
            # Note: caller should parse this JSON string
        except Exception as e:
            print(f"Error in judge: {e}")
            return {}
