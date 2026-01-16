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
        self, references: List[str], candidates: List[str]
    ) -> Dict[str, float]:
        # BLEU Score
        bleu = sacrebleu.corpus_bleu(candidates, [references])

        # BERTScore
        # Suppress some warnings for cleaner output if desired, or handle device checks
        P, R, F1 = score(candidates, references, lang="vi", verbose=False)

        return {"BLEU": bleu.score, "BERTScore_F1": F1.mean().item()}

    def llm_judge(self, source: str, reference: str, candidate: str) -> Dict[str, any]:
        prompt = f"""
You are a professional translator and evaluator.
Rate the following Vietnamese translation of a Sanskrit text on a scale from 1 to 5 (5 is best) for:
1. Accuracy (meaning preservation)
2. Fluency (natural Vietnamese)

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
