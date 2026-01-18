from typing import List, Dict
import json
import sacrebleu
from bert_score import score
import litellm
from dotenv import load_dotenv

load_dotenv()


class Evaluator:
    def __init__(self, judge_model: str = "gemini/gemini-3-flash-preview"):
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

    def batch_llm_judge(
        self,
        sources: List[str],
        references: List[str],
        candidates: List[str],
        source_lang: str = "Sanskrit",
        batch_size: int = 20,
    ) -> List[Dict[str, any]]:
        """
        Evaluate multiple translations in batched API calls to reduce RPM usage.

        Args:
            sources: List of source texts
            references: List of reference translations
            candidates: List of candidate translations
            source_lang: Source language name
            batch_size: Number of items to evaluate per API call (default: 5)

        Returns:
            List of judgement dictionaries for each item
        """
        all_results = []
        total = len(sources)

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_sources = sources[batch_start:batch_end]
            batch_refs = references[batch_start:batch_end]
            batch_candidates = candidates[batch_start:batch_end]

            print(
                f"  Judging batch {batch_start + 1}-{batch_end} of {total}...", end="\r"
            )

            # Build batch prompt
            items_text = ""
            for i, (src, ref, cand) in enumerate(
                zip(batch_sources, batch_refs, batch_candidates)
            ):
                items_text += f"""
--- Item {i + 1} ---
Source ({source_lang}): {src}
Reference (Vietnamese): {ref}
Candidate (Vietnamese): {cand}
"""

            prompt = f"""You are a professional translator and evaluator.
Rate EACH of the following Vietnamese translations of {source_lang} texts on a scale from 1 to 5 for:
1. Accuracy (Meaning preservation)
2. Fluency (Natural Vietnamese)

Rubric:
1: Completely incorrect, meaningless, or hallucinated.
2: Major errors, missing key information, or very awkward phrasing.
3: Generally preserves meaning but has noticeable errors or unnatural phrasing.
4: Accurate and readable, but with minor imperfections or slight awkwardness.
5: Perfect translation, professionally accurate and native-level fluency.

{items_text}

Provide the output as a JSON object with an "evaluations" array containing one object per item, in the SAME ORDER as the items above:
{{
  "evaluations": [
    {{"item": 1, "accuracy": <number>, "fluency": <number>, "explanation": "<short explanation>"}},
    {{"item": 2, "accuracy": <number>, "fluency": <number>, "explanation": "<short explanation>"}},
    ...
  ]
}}
"""
            try:
                response = litellm.completion(
                    model=self.judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )
                result_text = response.choices[0].message.content

                # Parse the batch response
                clean_json = (
                    result_text.replace("```json", "").replace("```", "").strip()
                )
                parsed = json.loads(clean_json)
                evaluations = parsed.get("evaluations", [])

                # Map results back to individual judgements
                for i, eval_item in enumerate(evaluations):
                    result_json = json.dumps(
                        {
                            "accuracy": eval_item.get("accuracy", 0),
                            "fluency": eval_item.get("fluency", 0),
                            "explanation": eval_item.get("explanation", ""),
                        }
                    )
                    all_results.append(result_json)

                # Handle case where fewer results returned than expected
                while len(all_results) < batch_end:
                    all_results.append(
                        '{"accuracy": 0, "fluency": 0, "explanation": "Batch parse error"}'
                    )

            except Exception as e:
                print(f"\nError in batch judge: {e}")
                # Fill with error results for this batch
                for _ in range(batch_end - batch_start):
                    all_results.append(
                        '{"accuracy": 0, "fluency": 0, "explanation": "API error"}'
                    )

        print()  # New line after progress
        return all_results

    def llm_judge(
        self, source: str, reference: str, candidate: str, source_lang: str = "Sanskrit"
    ) -> Dict[str, any]:
        """Single-item evaluation (kept for backwards compatibility)."""
        prompt = f"""
You are a professional translator and evaluator.
Rate the following Vietnamese translation of a {source_lang} text on a scale from 1 to 5 for:
1. Accuracy (Meaning preservation)
2. Fluency (Natural Vietnamese)

Rubric:
1: Completely incorrect, meaningless, or hallucinated.
2: Major errors, missing key information, or very awkward phrasing.
3: Generally preserves meaning but has noticeable errors or unnatural phrasing.
4: Accurate and readable, but with minor imperfections or slight awkwardness.
5: Perfect translation, professionally accurate and native-level fluency.

Source ({source_lang}): {source}
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
