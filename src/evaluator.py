from typing import List, Dict, Optional, TYPE_CHECKING

import json
import sacrebleu
from bert_score import score
import numpy as np
import litellm
from litellm.exceptions import RateLimitError
import time
from dotenv import load_dotenv

load_dotenv()


class Evaluator:
    def __init__(
        self,
        judge_model: str = "gemini/gemini-2.5-flash",
        temperature: float = 0.1,
        rubric: Optional[str] = None,
        single_judge_prompt_template: Optional[str] = None,
        batch_judge_prompt_template: Optional[str] = None,
    ):
        self.judge_model = judge_model
        self.temperature = temperature

        # Fallback to local prompts if not provided
        if (
            not rubric
            or not single_judge_prompt_template
            or not batch_judge_prompt_template
        ):
            from system_prompts.evaluator.current import (
                EVALUATION_RUBRIC,
                BATCH_JUDGE_PROMPT,
                SINGLE_JUDGE_PROMPT,
            )

        self.rubric = rubric or EVALUATION_RUBRIC
        self.single_judge_prompt_template = (
            single_judge_prompt_template or SINGLE_JUDGE_PROMPT
        )
        self.batch_judge_prompt_template = (
            batch_judge_prompt_template or BATCH_JUDGE_PROMPT
        )

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

    def calculate_item_metrics(
        self, references: List[List[str]], candidates: List[str]
    ) -> Dict[str, List[float]]:
        """
        Calculate per-item BLEU and BERTScore metrics.

        Args:
            references: List of reference lists, where each inner list contains
                       all reference translations for that position
            candidates: List of candidate translations

        Returns:
            Dictionary with 'BLEU' and 'BERTScore_F1' keys, each containing
            a list of scores per item
        """
        # Per-item sentence BLEU scores
        sentence_bleu_scores = []
        for i, cand in enumerate(candidates):
            # Get references for this item from each reference list
            item_refs = [ref_list[i] for ref_list in references if i < len(ref_list)]
            if item_refs and cand and cand.strip():
                # sacrebleu.sentence_bleu expects refs as list of strings
                sent_bleu = sacrebleu.sentence_bleu(cand, item_refs)
                sentence_bleu_scores.append(sent_bleu.score)
            else:
                sentence_bleu_scores.append(0.0)

        # Per-item BERTScore (already returns per-item scores)
        primary_ref = references[0] if isinstance(references[0], list) else references
        P, R, F1 = score(candidates, primary_ref, lang="vi", verbose=False)
        bertscore_f1_list = F1.tolist()

        return {
            "BLEU": sentence_bleu_scores,
            "BERTScore_F1": bertscore_f1_list,
        }

    def batch_llm_judge(
        self,
        sources: List[str],
        references: List[str],
        candidates: List[str],
        source_lang: str = "Sanskrit",
        batch_size: int = 20,
        model_id: str = "",
        session_id: Optional[str] = None,
    ) -> List[Dict[str, any]]:
        """
        Evaluate multiple translations in batched API calls to reduce RPM usage.

        Args:
            sources: List of source texts
            references: List of reference translations
            candidates: List of candidate translations
            source_lang: Source language name
            batch_size: Number of items to evaluate per API call (default: 20)
            cache: Optional BenchmarkCache instance for caching judgements
            model_id: Model that produced the translations (for cache key)

        Returns:
            List of judgement dictionaries for each item
        """
        all_results: List[Optional[str]] = [None] * len(sources)
        total = len(sources)
        # Prepare items for judging (LiteLLM handles caching)
        items_to_judge: List[tuple[int, str, str, str]] = []  # (idx, src, ref, cand)
        for idx, (src, ref, cand) in enumerate(zip(sources, references, candidates)):
            items_to_judge.append((idx, src, ref, cand))

        if not items_to_judge:
            return []

        if not items_to_judge:
            # All judgements were cached
            return [
                r
                if r is not None
                else '{"accuracy": 0, "fluency": 0, "explanation": "Missing"}'
                for r in all_results
            ]

        # Second pass: batch judge remaining items
        for batch_start in range(0, len(items_to_judge), batch_size):
            batch_end = min(batch_start + batch_size, len(items_to_judge))
            batch_items = items_to_judge[batch_start:batch_end]

            progress_start = batch_start + 1
            progress_end = batch_end
            print(
                f"  Judging batch {progress_start}-{progress_end} of {total}...",
                end="\r",
            )

            # Build batch prompt
            items_text = ""
            for i, (_, src, ref, cand) in enumerate(batch_items):
                items_text += f"""
--- Item {i + 1} ---
Source ({source_lang}): {src}
Reference (Vietnamese): {ref}
Candidate (Vietnamese): {cand}
"""

            prompt = self.batch_judge_prompt_template.format(
                source_lang=source_lang,
                rubric=self.rubric,
                items_text=items_text,
            )
            # Retry logic for rate limits: 60s, 120s, 180s
            retry_delays = [60, 120, 180]
            max_retries = len(retry_delays)
            response = None

            for attempt in range(max_retries + 1):
                try:
                    response = litellm.completion(
                        model=self.judge_model,
                        messages=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"},
                        temperature=self.temperature,
                        metadata={
                            "session_id": session_id,
                            "tags": ["evaluation", self.judge_model, source_lang],
                        },
                    )
                    break  # Success, exit retry loop
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

                # Parse the batch response
                clean_json = (
                    result_text.replace("```json", "").replace("```", "").strip()
                )
                parsed = json.loads(clean_json)
                if isinstance(parsed, list):
                    evaluations = parsed
                else:
                    evaluations = parsed.get("evaluations", [])

                # Map results back to original indices
                for i, eval_item in enumerate(evaluations):
                    if i >= len(batch_items):
                        break
                    original_idx, src, ref, cand = batch_items[i]
                    result_json = json.dumps(
                        {
                            "accuracy": eval_item.get("accuracy", 0),
                            "fluency": eval_item.get("fluency", 0),
                            "explanation": eval_item.get("explanation", ""),
                        }
                    )
                    all_results[original_idx] = result_json

                # Handle case where fewer results returned than expected
                for i in range(len(evaluations), len(batch_items)):
                    original_idx = batch_items[i][0]
                    all_results[original_idx] = (
                        '{"accuracy": 0, "fluency": 0, "explanation": "Batch parse error"}'
                    )

            except Exception as e:
                print(f"\nError in batch judge: {e}")
                # Fill with error results for this batch
                for original_idx, _, _, _ in batch_items:
                    if all_results[original_idx] is None:
                        all_results[original_idx] = (
                            '{"accuracy": 0, "fluency": 0, "explanation": "API error"}'
                        )

        print()  # New line after progress
        return [
            r
            if r is not None
            else '{"accuracy": 0, "fluency": 0, "explanation": "Missing"}'
            for r in all_results
        ]

    def llm_judge(
        self,
        source: str,
        reference: str,
        candidate: str,
        source_lang: str = "Sanskrit",
        session_id: Optional[str] = None,
    ) -> Dict[str, any]:
        """Single-item evaluation (kept for backwards compatibility)."""
        prompt = self.single_judge_prompt_template.format(
            source_lang=source_lang,
            rubric=self.rubric,
            source=source,
            reference=reference,
            candidate=candidate,
        )
        try:
            response = litellm.completion(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=self.temperature,
                metadata={
                    "session_id": session_id,
                    "tags": ["evaluation", self.judge_model, source_lang],
                },
            )
            return response.choices[0].message.content
            # Note: caller should parse this JSON string
        except Exception as e:
            print(f"Error in judge: {e}")
            return {}
