"""
Evaluator module for LLM-based translation quality evaluation.

Extends BaseLLMClient to provide evaluation-specific functionality
while inheriting shared LLM logic (retry, timing, batch processing).
Also provides automated metrics (BLEU, BERTScore).
"""

import json
from typing import List, Dict, Optional, Any

import sacrebleu
from bert_score import score
from dotenv import load_dotenv

from llm_client import BaseLLMClient
from response_parser import validate_pydantic
from schemas import BatchJudgementResult
from prompt_manager import render_prompt

load_dotenv()


class Evaluator(BaseLLMClient):
    """
    LLM-based evaluator for translation quality assessment.

    Provides:
    - Automated metrics (BLEU, BERTScore)
    - LLM judge evaluation for accuracy and fluency

    Inherits retry logic and batch processing from BaseLLMClient.
    """

    def __init__(
        self,
        judge_model: str = "gemini/gemini-2.5-flash",
        temperature: float = 0.1,
        rubric: Optional[str] = None,
        single_judge_prompt_template: Optional[str] = None,
        batch_judge_prompt_template: Optional[str] = None,
    ):
        super().__init__(
            model_name=judge_model,
            temperature=temperature,
        )
        self._load_prompt_templates(
            rubric, single_judge_prompt_template, batch_judge_prompt_template
        )

    def _load_prompt_templates(
        self,
        rubric: Optional[str],
        single_judge_prompt_template: Optional[str],
        batch_judge_prompt_template: Optional[str],
    ) -> None:
        """Load default templates if not provided."""
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
        else:
            self.rubric = rubric
            self.single_judge_prompt_template = single_judge_prompt_template
            self.batch_judge_prompt_template = batch_judge_prompt_template

    # =========================================================================
    # Automated Metrics (BLEU, BERTScore)
    # =========================================================================

    def calculate_metrics(
        self, references: List[List[str]], candidates: List[str]
    ) -> Dict[str, float]:
        """
        Calculate corpus-level BLEU and BERTScore.

        Args:
            references: List of reference lists (multiple references per item)
            candidates: List of candidate translations

        Returns:
            Dictionary with BLEU and BERTScore_F1 scores
        """
        # BLEU Score
        bleu = sacrebleu.corpus_bleu(candidates, references)

        # BERTScore
        primary_ref = references[0] if isinstance(references[0], list) else references

        # Calculate BERTScore (sync, CPU/GPU bound)
        P, R, F1 = score(candidates, primary_ref, lang="vi", verbose=False)

        return {"BLEU": bleu.score, "BERTScore_F1": F1.mean().item()}

    def calculate_item_metrics(
        self, references: List[List[str]], candidates: List[str]
    ) -> Dict[str, List[float]]:
        """
        Calculate per-item BLEU and BERTScore.

        Args:
            references: List of reference lists
            candidates: List of candidate translations

        Returns:
            Dictionary with lists of per-item scores
        """
        sentence_bleu_scores = []
        for i, cand in enumerate(candidates):
            item_refs = [ref_list[i] for ref_list in references if i < len(ref_list)]
            if item_refs and cand and cand.strip():
                sent_bleu = sacrebleu.sentence_bleu(cand, item_refs)
                sentence_bleu_scores.append(sent_bleu.score)
            else:
                sentence_bleu_scores.append(0.0)

        primary_ref = references[0] if isinstance(references[0], list) else references
        P, R, F1 = score(candidates, primary_ref, lang="vi", verbose=False)
        bertscore_f1_list = F1.tolist()

        return {
            "BLEU": sentence_bleu_scores,
            "BERTScore_F1": bertscore_f1_list,
        }

    # =========================================================================
    # LLM Judge (BaseLLMClient implementation)
    # =========================================================================

    def _build_prompt(
        self,
        batch_items: List[tuple[int, str, str, str]],
        source_lang: str,
        **kwargs,
    ) -> str:
        """
        Build evaluation prompt from batch items.

        Args:
            batch_items: List of (idx, source, reference, candidate) tuples
            source_lang: Source language name

        Returns:
            Formatted prompt string
        """
        items_text = ""
        for i, (_, src, ref, cand) in enumerate(batch_items):
            items_text += f"""
--- Item {i + 1} ---
Source ({source_lang}): {src}
Reference (Vietnamese): {ref}
Candidate (Vietnamese): {cand}
"""

        return render_prompt(
            self.batch_judge_prompt_template,
            source_lang=source_lang,
            rubric=self.rubric,
            items_text=items_text,
        )

    def _parse_response(self, content: str, batch_size: int) -> List[str]:
        """
        Parse BatchJudgementResult into list of JSON strings.

        Args:
            content: Cleaned JSON response content
            batch_size: Expected number of judgements

        Returns:
            List of JSON strings with accuracy, fluency, explanation
        """
        try:
            result = validate_pydantic(content, BatchJudgementResult)

            judgement_strings = []
            for item in result.evaluations:
                judgement_strings.append(
                    json.dumps(
                        {
                            "accuracy": item.accuracy,
                            "fluency": item.fluency,
                            "explanation": item.explanation,
                        }
                    )
                )

            return judgement_strings

        except Exception as e:
            print(f"Judge Schema Validation Failed: {e}")
            return [
                json.dumps(
                    {"accuracy": 0, "fluency": 0, "explanation": "Validation Error"}
                )
            ] * batch_size

    async def batch_llm_judge(
        self,
        sources: List[str],
        references: List[str],
        candidates: List[str],
        source_lang: str = "Sanskrit",
        batch_size: int = 20,
        model_id: str = "",
        session_id: Optional[str] = None,
    ) -> List[str]:
        """
        Async batch LLM judge evaluation.

        Args:
            sources: Source texts
            references: Reference translations
            candidates: Candidate translations to evaluate
            source_lang: Source language
            batch_size: Items per batch
            model_id: Model identifier (for metadata)
            session_id: Optional session ID

        Returns:
            List of JSON strings with judgement results
        """
        # Prepare items as tuples: (idx, src, ref, cand)
        items_to_judge = [
            (idx, src, ref, cand)
            for idx, (src, ref, cand) in enumerate(zip(sources, references, candidates))
        ]

        if not items_to_judge:
            return []

        default_result = json.dumps(
            {"accuracy": 0, "fluency": 0, "explanation": "Missing"}
        )

        results, _ = await self.process_batches(
            items=items_to_judge,
            batch_size=batch_size,
            session_id=session_id,
            default_result=default_result,
            source_lang=source_lang,
        )

        return results

    def llm_judge(
        self,
        source: str,
        reference: str,
        candidate: str,
        source_lang: str = "Sanskrit",
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Single-item evaluation (placeholder for backwards compatibility)."""
        pass
