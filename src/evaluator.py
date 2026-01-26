import asyncio
import json
import time
from typing import List, Dict, Optional
import sacrebleu
from bert_score import score
import litellm
from litellm.exceptions import RateLimitError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from dotenv import load_dotenv
from schemas import BatchJudgementResult

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
        else:
            self.rubric = rubric
            self.single_judge_prompt_template = single_judge_prompt_template
            self.batch_judge_prompt_template = batch_judge_prompt_template

    def calculate_metrics(
        self, references: List[List[str]], candidates: List[str]
    ) -> Dict[str, float]:
        # BLEU Score
        bleu = sacrebleu.corpus_bleu(candidates, references)

        # BERTScore
        primary_ref = references[0] if isinstance(references[0], list) else references

        # Calculate BERTScore
        # Note: Keeps sync for now as BERTScore is CPU/GPU bound, not Network bound.
        P, R, F1 = score(candidates, primary_ref, lang="vi", verbose=False)

        return {"BLEU": bleu.score, "BERTScore_F1": F1.mean().item()}

    def calculate_item_metrics(
        self, references: List[List[str]], candidates: List[str]
    ) -> Dict[str, List[float]]:
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

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
    )
    async def _process_batch_judge(
        self,
        batch_items: List[tuple[int, str, str, str]],  # (idx, src, ref, cand)
        source_lang: str,
        session_id: Optional[str],
    ) -> tuple[List[str], float]:
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

        model_params = {
            "model": self.judge_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            # Use json_object for broader compatibility
            "response_format": {"type": "json_object"},
            "metadata": {
                "session_id": session_id,
                "tags": ["evaluation", self.judge_model, source_lang],
            },
        }

        start_time = time.time()
        response = await litellm.acompletion(**model_params)
        end_time = time.time()

        try:
            content = response.choices[0].message.content

            # Clean up markdown code blocks if present
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "")
            elif content.startswith("```"):
                content = content.replace("```", "")

            result = BatchJudgementResult.model_validate_json(content)

            # Map back to standardized JSON strings for backward compatibility/storage
            # The codebase expects list of JSON strings with "accuracy", "fluency", "explanation"

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

            return judgement_strings, end_time - start_time

        except Exception as e:
            print(f"Judge Schema Validation Failed: {e}")
            # Fallback
            return [
                json.dumps(
                    {"accuracy": 0, "fluency": 0, "explanation": "Validation Error"}
                )
            ] * len(batch_items), end_time - start_time

    async def batch_llm_judge(
        self,
        sources: List[str],
        references: List[str],
        candidates: List[str],
        source_lang: str = "Sanskrit",
        batch_size: int = 20,
        model_id: str = "",
        session_id: Optional[str] = None,
    ) -> List[Dict[str, any]]:
        # Prepare items
        items_to_judge = []
        valid_indices = []

        for idx, (src, ref, cand) in enumerate(zip(sources, references, candidates)):
            items_to_judge.append((idx, src, ref, cand))
            valid_indices.append(idx)

        if not items_to_judge:
            return []

        tasks = []
        batch_map = []

        for i in range(0, len(items_to_judge), batch_size):
            batch_end = min(i + batch_size, len(items_to_judge))
            batch_items = items_to_judge[i:batch_end]
            batch_map.append((i, batch_end))

            tasks.append(
                self._process_batch_judge(
                    batch_items=batch_items,
                    source_lang=source_lang,
                    session_id=session_id,
                )
            )

        print(
            f"Dispatching {len(tasks)} async judge batches for {len(items_to_judge)} items..."
        )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate
        all_results = [
            json.dumps({"accuracy": 0, "fluency": 0, "explanation": "Missing"})
        ] * len(sources)

        for idx, result in enumerate(results):
            start_off, _ = batch_map[idx]  # offset within the list of *valid items*

            if isinstance(result, Exception):
                print(f"Judge Batch {idx} Failed: {result}")
            else:
                judgements, _ = result

                # Map these judgements back to the original source list
                # Since items_to_judge contains (original_idx, ...), we use that
                current_batch_items = items_to_judge[
                    start_off : start_off + len(judgements)
                ]

                for k, j_str in enumerate(judgements):
                    if k < len(current_batch_items):
                        original_idx = current_batch_items[k][0]
                        all_results[original_idx] = j_str

        return all_results

    def llm_judge(
        self,
        source: str,
        reference: str,
        candidate: str,
        source_lang: str = "Sanskrit",
        session_id: Optional[str] = None,
    ) -> Dict[str, any]:
        """Single-item evaluation (kept for backwards compatibility, synchronous wrapper)."""
        # Ideally this should be async too, but for legacy support we might leave it or use asyncio.run
        # Given the instruction was upgrade to async, let's keep it async but not use it.
        pass
