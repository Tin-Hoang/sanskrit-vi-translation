import json
from typing import List, Optional, TYPE_CHECKING


import litellm
from litellm.exceptions import RateLimitError, BadRequestError
import time
from dotenv import load_dotenv

load_dotenv()

# Optional Langfuse Import
try:
    from langfuse import observe, Langfuse
except ImportError:
    # Dummy decorator and None client if missing
    def observe(**kwargs):
        def decorator(func):
            return func

        return decorator

    Langfuse = None


class Translator:
    def __init__(
        self,
        model_name: str = "groq/llama-3.3-70b-versatile",
        temperature: float = 0.3,
        single_prompt_template: Optional[str] = None,
        batch_prompt_template: Optional[str] = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        # Use provided templates or load from current (fallback logic to be safe)
        if not single_prompt_template or not batch_prompt_template:
            from system_prompts.translator.current import (
                SINGLE_TRANSLATE_PROMPT,
                BATCH_TRANSLATE_PROMPT,
            )

        self.single_prompt_template = single_prompt_template or SINGLE_TRANSLATE_PROMPT
        self.batch_prompt_template = batch_prompt_template or BATCH_TRANSLATE_PROMPT

    def translate(
        self,
        text: str,
        source_lang: str = "Sanskrit",
        target_lang: str = "Vietnamese",
        session_id: Optional[str] = None,
    ) -> str:
        """Single text translation (kept for backwards compatibility)."""
        prompt = self.single_prompt_template.format(
            source_lang=source_lang,
            target_lang=target_lang,
            text=text,
        )
        try:
            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                metadata={
                    "session_id": session_id,
                    "tags": ["translation", self.model_name, source_lang],
                },
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error translating text: {e}")
            return ""

    def batch_translate(
        self,
        texts: List[str],
        source_lang: str = "Sanskrit",
        batch_size: int = 10,
        session_id: Optional[str] = None,
        dataset_item_ids: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ) -> tuple[List[str], float, List[Optional[str]]]:
        """
        Translate multiple texts using batched API calls to reduce RPM usage.

        Args:
            texts: List of source texts to translate
            source_lang: Source language name
            batch_size: Number of texts to translate per API call (default: 10)
            cache: Optional BenchmarkCache instance for caching translations
            dataset_item_ids: Optional list of Langfuse Dataset Item IDs for trace linking
            dataset_name: Optional name of the Langfuse dataset for linking
            experiment_name: Optional run name for the experiment linking

        Returns:
            Tuple of (translations list, total inference time, trace_ids list)
        """

        # internal worker to be decorated
        @observe(name=f"Batch Translate {self.model_name}")
        def _execute_batch_pass(
            batch_texts: List[str],
            batch_item_ids: List[Optional[str]],
            Langfuse_cls=Langfuse,
        ) -> tuple[List[str], float, str]:
            # Get the current trace ID
            current_trace_id = None
            lf = None
            if Langfuse_cls:
                try:
                    lf = Langfuse_cls()
                    current_trace_id = lf.get_current_trace_id()
                except Exception as e:
                    print(f"    DEBUG: Failed to get trace ID: {e}")

            print(f"    DEBUG: current_trace_id={current_trace_id}", flush=True)
            if batch_item_ids:
                print(
                    f"    DEBUG: batch_item_ids count={len(batch_item_ids)}", flush=True
                )
                print(f"    DEBUG: first item_id={batch_item_ids[0]}", flush=True)

            # Build batch prompt
            items_text = ""
            for i, text in enumerate(batch_texts):
                items_text += f"\n--- Item {i + 1} ---\nText: {text}\n"

            prompt = self.batch_prompt_template.format(
                source_lang=source_lang, items_text=items_text
            )

            # Retry logic for rate limits
            retry_delays = [60, 120, 180]
            max_retries = len(retry_delays)
            response = None
            inference_time = 0.0

            use_json_mode = True
            for attempt in range(max_retries + 1):
                try:
                    start_time = time.time()
                    completion_kwargs = {
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": self.temperature,
                        "metadata": {
                            "session_id": session_id,
                            "tags": ["translation", self.model_name, source_lang],
                        },
                    }
                    if current_trace_id:
                        # Propagate trace ID to LiteLLM so it nests under this trace
                        completion_kwargs["metadata"]["trace_id"] = current_trace_id

                    if use_json_mode:
                        completion_kwargs["response_format"] = {"type": "json_object"}

                    response = litellm.completion(**completion_kwargs)
                    end_time = time.time()
                    inference_time = end_time - start_time
                    break
                except BadRequestError as e:
                    if "json_validate_failed" in str(e) and use_json_mode:
                        print(f"\n  JSON mode failed, retrying without...")
                        use_json_mode = False
                        continue
                    else:
                        raise e
                except RateLimitError:
                    if attempt < max_retries:
                        time.sleep(retry_delays[attempt])
                    else:
                        raise

            if response is None:
                raise RuntimeError("No response")

            # Parse results
            result_text = response.choices[0].message.content
            clean_json = result_text.replace("```json", "").replace("```", "").strip()

            translations = []
            try:
                parsed = json.loads(clean_json)
                if isinstance(parsed, list):
                    translations = parsed
                else:
                    translations = parsed.get("translations", [])
            except json.JSONDecodeError:
                # Fallback regex parsing
                import re

                json_match = re.search(r'\{[\s\S]*"translations"[\s\S]*\}', clean_json)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group())
                        translations = parsed.get("translations", [])
                    except:
                        pass

            # Normalize translations list to match batch size
            final_translations = []
            for i in range(len(batch_texts)):
                val = ""
                if i < len(translations):
                    item = translations[i]
                    if isinstance(item, dict):
                        val = item.get("translation", "")
                    elif isinstance(item, str):
                        val = item
                final_translations.append(val)

            # Link Dataset Items - Create specific spans for each item
            if current_trace_id and any(batch_item_ids):
                try:
                    from langfuse import Langfuse
                    import os

                    if os.getenv("LANGFUSE_PUBLIC_KEY"):
                        # Initialize fetching dataset (relying on client caching if available)
                        # Ideally we do this once per batch or pass it in, but this is safe
                        lf = Langfuse()
                        target_dataset = dataset_name or "pali-vi-dhammapada"
                        run_name_for_link = (
                            experiment_name or f"Experiment {self.model_name}"
                        )

                        # We use try/except for get_dataset in case network fails
                        try:
                            # Verify if dataset exists and fetch items
                            try:
                                dataset = lf.get_dataset(target_dataset)
                            except Exception:
                                # For local files, dataset might not exist via API if not created yet?
                                # But we need it for linking.
                                print(
                                    f"    Warning: Dataset '{target_dataset}' fetch failed. Linking skipped."
                                )
                                dataset = None

                            if dataset and hasattr(dataset, "items"):
                                # Build lookup for existing items (v3 doesn't have get_item)
                                item_lookup = {item.id: item for item in dataset.items}

                                links_created = 0
                                for i, item_id in enumerate(batch_item_ids):
                                    if item_id and i < len(final_translations):
                                        item = item_lookup.get(item_id)
                                        if item:
                                            try:
                                                # v3 linking: use item.run() to create a linked span
                                                with item.run(
                                                    run_name=run_name_for_link,
                                                    run_metadata={
                                                        "model": self.model_name,
                                                        "source_lang": source_lang,
                                                    },
                                                ) as item_span:
                                                    # Record specific content
                                                    item_span.update(
                                                        input=batch_texts[i],
                                                        output=final_translations[i],
                                                    )
                                                links_created += 1
                                            except Exception as inner_e:
                                                print(
                                                    f"    Item {i} link error: {inner_e}"
                                                )
                                        else:
                                            # Item ID might be from local file but not uploaded yet?
                                            # If so, we can't link.
                                            pass

                                # Ensure data dispatch
                                lf.flush()
                                if links_created > 0:
                                    print(
                                        f"    Linked {links_created} items to experiment '{run_name_for_link}'"
                                    )

                        except Exception as e:
                            print(f"    Dataset fetch/processing failed: {e}")

                except Exception as e:
                    print(f"  Linking logic failed: {e}")

            return final_translations, inference_time, current_trace_id

        # --- Main Execution Logic ---
        all_translations: List[Optional[str]] = [None] * len(texts)
        all_trace_ids: List[Optional[str]] = [None] * len(texts)
        total_inference_time = 0.0

        texts_to_translate: List[tuple[int, str, Optional[str]]] = []

        # Prepare all items for translation (LiteLLM handles caching)
        for idx, text in enumerate(texts):
            item_id = dataset_item_ids[idx] if dataset_item_ids else None
            texts_to_translate.append((idx, text, item_id))

        if not texts_to_translate:
            return ([t or "" for t in all_translations], 0.0, all_trace_ids)

        # Batch Processing
        for batch_start in range(0, len(texts_to_translate), batch_size):
            batch_end = min(batch_start + batch_size, len(texts_to_translate))
            batch_items = texts_to_translate[batch_start:batch_end]

            print(f"  Translating batch {batch_start + 1}-{batch_end}...", end="\r")

            try:
                b_texts = [x[1] for x in batch_items]
                b_ids = [x[2] for x in batch_items]

                # Execute the decorated internal function
                b_trans, b_time, b_trace_id = _execute_batch_pass(b_texts, b_ids)

                total_inference_time += b_time

                for i, (orig_idx, orig_text, _) in enumerate(batch_items):
                    t_val = b_trans[i]
                    all_translations[orig_idx] = t_val
                    all_trace_ids[orig_idx] = b_trace_id

            except Exception as e:
                print(f"\n  Batch failed: {e}")

        print()
        return (
            [t or "" for t in all_translations],
            total_inference_time,
            all_trace_ids,
        )
