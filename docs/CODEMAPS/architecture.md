<!-- Generated: 2026-04-18 | Files scanned: 18 | Token estimate: ~600 -->

# Architecture

Buddhist text translation benchmark — evaluates LLMs translating Sanskrit/Pali → Vietnamese, scored against classical human translations.

## Execution Flow

```
CLI (Hydra)
  └─ main.py::async_main()
       ├─ load_input_data()          # CSV file OR Langfuse dataset name
       │    └─ sync_local_dataset()  # auto-upsert to Langfuse if new/larger
       ├─ PromptManager.get_prompt() # Langfuse live prompts → local fallback
       ├─ Evaluator(judge_model)     # init once, reused across all models
       └─ run_benchmark() [per dataset]
            └─ [per model in config]
                 ├─ Translator.batch_translate()   # async parallel batches
                 ├─ Evaluator.calculate_metrics()  # BLEU + BERTScore (sync)
                 ├─ Evaluator.batch_llm_judge()    # async parallel batches
                 └─ Langfuse score upload          # per-item traces + scores
```

## Key Abstractions

```
BaseLLMClient (llm_client.py)
  ├─ Translator   (translator.py)  — wraps batch_translate()
  └─ Evaluator    (evaluator.py)   — wraps batch_llm_judge() + BLEU/BERTScore
```

Both share: retry-on-rate-limit (tenacity), async parallel batching, LiteLLM routing.

## Batch Parallelism

Items → chunked into batches → `asyncio.gather(*tasks)` → results merged back.
Default batch sizes: translation=10, judge=30 (config.yaml overridable).

## Caching

LiteLLM disk cache at `.litellm_cache/` — keyed automatically by model+prompt.
Legacy JSON cache in `cache/` is superseded but still cleared by `clear_cache=true`.

## Prompt Versioning

`src/system_prompts/{translator,evaluator}/{v1,v2,current}.py`
Live overrides fetched from Langfuse (5min TTL cache); falls back to `current.py`.
