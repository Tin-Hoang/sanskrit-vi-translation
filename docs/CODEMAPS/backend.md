<!-- Generated: 2026-04-18 | Files scanned: 18 | Token estimate: ~700 -->

# Backend / Pipeline Modules

## Core Pipeline

| File | Lines | Role |
|------|-------|------|
| `src/main.py` | 520 | Hydra entry point; orchestrates full benchmark loop |
| `src/llm_client.py` | 271 | `BaseLLMClient` — shared LiteLLM async call, retry, batching |
| `src/translator.py` | 174 | `Translator(BaseLLMClient)` — prompt build + response parse for translation |
| `src/evaluator.py` | 270 | `Evaluator(BaseLLMClient)` — BLEU/BERTScore + LLM judge |
| `src/schemas.py` | 47 | Pydantic models: `BatchTranslationResult`, `BatchJudgementResult` |
| `src/response_parser.py` | 111 | JSON cleanup, safe parse, Pydantic validation, double-encode fix |
| `src/prompt_manager.py` | 111 | `PromptManager` — Langfuse fetch + Jinja2/format render |
| `src/dataset_loader.py` | 262 | Load CSV or Langfuse dataset; auto-sync/upsert |
| `src/observability.py` | 79 | `init_langfuse()` — sets `litellm.callbacks = ["langfuse_otel"]` |
| `src/utils.py` | ~50 | `load_data`, `save_results`, `identify_columns` |
| `src/cache.py` | — | Legacy JSON cache (superseded by LiteLLM disk cache) |

## Key Function Signatures

```python
# main.py
load_input_data(input_arg, base_dir) -> (DataFrame, dataset_name, source_lang)
run_benchmark(dataset_name, source_lang, df, translator_prompts, evaluator,
              output_prefix, base_dir, models_config, ...) -> (results_df, stats)

# llm_client.py
BaseLLMClient._call_llm(prompt, session_id, tags, trace_id) -> (content, elapsed)
BaseLLMClient.process_batches(items, batch_size, ...) -> (results, total_time)

# translator.py
Translator.batch_translate(texts, source_lang, batch_size, ...) -> (translations, time, trace_ids)

# evaluator.py
Evaluator.calculate_metrics(references, candidates) -> {BLEU, BERTScore_F1}
Evaluator.batch_llm_judge(sources, references, candidates, source_lang, ...) -> [json_str]
```

## Model Config Shape (config.yaml)

```yaml
translator:
  models:
    - id: "groq/llama-3.3-70b-versatile"  # LiteLLM model string
      name: "Llama-3.3-70B"               # display name
      temperature: 0.3
      api_base: ...                        # optional, for vLLM
      api_key: ...                         # optional override
      reasoning_effort: "medium"           # optional extra param (OpenAI o-models)
```

## Crawlers (Data Collection, not benchmark)

```
src/crawlers/base.py          — BaseCrawler abstract class
src/crawlers/budsas.py        — budsas.net scraper
src/crawlers/thuvienhoasen.py — thuvienhoasen.org scraper
src/run_crawler.py            — CLI entry for crawlers
src/align_data.py             — auto-align crawled texts
src/clean_data.py             — post-crawl cleaning
src/manual_align.py           — manual alignment helpers
```

## Scripts

```
scripts/list_groq_models.py              — list available Groq models
scripts/upload_all_datasets_to_langfuse.py — bulk upload CSVs
scripts/sync_prompts.py                  — sync local prompts → Langfuse
scripts/split_parallel_dataset.py        — split parallel dataset
```
