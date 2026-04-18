# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run benchmark (local file)
uv run src/main.py input=data/sanskrit_vi_heart_sutra.csv

# Run benchmark (Langfuse dataset name)
uv run src/main.py input=sanskrit-vi-heart-sutra

# Common overrides
uv run src/main.py input=data/sanskrit_vi_heart_sutra.csv limit=2
uv run src/main.py input=data/pali_vi_dhammapada.csv clear_cache=true
uv run src/main.py input=data/pali_vi_dhammapada.csv no_cache=true batch_size=5

# Install with vLLM support
uv sync --extra vllm

# Start vLLM server
./scripts/serve_vllm.sh --model Qwen/Qwen2.5-32B-Instruct

# vLLM monitoring
cd vllm_monitoring && docker compose up -d

# Run tests
pytest --cov=src --cov-report=term-missing

# Utilities
uv run scripts/list_groq_models.py
uv run scripts/upload_all_datasets_to_langfuse.py
```

## Architecture

This is a **Buddhist text translation benchmark** — it evaluates LLMs translating Sanskrit/Pali into Vietnamese, scored against classical human translations.

### Execution Flow

`main.py` (Hydra entry point) → loads data → runs `run_benchmark()` per dataset → for each model: translate → BLEU/BERTScore → LLM Judge → write scores to Langfuse

### Key Modules

| File | Role |
|------|------|
| `src/main.py` | Entry point; Hydra config; orchestrates benchmark loop |
| `src/llm_client.py` | `BaseLLMClient` — shared LiteLLM call logic, retry, timing |
| `src/translator.py` | `Translator(BaseLLMClient)` — batch translation, prompt rendering |
| `src/evaluator.py` | `Evaluator` — BLEU, BERTScore, async LLM-as-a-Judge |
| `src/schemas.py` | Pydantic models: `BatchTranslationResult`, `BatchJudgementResult` |
| `src/prompt_manager.py` | `PromptManager` — fetches prompts from Langfuse, falls back to local |
| `src/dataset_loader.py` | Load from local CSV or Langfuse dataset; auto-syncs new items |
| `src/cache.py` | Legacy JSON cache (superseded by LiteLLM disk cache) |
| `src/observability.py` | Langfuse init; trace/score helpers |
| `src/response_parser.py` | JSON extraction, Pydantic validation, double-encoding fixes |

### Configuration (`config.yaml`)

All settings are Hydra-managed. Key top-level keys:
- `input`: file path or Langfuse dataset name
- `translator.models`: list of `{id, name, temperature, api_base?, api_key?}`
- `judge.model` / `judge.temperature`: LLM Judge config
- `batch_size`, `limit`, `no_cache`, `clear_cache`

To add a model, append an entry to `translator.models` in `config.yaml`.

### Prompt System

Prompts live in `src/system_prompts/{translator,evaluator}/{v1.py,current.py}`. `PromptManager` fetches live versions from Langfuse (prompt names: `translator-single`, `translator-batch`, `evaluator-rubric`, `evaluator-single`, `evaluator-batch`) and falls back to `current.py` constants if Langfuse is unavailable.

### Caching

LiteLLM disk cache is enabled by default at `.litellm_cache/`. Cache keying is automatic via LiteLLM. Use `clear_cache=true` to wipe it; `no_cache=true` to bypass.

### Langfuse Integration

- **Tracing**: all LLM calls traced via `@observe` decorator
- **Datasets**: local CSVs auto-synced to Langfuse on each run
- **Scores**: per-item BLEU, BERTScore, judge-accuracy, judge-fluency written back to dataset runs
- Requires `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in `.env`; gracefully degrades if absent

### Supported Benchmark Tasks

| Task key | Dataset file | Source language |
|----------|-------------|-----------------|
| `sanskrit-vi` | `data/sanskrit_vi_heart_sutra.csv` | Sanskrit |
| `pali-vi` | `data/pali_vi_dhammapada.csv` | Pali |
| `compare` | `data/dhammapada_udanavarga_parallel.csv` | Both (parallel) |

Source language is auto-detected from column names (column containing `pali` → Pali; default → Sanskrit).

## Environment Variables

See `.env.example`. Required keys depend on which providers are used:
- `GROQ_API_KEY` — Groq-hosted models
- `GEMINI_API_KEY` — Gemini judge / translator
- `OPENAI_API_KEY` — OpenAI models
- `XAI_API_KEY` — xAI/Grok models
- `DEEPSEEK_API_KEY` — DeepSeek models
- `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` — observability (optional)
