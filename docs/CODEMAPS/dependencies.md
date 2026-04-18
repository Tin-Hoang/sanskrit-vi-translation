<!-- Generated: 2026-04-18 | Files scanned: 18 | Token estimate: ~400 -->

# Dependencies

## External Services

| Service | Purpose | Credentials |
|---------|---------|-------------|
| Langfuse Cloud | Tracing, dataset management, prompt registry, score storage | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` |
| Groq | Hosted LLMs (Llama, Kimi-K2, Qwen3, GPT-OSS) | `GROQ_API_KEY` |
| Google Gemini | LLM judge + translator | `GEMINI_API_KEY` |
| OpenAI | GPT models | `OPENAI_API_KEY` |
| xAI | Grok models | `XAI_API_KEY` |
| DeepSeek | DeepSeek models | `DEEPSEEK_API_KEY` |
| vLLM (local) | Self-hosted OpenAI-compatible inference | `api_base`, `api_key: EMPTY` in config |

## Python Packages

| Package | Role |
|---------|------|
| `litellm` | Universal LLM router (all provider calls go through this) |
| `langfuse` | Tracing (`@observe`), dataset CRUD, prompt registry |
| `hydra-core` | Config management + CLI overrides |
| `pydantic` | Response schema validation |
| `tenacity` | Retry-on-rate-limit decorator |
| `sacrebleu` | BLEU score (corpus + sentence) |
| `bert-score` | BERTScore F1 for Vietnamese |
| `torch` | Required by bert-score |
| `jinja2` | Prompt template rendering |
| `pandas` | DataFrame for dataset I/O |
| `python-dotenv` | `.env` file loading |
| `beautifulsoup4` / `cloudscraper` | Web crawlers (data collection only) |
| `opentelemetry-*` | OTEL exporter for Langfuse tracing |

## vLLM Monitoring Stack

```
vllm_monitoring/docker-compose.yml
  ├─ Prometheus  — scrapes vLLM `/metrics` endpoint
  └─ Grafana     — dashboard (see docs/grafana_vllm_dashboard.png)
```

## Prompt Registry (Langfuse)

| Prompt Name | Used By |
|-------------|---------|
| `translator-single` | Translator (single-item, unused in batch flow) |
| `translator-batch` | Translator.batch_translate() |
| `evaluator-rubric` | Evaluator (injected into judge prompt) |
| `evaluator-single` | Evaluator (single-item, backwards compat) |
| `evaluator-batch` | Evaluator.batch_llm_judge() |

Local fallbacks: `src/system_prompts/{translator,evaluator}/current.py`
