# Buddhist Text Translation Benchmark
This benchmark evaluates how well AI models translate sacred texts from **Sanskrit** and **Pali** into Vietnamese, comparing them against classical translations by renowned Buddhist monks.

<p align="center">
  <img src="docs/sanskrit_vi_banner.jpg" alt="Sanskrit-Vietnamese Translation Benchmark" width="100%" />
</p>

## 🔬 Research Questions

1. Can modern LLMs translate ancient Buddhist scriptures as well as human scholars?
2. Which current LLM achieves the highest translation accuracy?
3. Which is the better source language to translate into Vietnamese: Sanskrit or Pali?

##  Supported Benchmarks

| Task | Description | Source Text |
|------|-------------|-------------|
| `sanskrit-vi` | Sanskrit → Vietnamese (Heart Sutra) | `sanskrit_vi_heart_sutra.csv` |
| `pali-vi` | Pali → Vietnamese (Dhammapada) | `pali_vi_dhammapada.csv` |
| `compare` | Pali vs Sanskrit comparison | `dhammapada_udanavarga_parallel.csv` |


## 📊 Results

### Sanskrit → Vietnamese (Heart Sutra | Bát Nhã Tâm Kinh)

**Date**: 2026-01-19 01:11:42

**Judge Model**: gemini/gemini-3-flash-preview

**Dataset**: sanskrit_vi_heart_sutra.csv (18 samples)

| Source   | Model                  |   BLEU ↑ |   BERTScore ↑ |   LLM Judge Accuracy (1-5) ↑ |   LLM Judge Fluency (1-5) ↑ |   Time (s) ↓ |
|:---------|:-----------------------|---------:|--------------:|-----------------------------:|----------------------------:|-------------:|
| Sanskrit | Llama-3.3-70b          |    15.78 |          0.73 |                         4.44 |                        4.44 |         3.32 |
| Sanskrit | GPT-OSS-120b           |    10.30 |          0.69 |                         4.28 |                        4.17 |         9.52 |
| Sanskrit | Kimi-k2                |    27.13 |          0.76 |                         5.00 |                        4.94 |         4.36 |
| Sanskrit | Qwen3-32b              |    18.69 |          0.75 |                         4.00 |                        4.50 |         7.24 |
| Sanskrit | Gemini-3-Flash-Preview |    41.84 |          0.76 |                         5.00 |                        5.00 |         5.50 |

### Pali → Vietnamese (Dhammapada | Kinh Pháp Cú)

**Date**: 2026-01-19 01:45:20

**Judge Model**: gemini/gemini-3-flash-preview

**Dataset**: pali_vi_dhammapada.csv (20 samples)

| Source   | Model                  |   BLEU ↑ |   BERTScore ↑ |   LLM Judge Accuracy (1-5) ↑ |   LLM Judge Fluency (1-5) ↑ |   Time (s) ↓ |
|:---------|:-----------------------|---------:|--------------:|-----------------------------:|----------------------------:|-------------:|
| Pali     | Llama-3.3-70b          |    14.00 |          0.72 |                         3.55 |                        4.05 |         4.86 |
| Pali     | GPT-OSS-120b           |     7.89 |          0.72 |                         2.75 |                        3.20 |         9.99 |
| Pali     | Kimi-k2                |    16.82 |          0.77 |                         4.75 |                        4.60 |         7.30 |
| Pali     | Qwen3-32b              |    11.59 |          0.74 |                         3.75 |                        4.00 |         5.38 |
| Pali     | Gemini-3-Flash-Preview |    37.65 |          0.84 |                         5.00 |                        4.95 |         7.79 |

### Pali vs Sanskrit Comparison (Dhammapada - Udanavarga | Kinh Pháp Cú)

**Date**: 2026-01-19 01:54:12

**Judge Model**: gemini/gemini-3-flash-preview

**Dataset**: dhammapada_udanavarga_parallel.csv (20 samples)

| Source   | Model                  |   BLEU ↑ |   BERTScore ↑ |   LLM Judge Accuracy (1-5) ↑ |   LLM Judge Fluency (1-5) ↑ |   Time (s) ↓ |
|:---------|:-----------------------|---------:|--------------:|-----------------------------:|----------------------------:|-------------:|
| Pali     | Llama-3.3-70b          |    10.62 |          0.71 |                         3.60 |                        3.85 |         4.81 |
| Pali     | GPT-OSS-120b           |     9.42 |          0.71 |                         2.95 |                        3.35 |        11.25 |
| Pali     | Kimi-k2                |    18.02 |          0.77 |                         4.80 |                        4.70 |         6.66 |
| Pali     | Qwen3-32b              |    13.08 |          0.75 |                         3.65 |                        3.95 |         5.25 |
| Pali     | Gemini-3-Flash-Preview |    39.16 |          0.84 |                         5.00 |                        4.60 |         7.61 |
| Sanskrit | Llama-3.3-70b          |    11.98 |          0.70 |                         3.65 |                        3.55 |         4.50 |
| Sanskrit | GPT-OSS-120b           |     6.54 |          0.66 |                         2.83 |                        2.89 |        10.75 |
| Sanskrit | Kimi-k2                |    12.61 |          0.70 |                         4.42 |                        4.37 |         5.87 |
| Sanskrit | Qwen3-32b              |     7.40 |          0.71 |                         3.80 |                        3.80 |         7.60 |
| Sanskrit | Gemini-3-Flash-Preview |    32.35 |          0.78 |                         5.00 |                        5.00 |         8.50 |

## 📂 Structure
- `data/`:
    - `sanskrit_vi_heart_sutra.csv`: Sanskrit Heart Sutra benchmark (18 lines, multi-ref).
    - `pali_vi_dhammapada.csv`: Pali Dhammapada benchmark (TBD).
    - `dhammapada_udanavarga_parallel.csv`: Parallel Pali/Sanskrit verses (TBD).
- `src/`:
    - `main.py`: **Entry point**. Runs benchmark with configurable inputs.
    - `translator.py`: LLM translation wrapper (supports multiple source languages).
    - `evaluator.py`: BLEU, BERTScore, and LLM Judge evaluation.
    - `cache.py`: Caching module for rate limit resilience.
    - `crawlers/`: Data collection utilities.
    - `system_prompts/`: Versioned prompt templates.
        - `translator/v1.py`, `current.py`: Translation prompts.
        - `evaluator/v1.py`, `current.py`: LLM Judge prompts.

## 🚀 Usage

### 1. Setup
```bash
uv sync
```
Create a `.env` file at the root:
```
GROQ_API_KEY=your_key_here
```

### 2. Run Benchmark

**Smart Loading:**
The benchmark automatically detects if the input is a local file or a Langfuse dataset name.

```bash
# Option A: Run with Local Data File
uv run src/main.py input=data/sanskrit_vi_heart_sutra.csv

# Option B: Run with Langfuse Dataset Name
uv run src/main.py input=sanskrit-vi-heart-sutra
```

**Configuration via CLI (Hydra):**
You can override any configuration from `config.yaml` directly on the command line:

```bash
# Change limit and judge temperature
uv run src/main.py input=data/sanskrit_vi_heart_sutra.csv limit=2 judge.temperature=0.5

# Change Batch Size
uv run src/main.py input=data/sanskrit_vi_heart_sutra.csv batch_size=5
```

### Caching (Rate Limit Handling)

The benchmark caches API outputs to handle rate limits gracefully.

```bash
# Default: caching enabled, resumes from cache
uv run src/main.py input=data/sanskrit_vi_heart_sutra.csv

# Clear cache and start fresh
uv run src/main.py input=data/sanskrit_vi_heart_sutra.csv clear_cache=true

# Disable caching entirely
uv run src/main.py input=data/sanskrit_vi_heart_sutra.csv no_cache=true
```

Cache files are stored in `cache/` (auto-created).

### Langfuse Observability (Optional)

Enable LLM tracing with [Langfuse](https://langfuse.com) (free) to monitor all translation and evaluation calls:

<p align="center">
  <img src="docs/langfuse-screenshot.png" alt="Langfuse Tracing Screenshot" width="100%" />
</p>

1. **Create a Langfuse account** at https://cloud.langfuse.com
2. **Create a project** and get your API keys from Settings → API Keys
3. **Configure `.env`**:
   ```bash
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   ```

When configured, you'll see:
```
✅ Langfuse tracing enabled
   View traces at: https://cloud.langfuse.com
```

### Langfuse Datasets (Data Management)

The benchmark supports auto dataset uploading into Langfuse:
-   **Auto-Sync**: When running with a local file, it automatically checks if the corresponding Langfuse dataset exists. If the local file has more items, it upserts them.
-   **Schema Enforcement**: All datasets utilize Langfuse's Native Schema Enforcement to ensure data quality:
    -   `input`: String (Source text)
    -   `expected_output`: Dictionary (Reference translations)

### 3. Output Files
Each task generates:
- `results_{dataset_name}_{timestamp}.csv`: Detailed translations and judgments

## 🧠 Methodology
- **Translation Models**:
    - `groq/llama-3.3-70b-versatile`
    - `groq/openai/gpt-oss-120b`
    - `groq/moonshotai/kimi-k2-instruct-0905`
    - `groq/qwen/qwen3-32b`
    - `gemini/gemini-3-flash-preview`
- **Evaluation**:
    - **Quantitative**: BLEU (corpus), BERTScore (semantic F1)
    - **Qualitative**: LLM-as-a-Judge (5-point accuracy/fluency rubric), using `gemini/gemini-3-flash-preview`

The `compare` task enables direct comparison by running both source languages on parallel texts (e.g., Dhammapada vs Udanavarga).

## TODO

- Beyond Translation -> Semantic Understanding, verify ability of LLM to interprete the Buddhist texts.
- Data Expansion
- Add more models
- Add more metrics
