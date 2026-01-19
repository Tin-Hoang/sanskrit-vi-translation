# Buddhist Text Translation Benchmark

This project benchmarks the performance of LLMs on translating Buddhist texts from **Sanskrit** and **Pali** into Vietnamese.

## 📊 Supported Benchmarks

| Task | Description | Source Text |
|------|-------------|-------------|
| `sanskrit-vi` | Sanskrit → Vietnamese (Heart Sutra) | `sanskrit_vi_heart_sutra.csv` |
| `pali-vi` | Pali → Vietnamese (Dhammapada) | `pali_vi_dhammapada.csv` |
| `compare` | Pali vs Sanskrit comparison | `dhammapada_udanavarga_parallel.csv` |


## Results

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

| Source   | Model            |   BLEU ↑ |   BERTScore ↑ |   LLM Judge Accuracy (1-5) ↑ |   LLM Judge Fluency (1-5) ↑ |   Time (s) ↓ |
|:---------|:-----------------|---------:|--------------:|-----------------------------:|----------------------------:|-------------:|
| Pali     | Llama-3.3-70b    |     3.41 |          0.71 |                         3.05 |                        4.15 |         5.39 |
| Pali     | GPT-OSS-120b     |     2.25 |          0.73 |                         2.45 |                        3.00 |        10.60 |
| Pali     | Kimi-k2          |     5.89 |          0.75 |                         4.80 |                        4.90 |         7.64 |
| Pali     | Qwen3-32b        |     4.88 |          0.73 |                         3.10 |                        4.00 |         6.09 |
| Pali     | Gemini-2.5-Flash |    12.34 |          0.77 |                         5.00 |                        5.00 |        15.56 |

### Pali vs Sanskrit Comparison (Dhammapada - Udanavarga | Kinh Pháp Cú)

| Source   | Model         |   BLEU ↑ |   BERTScore ↑ |   LLM Judge Accuracy (1-5) ↑ |   LLM Judge Fluency (1-5) ↑ |   Time (s) ↓ |
|:---------|:--------------|---------:|--------------:|-----------------------------:|----------------------------:|-------------:|
| Pali     | Llama-3.3-70b |     3.00 |          0.72 |                         3.65 |                        4.05 |        10.55 |
| Pali     | GPT-OSS-120b  |     2.49 |          0.72 |                         3.45 |                        3.90 |        20.55 |
| Pali     | Kimi-k2       |     7.91 |          0.76 |                         4.65 |                        4.50 |        13.16 |
| Pali     | Qwen3-32b     |     0.21 |          0.56 |                         4.40 |                        4.45 |        45.76 |
| Sanskrit | Llama-3.3-70b |     4.24 |          0.71 |                         3.55 |                        3.80 |         8.56 |
| Sanskrit | GPT-OSS-120b  |     1.85 |          0.70 |                         3.00 |                        3.45 |        21.58 |
| Sanskrit | Kimi-k2       |     2.69 |          0.73 |                         4.00 |                        4.05 |        12.99 |
| Sanskrit | Qwen3-32b     |     0.19 |          0.55 |                         3.50 |                        3.60 |        59.62 |


## 📂 Structure
- `data/`:
    - `sanskrit_vi_heart_sutra.csv`: Sanskrit Heart Sutra benchmark (18 lines, multi-ref).
    - `pali_vi_dhammapada.csv`: Pali Dhammapada benchmark (TBD).
    - `dhammapada_udanavarga_parallel.csv`: Parallel Pali/Sanskrit verses (TBD).
- `src/`:
    - `main.py`: **Entry point**. Runs benchmark with configurable tasks.
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

**Sanskrit → Vietnamese (default):**
```bash
uv run src/main.py --task sanskrit-vi
```

**Pali → Vietnamese:**
```bash
uv run src/main.py --task pali-vi
```

**Comparison Mode (both languages on parallel data):**
```bash
uv run src/main.py --task compare
```

**Custom data file:**
```bash
uv run src/main.py --task sanskrit-vi --data /path/to/custom.csv
```

### Caching (Rate Limit Handling)

The benchmark caches API outputs to handle rate limits gracefully. If interrupted, re-run to resume from cached results.

```bash
# Default: caching enabled, resumes from cache
uv run src/main.py --task pali-vi

# Clear cache and start fresh
uv run src/main.py --task pali-vi --clear-cache

# Disable caching entirely
uv run src/main.py --task pali-vi --no-cache
```

Cache files are stored in `cache/` (auto-created).

### 3. Output Files
Each task generates:
- `results_{task}_benchmark.csv`: Detailed translations and judgments
- `BENCHMARK_REPORT_{TASK}.md`: Summary report with metrics

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

## 🔬 Research Question
> Which original language (Sanskrit or Pali) produces better Vietnamese translations of Buddhist texts?

The `compare` task enables direct comparison by running both source languages on parallel texts (e.g., Dhammapada vs Udanavarga).

