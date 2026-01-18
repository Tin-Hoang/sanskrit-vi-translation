# Buddhist Text Translation Benchmark

This project benchmarks the performance of LLMs on translating Buddhist texts from **Sanskrit** and **Pali** into Vietnamese.

## 📊 Supported Benchmarks

| Task | Description | Source Text |
|------|-------------|-------------|
| `sanskrit-vi` | Sanskrit → Vietnamese (Heart Sutra) | `sanskrit_vi_heart_sutra.csv` |
| `pali-vi` | Pali → Vietnamese (Dhammapada) | `pali_vi_dhammapada.csv` |
| `compare` | Pali vs Sanskrit comparison | `dhammapada_udanavarga_parallel.csv` |


## Results

### Sanskrit → Vietnamese

| Model         |   BLEU ↑ |   BERTScore ↑ |   LLM Judge Accuracy (1-5) ↑ |   LLM Judge Fluency (1-5) ↑ |   Time (s) ↓ |
|:--------------|---------:|--------------:|-----------------------------:|----------------------------:|-------------:|
| Llama-3.3-70b |     7.37 |          0.70 |                         4.33 |                        3.89 |         8.38 |
| GPT-OSS-120b  |     9.79 |          0.69 |                         3.94 |                        3.89 |        18.39 |
| Kimi-k2       |    21.54 |          0.74 |                         0.28 |                        0.22 |         9.38 |
| Qwen3-32b     |     0.59 |          0.54 |                         0.00 |                        0.00 |        37.59 |


### Pali → Vietnamese

| Source   | Model         |   BLEU ↑ |   BERTScore ↑ |   LLM Judge Accuracy (1-5) ↑ |   LLM Judge Fluency (1-5) ↑ |   Time (s) ↓ |
|:---------|:--------------|---------:|--------------:|-----------------------------:|----------------------------:|-------------:|
| Pali     | Llama-3.3-70b |     3.05 |          0.71 |                         3.60 |                        4.25 |        10.20 |
| Pali     | GPT-OSS-120b  |     2.24 |          0.72 |                         1.60 |                        1.85 |        22.83 |
| Pali     | Kimi-k2       |     5.02 |          0.76 |                         0.00 |                        0.00 |        13.31 |
| Pali     | Qwen3-32b     |     0.21 |          0.56 |                         0.00 |                        0.00 |        47.73 |


### Pali vs Sanskrit Comparison

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
- **Evaluation**:
    - **Quantitative**: BLEU (corpus), BERTScore (semantic F1)
    - **Qualitative**: LLM-as-a-Judge (5-point accuracy/fluency rubric)

## 🔬 Research Question
> Which original language (Sanskrit or Pali) produces better Vietnamese translations of Buddhist texts?

The `compare` task enables direct comparison by running both source languages on parallel texts (e.g., Dhammapada vs Udanavarga).

