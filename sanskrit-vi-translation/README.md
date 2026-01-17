# Sanskrit to Vietnamese Translation Benchmark

This project benchmarks the performance of LLMs on translating Buddhist Sanskrit texts into Vietnamese, focusing on the **Heart Sutra**.

## 📊 Status & Results
**Current Benchmark (`results_benchmark.csv`):**
Date: 2026-01-17

| Model         |   BLEU ↑ |   BERTScore ↑ |   LLM Judge Accuracy (1-5) ↑ |   LLM Judge Fluency (1-5) ↑ |   Time (s) ↓ |
|:--------------|---------:|--------------:|-----------------------------:|----------------------------:|-------------:|
| Llama-3.3-70b |     9.34 |          0.72 |                         4.17 |                        3.89 |        60.96 |
| GPT-OSS-120b  |     7.18 |          0.69 |                         4.11 |                        4.06 |        71.44 |
| Kimi-k2       |    21.52 |          0.74 |                         4.50 |                        4.17 |        63.57 |
| Qwen3-32b     |     0.48*|          0.53*|                         4.83 |                        4.83 |        84.16 |

> **Note**: Qwen3-32b output included reasoning traces (`<think>...`), which heavily impacted automated metrics (BLEU/BERTScore) but was correctly rated by the LLM Judge.

## 📂 Structure
- `data/`:
    - `sanskrit_vi_heart_sutra.csv`: The **main benchmark dataset** (18 lines, multi-ref).
    - `heart_sutra_sanskrit_vi_parallel.csv`: Initial small dataset (2 lines).
    - `heart_sutra_crawled_raw.csv`: Raw data from `budsas.net`.
    - `heart_sutra_vietnamese_candidates.csv`: Intermediate cleaned segments.
- `src/`:
    - `main.py`: **Entry point**. Runs benchmark on multiple models.
    - `crawlers/`:
        - `budsas.py`: Crawler for `budsas.net` (LiteSpeed, cleaner).
        - `thuvienhoasen.py`: (Deprecated) Crawler for `thuvienhoasen.org` (Cloudflare blocked).
    - `run_crawler.py`: Utility to run crawlers.
    - `clean_data.py`: Cleans and segments raw crawled data.
    - `align_data.py`: Aligns cleaned Vietnamese text with standard Sanskrit source.

## 🚀 Usage

### 1. Setup
Ensure you are using `uv` and have installed dependencies:
```bash
uv sync
```
Create a `.env` file at the root with your API key:
```
GROQ_API_KEY=your_key_here
```

### 2. Run Benchmark
To run the full pipeline on the extended dataset:
```bash
uv run sanskrit-vi-translation/src/main.py
```
This will:
1. Load `data/sanskrit_vi_heart_sutra.csv`.
2. Translate using 4 models (Llama-3, GPT-OSS, Kimi-k2, Qwen3).
3. Evaluate against references.
4. Generate `BENCHMARK_REPORT.md` and `results_benchmark.csv`.

### 3. Data Collection (Optional)
To regenerate the dataset from scratch:

**Step A: Crawl**
```bash
uv run sanskrit-vi-translation/src/run_crawler.py --source budsas --url "https://budsas.net/uni/u-kinh-bt-ngan/bntk.htm"
```
**Step B: Clean & Segment**
```bash
python3 sanskrit-vi-translation/src/clean_data.py
```
**Step C: Align**
```bash
python3 sanskrit-vi-translation/src/align_data.py
```

## 🧠 Methodology
- **Translation Models**:
    - `groq/llama-3.3-70b-versatile`
    - `groq/openai/gpt-oss-120b`
    - `groq/moonshotai/kimi-k2-instruct-0905`
    - `groq/qwen/qwen3-32b`
- **Evaluation**:
    - **Quantitative**:
        - **BLEU**: Corpuse score against multiple references.
        - **BERTScore**: Semantic similarity (F1) against primary reference (Hán-Việt).
    - **Qualitative**: LLM-as-a-Judge (Llama-3) rating accuracy/fluency.
