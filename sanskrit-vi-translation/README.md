# Sanskrit to Vietnamese Translation Benchmark

This project benchmarks the performance of LLMs on translating Buddhist Sanskrit texts into Vietnamese, focusing on the **Heart Sutra**.

## 📊 Status & Results
**Current Benchmark (`results_extended.csv`):**
- **Dataset**: Full Heart Sutra (18 parallels), aligned with 4 diverse human reference translations (Hán-Việt, Modern, Poetic, Scholarly).
- **Source**: `budsas.net` (Crawled & Aligned).
- **Metrics**:
  - **BLEU**: ~1.42 (Low due to diversity/literalness difference).
  - **BERTScore**: ~0.66 (High semantic preservation).

## 📂 Structure
- `data/`:
    - `sanskrit_vi_heart_sutra.csv`: The **main benchmark dataset** (18 lines, multi-ref).
    - `heart_sutra_sanskrit_vi_parallel.csv`: Initial small dataset (2 lines).
    - `heart_sutra_crawled_raw.csv`: Raw data from `budsas.net`.
    - `heart_sutra_vietnamese_candidates.csv`: Intermediate cleaned segments.
- `src/`:
    - `main.py`: **Entry point**. Runs translation, evaluation, and saves results.
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
1. Load `data/sanskrit_vi_heart_sutra.csv` (automagically falls back to standard if missing).
2. Translate Sanskrit texts using Llama-3 (via Groq).
3. Evaluate against **all 4 reference columns** (`ref_han_viet`, `ref_viet_modern`, etc.).
4. Save results to `results_extended.csv`.

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
- **Translation Model**: Llama-3.3-70b-versatile (via Groq key).
- **Evaluation**:
    - **Quantitative**:
        - **BLEU**: Corpuse score against multiple references.
        - **BERTScore**: Semantic similarity (F1) against primary reference (Hán-Việt).
    - **Qualitative**: LLM-as-a-Judge (Llama-3) rating accuracy/fluency.
