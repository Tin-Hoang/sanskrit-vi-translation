# Buddhist Text Translation Benchmark

This project benchmarks the performance of LLMs on translating Buddhist texts from **Sanskrit** and **Pali** into Vietnamese.

## 📊 Supported Benchmarks

| Task | Description | Source Text |
|------|-------------|-------------|
| `sanskrit-vi` | Sanskrit → Vietnamese (Heart Sutra) | `sanskrit_vi_heart_sutra.csv` |
| `pali-vi` | Pali → Vietnamese (Dhammapada) | `pali_vi_dhammapada.csv` |
| `compare` | Pali vs Sanskrit comparison | `dhammapada_udanavarga_parallel.csv` |

## 📂 Structure
- `data/`:
    - `sanskrit_vi_heart_sutra.csv`: Sanskrit Heart Sutra benchmark (18 lines, multi-ref).
    - `pali_vi_dhammapada.csv`: Pali Dhammapada benchmark (TBD).
    - `dhammapada_udanavarga_parallel.csv`: Parallel Pali/Sanskrit verses (TBD).
- `src/`:
    - `main.py`: **Entry point**. Runs benchmark with configurable tasks.
    - `translator.py`: LLM translation wrapper (supports multiple source languages).
    - `evaluator.py`: BLEU, BERTScore, and LLM Judge evaluation.
    - `crawlers/`: Data collection utilities.

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

