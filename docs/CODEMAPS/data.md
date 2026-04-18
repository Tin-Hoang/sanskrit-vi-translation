<!-- Generated: 2026-04-18 | Files scanned: 18 | Token estimate: ~450 -->

# Data

## Local Dataset Files (`data/`)

| File | Task Key | Source Lang | Description |
|------|----------|-------------|-------------|
| `sanskrit_vi_heart_sutra.csv` | `sanskrit-vi` | Sanskrit | Heart Sutra verses |
| `pali_vi_dhammapada.csv` | `pali-vi` | Pali | Dhammapada verses |
| `dhammapada_udanavarga_parallel.csv` | `compare` | Both | Parallel Dhammapada/Udanavarga |
| `pali_vi_dhammapada_18verses.csv` | — | Pali | Subset (18 verses) |
| `sanskrit_vi_udanavarga_18verses.csv` | — | Sanskrit | Subset (18 verses) |
| `crawled_raw.csv` | — | — | Raw crawled text (pre-cleaning) |

## Column Conventions

```
input column  : detected by name (contains "pali" → Pali source; else Sanskrit)
ref columns   : all other non-metadata columns (multiple refs supported)
metadata cols : "id", "source", "notes", etc.
```

## Pydantic Schemas (`src/schemas.py`)

```python
TranslationItem       : translation (str), original_text?, thinking_process?
BatchTranslationResult: translations: List[TranslationItem]

JudgementItem         : accuracy (1-5), fluency (1-5), explanation (str)
BatchJudgementResult  : evaluations: List[JudgementItem]
```

## Langfuse Dataset Schema

Each item stored with:
```
input          : {text, source_lang, target_lang}
expected_output: {ref_col_name: ref_text, ...}  # dict of references
metadata       : {id, notes, ...}
```

## Results Output

```
results/results_{dataset_name}_{timestamp}.csv   — full per-item translations + judgements
```

Per-item Langfuse scores written: `bleu`, `bertscore`, `judge-accuracy`, `judge-fluency`

## Benchmark Metrics

| Metric | Level | Direction |
|--------|-------|-----------|
| BLEU (sacrebleu) | corpus + sentence | ↑ |
| BERTScore F1 (lang=vi) | corpus + per-item | ↑ |
| LLM Judge Accuracy (1-5) | per-item avg | ↑ |
| LLM Judge Fluency (1-5) | per-item avg | ↑ |
| Translation Time (s) | total | ↓ |
