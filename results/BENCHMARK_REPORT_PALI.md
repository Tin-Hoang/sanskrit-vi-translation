# Pali-Vietnamese (Dhammapada) Benchmark Results

**Date**: 2026-01-19 01:45:20
**Judge Model**: gemini/gemini-3-flash-preview
**Dataset**: pali_vi_dhammapada.csv (20 samples)

## Performance Summary

| Source   | Model                  |   BLEU ↑ |   BERTScore ↑ |   LLM Judge Accuracy (1-5) ↑ |   LLM Judge Fluency (1-5) ↑ |   Time (s) ↓ |
|:---------|:-----------------------|---------:|--------------:|-----------------------------:|----------------------------:|-------------:|
| Pali     | Llama-3.3-70b          |    14.00 |          0.72 |                         3.55 |                        4.05 |         4.86 |
| Pali     | GPT-OSS-120b           |     7.89 |          0.72 |                         2.75 |                        3.20 |         9.99 |
| Pali     | Kimi-k2                |    16.82 |          0.77 |                         4.75 |                        4.60 |         7.30 |
| Pali     | Qwen3-32b              |    11.59 |          0.74 |                         3.75 |                        4.00 |         5.38 |
| Pali     | Gemini-3-Flash-Preview |    37.65 |          0.84 |                         5.00 |                        4.95 |         7.79 |

*Evaluation powered by LLM Judge using a 5-point rubric for Accuracy and Fluency.*
