# Pali-Vietnamese (Dhammapada) Benchmark Results

**Date**: 2026-01-18 13:02:16
**Judge Model**: gemini/gemini-2.5-flash
**Dataset**: pali_vi_dhammapada.csv (20 samples)

## Performance Summary

| Source   | Model            |   BLEU ↑ |   BERTScore ↑ |   LLM Judge Accuracy (1-5) ↑ |   LLM Judge Fluency (1-5) ↑ |   Time (s) ↓ |
|:---------|:-----------------|---------:|--------------:|-----------------------------:|----------------------------:|-------------:|
| Pali     | Llama-3.3-70b    |     3.41 |          0.71 |                         3.05 |                        4.15 |         5.39 |
| Pali     | GPT-OSS-120b     |     2.25 |          0.73 |                         2.45 |                        3.00 |        10.60 |
| Pali     | Kimi-k2          |     5.89 |          0.75 |                         4.80 |                        4.90 |         7.64 |
| Pali     | Qwen3-32b        |     4.88 |          0.73 |                         3.10 |                        4.00 |         6.09 |
| Pali     | Gemini-2.5-Flash |    12.34 |          0.77 |                         5.00 |                        5.00 |        15.56 |

*Evaluation powered by LLM Judge (Gemini 2.0 Flash) using a 5-point rubric for Accuracy and Fluency.*
