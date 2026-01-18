# Pali-Vietnamese (Dhammapada) Benchmark Results

**Date**: 2026-01-17 19:35:49
**Judge Model**: groq/llama-3.3-70b-versatile
**Dataset**: pali_vi_dhammapada.csv (20 samples)

## Performance Summary

| Source   | Model         |   BLEU ↑ |   BERTScore ↑ |   LLM Judge Accuracy (1-5) ↑ |   LLM Judge Fluency (1-5) ↑ |   Time (s) ↓ |
|:---------|:--------------|---------:|--------------:|-----------------------------:|----------------------------:|-------------:|
| Pali     | Llama-3.3-70b |     3.05 |          0.71 |                         3.60 |                        4.25 |        10.20 |
| Pali     | GPT-OSS-120b  |     2.24 |          0.72 |                         1.60 |                        1.85 |        22.83 |
| Pali     | Kimi-k2       |     5.02 |          0.76 |                         0.00 |                        0.00 |        13.31 |
| Pali     | Qwen3-32b     |     0.21 |          0.56 |                         0.00 |                        0.00 |        47.73 |

*Evaluation powered by LLM Judge (Llama-3.3-70b) using a 5-point rubric for Accuracy and Fluency.*
