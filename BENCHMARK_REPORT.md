# Sanskrit-Vietnamese Translation Benchmark Results

**Date**: 2026-01-17 10:41:55
**Judge Model**: groq/llama-3.3-70b-versatile
**Dataset**: sanskrit_vi_heart_sutra.csv (18 samples)

## Performance Summary

| Model         |   BLEU ↑ |   BERTScore ↑ |   LLM Judge Accuracy (1-5) ↑ |   LLM Judge Fluency (1-5) ↑ |   Time (s) ↓ |
|:--------------|---------:|--------------:|-----------------------------:|----------------------------:|-------------:|
| Llama-3.3-70b |     9.34 |          0.72 |                         4.17 |                        3.89 |        60.96 |
| GPT-OSS-120b  |     7.18 |          0.69 |                         4.11 |                        4.06 |        71.44 |
| Kimi-k2       |    21.52 |          0.74 |                         4.50 |                        4.17 |        63.57 |
| Qwen3-32b     |     0.48 |          0.53 |                         4.83 |                        4.83 |        84.16 |

*Evaluation powered by LLM Judge (Llama-3.3-70b) using a 5-point rubric for Accuracy and Fluency.*
