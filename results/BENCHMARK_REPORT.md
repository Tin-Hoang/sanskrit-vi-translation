# Sanskrit-Vietnamese Translation Benchmark Results

**Date**: 2026-01-17 12:11:46
**Judge Model**: groq/llama-3.3-70b-versatile
**Dataset**: sanskrit_vi_heart_sutra.csv (18 samples)

## Performance Summary

| Model         |   BLEU ↑ |   BERTScore ↑ |   LLM Judge Accuracy (1-5) ↑ |   LLM Judge Fluency (1-5) ↑ |   Time (s) ↓ |
|:--------------|---------:|--------------:|-----------------------------:|----------------------------:|-------------:|
| Llama-3.3-70b |     7.37 |          0.70 |                         4.33 |                        3.89 |         8.38 |
| GPT-OSS-120b  |     9.79 |          0.69 |                         3.94 |                        3.89 |        18.39 |
| Kimi-k2       |    21.54 |          0.74 |                         0.28 |                        0.22 |         9.38 |
| Qwen3-32b     |     0.59 |          0.54 |                         0.00 |                        0.00 |        37.59 |

*Evaluation powered by LLM Judge (Llama-3.3-70b) using a 5-point rubric for Accuracy and Fluency.*
