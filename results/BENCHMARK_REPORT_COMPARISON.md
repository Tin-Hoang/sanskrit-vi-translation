# Pali vs Sanskrit Comparison Benchmark Results

**Date**: 2026-01-17 15:46:09
**Judge Model**: groq/llama-3.3-70b-versatile
**Dataset**: dhammapada_udanavarga_parallel.csv (20 samples)

## Performance Summary

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

*Evaluation powered by LLM Judge (Llama-3.3-70b) using a 5-point rubric for Accuracy and Fluency.*
