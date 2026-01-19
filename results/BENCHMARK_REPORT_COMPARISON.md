# Pali vs Sanskrit Comparison Benchmark Results

**Date**: 2026-01-19 01:54:12
**Judge Model**: gemini/gemini-3-flash-preview
**Dataset**: dhammapada_udanavarga_parallel.csv (20 samples)

## Performance Summary

| Source   | Model                  |   BLEU ↑ |   BERTScore ↑ |   LLM Judge Accuracy (1-5) ↑ |   LLM Judge Fluency (1-5) ↑ |   Time (s) ↓ |
|:---------|:-----------------------|---------:|--------------:|-----------------------------:|----------------------------:|-------------:|
| Pali     | Llama-3.3-70b          |    10.62 |          0.71 |                         3.60 |                        3.85 |         4.81 |
| Pali     | GPT-OSS-120b           |     9.42 |          0.71 |                         2.95 |                        3.35 |        11.25 |
| Pali     | Kimi-k2                |    18.02 |          0.77 |                         4.80 |                        4.70 |         6.66 |
| Pali     | Qwen3-32b              |    13.08 |          0.75 |                         3.65 |                        3.95 |         5.25 |
| Pali     | Gemini-3-Flash-Preview |    39.16 |          0.84 |                         5.00 |                        4.60 |         7.61 |
| Sanskrit | Llama-3.3-70b          |    11.98 |          0.70 |                         3.65 |                        3.55 |         4.50 |
| Sanskrit | GPT-OSS-120b           |     6.54 |          0.66 |                         2.83 |                        2.89 |        10.75 |
| Sanskrit | Kimi-k2                |    12.61 |          0.70 |                         4.42 |                        4.37 |         5.87 |
| Sanskrit | Qwen3-32b              |     7.40 |          0.71 |                         3.80 |                        3.80 |         7.60 |
| Sanskrit | Gemini-3-Flash-Preview |    32.35 |          0.78 |                         5.00 |                        5.00 |         8.50 |

*Evaluation powered by LLM Judge using a 5-point rubric for Accuracy and Fluency.*
