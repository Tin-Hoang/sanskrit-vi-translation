# Sanskrit-Vietnamese Heart Sutra Benchmark Results

**Date**: 2026-01-18 12:50:56
**Judge Model**: gemini/gemini-2.5-flash
**Dataset**: sanskrit_vi_heart_sutra.csv (18 samples)

## Performance Summary

| Source   | Model            |   BLEU ↑ |   BERTScore ↑ |   LLM Judge Accuracy (1-5) ↑ |   LLM Judge Fluency (1-5) ↑ |   Time (s) ↓ |
|:---------|:-----------------|---------:|--------------:|-----------------------------:|----------------------------:|-------------:|
| Sanskrit | Llama-3.3-70b    |    14.89 |          0.71 |                         4.28 |                        4.61 |         3.37 |
| Sanskrit | GPT-OSS-120b     |     9.55 |          0.71 |                         3.17 |                        3.56 |         8.66 |
| Sanskrit | Kimi-k2          |    33.21 |          0.76 |                         4.89 |                        4.94 |         4.42 |
| Sanskrit | Qwen3-32b        |    19.27 |          0.75 |                         4.00 |                        4.56 |         7.55 |
| Sanskrit | Gemini-2.5-Flash |    36.40 |          0.78 |                         4.94 |                        4.94 |        11.71 |

*Evaluation powered by LLM Judge (Gemini 2.0 Flash) using a 5-point rubric for Accuracy and Fluency.*
