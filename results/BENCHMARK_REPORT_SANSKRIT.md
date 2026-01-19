# Sanskrit-Vietnamese Heart Sutra Benchmark Results

**Date**: 2026-01-19 01:11:42
**Judge Model**: gemini/gemini-3-flash-preview
**Dataset**: sanskrit_vi_heart_sutra.csv (18 samples)

## Performance Summary

| Source   | Model                  |   BLEU ↑ |   BERTScore ↑ |   LLM Judge Accuracy (1-5) ↑ |   LLM Judge Fluency (1-5) ↑ |   Time (s) ↓ |
|:---------|:-----------------------|---------:|--------------:|-----------------------------:|----------------------------:|-------------:|
| Sanskrit | Llama-3.3-70b          |    15.78 |          0.73 |                         4.44 |                        4.44 |         3.32 |
| Sanskrit | GPT-OSS-120b           |    10.30 |          0.69 |                         4.28 |                        4.17 |         9.52 |
| Sanskrit | Kimi-k2                |    27.13 |          0.76 |                         5.00 |                        4.94 |         4.36 |
| Sanskrit | Qwen3-32b              |    18.69 |          0.75 |                         4.00 |                        4.50 |         7.24 |
| Sanskrit | Gemini-3-Flash-Preview |    41.84 |          0.76 |                         5.00 |                        5.00 |         5.50 |

*Evaluation powered by LLM Judge (Gemini 2.0 Flash) using a 5-point rubric for Accuracy and Fluency.*
