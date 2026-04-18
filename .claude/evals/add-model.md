# EVAL: add-model

Capability eval for the most common task in this repo: adding a new LLM to the benchmark.

## Task Description

Add a new model entry to `config.yaml` under `translator.models` and verify it runs.

## Code-Based Graders

```bash
# 1. Model appears in config
grep -q "id:.*<new-model-id>" config.yaml && echo "PASS" || echo "FAIL"

# 2. Config parses without error
uv run python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('config.yaml')
models = cfg.translator.models
assert any(m.id == '<new-model-id>' for m in models), 'Model not found'
print('PASS')
"

# 3. Smoke-run with limit=1 (replace <dataset> and <model-id>)
uv run src/main.py input=data/pali_vi_dhammapada.csv limit=1 \
  'translator.models=[{id: "<new-model-id>", name: "Test", temperature: 0.3}]' \
  && echo "PASS" || echo "FAIL"
```

## Success Criteria

- [ ] `config.yaml` entry has `id`, `name`, `temperature` at minimum
- [ ] Config loads without OmegaConf error
- [ ] Smoke run produces non-empty translation for 1 item
- [ ] Judge scores are > 0 (not default 0/0 from failed parse)
- [ ] No Python exception in output

## Common Failure Modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Translation failed` | Wrong LiteLLM model string | Check provider prefix (`groq/`, `gemini/`, etc.) |
| Judge scores all 0 | Response parse failed | Check model supports `response_format: json_object` |
| `RateLimitError` | API quota hit | Add `api_key` override or reduce `batch_size` |
| `AuthenticationError` | Missing API key in `.env` | Add key to `.env` |

## pass@k Target

- pass@3 >= 90% (at least one successful run in 3 attempts)
