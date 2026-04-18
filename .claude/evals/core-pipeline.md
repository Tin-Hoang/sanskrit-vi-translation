# EVAL: core-pipeline

Regression evals for the deterministic parts of the benchmark pipeline.
Run these after ANY change to src/ to catch regressions.

## Code-Based Graders (automated)

```bash
# Run all unit tests
cd /home/tin/Research/ml-sandbox
uv run pytest tests/ -v && echo "PASS" || echo "FAIL"

# Targeted graders
uv run pytest tests/test_response_parser.py -v   # JSON parsing + Pydantic validation
uv run pytest tests/test_schemas.py -v           # Schema bounds + field validation
uv run pytest tests/test_utils.py -v             # Column detection + CSV loading
uv run pytest tests/test_prompt_manager.py -v   # Jinja2 + format() rendering
```

## Regression Evals

| Eval | Grader | Target |
|------|--------|--------|
| JSON cleanup (markdown code blocks) | `test_response_parser::TestCleanJsonResponse` | pass^3 = 100% |
| Double-encoded list fix | `test_response_parser::TestFixDoubleEncodedList` | pass^3 = 100% |
| Pydantic schema validation | `test_schemas` | pass^3 = 100% |
| Score bounds enforcement (1-5) | `test_schemas::TestJudgementItem` | pass^3 = 100% |
| Column detection (pali/sanskrit/ref) | `test_utils::TestIdentifyColumns` | pass^3 = 100% |
| Prompt Jinja2/format rendering | `test_prompt_manager` | pass^3 = 100% |

## Capability Evals (model-graded, run manually)

### smoke-test-translation
```
Task: Run `uv run src/main.py input=data/pali_vi_dhammapada.csv limit=1`
with a single model.

Success Criteria:
  - [ ] Exits without exception
  - [ ] Produces a row in results/ CSV
  - [ ] Translation column is non-empty
  - [ ] Judge scores are numeric (not 0/0)
```

### prompt-change-safe
```
Task: Modify a system prompt in src/system_prompts/ and re-run limit=2.

Success Criteria:
  - [ ] Response parser does not crash
  - [ ] New prompt content is reflected in LLM call (check Langfuse trace)
  - [ ] Scores are still valid numbers
```

## Metrics Targets

- Regression (code graders): pass^3 = 100%
- Capability (smoke test): pass@3 >= 90%
