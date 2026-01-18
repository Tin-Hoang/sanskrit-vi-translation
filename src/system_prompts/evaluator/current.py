"""
Current evaluator prompts - Re-exports the latest version.

To update to a new version, change the import below.
"""

from .v1 import EVALUATION_RUBRIC, BATCH_JUDGE_PROMPT, SINGLE_JUDGE_PROMPT

__all__ = ["EVALUATION_RUBRIC", "BATCH_JUDGE_PROMPT", "SINGLE_JUDGE_PROMPT"]
