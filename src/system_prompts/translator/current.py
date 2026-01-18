"""
Current translator prompts - Re-exports the latest version.

To update to a new version, change the import below.
"""

from .v1 import SINGLE_TRANSLATE_PROMPT, BATCH_TRANSLATE_PROMPT

__all__ = ["SINGLE_TRANSLATE_PROMPT", "BATCH_TRANSLATE_PROMPT"]
