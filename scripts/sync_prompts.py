"""
Sync Prompts Script

Bidirectional sync between local prompt files and Langfuse.

Usage:
    python scripts/sync_prompts.py --push   # Upload local prompts to Langfuse
    python scripts/sync_prompts.py --pull   # Update local files from Langfuse
"""

import argparse
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add src to path
from langfuse import Langfuse
from src.system_prompts.translator.current import (
    SINGLE_TRANSLATE_PROMPT,
    BATCH_TRANSLATE_PROMPT,
)
from src.system_prompts.evaluator.current import (
    EVALUATION_RUBRIC,
    BATCH_JUDGE_PROMPT,
    SINGLE_JUDGE_PROMPT,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Map prompt names to local variables
PROMPT_MAP = {
    "translator-single": SINGLE_TRANSLATE_PROMPT,
    "translator-batch": BATCH_TRANSLATE_PROMPT,
    "evaluator-rubric": EVALUATION_RUBRIC,
    "evaluator-batch": BATCH_JUDGE_PROMPT,
    "evaluator-single": SINGLE_JUDGE_PROMPT,
}

# Map prompt names to file paths (for pulling)
# Note: For simplicity, we are writing back to specific files.
# In a more complex setup, this could be dynamic.
FILE_MAP = {
    "translator-single": "src/system_prompts/translator/v1.py",
    "translator-batch": "src/system_prompts/translator/v1.py",
    "evaluator-rubric": "src/system_prompts/evaluator/v2.py",
    "evaluator-batch": "src/system_prompts/evaluator/v2.py",
    "evaluator-single": "src/system_prompts/evaluator/v2.py",
}


def safe_read_file(path: str) -> str:
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace for comparison (strip trailing, normalize newlines)."""
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def push_prompts(langfuse: Langfuse, force: bool = False):
    """Upload local prompts to Langfuse (only if changed).

    Args:
        langfuse: Langfuse client
        force: If True, upload all prompts regardless of changes
    """
    logger.info("🚀 Checking prompts for changes...")

    updated_count = 0
    skipped_count = 0
    new_count = 0

    for name, local_content in PROMPT_MAP.items():
        # Try to get current version from Langfuse
        remote_content = None
        try:
            prompt = langfuse.get_prompt(name, label="production")
            remote_content = prompt.compile()
        except Exception:
            # Prompt doesn't exist yet
            pass

        # Compare normalized content
        local_normalized = normalize_whitespace(local_content)
        remote_normalized = (
            normalize_whitespace(remote_content) if remote_content else None
        )

        if not force and remote_normalized == local_normalized:
            logger.info(f"  ⏭️  '{name}' - unchanged, skipping")
            skipped_count += 1
            continue

        # Upload the prompt
        status = "🆕 new" if remote_content is None else "✅ updated"
        logger.info(f"  {status}: '{name}'")

        langfuse.create_prompt(
            name=name,
            prompt=local_content,
            labels=["production"],
            type="text",
        )

        if remote_content is None:
            new_count += 1
        else:
            updated_count += 1

    logger.info(
        f"\n📊 Summary: {updated_count} updated, {new_count} new, {skipped_count} unchanged"
    )


def pull_prompts(langfuse: Langfuse):
    """Update local files from Langfuse production prompts."""
    logger.info("📥 Pulling prompts from Langfuse...")

    # We will read the target files, simple string replace the content
    # WARNING: This is a simple implementation. A more robust one would use AST.
    # For now, we assume the file structure hasn't drifted significantly.

    # Ideally, we would update the constants in the files.
    # But since these are Python files, "pulling" back into code is risky with simple regex.
    # A safer approach for a "Pull" workflow is to write to JSON files instead of Python files,
    # OR mainly use Pull to see what changed.

    # For this implementation, let's write to a new 'pulled' directory to avoid overwriting code destructively suitable for manual review.
    # Alternatively, we can just print the diffs.

    output_dir = Path("src/system_prompts/pulled")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"  Writing pulled prompts to {output_dir} for review...")

    for name in PROMPT_MAP.keys():
        try:
            prompt = langfuse.get_prompt(name, label="production")
            content = prompt.compile()

            # Write to individual file
            file_path = output_dir / f"{name}.txt"
            with open(file_path, "w") as f:
                f.write(content)

            logger.info(f"  - Saved '{name}' to {file_path}")

        except Exception as e:
            logger.error(f"  ❌ Failed to pull '{name}': {e}")

    logger.info("✅ Pull complete! Please review changes in src/system_prompts/pulled/")
    logger.info("  To apply changes, manually copy content to src/system_prompts/.")


def main():
    parser = argparse.ArgumentParser(description="Sync prompts with Langfuse")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--push",
        action="store_true",
        help="Push local prompts to Langfuse (only changed)",
    )
    group.add_argument(
        "--pull", action="store_true", help="Pull production prompts from Langfuse"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force push all prompts, ignoring change detection",
    )

    args = parser.parse_args()

    try:
        langfuse = Langfuse()
    except Exception as e:
        logger.error(f"❌ Failed to initialize Langfuse: {e}")
        logger.error("  Check LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY env vars.")
        sys.exit(1)

    if args.push:
        push_prompts(langfuse, force=args.force)
    elif args.pull:
        pull_prompts(langfuse)


if __name__ == "__main__":
    main()
