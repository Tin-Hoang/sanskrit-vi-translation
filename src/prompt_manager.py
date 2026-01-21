"""
Prompt Manager for Langfuse Integration.

Handles fetching prompts from Langfuse with local fallback support.
"""

import os
import logging
from typing import Optional, Any
from langfuse import Langfuse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptManager:
    """
    Manages retrieval of prompts from Langfuse with local fallback.
    """

    def __init__(self):
        """
        Initialize Langfuse client if credentials are available.
        """
        self.langfuse: Optional[Langfuse] = None
        self._init_langfuse()

    def _init_langfuse(self):
        """Initialize the Langfuse client."""
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")

        if public_key and secret_key:
            try:
                self.langfuse = Langfuse()
                logger.info("✅ Langfuse PromptManager initialized")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize Langfuse client: {e}")
        else:
            logger.warning(
                "⚠️ Langfuse credentials not found. Using local fallbacks only."
            )

    def get_prompt(
        self, name: str, fallback: str = "", label: str = "production"
    ) -> str:
        """
        Get a prompt template by name.

        Args:
            name: The name of the prompt in Langfuse.
            fallback: The local string to use if Langfuse fails or is not configured.
            label: The Langfuse label to fetch (default: "production").

        Returns:
            The prompt template string.
        """
        if self.langfuse:
            try:
                # Fetch prompt from Langfuse
                # cache_ttl_seconds enables internal caching in the SDK
                prompt = self.langfuse.get_prompt(
                    name, label=label, cache_ttl_seconds=300
                )
                return prompt.compile()  # Returns the raw template if no vars provided
            except Exception as e:
                logger.warning(f"⚠️ Failed to fetch prompt '{name}' from Langfuse: {e}")
                logger.info(f"Using local fallback for '{name}'")

        return fallback

    def compile_prompt(self, name: str, fallback: str = "", **kwargs) -> str:
        """
        Get and compile a prompt with variables.

        Args:
            name: Prompt name.
            fallback: Local fallback template.
            **kwargs: Variables to format the prompt with.

        Returns:
            Formatted prompt string.
        """
        template = self.get_prompt(name, fallback)
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing variable for prompt '{name}': {e}")
            return template  # Return unformatted template on error, or raise?
        except Exception as e:
            logger.error(f"Error formatting prompt '{name}': {e}")
            return template
