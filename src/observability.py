"""
Langfuse Observability Integration for Sanskrit-Vi Translation Benchmark.

This module provides LLM observability via Langfuse Cloud.
All LiteLLM calls (translation and evaluation) are automatically logged.

Environment Variables Required:
    LANGFUSE_PUBLIC_KEY: Your Langfuse public key
    LANGFUSE_SECRET_KEY: Your Langfuse secret key
    LANGFUSE_HOST: Langfuse host (default: https://cloud.langfuse.com)

    To get your keys:
        1. Sign up at https://cloud.langfuse.com
        2. Create a project
        3. Go to Settings -> API Keys
        4. Copy your public and secret keys

Usage:
    from observability import init_langfuse

    # Call once at application startup
    init_langfuse()

    # All subsequent LiteLLM calls are automatically traced
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Suppress harmless OpenTelemetry span warnings from LiteLLM's OTEL callback
# These occur when attributes are set on ended spans, which is a known issue
# in async tracing scenarios but doesn't affect functionality
logging.getLogger("opentelemetry.sdk.trace").setLevel(logging.ERROR)


def init_langfuse() -> bool:
    """
    Initialize Langfuse tracing for all LLM calls.

    Returns:
        True if Langfuse was initialized successfully, False otherwise
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        print(
            "⚠️  LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY not set - tracing disabled"
        )
        print("   Get your keys from: https://cloud.langfuse.com")
        return False

    try:
        import litellm

        # Map LANGFUSE_HOST to LANGFUSE_OTEL_HOST if not already set
        if host and "LANGFUSE_OTEL_HOST" not in os.environ:
            os.environ["LANGFUSE_OTEL_HOST"] = host

        # Set Langfuse OTEL as callback for all LiteLLM calls
        # Using 'langfuse_otel' as per https://langfuse.com/integrations/frameworks/litellm-sdk
        litellm.callbacks = ["langfuse_otel"]

        print(f"✅ Langfuse tracing enabled (via OTEL)")
        print(f"   View traces at: {host}")
        return True

    except ImportError as e:
        print(f"⚠️  Required packages not installed: {e}")
        print("   Run: uv add langfuse")
        return False
    except Exception as e:
        print(f"⚠️  Failed to initialize Langfuse: {e}")
        return False
