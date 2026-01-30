"""
Base LLM Client providing shared functionality for LiteLLM-based services.

This module provides:
- BaseLLMClient: Abstract base class with retry logic, timing, and common parameters
- Shared configuration for tenacity retry decorators
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic, Tuple

import litellm
from litellm.exceptions import RateLimitError
from pydantic import BaseModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from response_parser import clean_json_response

T = TypeVar("T", bound=BaseModel)


# Shared retry decorator configuration
def llm_retry():
    """Standard retry decorator for LLM calls with rate limit handling."""
    return retry(
        retry=retry_if_exception_type(RateLimitError),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
    )


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM-powered services.

    Provides shared functionality:
    - Model parameter building
    - Retry logic for rate limits
    - Timing for LLM calls
    - Response cleaning utilities

    Subclasses must implement:
    - _build_prompt(): Build domain-specific prompts
    - _parse_response(): Parse responses into domain objects
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.3,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.api_base = api_base
        self.api_key = api_key
        self.extra_params = extra_params or {}

    def _build_model_params(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build common model parameters for litellm.acompletion.

        Args:
            prompt: The user prompt content
            session_id: Optional session ID for tracking
            tags: Optional list of tags for metadata
            trace_id: Optional trace ID for observability

        Returns:
            Dictionary of parameters ready for litellm.acompletion
        """
        params = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "metadata": {
                "session_id": session_id,
                "tags": tags or [],
                "trace_id": trace_id,
            },
        }

        if self.api_base:
            params["api_base"] = self.api_base
        if self.api_key:
            params["api_key"] = self.api_key

        # Merge extra parameters (reasoning_effort, top_p, etc.)
        params.update(self.extra_params)

        return params

    async def _call_llm(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        trace_id: Optional[str] = None,
    ) -> Tuple[str, float]:
        """
        Make an async LLM call with timing.

        Args:
            prompt: The prompt to send
            session_id: Optional session ID
            tags: Optional tags for metadata
            trace_id: Optional trace ID

        Returns:
            Tuple of (response_content, elapsed_time_seconds)
        """
        model_params = self._build_model_params(
            prompt=prompt,
            session_id=session_id,
            tags=tags,
            trace_id=trace_id,
        )

        start_time = time.time()
        response = await litellm.acompletion(**model_params)
        end_time = time.time()

        content = response.choices[0].message.content
        cleaned_content = clean_json_response(content)

        return cleaned_content, end_time - start_time

    @abstractmethod
    def _build_prompt(self, **kwargs) -> str:
        """
        Build the prompt for the LLM call.

        Subclasses implement this with domain-specific prompt construction.
        """
        pass

    @abstractmethod
    def _parse_response(self, content: str, batch_size: int) -> List[Any]:
        """
        Parse the LLM response into domain-specific results.

        Args:
            content: The cleaned JSON response content
            batch_size: Expected number of items in the batch

        Returns:
            List of parsed results
        """
        pass

    async def _process_batch_with_retry(
        self,
        batch_items: List[Any],
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        **prompt_kwargs,
    ) -> Tuple[List[Any], float]:
        """
        Process a batch with retry logic.

        This method applies the standard retry decorator and handles
        the full batch processing flow.

        Args:
            batch_items: Items to process in this batch
            session_id: Optional session ID
            trace_id: Optional trace ID
            **prompt_kwargs: Additional kwargs passed to _build_prompt

        Returns:
            Tuple of (parsed_results, elapsed_time)
        """

        @llm_retry()
        async def _inner():
            prompt = self._build_prompt(batch_items=batch_items, **prompt_kwargs)

            content, elapsed = await self._call_llm(
                prompt=prompt,
                session_id=session_id,
                tags=[self.model_name],
                trace_id=trace_id,
            )

            results = self._parse_response(content, len(batch_items))
            return results, elapsed

        return await _inner()

    async def process_batches(
        self,
        items: List[Any],
        batch_size: int = 10,
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        default_result: Any = None,
        **prompt_kwargs,
    ) -> Tuple[List[Any], float]:
        """
        Process items in parallel batches.

        Args:
            items: All items to process
            batch_size: Number of items per batch
            session_id: Optional session ID
            trace_id: Optional trace ID
            default_result: Default value for failed items
            **prompt_kwargs: Additional kwargs for prompt building

        Returns:
            Tuple of (all_results, total_time)
        """
        if not items:
            return [], 0.0

        tasks = []
        batch_map = []

        for i in range(0, len(items), batch_size):
            batch_end = min(i + batch_size, len(items))
            batch_items = items[i:batch_end]
            batch_map.append((i, batch_end))

            tasks.append(
                self._process_batch_with_retry(
                    batch_items=batch_items,
                    session_id=session_id,
                    trace_id=trace_id,
                    **prompt_kwargs,
                )
            )

        print(f"Dispatching {len(tasks)} async batches for {len(items)} items...")

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        all_results = [default_result] * len(items)
        total_time = 0.0

        for idx, result in enumerate(results):
            start_idx, end_idx = batch_map[idx]

            if isinstance(result, Exception):
                print(f"Batch {idx} failed: {result}")
            else:
                batch_results, batch_time = result
                total_time += batch_time

                for k, res in enumerate(batch_results):
                    if start_idx + k < len(all_results):
                        all_results[start_idx + k] = res

        return all_results, total_time
