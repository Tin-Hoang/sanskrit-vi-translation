"""
Caching module for benchmark outputs to handle rate limits gracefully.

Provides JSON-based caching for:
- Translation outputs (per model, per task)
- LLM Judge outputs (per model pair, per task)
"""

import hashlib
import json
from pathlib import Path
from typing import Optional
from datetime import datetime


class BenchmarkCache:
    """Cache manager for benchmark translations and judgements."""

    def __init__(
        self,
        cache_dir: Path,
        task_name: str,
        translator_prompt_hash: str = "",
        evaluator_prompt_hash: str = "",
    ):
        """
        Initialize cache for a specific benchmark task.

        Args:
            cache_dir: Root cache directory (e.g., project_root/cache)
            task_name: Name of the benchmark task (e.g., "sanskrit-vi")
            translator_prompt_hash: Hash of translator prompts (for cache invalidation)
            evaluator_prompt_hash: Hash of evaluator prompts (for cache invalidation)
        """
        self.cache_dir = cache_dir
        self.task_name = task_name
        self.translator_prompt_hash = (
            translator_prompt_hash[:8] if translator_prompt_hash else "default"
        )
        self.evaluator_prompt_hash = (
            evaluator_prompt_hash[:8] if evaluator_prompt_hash else "default"
        )
        self.translations_dir = cache_dir / "translations"
        self.judgements_dir = cache_dir / "judgements"

        # Create directories
        self.translations_dir.mkdir(parents=True, exist_ok=True)
        self.judgements_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache (loaded lazily)
        self._translation_caches: dict[str, dict] = {}
        self._judgement_caches: dict[str, dict] = {}

    @staticmethod
    def _hash_content(content: str) -> str:
        """Generate a short hash for content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @staticmethod
    def _model_hash(model_id: str) -> str:
        """Generate a filesystem-safe hash for model ID."""
        # Use a short hash to avoid long filenames
        return hashlib.md5(model_id.encode()).hexdigest()[:8]

    @staticmethod
    def _short_model_name(model_id: str) -> str:
        """Extract a short, filesystem-safe model name for readability."""
        # Get last part of model path (e.g., 'groq/llama-3.3-70b' -> 'llama-3.3-70b')
        name = model_id.split("/")[-1]
        # Remove common suffixes and sanitize
        name = name.replace("-instruct", "").replace("-versatile", "")
        # Keep only alphanumeric, dash, underscore
        name = "".join(c if c.isalnum() or c in "-_" else "" for c in name)
        # Limit length
        return name[:20]

    def _get_translation_cache_path(self, model_id: str) -> Path:
        """Get path to translation cache file for a model."""
        short_name = self._short_model_name(model_id)
        model_hash = self._model_hash(model_id)
        prompt_hash = self.translator_prompt_hash
        return (
            self.translations_dir
            / f"{self.task_name}_{short_name}_{model_hash}_p{prompt_hash}.json"
        )

    def _get_judgement_cache_path(self, model_id: str, judge_model: str) -> Path:
        """Get path to judgement cache file for a model pair."""
        short_name = self._short_model_name(model_id)
        judge_short = self._short_model_name(judge_model)
        model_hash = self._model_hash(model_id)
        judge_hash = self._model_hash(judge_model)
        prompt_hash = self.evaluator_prompt_hash
        return (
            self.judgements_dir
            / f"{self.task_name}_{short_name}_{model_hash}_judge_{judge_short}_{judge_hash}_p{prompt_hash}.json"
        )

    def _load_translation_cache(self, model_id: str) -> dict:
        """Load translation cache from disk."""
        if model_id in self._translation_caches:
            return self._translation_caches[model_id]

        cache_path = self._get_translation_cache_path(model_id)
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)
                    self._translation_caches[model_id] = data
                    return data
            except (json.JSONDecodeError, IOError):
                pass

        # Initialize empty cache
        self._translation_caches[model_id] = {
            "metadata": {
                "model": model_id,
                "task": self.task_name,
                "created": datetime.now().isoformat(),
            },
            "entries": {},
        }
        return self._translation_caches[model_id]

    def _load_judgement_cache(self, model_id: str, judge_model: str) -> dict:
        """Load judgement cache from disk."""
        cache_key = f"{model_id}:{judge_model}"
        if cache_key in self._judgement_caches:
            return self._judgement_caches[cache_key]

        cache_path = self._get_judgement_cache_path(model_id, judge_model)
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)
                    self._judgement_caches[cache_key] = data
                    return data
            except (json.JSONDecodeError, IOError):
                pass

        # Initialize empty cache
        self._judgement_caches[cache_key] = {
            "metadata": {
                "model": model_id,
                "judge": judge_model,
                "task": self.task_name,
                "created": datetime.now().isoformat(),
            },
            "entries": {},
        }
        return self._judgement_caches[cache_key]

    def _save_translation_cache(self, model_id: str):
        """Persist translation cache to disk."""
        if model_id not in self._translation_caches:
            return
        cache_path = self._get_translation_cache_path(model_id)
        with open(cache_path, "w") as f:
            json.dump(
                self._translation_caches[model_id], f, ensure_ascii=False, indent=2
            )

    def _save_judgement_cache(self, model_id: str, judge_model: str):
        """Persist judgement cache to disk."""
        cache_key = f"{model_id}:{judge_model}"
        if cache_key not in self._judgement_caches:
            return
        cache_path = self._get_judgement_cache_path(model_id, judge_model)
        with open(cache_path, "w") as f:
            json.dump(
                self._judgement_caches[cache_key], f, ensure_ascii=False, indent=2
            )

    # === Public API ===

    def get_translation(
        self, model_id: str, source_text: str
    ) -> Optional[tuple[str, float]]:
        """
        Get cached translation if available.

        Args:
            model_id: The model used for translation
            source_text: The source text that was translated

        Returns:
            Tuple of (translation, time_seconds), or None if not cached
        """
        cache = self._load_translation_cache(model_id)
        content_hash = self._hash_content(source_text)
        entry = cache["entries"].get(content_hash)
        if entry is None:
            return None
        # Handle legacy cache entries (plain strings)
        if isinstance(entry, str):
            return (entry, 0.0)
        return (entry.get("translation", ""), entry.get("time_seconds", 0.0))

    def set_translation(
        self, model_id: str, source_text: str, translation: str, time_seconds: float
    ):
        """
        Cache a translation result with its processing time.

        Args:
            model_id: The model used for translation
            source_text: The source text that was translated
            translation: The resulting translation
            time_seconds: Time taken to generate this translation
        """
        cache = self._load_translation_cache(model_id)
        content_hash = self._hash_content(source_text)
        cache["entries"][content_hash] = {
            "translation": translation,
            "time_seconds": time_seconds,
        }
        self._save_translation_cache(model_id)

    def get_judgement(
        self, model_id: str, judge_model: str, source: str, candidate: str
    ) -> Optional[str]:
        """
        Get cached judgement if available.

        Args:
            model_id: The model that produced the translation
            judge_model: The model used for judging
            source: The source text
            candidate: The candidate translation being judged

        Returns:
            Cached judgement JSON string, or None if not cached
        """
        cache = self._load_judgement_cache(model_id, judge_model)
        # Hash both source and candidate to uniquely identify the judgement
        content_hash = self._hash_content(f"{source}|||{candidate}")
        return cache["entries"].get(content_hash)

    def set_judgement(
        self,
        model_id: str,
        judge_model: str,
        source: str,
        candidate: str,
        judgement: str,
    ):
        """
        Cache a judgement result.

        Args:
            model_id: The model that produced the translation
            judge_model: The model used for judging
            source: The source text
            candidate: The candidate translation being judged
            judgement: The judgement result (JSON string)
        """
        cache = self._load_judgement_cache(model_id, judge_model)
        content_hash = self._hash_content(f"{source}|||{candidate}")
        cache["entries"][content_hash] = judgement
        self._save_judgement_cache(model_id, judge_model)

    def get_translation_stats(self, model_id: str) -> tuple[int, int]:
        """Get (cached_count, total_entries) for translation cache."""
        cache = self._load_translation_cache(model_id)
        count = len(cache["entries"])
        return count, count

    def clear(self):
        """Clear all caches for this task."""
        for cache_file in self.translations_dir.glob(f"{self.task_name}_*.json"):
            cache_file.unlink()
        for cache_file in self.judgements_dir.glob(f"{self.task_name}_*.json"):
            cache_file.unlink()

        self._translation_caches.clear()
        self._judgement_caches.clear()
        print(f"Cleared cache for task: {self.task_name}")
