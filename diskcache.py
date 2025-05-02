import os
import pickle
from typing import Any, Optional

import numpy as np
from jax.typing import ArrayLike as Array

# TODO: Implement smart retrieval from disk of precomputed FFT matrices for (c_type, n, k) tuples. 
# Required for meaningful deployment for fast preprocessing.
class FFSATCache:
    """Unified caching system for FFSAT."""

    def __init__(self, cache_file: Optional[str] = None):
        self.cache_file = cache_file
        self.memory_cache: dict[tuple, Any] = {}
        self.modified = False

        if cache_file and os.path.exists(cache_file):
            self._load_cache()

    def _load_cache(self) -> None:
        """Load cache from disk."""
        try:
            with open(self.cache_file, "rb") as f:
                self.memory_cache = pickle.load(f)
        except (pickle.PickleError, EOFError):
            print(f"Warning: Could not load cache from {self.cache_file}, starting with empty cache")
            self.memory_cache = {}

    def _save_cache(self) -> None:
        """Save cache to disk."""
        if not self.cache_file:
            return

        # Only save if there have been modifications
        if not self.modified:
            return

        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.memory_cache, f)
            self.modified = False
        except (pickle.PickleError, IOError) as e:
            print(f"Warning: Failed to save cache to {self.cache_file}: {e}")

    def get(self, key: tuple) -> Optional[Any]:
        """Get a value from the cache."""
        return self.memory_cache.get(key)

    def put(self, key: tuple, value: Any) -> None:
        """Put a value in the cache."""
        self.memory_cache[key] = value
        self.modified = True

    def __contains__(self, key: tuple) -> bool:
        """Check if a key is in the cache."""
        return key in self.memory_cache

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Save cache to disk when exiting context."""
        self._save_cache()
