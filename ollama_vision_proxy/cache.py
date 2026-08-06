"""SHA256-keyed transcription cache with single-flight semantics.

Conversation history resends the same image on every turn, and Claude Code can
have several requests in flight at once (subagents, the haiku title call). So the
cache must both remember results and collapse concurrent duplicate work, while
never blocking one image behind another and never caching a failure.
"""

from __future__ import annotations

import hashlib
import threading
from typing import Callable, Dict, List, TypeVar

T = TypeVar("T")


class TranscriptionCache:
    """Thread-safe memo keyed by the SHA256 of the payload."""

    def __init__(self) -> None:
        self._values: Dict[str, object] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._guard = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key_for(payload: str) -> str:
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get_or_compute(self, payload: str, compute: Callable[[], T]) -> T:
        """Return the cached value for `payload`, computing it at most once.

        Concurrent callers for the same payload wait for the first one. Callers
        for different payloads never block each other. If `compute` raises, the
        exception propagates and nothing is cached.
        """
        key = self.key_for(payload)

        with self._guard:
            if key in self._values:
                self.hits += 1
                return self._values[key]  # type: ignore[return-value]
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock

        with lock:
            with self._guard:
                if key in self._values:
                    self.hits += 1
                    return self._values[key]  # type: ignore[return-value]
                self.misses += 1

            try:
                value = compute()
            except BaseException:
                with self._guard:
                    self._locks.pop(key, None)
                raise

            with self._guard:
                self._values[key] = value
                self._locks.pop(key, None)
            return value

    def keys(self) -> List[str]:
        with self._guard:
            return list(self._values)

    def __len__(self) -> int:
        with self._guard:
            return len(self._values)

    def __contains__(self, payload: str) -> bool:
        with self._guard:
            return self.key_for(payload) in self._values
