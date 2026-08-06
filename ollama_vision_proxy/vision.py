"""Transcribe images with a local Ollama vision model.

Uses Ollama's native `/api/chat` endpoint (not the Anthropic-compatible
`/v1/messages` layer) because the native shape is what accepts an `images`
array, and routing the vision call through the compat layer would put it back
through the very translation we are working around.

Transcription is fail-soft on purpose. Propagating a vision error would
reproduce the bug this proxy exists to fix, where one bad content block breaks
every subsequent turn in the session.
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx

from .cache import TranscriptionCache
from .transform import ImageBlock

logger = logging.getLogger(__name__)

DEFAULT_VISION_MODEL = "gemma3:4b"

DEFAULT_PROMPT = (
    "Describe this image in detail. If it contains text, transcribe the text "
    "exactly."
)

#: Generous, because a cold vision model has to load before it can answer.
DEFAULT_TIMEOUT = 180.0


class VisionError(Exception):
    """The vision model did not return a usable description."""


class VisionTranscriber:
    """Callable that turns an `ImageBlock` into a description string."""

    def __init__(
        self,
        model: str = DEFAULT_VISION_MODEL,
        upstream_url: str = "http://127.0.0.1:11434",
        client: Optional[httpx.Client] = None,
        timeout: float = DEFAULT_TIMEOUT,
        prompt: str = DEFAULT_PROMPT,
        cache: Optional[TranscriptionCache] = None,
    ) -> None:
        self.model = model
        self.upstream_url = upstream_url.rstrip("/")
        self.timeout = timeout
        self.prompt = prompt
        self.cache = cache if cache is not None else TranscriptionCache()
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(timeout=timeout)

    def __call__(self, block: ImageBlock) -> str:
        return self.transcribe(block)

    def transcribe(self, block: ImageBlock) -> str:
        """Describe `block`, returning a placeholder string on any failure."""
        if not block.data:
            logger.warning("image block has no inline base64 data; skipping")
            return _failure("no inline image data")

        try:
            return self.cache.get_or_compute(block.data, lambda: self._request(block))
        except Exception as exc:  # noqa: BLE001 - fail-soft is the contract
            logger.warning("vision transcription failed: %s", exc)
            return _failure(_reason(exc))

    def _request(self, block: ImageBlock) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": self.prompt, "images": [block.data]}
            ],
            "stream": False,
        }
        response = self._client.post(
            f"{self.upstream_url}/api/chat", json=payload, timeout=self.timeout
        )
        response.raise_for_status()

        data = response.json()
        message = data.get("message") if isinstance(data, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise VisionError("the vision model returned no description")
        return content.strip()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "VisionTranscriber":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _failure(reason: str) -> str:
    return f"transcription failed ({reason})"


def _reason(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"vision model returned HTTP {exc.response.status_code}"
    if isinstance(exc, httpx.TimeoutException):
        return "vision model timed out"
    if isinstance(exc, httpx.HTTPError):
        return "could not reach the vision model"
    if isinstance(exc, VisionError):
        return str(exc)
    if isinstance(exc, ValueError):
        return "vision model returned a malformed response"
    return type(exc).__name__
