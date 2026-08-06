"""Transcribe images with a local Ollama vision model, in two steps.

Step one asks the model what kind of image this is. Step two asks for the
description that kind deserves: verbatim text and application names for a
screenshot, subjects then background for a photograph. One generic prompt served
both badly, padding prose and inventing proper nouns.

Sampling is greedy. The model ships with temperature 1, which made transcription
a lottery: the same image produced different text run to run, and on one run the
model declared an image had no text when it plainly did.

Uses Ollama's native /api/chat rather than the Anthropic-compatible layer,
because that is the shape which accepts an images array.

Transcription is fail-soft throughout. Propagating a vision error would recreate
the bug this proxy exists to fix, where one bad block breaks every later turn.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

from .cache import TranscriptionCache
from .exif import extract_exif
from .geocode import ReverseGeocoder
from .metadata import render_metadata
from .prompts import ImageKind, classify_prompt, describe_prompt, parse_kind
from .transform import ImageBlock

logger = logging.getLogger(__name__)

DEFAULT_VISION_MODEL = "gemma3:4b"

#: Generous, because a cold vision model has to load before it can answer.
DEFAULT_TIMEOUT = 180.0

#: Greedy decoding: this is a reporting task, not a creative one.
DEFAULT_OPTIONS: Dict[str, Any] = {"temperature": 0}

#: The classifier only has to emit one word.
CLASSIFY_OPTIONS: Dict[str, Any] = {"temperature": 0, "num_predict": 8}


class VisionError(Exception):
    """The vision model did not return a usable description."""


@dataclass
class Transcription:
    """A description, plus the metadata block that follows it (if any)."""

    description: str
    metadata: Optional[str] = None
    kind: Optional[ImageKind] = None


class VisionTranscriber:
    """Callable that turns an `ImageBlock` into a `Transcription`."""

    def __init__(
        self,
        model: str = DEFAULT_VISION_MODEL,
        upstream_url: str = "http://127.0.0.1:11434",
        client: Optional[httpx.Client] = None,
        timeout: float = DEFAULT_TIMEOUT,
        prompt: Optional[str] = None,
        cache: Optional[TranscriptionCache] = None,
        geocoder: Optional[ReverseGeocoder] = None,
        classify: bool = True,
    ) -> None:
        self.model = model
        self.upstream_url = upstream_url.rstrip("/")
        self.timeout = timeout
        #: When set, overrides the per-kind prompt entirely.
        self.prompt = prompt
        self.classify = classify
        self.cache = cache if cache is not None else TranscriptionCache()
        self.geocoder = geocoder
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(timeout=timeout)

    def __call__(self, block: ImageBlock) -> Transcription:
        return self.transcribe(block)

    def transcribe(self, block: ImageBlock) -> Transcription:
        """Describe `block`, never raising."""
        if not block.data:
            logger.warning("image block has no inline base64 data; skipping")
            return Transcription(_failure("no inline image data"))

        try:
            return self.cache.get_or_compute(block.data, lambda: self._pipeline(block))
        except Exception as exc:  # noqa: BLE001 - fail-soft is the contract
            logger.warning("vision transcription failed: %s", exc)
            return Transcription(_failure(_reason(exc)))

    def _pipeline(self, block: ImageBlock) -> Transcription:
        kind = self._classify(block)
        description = self._describe(block, kind)
        return Transcription(
            description=description,
            metadata=self._metadata(block),
            kind=kind,
        )

    def _classify(self, block: ImageBlock) -> ImageKind:
        """Never fatal: an unknown kind just means the generic prompt."""
        if not self.classify or self.prompt is not None:
            return ImageKind.OTHER
        try:
            reply = self._chat(classify_prompt(), block, CLASSIFY_OPTIONS)
        except Exception as exc:  # noqa: BLE001
            logger.debug("classification failed, using the generic prompt: %s", exc)
            return ImageKind.OTHER
        kind = parse_kind(reply)
        logger.debug("classified image as %s (reply %r)", kind.value, reply)
        return kind

    def _describe(self, block: ImageBlock, kind: ImageKind) -> str:
        prompt = self.prompt if self.prompt is not None else describe_prompt(kind)
        return self._chat(prompt, block, DEFAULT_OPTIONS)

    def _metadata(self, block: ImageBlock) -> Optional[str]:
        """Best effort: metadata is a bonus, never a reason to fail."""
        try:
            raw = base64.b64decode(block.data or "", validate=False)
        except Exception as exc:  # noqa: BLE001
            logger.debug("could not decode image bytes for EXIF: %s", exc)
            return None

        exif = extract_exif(raw)
        if exif is None or not exif.has_gps:
            return None

        address = None
        if self.geocoder is not None:
            address = self.geocoder.lookup(exif.latitude, exif.longitude)
        return render_metadata(exif, address)

    def _chat(
        self, prompt: str, block: ImageBlock, options: Dict[str, Any]
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt, "images": [block.data]}
            ],
            "stream": False,
            "options": options,
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
        if self.geocoder is not None:
            self.geocoder.close()

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
