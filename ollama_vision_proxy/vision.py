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

#: qwen3-vl:4b has better OCR (6/7 vs 5/7 on the benchmark, and 6/6 vs 5/6 on a
#: real terminal screenshot), and it was approved as the default. It is NOT the
#: default, because measuring the fixed pipeline end to end contradicted the
#: premise of that approval:
#:
#:   gemma3:4b      describe   20.8s     478 chars, no thinking
#:   qwen3-vl:4b    describe  202.2s     552 chars, 15759 chars of thinking
#:
#: The benchmark had flattered it. Its classifier was returning an empty string,
#: so every image fell back to the light generic prompt. Repairing the classifier
#: routed screenshots to the demanding screenshot prompt, which sends a thinking
#: model into 4000 tokens of deliberation for a 552-character answer. With the
#: 60s backstop that means screenshots always time out, so this default would be
#: broken for the exact case it was chosen to improve.
#:
#: Worth noting for whoever revisits this: qwen3-vl:4b on the GENERIC prompt was
#: 6/6 at 30.6s, which beats gemma3:4b on both counts. Not specialising the
#: prompt for thinking models looks like the fix, but it needs measuring.
DEFAULT_VISION_MODEL = "gemma3:4b"

#: A backstop, not a budget. Thinking models are erratic: the same describe call
#: measured 30.6s once and 248.8s another time, and a slow transcription blocks
#: the request it belongs to. Exceeding this is fail-soft, so the block becomes a
#: placeholder rather than an error. Raise it with --vision-timeout if a large
#: model on a cold load needs longer.
DEFAULT_TIMEOUT = 60.0

#: Describing one image needs a few thousand tokens, so the context must be set
#: explicitly. Without it the server default applies, and a host configured with
#: OLLAMA_CONTEXT_LENGTH=262144 made a 1.9 GB model reserve 32 GB of KV cache on
#: a 39 GB machine: it evicted every other model, swapped, and ran twice as slow
#: for byte-identical output.
DEFAULT_NUM_CTX = 8192


#: Enough for a thinking model to reason and then state its answer. Capped at 8
#: it emitted nothing at all: every token went into the reasoning and the reply
#: was truncated before the model ever committed to a category.
CLASSIFY_MAX_TOKENS = 512


def _options(num_ctx: int, **extra: Any) -> Dict[str, Any]:
    """Greedy decoding, with the context pinned to what the task needs."""
    return {"temperature": 0, "num_ctx": num_ctx, **extra}


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
        num_ctx: int = DEFAULT_NUM_CTX,
    ) -> None:
        self.model = model
        self.upstream_url = upstream_url.rstrip("/")
        self.timeout = timeout
        #: When set, overrides the per-kind prompt entirely.
        self.prompt = prompt
        self.classify = classify
        self.num_ctx = num_ctx
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
            reply = self._chat(
                classify_prompt(), block, _options(self.num_ctx, num_predict=CLASSIFY_MAX_TOKENS)
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("classification failed, using the generic prompt: %s", exc)
            return ImageKind.OTHER
        kind = parse_kind(reply)
        logger.debug("classified image as %s (reply %r)", kind.value, reply)
        return kind

    def _describe(self, block: ImageBlock, kind: ImageKind) -> str:
        prompt = self.prompt if self.prompt is not None else describe_prompt(kind)
        return self._chat(prompt, block, _options(self.num_ctx))

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
        reply = _reply_text(message)
        if not reply:
            raise VisionError("the vision model returned no description")
        return reply

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
        if self.geocoder is not None:
            self.geocoder.close()

    def __enter__(self) -> "VisionTranscriber":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _reply_text(message: Any) -> str:
    """The model's answer, from `content` or else from `thinking`.

    Thinking models put their reasoning in a separate `thinking` field and can
    leave `content` empty, which read as "no description" and lost an answer that
    had in fact been produced.
    """
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    thinking = message.get("thinking")
    if isinstance(thinking, str) and thinking.strip():
        logger.debug("model left content empty; falling back to its thinking")
        return thinking.strip()
    return ""


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
