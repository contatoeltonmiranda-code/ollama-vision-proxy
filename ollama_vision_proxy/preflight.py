"""Startup checks, so failures surface as advice instead of a runtime 400.

The important check is capability, not presence. A model can be pulled and still
have no vision (gemma3:1b is a live example), and pointing the proxy at one of
those fails on the first pasted image rather than at startup.
"""

from __future__ import annotations

import enum
import logging
import shutil
import subprocess
from typing import Callable, List, Optional

import httpx

logger = logging.getLogger(__name__)

VISION_CAPABILITY = "vision"
CHECK_TIMEOUT = 10.0


class PreflightError(Exception):
    """A startup check failed. The message is meant to be shown to the user."""


class VisionModelStatus(enum.Enum):
    OK = "ok"
    MISSING = "missing"
    NO_VISION = "no_vision"


def _http(client: Optional[httpx.Client]) -> httpx.Client:
    return client if client is not None else httpx.Client(timeout=CHECK_TIMEOUT)


def check_ollama_running(
    upstream_url: str, client: Optional[httpx.Client] = None
) -> str:
    """Return the Ollama version, or raise `PreflightError`."""
    base = upstream_url.rstrip("/")
    try:
        response = _http(client).get(f"{base}/api/version")
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001 - all failures are the same advice
        raise PreflightError(
            f"Cannot reach the Ollama server at {base} ({exc}). "
            "Start it with `ollama serve` and try again."
        ) from exc
    version = payload.get("version") if isinstance(payload, dict) else None
    return version if isinstance(version, str) else "unknown"


def model_capabilities(
    model: str, upstream_url: str, client: Optional[httpx.Client] = None
) -> List[str]:
    """Capabilities reported by `/api/show`; empty if the model is unknown."""
    base = upstream_url.rstrip("/")
    try:
        response = _http(client).post(f"{base}/api/show", json={"model": model})
        if response.status_code != 200:
            return []
        payload = response.json()
    except Exception as exc:  # noqa: BLE001
        logger.debug("could not read capabilities for %s: %s", model, exc)
        return []
    capabilities = payload.get("capabilities") if isinstance(payload, dict) else None
    if not isinstance(capabilities, list):
        return []
    return [item for item in capabilities if isinstance(item, str)]


def vision_model_status(
    model: str, upstream_url: str, client: Optional[httpx.Client] = None
) -> VisionModelStatus:
    capabilities = model_capabilities(model, upstream_url, client)
    if VISION_CAPABILITY in capabilities:
        return VisionModelStatus.OK
    if not capabilities:
        return VisionModelStatus.MISSING
    return VisionModelStatus.NO_VISION


def list_vision_models(
    upstream_url: str, client: Optional[httpx.Client] = None
) -> List[str]:
    """Pulled models that report the vision capability."""
    base = upstream_url.rstrip("/")
    http = _http(client)
    try:
        response = http.get(f"{base}/api/tags")
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001
        logger.debug("could not list models: %s", exc)
        return []
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        return []
    names = [
        entry["name"]
        for entry in models
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    ]
    return [
        name
        for name in names
        if VISION_CAPABILITY in model_capabilities(name, base, http)
    ]


def check_vision_model(
    model: str, upstream_url: str, client: Optional[httpx.Client] = None
) -> None:
    """Raise `PreflightError` unless `model` exists and supports vision."""
    status = vision_model_status(model, upstream_url, client)
    if status is VisionModelStatus.OK:
        return

    if status is VisionModelStatus.MISSING:
        message = (
            f"The vision model '{model}' is not available in Ollama. "
            f"Pull it with `ollama pull {model}`."
        )
    else:
        capabilities = model_capabilities(model, upstream_url, client)
        message = (
            f"The model '{model}' is pulled but does not support vision "
            f"(capabilities: {', '.join(capabilities)}). Pick a vision model."
        )

    alternatives = list_vision_models(upstream_url, client)
    if alternatives:
        message += (
            " Already pulled and vision-capable: " + ", ".join(alternatives) + "."
        )
    raise PreflightError(message)


def pull_model(model: str, ollama_bin: str = "ollama") -> int:
    """Run `ollama pull <model>`, streaming its progress to the terminal."""
    binary = shutil.which(ollama_bin) or ollama_bin
    logger.info("pulling %s", model)
    return subprocess.call([binary, "pull", model])


def find_claude_cli(which: Callable[[str], Optional[str]] = shutil.which) -> str:
    """Locate the `claude` executable, or raise `PreflightError`."""
    path = which("claude")
    if path:
        return path
    raise PreflightError(
        "Could not find the `claude` CLI on PATH. Install Claude Code "
        "(https://claude.com/claude-code) or add it to PATH."
    )
