"""The `ovp` command line entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional, Sequence

import httpx

from . import __version__
from .geocode import NOMINATIM_URL, ReverseGeocoder
from .launcher import build_claude_env, run_claude, split_passthrough
from .preflight import (
    CHECK_TIMEOUT,
    PreflightError,
    VisionModelStatus,
    check_ollama_running,
    check_vision_model,
    find_claude_cli,
    pull_model,
    vision_model_status,
)
from .proxy import DEFAULT_PROXY_PORT, DEFAULT_UPSTREAM_URL, ProxyServer
from .vision import (
    DEFAULT_NUM_CTX,
    DEFAULT_TIMEOUT,
    DEFAULT_VISION_MODEL,
    VisionTranscriber,
)

logger = logging.getLogger("ovp")

EXIT_PREFLIGHT_FAILED = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ovp",
        description=(
            "Run Claude Code against a text-only Ollama model with image "
            "support, by transcribing images with a local vision model first."
        ),
    )
    parser.add_argument("--version", action="version", version=f"ovp {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    launch = subcommands.add_parser(
        "launch",
        help="start the proxy and run the claude CLI through it",
        description=(
            "Everything after a bare -- is passed straight to the claude CLI, "
            "for example: ovp launch --target-model glm-5.2:cloud -- --agent manager"
        ),
    )
    _add_launch_arguments(launch)
    return parser


def _add_launch_arguments(launch: argparse.ArgumentParser) -> None:
    launch.add_argument(
        "--target-model",
        required=True,
        help="the text-only model Claude Code should talk to, e.g. glm-5.2:cloud",
    )
    launch.add_argument(
        "--vision-model",
        default=DEFAULT_VISION_MODEL,
        help=f"local Ollama vision model used for transcription (default: {DEFAULT_VISION_MODEL})",
    )
    launch.add_argument(
        "--proxy-port",
        type=int,
        default=DEFAULT_PROXY_PORT,
        help=f"port for this proxy to listen on (default: {DEFAULT_PROXY_PORT})",
    )
    launch.add_argument(
        "--upstream-url",
        default=DEFAULT_UPSTREAM_URL,
        help=f"the Ollama server to forward to (default: {DEFAULT_UPSTREAM_URL})",
    )
    launch.add_argument(
        "--vision-timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"seconds to wait for a transcription (default: {DEFAULT_TIMEOUT:.0f})",
    )
    launch.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="pull a missing vision model without asking",
    )
    _add_output_arguments(launch)


def _add_output_arguments(launch: argparse.ArgumentParser) -> None:
    launch.add_argument(
        "--vision-context",
        type=int,
        default=DEFAULT_NUM_CTX,
        help="context window for the vision model (default: "
        f"{DEFAULT_NUM_CTX}). Set explicitly so a large OLLAMA_CONTEXT_LENGTH on "
        "the host cannot reserve tens of GB of KV cache per image",
    )
    launch.add_argument(
        "--geocode-url",
        default=NOMINATIM_URL,
        help="reverse geocoding endpoint used to name a photo's location "
        f"(default: {NOMINATIM_URL})",
    )
    launch.add_argument(
        "--no-geocode",
        action="store_true",
        help="skip the address lookup; coordinates are still reported",
    )
    launch.add_argument(
        "--log-file",
        default=None,
        help="write full logs here instead of the terminal, so the Claude Code "
        "interface stays clean",
    )
    launch.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="debug logging, kept on the terminal even during the session",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    own_args, claude_args = split_passthrough(raw)
    args = build_parser().parse_args(own_args)

    _configure_logging(args.verbose, args.log_file)

    try:
        return _launch(args, claude_args)
    except PreflightError as exc:
        logger.error("%s", exc)
        return EXIT_PREFLIGHT_FAILED
    except KeyboardInterrupt:
        return 130


def _launch(args: argparse.Namespace, claude_args: List[str]) -> int:
    # One client for every startup check, so nothing is left unclosed.
    with httpx.Client(timeout=CHECK_TIMEOUT) as checks:
        version = check_ollama_running(args.upstream_url, checks)
        logger.info("ollama %s reachable at %s", version, args.upstream_url)
        _ensure_vision_model(args, checks)

    claude_path = find_claude_cli()
    logger.debug("using claude at %s", claude_path)

    transcriber, proxy = _build_pipeline(args)

    try:
        proxy.start()
    except OSError as exc:
        transcriber.close()
        raise PreflightError(
            f"Cannot listen on port {args.proxy_port} ({exc}). Another process "
            "may already be using it; pass --proxy-port to pick another."
        ) from exc

    try:
        logger.info(
            "proxying %s to %s, transcribing images with %s",
            proxy.url,
            args.upstream_url,
            args.vision_model,
        )
        env = build_claude_env(args.target_model, proxy.port)
        # claude owns the terminal from here on. Anything we write to stderr
        # lands in the middle of its interface, so go quiet for the session.
        set_terminal_logging(False)
        return run_claude(claude_path, claude_args, env)
    finally:
        set_terminal_logging(True)
        proxy.stop()
        transcriber.close()
        logger.info(
            "transcription cache: %d hit(s), %d miss(es)",
            transcriber.cache.hits,
            transcriber.cache.misses,
        )


def _build_pipeline(args: argparse.Namespace):
    """The transcriber and the proxy that will call it."""
    transcriber = VisionTranscriber(
        model=args.vision_model,
        upstream_url=args.upstream_url,
        timeout=args.vision_timeout,
        num_ctx=args.vision_context,
        geocoder=ReverseGeocoder(url=args.geocode_url, enabled=not args.no_geocode),
    )
    proxy = ProxyServer(
        port=args.proxy_port,
        upstream_url=args.upstream_url,
        transcriber=transcriber,
    )
    return transcriber, proxy


def _ensure_vision_model(
    args: argparse.Namespace, client: Optional[httpx.Client] = None
) -> None:
    status = vision_model_status(args.vision_model, args.upstream_url, client)
    if status is VisionModelStatus.OK:
        logger.debug("vision model %s is ready", args.vision_model)
        return

    if status is VisionModelStatus.NO_VISION:
        # Pulling cannot fix a model that simply has no vision support.
        check_vision_model(args.vision_model, args.upstream_url, client)
        return

    question = f"Vision model '{args.vision_model}' is not pulled. Pull it now?"
    if not (args.yes or _confirm(question)):
        raise PreflightError(
            "Cannot transcribe images without a vision model. Pull one with "
            f"`ollama pull {args.vision_model}`, or choose another with --vision-model."
        )

    if pull_model(args.vision_model) != 0:
        raise PreflightError(f"`ollama pull {args.vision_model}` failed.")
    check_vision_model(args.vision_model, args.upstream_url, client)


def _confirm(question: str) -> bool:
    if not sys.stdin.isatty():
        return False
    sys.stderr.write(f"ovp: {question} [y/N] ")
    sys.stderr.flush()
    try:
        answer = input()
    except EOFError:
        return False
    return answer.strip().lower() in ("y", "yes")


#: The stderr handler, kept so it can be detached while claude owns the screen.
_terminal_handler: Optional[logging.Handler] = None
_keep_terminal_logging = False


def _configure_logging(verbose: bool, log_file: Optional[str] = None) -> None:
    global _terminal_handler, _keep_terminal_logging
    _keep_terminal_logging = verbose

    handlers: List[logging.Handler] = []
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        handlers.append(file_handler)

    # Without this, detaching the stderr handler can leave the root logger with
    # no handlers at all, and logging falls back to logging.lastResort, which
    # prints WARNING and above straight to stderr. A NullHandler keeps the
    # fallback from ever engaging, so silence really is silence.
    handlers.append(logging.NullHandler())

    _terminal_handler = logging.StreamHandler(sys.stderr)
    _terminal_handler.setFormatter(logging.Formatter("ovp: %(message)s"))
    handlers.append(_terminal_handler)

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        handlers=handlers,
        force=True,
    )
    # httpx logs every request at INFO, which would echo Claude Code's traffic.
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(
            logging.DEBUG if verbose else logging.WARNING
        )


def set_terminal_logging(enabled: bool) -> None:
    """Attach or detach the stderr handler.

    Startup messages are useful, but once claude is drawing its interface any
    further stderr write appears inside it. With -v the caller has asked for the
    noise, so it stays. A --log-file keeps the full record either way.
    """
    if _terminal_handler is None or _keep_terminal_logging:
        return
    root = logging.getLogger()
    if enabled and _terminal_handler not in root.handlers:
        root.addHandler(_terminal_handler)
    elif not enabled and _terminal_handler in root.handlers:
        root.removeHandler(_terminal_handler)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
