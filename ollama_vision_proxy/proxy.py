"""The HTTP proxy that sits between Claude Code and the Ollama server.

Only `POST /v1/messages` is inspected. Its JSON body has every image block
replaced with a text description, the body is re-serialised, and the request is
forwarded upstream. Every other request, and any body we cannot parse, is
forwarded byte for byte.

Server-sent-event responses are relayed chunk by chunk as they arrive rather
than buffered, so Claude Code renders tokens while the model is still producing
them.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import httpx

from . import __version__
from .transform import Transcriber, has_images, transform_request

logger = logging.getLogger(__name__)

DEFAULT_PROXY_PORT = 11435
DEFAULT_UPSTREAM_URL = "http://127.0.0.1:11434"

MESSAGES_PATH = "/v1/messages"

#: Headers that describe a single hop and must not be forwarded.
HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)

#: Read timeout is disabled: a streamed completion can legitimately take minutes.
UPSTREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=60.0, pool=60.0)

#: A client vanishing is routine, not an error. ConnectionError covers
#: ConnectionResetError, BrokenPipeError, and the ConnectionAbortedError that
#: Windows raises for the same situation.
DISCONNECT_ERRORS = (ConnectionError, TimeoutError)


class _QuietThreadingHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that does not dump tracebacks to stderr.

    The default `handle_error` prints a bare traceback, and with several
    connections dropping at once the threads interleave their output and wreck
    the terminal that Claude Code is drawing its interface in.
    """

    daemon_threads = True

    def handle_error(self, request, client_address) -> None:
        error = sys.exc_info()[1]
        if isinstance(error, DISCONNECT_ERRORS):
            logger.debug("client %s went away: %s", client_address, error)
            return
        # One atomic logging call, so concurrent threads cannot interleave.
        logger.exception("error while handling a request from %s", client_address)


class ProxyServer:
    """A threaded HTTP proxy. `port=0` binds an ephemeral port."""

    def __init__(
        self,
        port: int = DEFAULT_PROXY_PORT,
        upstream_url: str = DEFAULT_UPSTREAM_URL,
        transcriber: Optional[Transcriber] = None,
        host: str = "127.0.0.1",
        client: Optional[httpx.Client] = None,
    ) -> None:
        if transcriber is None:
            raise ValueError("a transcriber is required")
        self.host = host
        self._requested_port = port
        self.upstream_url = upstream_url.rstrip("/")
        self.transcriber = transcriber
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(
            timeout=UPSTREAM_TIMEOUT
        )
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "ProxyServer":
        httpd = _QuietThreadingHTTPServer((self.host, self._requested_port), _Handler)
        httpd.proxy = self  # type: ignore[attr-defined]
        self._httpd = httpd
        self._thread = threading.Thread(
            target=httpd.serve_forever, name="ovp-proxy", daemon=True
        )
        self._thread.start()
        logger.info("proxy listening on %s, forwarding to %s", self.url, self.upstream_url)
        return self

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        if self._owns_client:
            self._client.close()

    @property
    def port(self) -> int:
        if self._httpd is None:
            return self._requested_port
        return self._httpd.server_port

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def client(self) -> httpx.Client:
        return self._client

    def __enter__(self) -> "ProxyServer":
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"ollama-vision-proxy/{__version__}"

    # BaseHTTPRequestHandler writes to stderr by default; route through logging.
    def log_message(self, fmt: str, *args: object) -> None:
        logger.debug("%s - %s", self.address_string(), fmt % args)

    def handle(self) -> None:
        """Serve the keep-alive request loop, tolerating a vanishing client.

        The stdlib only catches TimeoutError around the read of the next request
        line, so an abortive close (RST) on an idle pooled connection escapes as
        ConnectionResetError. The same applies to the flush the stdlib performs
        after a handler returns, which is outside our own write guards.
        """
        try:
            super().handle()
        except DISCONNECT_ERRORS as exc:
            logger.debug("client connection dropped: %s", exc)
            self.close_connection = True

    def do_GET(self) -> None:
        self._proxy()

    def do_POST(self) -> None:
        self._proxy()

    def do_PUT(self) -> None:
        self._proxy()

    def do_PATCH(self) -> None:
        self._proxy()

    def do_DELETE(self) -> None:
        self._proxy()

    def do_HEAD(self) -> None:
        self._proxy()

    def do_OPTIONS(self) -> None:
        self._proxy()

    @property
    def _proxy_server(self) -> ProxyServer:
        return self.server.proxy  # type: ignore[attr-defined,no-any-return]

    def _proxy(self) -> None:
        body = self._read_body()
        if self.command == "POST" and self._is_messages_request():
            body = self._transcribe_images(body)

        headers = self._upstream_headers(body)
        target = self._proxy_server.upstream_url + self.path

        try:
            with self._proxy_server.client.stream(
                self.command, target, content=body, headers=headers
            ) as upstream:
                self._relay(upstream)
        except httpx.HTTPError as exc:
            logger.error("upstream request failed: %s", exc)
            self._send_json_error(502, f"Cannot reach the Ollama server: {exc}")

    def _read_body(self) -> bytes:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            return b""
        return self.rfile.read(length)

    def _is_messages_request(self) -> bool:
        """Exact match only, so /v1/messages/count_tokens passes through."""
        return self.path.split("?", 1)[0].rstrip("/") == MESSAGES_PATH

    def _transcribe_images(self, body: bytes) -> bytes:
        if not body:
            return body
        try:
            parsed = json.loads(body)
        except ValueError:
            logger.debug("body on %s is not JSON; forwarding unchanged", self.path)
            return body
        if not isinstance(parsed, dict) or not has_images(parsed):
            return body

        result = transform_request(parsed, self._proxy_server.transcriber)
        logger.info(
            "transcribed %d image(s), skipped %d",
            result.images_transcribed,
            result.images_skipped,
        )
        return json.dumps(result.body).encode("utf-8")

    def _upstream_headers(self, body: bytes) -> dict:
        headers = {
            name: value
            for name, value in self.headers.items()
            if name.lower() not in HOP_BY_HOP
            and name.lower() not in ("host", "content-length")
        }
        if body or self.command in ("POST", "PUT", "PATCH"):
            headers["Content-Length"] = str(len(body))
        return headers

    def _relay(self, upstream: httpx.Response) -> None:
        content_type = upstream.headers.get("content-type", "")
        if "text/event-stream" in content_type.lower():
            self._relay_stream(upstream)
        else:
            self._relay_buffered(upstream)

    def _relay_buffered(self, upstream: httpx.Response) -> None:
        body = b"".join(upstream.iter_raw())
        self._send_upstream_headers(upstream, extra={"Content-Length": str(len(body))})
        if self.command != "HEAD":
            self._write(body)

    def _relay_stream(self, upstream: httpx.Response) -> None:
        """Relay SSE using chunked framing, flushing every chunk immediately."""
        self._send_upstream_headers(upstream, extra={"Transfer-Encoding": "chunked"})
        try:
            for chunk in upstream.iter_raw():
                if not chunk:
                    continue
                self.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("client disconnected mid-stream")
            self.close_connection = True

    def _send_upstream_headers(self, upstream: httpx.Response, extra: dict) -> None:
        self.send_response(upstream.status_code)
        for name, value in upstream.headers.multi_items():
            lowered = name.lower()
            if lowered in HOP_BY_HOP or lowered == "content-length":
                continue
            self.send_header(name, value)
        for name, value in extra.items():
            self.send_header(name, value)
        self.end_headers()

    def _send_json_error(self, status: int, message: str) -> None:
        payload = json.dumps(
            {"type": "error", "error": {"type": "api_error", "message": message}}
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self._write(payload)

    def _write(self, data: bytes) -> None:
        try:
            self.wfile.write(data)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("client disconnected before the response was written")
            self.close_connection = True
