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

#: Bind port 0 and the OS hands out a free one. This is the default because the
#: proxy is private to one session: only the `claude` it spawns talks to it, and
#: that child is told the port through its environment. A fixed port would buy
#: nothing and would stop a second concurrent session from starting at all.
EPHEMERAL_PORT = 0
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

#: Generous cap for one chunk-size line; the stdlib uses the same order.
MAX_CHUNK_LINE = 65536


class _MalformedBody(Exception):
    """The request body could not be decoded."""


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
        port: int = EPHEMERAL_PORT,
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

    #: True once a status line is on the wire, so we never send a second one.
    _headers_sent = False

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
        self._headers_sent = False
        try:
            body = self._read_body()
        except _MalformedBody as exc:
            logger.error("could not read the request body: %s", exc)
            self._send_json_error(400, f"Malformed request body: {exc}")
            self.close_connection = True
            return

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
            if self._headers_sent:
                # A response is already committed on this connection. Writing a
                # second status line here would put "HTTP/1.1 502" inside the
                # body the client is still reading, so end the stream instead.
                self._finish_broken_stream(exc)
            else:
                self._send_json_error(502, f"Cannot reach the Ollama server: {exc}")

    def _read_body(self) -> bytes:
        encoding = (self.headers.get("Transfer-Encoding") or "").lower()
        if "chunked" in encoding:
            return self._read_chunked_body()
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            return b""
        return self.rfile.read(length)

    def _read_chunked_body(self) -> bytes:
        """Decode a chunked request body.

        Honouring only Content-Length silently dropped the payload AND left the
        chunk bytes in the socket, where they were then parsed as the next
        request, breaking every later request on the connection.
        """
        parts = []
        while True:
            line = self.rfile.readline(MAX_CHUNK_LINE)
            if not line:
                raise _MalformedBody("connection ended inside a chunked body")
            size_field = line.split(b";", 1)[0].strip()
            try:
                size = int(size_field, 16)
            except ValueError:
                raise _MalformedBody(f"bad chunk size {size_field!r}") from None
            if size == 0:
                self._consume_trailers()
                break
            chunk = self.rfile.read(size)
            if len(chunk) != size:
                raise _MalformedBody("chunked body ended early")
            parts.append(chunk)
            self.rfile.read(2)  # the CRLF that terminates the chunk
        return b"".join(parts)

    def _consume_trailers(self) -> None:
        while True:
            line = self.rfile.readline(MAX_CHUNK_LINE)
            if line in (b"\r\n", b"\n", b""):
                return

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
                self._write_chunk(chunk)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("client disconnected mid-stream")
            self.close_connection = True

    def _write_chunk(self, chunk: bytes) -> None:
        self.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
        self.wfile.flush()

    def _finish_broken_stream(self, exc: Exception) -> None:
        """End a stream whose headers are already sent, without a second response.

        The client is mid-SSE, so the only well-formed thing we can say is an
        error event followed by the terminating chunk.
        """
        payload = json.dumps(
            {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": f"Upstream stream failed: {exc}",
                },
            }
        )
        try:
            self._write_chunk(f"event: error\ndata: {payload}\n\n".encode("utf-8"))
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("client already gone while ending a broken stream")
        self.close_connection = True

    def _send_upstream_headers(self, upstream: httpx.Response, extra: dict) -> None:
        self._headers_sent = True
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
