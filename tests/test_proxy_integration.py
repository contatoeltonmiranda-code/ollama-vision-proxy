"""End-to-end proxy tests against a fake upstream Ollama server."""

import json
import socket
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from ollama_vision_proxy.proxy import ProxyServer
from ollama_vision_proxy.transform import wrap_transcription

SSE_EVENTS = [
    b'event: message_start\ndata: {"type":"message_start"}\n\n',
    b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0}\n\n',
    b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
]


class _QuietTestServer(ThreadingHTTPServer):
    """Fake upstream must not print tracebacks of its own.

    When the real client hangs up, the proxy correctly drops its upstream
    connection, which makes this fake server hit ConnectionResetError on its own
    idle keep-alive read. The default handler would dump that to stderr and mask
    what we are actually asserting, which is that the *proxy* stays silent.
    Unexpected errors are still recorded so nothing is hidden.
    """

    daemon_threads = True

    def handle_error(self, request, client_address):
        error = sys.exc_info()[1]
        if isinstance(error, (ConnectionError, TimeoutError)):
            return
        self.unexpected_errors.append(error)  # type: ignore[attr-defined]


class FakeUpstream:
    """Records what the proxy forwarded and replays a scripted response."""

    def __init__(self):
        self.requests = []
        self.mode = "json"
        self.status = 200
        self.slow_gap = 0.0
        self.unexpected_errors = []
        self._server = None
        self._thread = None

    def start(self):
        upstream = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def _record(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                upstream.requests.append(
                    {
                        "path": self.path,
                        "method": self.command,
                        "headers": dict(self.headers),
                        "raw": raw,
                    }
                )
                return raw

            def do_GET(self):
                self._record()
                self._respond_json({"version": "fake"})

            def do_POST(self):
                self._record()
                if upstream.mode == "sse":
                    self._respond_sse()
                elif upstream.mode == "sse_abort":
                    self._respond_sse_abort()
                elif upstream.mode == "sse_truncate":
                    self._respond_sse_truncate()
                elif upstream.mode == "status":
                    self._respond_json({"error": "upstream said no"}, upstream.status)
                else:
                    self._respond_json({"type": "message", "content": []})

            def _respond_json(self, payload, status=200):
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _respond_sse(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for index, event in enumerate(SSE_EVENTS):
                    if index > 0 and upstream.slow_gap:
                        time.sleep(upstream.slow_gap)
                    self.wfile.write(b"%x\r\n" % len(event) + event + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()

            def _respond_sse_abort(self):
                """Commit SSE headers, send one event, then reset the connection.

                This is the normal failure mode of a remote model: the stream
                dies after the proxy has already sent 200 to the client.
                """
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                event = SSE_EVENTS[0]
                self.wfile.write(b"%x\r\n" % len(event) + event + b"\r\n")
                self.wfile.flush()
                self.connection.setsockopt(
                    socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
                )
                self.connection.close()
                self.close_connection = True

            def _respond_sse_truncate(self):
                """Send one event then close gracefully, with no final chunk.

                Unlike the RST case, a FIN preserves already-delivered bytes, so
                this checks the good event survives while the stream still ends
                as a single well-formed response.
                """
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                event = SSE_EVENTS[0]
                self.wfile.write(b"%x\r\n" % len(event) + event + b"\r\n")
                self.wfile.flush()
                self.connection.shutdown(socket.SHUT_WR)
                self.close_connection = True

        self._server = _QuietTestServer(("127.0.0.1", 0), Handler)
        self._server.unexpected_errors = self.unexpected_errors
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    @property
    def url(self):
        return f"http://127.0.0.1:{self._server.server_port}"

    def stop(self):
        self._server.shutdown()
        self._server.server_close()

    def last_body(self):
        return json.loads(self.requests[-1]["raw"])


@pytest.fixture
def upstream():
    server = FakeUpstream().start()
    yield server
    server.stop()


@pytest.fixture
def proxy(upstream):
    def transcribe(block):
        return f"described({block.data})"

    server = ProxyServer(port=0, upstream_url=upstream.url, transcriber=transcribe)
    server.start()
    yield server
    server.stop()


def _image_request(data="IMGDATA", stream=False):
    return {
        "model": "glm-5.2:cloud",
        "max_tokens": 100,
        "stream": stream,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": data,
                        },
                    },
                ],
            }
        ],
    }


class TestImageTranscription:
    def test_upstream_receives_no_image_blocks(self, proxy, upstream):
        httpx.post(f"{proxy.url}/v1/messages", json=_image_request(), timeout=10)
        body = upstream.last_body()
        blocks = body["messages"][0]["content"]
        assert all(block["type"] != "image" for block in blocks)

    def test_image_is_replaced_with_its_description(self, proxy, upstream):
        httpx.post(f"{proxy.url}/v1/messages", json=_image_request("ABC"), timeout=10)
        blocks = upstream.last_body()["messages"][0]["content"]
        assert blocks[1] == {"type": "text", "text": wrap_transcription("described(ABC)")}

    def test_other_fields_survive(self, proxy, upstream):
        httpx.post(f"{proxy.url}/v1/messages", json=_image_request(), timeout=10)
        body = upstream.last_body()
        assert body["model"] == "glm-5.2:cloud"
        assert body["max_tokens"] == 100

    def test_content_length_is_recomputed(self, proxy, upstream):
        """The body grows when a short base64 becomes a long description."""
        httpx.post(f"{proxy.url}/v1/messages", json=_image_request("Q"), timeout=10)
        request = upstream.requests[-1]
        assert int(request["headers"]["Content-Length"]) == len(request["raw"])

    def test_text_only_request_is_forwarded_unchanged(self, proxy, upstream):
        payload = {"model": "m", "messages": [{"role": "user", "content": "plain"}]}
        httpx.post(f"{proxy.url}/v1/messages", json=payload, timeout=10)
        assert upstream.last_body() == payload

    def test_auth_headers_are_forwarded(self, proxy, upstream):
        httpx.post(
            f"{proxy.url}/v1/messages",
            json=_image_request(),
            headers={"x-api-key": "ollama", "anthropic-version": "2023-06-01"},
            timeout=10,
        )
        headers = upstream.requests[-1]["headers"]
        assert headers.get("x-api-key") == "ollama"
        assert headers.get("anthropic-version") == "2023-06-01"

    def test_proxy_delegates_caching_to_the_transcriber(self, upstream):
        calls = []

        def transcribe(block):
            calls.append(block.data)
            return "once"

        server = ProxyServer(port=0, upstream_url=upstream.url, transcriber=transcribe)
        server.start()
        try:
            for _ in range(3):
                httpx.post(
                    f"{server.url}/v1/messages", json=_image_request("SAME"), timeout=10
                )
        finally:
            server.stop()
        # The proxy passes the transcriber straight through; caching lives in
        # VisionTranscriber, so every call is expected to arrive here.
        assert calls == ["SAME", "SAME", "SAME"]


class TestResponses:
    def test_json_response_reaches_the_client(self, proxy):
        response = httpx.post(
            f"{proxy.url}/v1/messages", json=_image_request(), timeout=10
        )
        assert response.status_code == 200
        assert response.json() == {"type": "message", "content": []}

    def test_upstream_error_status_is_propagated(self, proxy, upstream):
        upstream.mode = "status"
        upstream.status = 400
        response = httpx.post(
            f"{proxy.url}/v1/messages", json=_image_request(), timeout=10
        )
        assert response.status_code == 400
        assert response.json()["error"] == "upstream said no"

    def test_sse_stream_arrives_intact_and_in_order(self, proxy, upstream):
        upstream.mode = "sse"
        with httpx.stream(
            "POST", f"{proxy.url}/v1/messages", json=_image_request(stream=True), timeout=10
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream"
            body = b"".join(response.iter_raw())
        assert body == b"".join(SSE_EVENTS)

    def test_sse_is_streamed_not_buffered(self, proxy, upstream):
        """First event must arrive before the upstream has finished sending."""
        upstream.mode = "sse"
        upstream.slow_gap = 0.5
        started = time.monotonic()
        first_byte_at = None
        with httpx.stream(
            "POST", f"{proxy.url}/v1/messages", json=_image_request(stream=True), timeout=15
        ) as response:
            for chunk in response.iter_raw():
                if chunk and first_byte_at is None:
                    first_byte_at = time.monotonic()
        total = time.monotonic() - started
        assert first_byte_at is not None
        assert first_byte_at - started < 0.4, "first event was buffered"
        assert total >= 0.5, "upstream did not actually stagger"


class TestPassthrough:
    def test_get_on_other_path_is_proxied(self, proxy, upstream):
        response = httpx.get(f"{proxy.url}/api/version", timeout=10)
        assert response.status_code == 200
        assert response.json() == {"version": "fake"}
        assert upstream.requests[-1]["path"] == "/api/version"

    def test_count_tokens_body_is_not_transformed(self, proxy, upstream):
        payload = _image_request()
        httpx.post(f"{proxy.url}/v1/messages/count_tokens", json=payload, timeout=10)
        assert upstream.last_body() == payload

    def test_unknown_post_path_is_not_transformed(self, proxy, upstream):
        payload = _image_request()
        httpx.post(f"{proxy.url}/v1/complete", json=payload, timeout=10)
        assert upstream.last_body() == payload
        assert upstream.requests[-1]["path"] == "/v1/complete"

    def test_query_string_is_preserved(self, proxy, upstream):
        httpx.get(f"{proxy.url}/api/tags?verbose=1", timeout=10)
        assert upstream.requests[-1]["path"] == "/api/tags?verbose=1"


class TestRobustness:
    def test_non_json_body_is_forwarded_untouched(self, proxy, upstream):
        response = httpx.post(
            f"{proxy.url}/v1/messages",
            content=b"not json",
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        assert response.status_code == 200
        assert upstream.requests[-1]["raw"] == b"not json"

    def test_empty_body_does_not_crash(self, proxy):
        response = httpx.post(f"{proxy.url}/v1/messages", content=b"", timeout=10)
        assert response.status_code == 200

    def test_json_array_body_is_forwarded_untouched(self, proxy, upstream):
        httpx.post(f"{proxy.url}/v1/messages", json=[1, 2, 3], timeout=10)
        assert upstream.last_body() == [1, 2, 3]

    def test_upstream_down_returns_502_not_a_crash(self, upstream):
        server = ProxyServer(
            port=0,
            upstream_url="http://127.0.0.1:1",
            transcriber=lambda block: "x",
        )
        server.start()
        try:
            response = httpx.post(
                f"{server.url}/v1/messages", json=_image_request(), timeout=10
            )
            assert response.status_code == 502
        finally:
            server.stop()

    def test_concurrent_requests_are_all_served(self, proxy):
        results = []

        def hit():
            response = httpx.post(
                f"{proxy.url}/v1/messages", json=_image_request(), timeout=15
            )
            results.append(response.status_code)

        threads = [threading.Thread(target=hit) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)
        assert results == [200] * 8


def _raw_exchange(port, request_bytes, reads=1):
    """Send raw bytes and read the whole response, so the wire is inspectable."""
    sock = socket.create_connection(("127.0.0.1", port), timeout=15)
    try:
        sock.sendall(request_bytes)
        sock.settimeout(10)
        received = b""
        while True:
            try:
                part = sock.recv(65536)
            except socket.timeout:
                break
            if not part:
                break
            received += part
            if reads and received.count(b"0\r\n\r\n") >= reads:
                break
        return received
    finally:
        sock.close()


def _chunked_request(path, payload, chunks=2):
    header = (
        f"POST {path} HTTP/1.1\r\nHost: x\r\n"
        "Content-Type: application/json\r\n"
        "Transfer-Encoding: chunked\r\n\r\n"
    ).encode()
    size = max(1, len(payload) // chunks)
    body = b""
    for start in range(0, len(payload), size):
        piece = payload[start : start + size]
        body += b"%x\r\n" % len(piece) + piece + b"\r\n"
    return header + body + b"0\r\n\r\n"


class TestMidStreamUpstreamFailure:
    """Once a 200 and SSE headers are committed, a later upstream failure must
    not put a second HTTP response inside the body the client is reading."""

    def test_only_one_http_response_is_written(self, proxy, upstream):
        upstream.mode = "sse_abort"
        payload = json.dumps(_image_request(stream=True)).encode()
        request = (
            b"POST /v1/messages HTTP/1.1\r\nHost: x\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload
        )
        raw = _raw_exchange(proxy.port, request)
        assert raw.count(b"HTTP/1.1") == 1, raw[:400]
        assert b"502" not in raw

    def test_stream_is_terminated_with_an_error_event(self, proxy, upstream):
        upstream.mode = "sse_abort"
        payload = json.dumps(_image_request(stream=True)).encode()
        request = (
            b"POST /v1/messages HTTP/1.1\r\nHost: x\r\n"
            b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload
        )
        raw = _raw_exchange(proxy.port, request)
        assert b"event: error" in raw
        # Not asserting the earlier event survived: an RST makes the receiver
        # discard its buffer, so losing it is TCP, not the proxy. The graceful
        # case below is where delivered bytes must be preserved.
        assert raw.rstrip().endswith(b"0")  # terminating zero-length chunk

    def test_graceful_truncation_keeps_delivered_events(self, proxy, upstream):
        upstream.mode = "sse_truncate"
        payload = json.dumps(_image_request(stream=True)).encode()
        request = (
            b"POST /v1/messages HTTP/1.1\r\nHost: x\r\n"
            b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload
        )
        raw = _raw_exchange(proxy.port, request)
        assert raw.count(b"HTTP/1.1") == 1, raw[:400]
        assert b"message_start" in raw
        assert b"event: error" in raw
        assert raw.rstrip().endswith(b"0")

    def test_upstream_failure_before_headers_still_yields_502(self, upstream):
        """The pre-headers path must keep returning a normal error response."""
        server = ProxyServer(
            port=0, upstream_url="http://127.0.0.1:1", transcriber=lambda b: "x"
        )
        server.start()
        try:
            response = httpx.post(
                f"{server.url}/v1/messages", json=_image_request(), timeout=10
            )
            assert response.status_code == 502
        finally:
            server.stop()


class TestChunkedRequestBodies:
    """Honouring only Content-Length dropped the payload and left the chunk
    bytes in the socket, where they were parsed as the next request."""

    def test_chunked_body_reaches_upstream_intact(self, proxy, upstream):
        payload = json.dumps(_image_request("CHUNKED")).encode()
        raw = _raw_exchange(proxy.port, _chunked_request("/v1/messages", payload))
        assert b"HTTP/1.1 200" in raw
        body = upstream.last_body()
        assert body["model"] == "glm-5.2:cloud"
        blocks = body["messages"][0]["content"]
        assert blocks[1] == {
            "type": "text",
            "text": wrap_transcription("described(CHUNKED)"),
        }

    def test_content_length_is_set_for_the_forwarded_request(self, proxy, upstream):
        payload = json.dumps(_image_request()).encode()
        _raw_exchange(proxy.port, _chunked_request("/v1/messages", payload))
        request = upstream.requests[-1]
        assert int(request["headers"]["Content-Length"]) == len(request["raw"])
        assert "chunked" not in (request["headers"].get("Transfer-Encoding") or "")

    def test_a_following_request_on_the_same_connection_still_works(
        self, proxy, upstream
    ):
        """The leftover-bytes bug swallowed every later request on the socket."""
        payload = json.dumps(_image_request("FIRST")).encode()
        second = json.dumps({"model": "m", "messages": []}).encode()
        request = _chunked_request("/v1/messages", payload) + (
            b"POST /v1/messages HTTP/1.1\r\nHost: x\r\n"
            b"Content-Length: " + str(len(second)).encode() + b"\r\n\r\n" + second
        )
        _raw_exchange(proxy.port, request, reads=0)
        paths = [entry["path"] for entry in upstream.requests]
        assert paths.count("/v1/messages") >= 2, upstream.requests
        assert upstream.last_body() == {"model": "m", "messages": []}

    def test_malformed_chunk_size_is_rejected_not_forwarded(self, proxy, upstream):
        before = len(upstream.requests)
        request = (
            b"POST /v1/messages HTTP/1.1\r\nHost: x\r\n"
            b"Transfer-Encoding: chunked\r\n\r\nZZZZ\r\n"
        )
        raw = _raw_exchange(proxy.port, request, reads=0)
        assert b"400" in raw
        assert len(upstream.requests) == before


class TestClientDisconnects:
    """Claude Code pools keep-alive connections and tears them down with a TCP
    RST. The stdlib only catches TimeoutError while reading the next request
    line, so an uncaught ConnectionResetError reaches socketserver.handle_error,
    which dumps a traceback to stderr and corrupts the terminal Claude Code is
    drawing in."""

    def _reset_after_request(self, proxy, count=3):
        for _ in range(count):
            sock = socket.create_connection(("127.0.0.1", proxy.port))
            sock.sendall(
                b"GET /api/version HTTP/1.1\r\nHost: x\r\n"
                b"Connection: keep-alive\r\n\r\n"
            )
            sock.recv(4096)
            # SO_LINGER with timeout 0 forces an abortive close (RST, not FIN),
            # while the server thread is blocked reading the next request.
            sock.setsockopt(
                socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
            )
            sock.close()
        time.sleep(0.8)

    def test_keepalive_reset_prints_nothing_to_stderr(self, proxy, capsys):
        self._reset_after_request(proxy)
        captured = capsys.readouterr()
        assert "Traceback" not in captured.err
        assert "ConnectionResetError" not in captured.err
        assert "Exception occurred during processing" not in captured.err

    def test_proxy_still_serves_after_a_reset(self, proxy):
        self._reset_after_request(proxy)
        response = httpx.get(f"{proxy.url}/api/version", timeout=10)
        assert response.status_code == 200

    def test_client_hangup_mid_request_is_silent(self, proxy, capsys):
        """Disconnecting after sending only a partial request line."""
        for _ in range(3):
            sock = socket.create_connection(("127.0.0.1", proxy.port))
            sock.sendall(b"POST /v1/messages HTTP/1.1\r\nHost: x\r\nContent-Len")
            sock.setsockopt(
                socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
            )
            sock.close()
        time.sleep(0.8)
        captured = capsys.readouterr()
        assert "Traceback" not in captured.err
        assert "Exception occurred during processing" not in captured.err

    def test_client_disconnects_while_streaming(self, proxy, upstream, capsys):
        """Hanging up mid-SSE must not raise on the write side either."""
        upstream.mode = "sse"
        upstream.slow_gap = 0.4
        with httpx.stream(
            "POST", f"{proxy.url}/v1/messages", json=_image_request(stream=True), timeout=15
        ) as response:
            next(response.iter_raw())  # take one chunk, then abandon the stream
        time.sleep(1.0)
        captured = capsys.readouterr()
        assert "Traceback" not in captured.err
        assert "Exception occurred during processing" not in captured.err
        # The quieted upstream must not be hiding a real failure.
        assert upstream.unexpected_errors == []


class TestLifecycle:
    def test_port_zero_picks_a_real_port(self, upstream):
        server = ProxyServer(
            port=0, upstream_url=upstream.url, transcriber=lambda block: "x"
        )
        server.start()
        try:
            assert server.port > 0
            assert server.url == f"http://127.0.0.1:{server.port}"
        finally:
            server.stop()

    def test_two_default_proxies_coexist_on_different_ports(self, upstream):
        # The regression this guards: a fixed default port let the first session
        # bind and made every later concurrent one die on "Address already in
        # use". Neither is given a port, so this exercises the default.
        first = ProxyServer(upstream_url=upstream.url, transcriber=lambda block: "x")
        second = ProxyServer(upstream_url=upstream.url, transcriber=lambda block: "x")
        first.start()
        try:
            second.start()
            try:
                assert first.port > 0
                assert second.port > 0
                assert first.port != second.port
            finally:
                second.stop()
        finally:
            first.stop()

    def test_stop_releases_the_port(self, upstream):
        server = ProxyServer(
            port=0, upstream_url=upstream.url, transcriber=lambda block: "x"
        )
        server.start()
        port = server.port
        server.stop()
        # Binding the same port again must succeed once stopped.
        again = ProxyServer(
            port=port, upstream_url=upstream.url, transcriber=lambda block: "x"
        )
        again.start()
        try:
            assert again.port == port
        finally:
            again.stop()
