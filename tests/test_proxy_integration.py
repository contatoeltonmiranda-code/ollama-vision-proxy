"""End-to-end proxy tests against a fake upstream Ollama server."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from ollama_vision_proxy.proxy import ProxyServer

SSE_EVENTS = [
    b'event: message_start\ndata: {"type":"message_start"}\n\n',
    b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0}\n\n',
    b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
]


class FakeUpstream:
    """Records what the proxy forwarded and replays a scripted response."""

    def __init__(self):
        self.requests = []
        self.mode = "json"
        self.status = 200
        self.slow_gap = 0.0
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

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
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
        assert blocks[1] == {"type": "text", "text": "[Image: described(ABC)]"}

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
