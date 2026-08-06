"""Tests for the Ollama vision client (fail-soft transcription)."""

import json

import httpx

from ollama_vision_proxy.transform import ImageBlock
from ollama_vision_proxy.vision import VisionTranscriber


def _block(data="BASE64DATA", media_type="image/png"):
    return ImageBlock(source_type="base64", media_type=media_type, data=data, url=None)


def _transcriber(handler, **kwargs):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return VisionTranscriber(
        model="gemma3:4b",
        upstream_url="http://127.0.0.1:11434",
        client=client,
        **kwargs,
    )


class TestRequestShape:
    def test_posts_to_native_api_chat(self):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            seen["method"] = request.method
            return httpx.Response(200, json={"message": {"content": "a cat"}})

        _transcriber(handler)(_block())
        assert seen["url"] == "http://127.0.0.1:11434/api/chat"
        assert seen["method"] == "POST"

    def test_payload_matches_ollama_native_format(self):
        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={"message": {"content": "a cat"}})

        _transcriber(handler)(_block(data="IMGDATA"))
        body = seen["body"]
        assert body["model"] == "gemma3:4b"
        assert body["stream"] is False
        assert len(body["messages"]) == 1
        assert body["messages"][0]["role"] == "user"
        assert body["messages"][0]["images"] == ["IMGDATA"]
        assert isinstance(body["messages"][0]["content"], str)
        assert body["messages"][0]["content"]

    def test_prompt_is_configurable(self):
        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={"message": {"content": "x"}})

        _transcriber(handler, prompt="CUSTOM PROMPT")(_block())
        assert seen["body"]["messages"][0]["content"] == "CUSTOM PROMPT"

    def test_upstream_url_trailing_slash_is_handled(self):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            return httpx.Response(200, json={"message": {"content": "x"}})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        VisionTranscriber(
            model="m", upstream_url="http://127.0.0.1:11434/", client=client
        )(_block())
        assert seen["url"] == "http://127.0.0.1:11434/api/chat"


class TestResponseParsing:
    def test_returns_the_description(self):
        def handler(request):
            return httpx.Response(200, json={"message": {"content": "a red bicycle"}})

        assert _transcriber(handler)(_block()) == "a red bicycle"

    def test_description_is_stripped(self):
        def handler(request):
            return httpx.Response(200, json={"message": {"content": "  spaced  \n"}})

        assert _transcriber(handler)(_block()) == "spaced"

    def test_multiline_description_is_preserved(self):
        def handler(request):
            return httpx.Response(200, json={"message": {"content": "line1\nline2"}})

        assert _transcriber(handler)(_block()) == "line1\nline2"


class TestFailSoft:
    """A vision failure must never break the request; that is the whole bug."""

    def test_http_error_yields_a_placeholder(self):
        def handler(request):
            return httpx.Response(500, text="boom")

        result = _transcriber(handler)(_block())
        assert "transcription failed" in result

    def test_model_not_found_yields_a_placeholder(self):
        def handler(request):
            return httpx.Response(404, json={"error": "model 'x' not found"})

        result = _transcriber(handler)(_block())
        assert "transcription failed" in result

    def test_connection_error_yields_a_placeholder(self):
        def handler(request):
            raise httpx.ConnectError("refused")

        result = _transcriber(handler)(_block())
        assert "transcription failed" in result

    def test_timeout_yields_a_placeholder(self):
        def handler(request):
            raise httpx.ReadTimeout("too slow")

        result = _transcriber(handler)(_block())
        assert "transcription failed" in result

    def test_malformed_json_yields_a_placeholder(self):
        def handler(request):
            return httpx.Response(200, text="not json at all")

        result = _transcriber(handler)(_block())
        assert "transcription failed" in result

    def test_missing_message_key_yields_a_placeholder(self):
        def handler(request):
            return httpx.Response(200, json={"unexpected": True})

        result = _transcriber(handler)(_block())
        assert "transcription failed" in result

    def test_empty_description_yields_a_placeholder(self):
        def handler(request):
            return httpx.Response(200, json={"message": {"content": "   "}})

        result = _transcriber(handler)(_block())
        assert "transcription failed" in result

    def test_block_without_data_never_calls_upstream(self):
        calls = []

        def handler(request):
            calls.append(1)
            return httpx.Response(200, json={"message": {"content": "x"}})

        block = ImageBlock(
            source_type="base64", media_type="image/png", data=None, url=None
        )
        result = _transcriber(handler)(block)
        assert calls == []
        assert "transcription failed" in result

    def test_never_raises(self):
        def handler(request):
            raise httpx.ConnectError("refused")

        transcriber = _transcriber(handler)
        # Must be usable directly as the transform_request callable.
        assert isinstance(transcriber(_block()), str)


class TestCaching:
    def test_identical_images_hit_upstream_once(self):
        calls = []

        def handler(request):
            calls.append(1)
            return httpx.Response(200, json={"message": {"content": "cached"}})

        transcriber = _transcriber(handler)
        assert transcriber(_block(data="SAME")) == "cached"
        assert transcriber(_block(data="SAME")) == "cached"
        assert transcriber(_block(data="SAME")) == "cached"
        assert len(calls) == 1

    def test_different_images_each_hit_upstream(self):
        calls = []

        def handler(request):
            calls.append(1)
            return httpx.Response(200, json={"message": {"content": "d"}})

        transcriber = _transcriber(handler)
        transcriber(_block(data="A"))
        transcriber(_block(data="B"))
        assert len(calls) == 2

    def test_failures_are_not_cached(self):
        state = {"n": 0}

        def handler(request):
            state["n"] += 1
            if state["n"] == 1:
                return httpx.Response(500, text="boom")
            return httpx.Response(200, json={"message": {"content": "recovered"}})

        transcriber = _transcriber(handler)
        assert "transcription failed" in transcriber(_block(data="K"))
        assert transcriber(_block(data="K")) == "recovered"
        assert state["n"] == 2


class TestTimeoutConfiguration:
    def test_timeout_is_applied_to_the_client(self):
        def handler(request):
            return httpx.Response(200, json={"message": {"content": "x"}})

        transcriber = _transcriber(handler, timeout=42.0)
        assert transcriber.timeout == 42.0


class TestIntegrationWithTransform:
    def test_plugs_into_transform_request(self):
        from ollama_vision_proxy.transform import transform_request

        def handler(request):
            return httpx.Response(200, json={"message": {"content": "a chart"}})

        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "Q",
                            },
                        }
                    ],
                }
            ]
        }
        result = transform_request(body, _transcriber(handler))
        assert result.body["messages"][0]["content"][0] == {
            "type": "text",
            "text": "[Image: a chart]",
        }
