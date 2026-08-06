"""Tests for the Ollama vision client (fail-soft transcription)."""

import json

import httpx

from ollama_vision_proxy.geocode import Address
from ollama_vision_proxy.prompts import ImageKind
from ollama_vision_proxy.transform import ImageBlock, wrap_transcription
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

        result = _transcriber(handler, classify=False)(_block())
        assert result.description == "a red bicycle"

    def test_description_is_stripped(self):
        def handler(request):
            return httpx.Response(200, json={"message": {"content": "  spaced  \n"}})

        assert _transcriber(handler, classify=False)(_block()).description == "spaced"

    def test_multiline_description_is_preserved(self):
        def handler(request):
            return httpx.Response(200, json={"message": {"content": "line1\nline2"}})

        assert _transcriber(handler, classify=False)(_block()).description == "line1\nline2"


class TestFailSoft:
    """A vision failure must never break the request; that is the whole bug."""

    def test_http_error_yields_a_placeholder(self):
        def handler(request):
            return httpx.Response(500, text="boom")

        result = _transcriber(handler, classify=False)(_block()).description
        assert "transcription failed" in result

    def test_model_not_found_yields_a_placeholder(self):
        def handler(request):
            return httpx.Response(404, json={"error": "model 'x' not found"})

        result = _transcriber(handler, classify=False)(_block()).description
        assert "transcription failed" in result

    def test_connection_error_yields_a_placeholder(self):
        def handler(request):
            raise httpx.ConnectError("refused")

        result = _transcriber(handler, classify=False)(_block()).description
        assert "transcription failed" in result

    def test_timeout_yields_a_placeholder(self):
        def handler(request):
            raise httpx.ReadTimeout("too slow")

        result = _transcriber(handler, classify=False)(_block()).description
        assert "transcription failed" in result

    def test_malformed_json_yields_a_placeholder(self):
        def handler(request):
            return httpx.Response(200, text="not json at all")

        result = _transcriber(handler, classify=False)(_block()).description
        assert "transcription failed" in result

    def test_missing_message_key_yields_a_placeholder(self):
        def handler(request):
            return httpx.Response(200, json={"unexpected": True})

        result = _transcriber(handler, classify=False)(_block()).description
        assert "transcription failed" in result

    def test_empty_description_yields_a_placeholder(self):
        def handler(request):
            return httpx.Response(200, json={"message": {"content": "   "}})

        result = _transcriber(handler, classify=False)(_block()).description
        assert "transcription failed" in result

    def test_block_without_data_never_calls_upstream(self):
        calls = []

        def handler(request):
            calls.append(1)
            return httpx.Response(200, json={"message": {"content": "x"}})

        block = ImageBlock(
            source_type="base64", media_type="image/png", data=None, url=None
        )
        result = _transcriber(handler, classify=False)(block).description
        assert calls == []
        assert "transcription failed" in result

    def test_never_raises(self):
        def handler(request):
            raise httpx.ConnectError("refused")

        transcriber = _transcriber(handler, classify=False)
        # Must be usable directly as the transform_request callable.
        assert isinstance(transcriber(_block()).description, str)


class TestCaching:
    def test_identical_images_hit_upstream_once(self):
        calls = []

        def handler(request):
            calls.append(1)
            return httpx.Response(200, json={"message": {"content": "cached"}})

        transcriber = _transcriber(handler, classify=False)
        for _ in range(3):
            assert transcriber(_block(data="SAME")).description == "cached"
        assert len(calls) == 1

    def test_different_images_each_hit_upstream(self):
        calls = []

        def handler(request):
            calls.append(1)
            return httpx.Response(200, json={"message": {"content": "d"}})

        transcriber = _transcriber(handler, classify=False)
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

        transcriber = _transcriber(handler, classify=False)
        assert "transcription failed" in transcriber(_block(data="K")).description
        assert transcriber(_block(data="K")).description == "recovered"
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
        result = transform_request(body, _transcriber(handler, classify=False))
        assert result.body["messages"][0]["content"][0] == {
            "type": "text",
            "text": wrap_transcription("a chart"),
        }


class TestTwoStepPipeline:
    """Classify first, then describe with the prompt that kind deserves."""

    def _recording_handler(self, kind_reply="SCREENSHOT", description="the text"):
        seen = []

        def handler(request):
            body = json.loads(request.content)
            prompt = body["messages"][0]["content"]
            seen.append(body)
            if "Classify this image" in prompt:
                return httpx.Response(200, json={"message": {"content": kind_reply}})
            return httpx.Response(200, json={"message": {"content": description}})

        return handler, seen

    def test_classifies_then_describes(self):
        handler, seen = self._recording_handler()
        result = _transcriber(handler)(_block())
        assert len(seen) == 2
        assert "Classify this image" in seen[0]["messages"][0]["content"]
        assert result.description == "the text"

    def test_screenshot_gets_the_screenshot_prompt(self):
        handler, seen = self._recording_handler(kind_reply="SCREENSHOT")
        result = _transcriber(handler)(_block())
        assert result.kind is ImageKind.SCREENSHOT
        assert "user interface" in seen[1]["messages"][0]["content"]

    def test_photo_gets_the_photo_prompt(self):
        handler, seen = self._recording_handler(kind_reply="PHOTO")
        result = _transcriber(handler)(_block())
        assert result.kind is ImageKind.PHOTO
        describe = seen[1]["messages"][0]["content"]
        assert "subjects" in describe and "background" in describe

    def test_diagram_gets_the_diagram_prompt(self):
        handler, seen = self._recording_handler(kind_reply="DIAGRAM")
        result = _transcriber(handler)(_block())
        assert result.kind is ImageKind.DIAGRAM
        assert "diagram" in seen[1]["messages"][0]["content"].lower()

    def test_classification_failure_falls_back_to_generic(self):
        """A failed classification must not lose the description."""
        state = {"n": 0}

        def handler(request):
            state["n"] += 1
            if state["n"] == 1:
                return httpx.Response(500, text="classifier down")
            return httpx.Response(200, json={"message": {"content": "still described"}})

        result = _transcriber(handler)(_block())
        assert result.description == "still described"
        assert result.kind is ImageKind.OTHER

    def test_an_explicit_prompt_skips_classification(self):
        handler, seen = self._recording_handler()
        _transcriber(handler, prompt="JUST THIS")(_block())
        assert len(seen) == 1
        assert seen[0]["messages"][0]["content"] == "JUST THIS"

    def test_both_calls_are_greedy(self):
        """Temperature 1 made transcription differ run to run."""
        handler, seen = self._recording_handler()
        _transcriber(handler)(_block())
        for body in seen:
            assert body["options"]["temperature"] == 0

    def test_classification_output_is_bounded(self):
        """Bounded so a rambling model cannot run away, but generous enough that
        a thinking model can reason and still reach its answer."""
        from ollama_vision_proxy.vision import CLASSIFY_MAX_TOKENS

        handler, seen = self._recording_handler()
        _transcriber(handler)(_block())
        assert seen[0]["options"]["num_predict"] == CLASSIFY_MAX_TOKENS
        assert 128 <= CLASSIFY_MAX_TOKENS <= 1024

    def test_pipeline_is_cached_as_a_whole(self):
        handler, seen = self._recording_handler()
        transcriber = _transcriber(handler)
        for _ in range(4):
            transcriber(_block(data="SAME"))
        assert len(seen) == 2  # one classify plus one describe, then all cached


class TestMetadataAttachment:
    def _jpeg_with_gps(self):
        # pytest's prepend import mode already puts tests/ on sys.path.
        from test_exif import build_exif_jpeg

        return build_exif_jpeg()

    def _b64(self, raw):
        import base64

        return base64.b64encode(raw).decode()

    def test_metadata_is_attached_when_gps_is_present(self):
        def handler(request):
            return httpx.Response(200, json={"message": {"content": "a photo"}})

        block = _block(data=self._b64(self._jpeg_with_gps()))
        result = _transcriber(handler, classify=False)(block)
        assert result.metadata is not None
        assert "<metadata>" in result.metadata
        assert "iPhone" in result.metadata

    def test_no_metadata_without_gps(self):
        from test_exif import build_exif_jpeg

        def handler(request):
            return httpx.Response(200, json={"message": {"content": "a screenshot"}})

        block = _block(data=self._b64(build_exif_jpeg(with_gps=False)))
        result = _transcriber(handler, classify=False)(block)
        assert result.metadata is None

    def test_no_metadata_for_a_plain_image(self):
        def handler(request):
            return httpx.Response(200, json={"message": {"content": "x"}})

        result = _transcriber(handler, classify=False)(_block(data="bm90YW5pbWFnZQ=="))
        assert result.metadata is None

    def test_geocoder_is_consulted_for_coordinates(self):
        seen = {}

        def handler(request):
            return httpx.Response(200, json={"message": {"content": "a photo"}})

        class FakeGeocoder:
            def lookup(self, latitude, longitude):
                seen["coords"] = (latitude, longitude)
                return Address(city="Toronto", country="Canada")

            def close(self):
                pass

        block = _block(data=self._b64(self._jpeg_with_gps()))
        result = _transcriber(handler, classify=False, geocoder=FakeGeocoder())(block)
        # The coordinates the EXIF fixture encodes, not a magic threshold.
        latitude, longitude = seen["coords"]
        assert abs(latitude - 43.6425) < 1e-4
        assert abs(longitude - (-79.387222)) < 1e-4
        assert "Toronto" in result.metadata

    def test_geocoder_failure_still_yields_coordinates(self):
        def handler(request):
            return httpx.Response(200, json={"message": {"content": "a photo"}})

        class DeadGeocoder:
            def lookup(self, latitude, longitude):
                return None

            def close(self):
                pass

        block = _block(data=self._b64(self._jpeg_with_gps()))
        result = _transcriber(handler, classify=False, geocoder=DeadGeocoder())(block)
        assert "lat" in result.metadata
        assert "city" not in result.metadata

    def test_metadata_reaches_the_request_outside_the_untrusted_wrapper(self):
        from ollama_vision_proxy.transform import TRANSCRIPTION_CLOSE, transform_request

        def handler(request):
            return httpx.Response(200, json={"message": {"content": "a photo"}})

        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": self._b64(self._jpeg_with_gps()),
                            },
                        }
                    ],
                }
            ]
        }
        result = transform_request(body, _transcriber(handler, classify=False))
        text = result.body["messages"][0]["content"][0]["text"]
        assert "<metadata>" in text
        # Trusted proxy data must sit after the untrusted description wrapper.
        assert text.index(TRANSCRIPTION_CLOSE) < text.index("<metadata>")


class TestContextWindow:
    """A host with a large OLLAMA_CONTEXT_LENGTH made a 1.9GB model reserve
    32GB of KV cache, so the context has to be pinned per request."""

    def _capture(self):
        seen = []

        def handler(request):
            seen.append(json.loads(request.content))
            body = seen[-1]
            if "Classify this image" in body["messages"][0]["content"]:
                return httpx.Response(200, json={"message": {"content": "PHOTO"}})
            return httpx.Response(200, json={"message": {"content": "described"}})

        return handler, seen

    def test_num_ctx_is_always_sent(self):
        handler, seen = self._capture()
        _transcriber(handler)(_block())
        assert seen, "no request captured"
        for body in seen:
            assert "num_ctx" in body["options"]

    def test_default_context_is_modest(self):
        from ollama_vision_proxy.vision import DEFAULT_NUM_CTX

        handler, seen = self._capture()
        _transcriber(handler)(_block())
        assert DEFAULT_NUM_CTX <= 32768, "a huge default defeats the purpose"
        for body in seen:
            assert body["options"]["num_ctx"] == DEFAULT_NUM_CTX

    def test_num_ctx_is_configurable(self):
        handler, seen = self._capture()
        _transcriber(handler, num_ctx=4096)(_block())
        for body in seen:
            assert body["options"]["num_ctx"] == 4096

    def test_classification_and_description_both_pinned(self):
        handler, seen = self._capture()
        _transcriber(handler, num_ctx=2048)(_block())
        assert len(seen) == 2
        assert seen[0]["options"]["num_ctx"] == 2048
        assert seen[1]["options"]["num_ctx"] == 2048

    def test_only_classification_bounds_output_length(self):
        handler, seen = self._capture()
        _transcriber(handler)(_block())
        assert "num_predict" in seen[0]["options"]
        assert "num_predict" not in seen[1]["options"]

    def test_greedy_sampling_survives_the_context_change(self):
        handler, seen = self._capture()
        _transcriber(handler, num_ctx=1024)(_block())
        for body in seen:
            assert body["options"]["temperature"] == 0


class TestThinkingModels:
    """qwen3-vl leaves message.content empty and puts the answer in thinking."""

    def test_thinking_is_used_when_content_is_empty(self):
        def handler(request):
            return httpx.Response(
                200,
                json={"message": {"content": "", "thinking": "a red bicycle"}},
            )

        result = _transcriber(handler, classify=False)(_block())
        assert result.description == "a red bicycle"

    def test_content_wins_when_both_are_present(self):
        def handler(request):
            return httpx.Response(
                200,
                json={"message": {"content": "the answer", "thinking": "musing"}},
            )

        result = _transcriber(handler, classify=False)(_block())
        assert result.description == "the answer"

    def test_both_empty_is_still_a_failure(self):
        def handler(request):
            return httpx.Response(200, json={"message": {"content": "", "thinking": ""}})

        result = _transcriber(handler, classify=False)(_block())
        assert "transcription failed" in result.description

    def test_classification_reads_a_thinking_reply(self):
        def handler(request):
            body = json.loads(request.content)
            if "Classify this image" in body["messages"][0]["content"]:
                return httpx.Response(
                    200,
                    json={
                        "message": {
                            "content": "",
                            "thinking": "options are SCREENSHOT, PHOTO; this is a PHOTO",
                        }
                    },
                )
            return httpx.Response(200, json={"message": {"content": "described"}})

        result = _transcriber(handler)(_block())
        assert result.kind is ImageKind.PHOTO

    def test_classification_gets_room_to_think(self):
        """Capped at 8 tokens the model emitted nothing at all."""
        from ollama_vision_proxy.vision import CLASSIFY_MAX_TOKENS

        seen = []

        def handler(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={"message": {"content": "PHOTO"}})

        _transcriber(handler)(_block())
        assert seen[0]["options"]["num_predict"] == CLASSIFY_MAX_TOKENS
        assert CLASSIFY_MAX_TOKENS >= 128


class TestDefaults:
    def test_default_model_is_one_that_finishes_in_time(self):
        """qwen3-vl:4b has better OCR but spends 202s deliberating on the
        screenshot prompt, so it always trips the 60s backstop."""
        from ollama_vision_proxy.vision import DEFAULT_TIMEOUT, DEFAULT_VISION_MODEL

        assert DEFAULT_VISION_MODEL == "gemma3:4b"
        # Guard the pairing: a default must be able to answer inside the backstop.
        assert DEFAULT_TIMEOUT >= 30.0

    def test_default_timeout_backstops_erratic_latency(self):
        from ollama_vision_proxy.vision import DEFAULT_TIMEOUT

        assert DEFAULT_TIMEOUT == 60.0
