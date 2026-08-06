"""Tests for startup checks: Ollama reachable, model actually vision-capable."""

import httpx
import pytest

from ollama_vision_proxy.preflight import (
    PreflightError,
    check_ollama_running,
    check_vision_model,
    find_claude_cli,
    list_vision_models,
    model_capabilities,
)


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


class TestOllamaRunning:
    def test_returns_version_when_up(self):
        def handler(request):
            assert str(request.url) == "http://127.0.0.1:11434/api/version"
            return httpx.Response(200, json={"version": "0.32.5"})

        assert check_ollama_running("http://127.0.0.1:11434", _client(handler)) == "0.32.5"

    def test_raises_when_connection_refused(self):
        def handler(request):
            raise httpx.ConnectError("refused")

        with pytest.raises(PreflightError) as exc:
            check_ollama_running("http://127.0.0.1:11434", _client(handler))
        assert "ollama serve" in str(exc.value)

    def test_raises_on_bad_status(self):
        def handler(request):
            return httpx.Response(500, text="broken")

        with pytest.raises(PreflightError):
            check_ollama_running("http://127.0.0.1:11434", _client(handler))


class TestModelCapabilities:
    def test_reads_capabilities_array(self):
        def handler(request):
            assert str(request.url) == "http://127.0.0.1:11434/api/show"
            return httpx.Response(200, json={"capabilities": ["completion", "vision"]})

        caps = model_capabilities("gemma3:4b", "http://127.0.0.1:11434", _client(handler))
        assert caps == ["completion", "vision"]

    def test_sends_the_model_name(self):
        import json

        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={"capabilities": ["vision"]})

        model_capabilities("qwen3.5:4b-mlx", "http://127.0.0.1:11434", _client(handler))
        assert seen["body"]["model"] == "qwen3.5:4b-mlx"

    def test_missing_model_returns_empty(self):
        def handler(request):
            return httpx.Response(404, json={"error": "model not found"})

        caps = model_capabilities("nope", "http://127.0.0.1:11434", _client(handler))
        assert caps == []


class TestCheckVisionModel:
    def test_passes_for_a_vision_model(self):
        def handler(request):
            if request.url.path == "/api/show":
                return httpx.Response(200, json={"capabilities": ["completion", "vision"]})
            return httpx.Response(404)

        check_vision_model("gemma3:4b", "http://127.0.0.1:11434", _client(handler))

    def test_rejects_a_text_only_model(self):
        """gemma3:1b is pulled but has no vision; catching this is the point."""

        def handler(request):
            if request.url.path == "/api/show":
                return httpx.Response(200, json={"capabilities": ["completion"]})
            if request.url.path == "/api/tags":
                return httpx.Response(200, json={"models": []})
            return httpx.Response(404)

        with pytest.raises(PreflightError) as exc:
            check_vision_model("gemma3:1b", "http://127.0.0.1:11434", _client(handler))
        message = str(exc.value)
        assert "gemma3:1b" in message
        assert "vision" in message

    def test_missing_model_mentions_ollama_pull(self):
        def handler(request):
            if request.url.path == "/api/show":
                return httpx.Response(404, json={"error": "not found"})
            if request.url.path == "/api/tags":
                return httpx.Response(200, json={"models": []})
            return httpx.Response(404)

        with pytest.raises(PreflightError) as exc:
            check_vision_model("minicpm-v", "http://127.0.0.1:11434", _client(handler))
        assert "ollama pull minicpm-v" in str(exc.value)

    def test_error_suggests_pulled_vision_models(self):
        def handler(request):
            if request.url.path == "/api/show":
                import json

                model = json.loads(request.content)["model"]
                if model == "gemma3:4b":
                    return httpx.Response(200, json={"capabilities": ["vision"]})
                return httpx.Response(404, json={"error": "not found"})
            if request.url.path == "/api/tags":
                return httpx.Response(
                    200, json={"models": [{"name": "gemma3:4b"}, {"name": "gemma3:1b"}]}
                )
            return httpx.Response(404)

        with pytest.raises(PreflightError) as exc:
            check_vision_model("minicpm-v", "http://127.0.0.1:11434", _client(handler))
        assert "gemma3:4b" in str(exc.value)


class TestListVisionModels:
    def test_filters_to_vision_capable_models(self):
        import json

        vision = {"gemma3:4b", "qwen3.5:4b-mlx"}

        def handler(request):
            if request.url.path == "/api/tags":
                return httpx.Response(
                    200,
                    json={
                        "models": [
                            {"name": "gemma3:4b"},
                            {"name": "gemma3:1b"},
                            {"name": "qwen3.5:4b-mlx"},
                        ]
                    },
                )
            if request.url.path == "/api/show":
                model = json.loads(request.content)["model"]
                caps = ["completion", "vision"] if model in vision else ["completion"]
                return httpx.Response(200, json={"capabilities": caps})
            return httpx.Response(404)

        found = list_vision_models("http://127.0.0.1:11434", _client(handler))
        assert set(found) == vision

    def test_returns_empty_when_ollama_unreachable(self):
        def handler(request):
            raise httpx.ConnectError("refused")

        assert list_vision_models("http://127.0.0.1:11434", _client(handler)) == []


class TestVisionModelStatus:
    """The CLI needs to tell "not pulled" apart from "pulled but text-only",
    because only the first is fixable by pulling."""

    def _handler(self, caps=None, status=200):
        def handler(request):
            if request.url.path == "/api/show":
                if status != 200:
                    return httpx.Response(status, json={"error": "not found"})
                return httpx.Response(200, json={"capabilities": caps or []})
            return httpx.Response(404)

        return handler

    def test_ok_for_vision_model(self):
        from ollama_vision_proxy.preflight import VisionModelStatus, vision_model_status

        status = vision_model_status(
            "gemma3:4b", "http://x", _client(self._handler(["completion", "vision"]))
        )
        assert status is VisionModelStatus.OK

    def test_missing_when_not_pulled(self):
        from ollama_vision_proxy.preflight import VisionModelStatus, vision_model_status

        status = vision_model_status(
            "minicpm-v", "http://x", _client(self._handler(status=404))
        )
        assert status is VisionModelStatus.MISSING

    def test_no_vision_when_pulled_but_text_only(self):
        from ollama_vision_proxy.preflight import VisionModelStatus, vision_model_status

        status = vision_model_status(
            "gemma3:1b", "http://x", _client(self._handler(["completion"]))
        )
        assert status is VisionModelStatus.NO_VISION


class TestFindClaudeCli:
    def test_returns_path_when_found(self):
        assert find_claude_cli(which=lambda name: "/usr/bin/claude") == "/usr/bin/claude"

    def test_looks_for_claude(self):
        seen = []

        def fake_which(name):
            seen.append(name)
            return "/usr/bin/claude"

        find_claude_cli(which=fake_which)
        assert "claude" in seen

    def test_raises_with_guidance_when_missing(self):
        with pytest.raises(PreflightError) as exc:
            find_claude_cli(which=lambda name: None)
        assert "claude" in str(exc.value).lower()
