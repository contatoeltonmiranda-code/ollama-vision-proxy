"""Tests for the CLI: preflight wiring, process lifecycle, cleanup."""

import pytest

from ollama_vision_proxy import cli
from ollama_vision_proxy.preflight import PreflightError, VisionModelStatus


class FakeTranscriber:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        self.cache = type("C", (), {"hits": 0, "misses": 0})()
        FakeTranscriber.instances.append(self)

    def close(self):
        self.closed = True


class FakeProxy:
    instances = []

    def __init__(self, port=0, upstream_url="", transcriber=None, **kwargs):
        self.requested_port = port
        self.upstream_url = upstream_url
        self.transcriber = transcriber
        self.started = False
        self.stopped = False
        self.start_error = None
        FakeProxy.instances.append(self)

    def start(self):
        if self.start_error:
            raise self.start_error
        self.started = True
        return self

    def stop(self):
        self.stopped = True

    @property
    def port(self):
        return self.requested_port or 55555

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"


@pytest.fixture
def wired(monkeypatch):
    """Patch every collaborator so main() can run without touching the world."""
    FakeTranscriber.instances = []
    FakeProxy.instances = []
    calls = {}

    monkeypatch.setattr(cli, "check_ollama_running", lambda url, client=None: "0.32.5")
    monkeypatch.setattr(
        cli,
        "vision_model_status",
        lambda model, url, client=None: VisionModelStatus.OK,
    )
    monkeypatch.setattr(cli, "find_claude_cli", lambda: "/usr/local/bin/claude")
    monkeypatch.setattr(cli, "VisionTranscriber", FakeTranscriber)
    monkeypatch.setattr(cli, "ProxyServer", FakeProxy)

    def fake_run(claude_path, claude_args, env):
        calls["path"] = claude_path
        calls["args"] = list(claude_args)
        calls["env"] = dict(env)
        return 0

    monkeypatch.setattr(cli, "run_claude", fake_run)
    return calls


class TestSuccessfulLaunch:
    def test_returns_the_child_exit_code(self, wired, monkeypatch):
        monkeypatch.setattr(cli, "run_claude", lambda *a, **k: 42)
        assert cli.main(["launch", "--target-model", "m"]) == 42

    def test_claude_is_pointed_at_the_bound_proxy_port(self, wired):
        cli.main(["launch", "--target-model", "glm-5.2:cloud", "--proxy-port", "12345"])
        assert wired["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:12345"

    def test_target_model_reaches_the_env(self, wired):
        cli.main(["launch", "--target-model", "glm-5.2:cloud"])
        assert wired["env"]["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "glm-5.2:cloud"
        assert wired["env"]["CLAUDE_CODE_SUBAGENT_MODEL"] == "glm-5.2:cloud"

    def test_args_after_the_separator_are_passed_through(self, wired):
        cli.main(["launch", "--target-model", "m", "--", "--agent", "manager"])
        assert wired["args"] == ["--agent", "manager"]

    def test_discovered_claude_path_is_used(self, wired):
        cli.main(["launch", "--target-model", "m"])
        assert wired["path"] == "/usr/local/bin/claude"

    def test_vision_model_flag_reaches_the_transcriber(self, wired):
        cli.main(["launch", "--target-model", "m", "--vision-model", "minicpm-v"])
        assert FakeTranscriber.instances[0].kwargs["model"] == "minicpm-v"

    def test_upstream_url_flag_reaches_the_proxy(self, wired):
        cli.main(
            ["launch", "--target-model", "m", "--upstream-url", "http://127.0.0.1:9"]
        )
        assert FakeProxy.instances[0].upstream_url == "http://127.0.0.1:9"


class TestCleanup:
    def test_proxy_is_stopped_and_transcriber_closed(self, wired):
        cli.main(["launch", "--target-model", "m"])
        assert FakeProxy.instances[0].stopped is True
        assert FakeTranscriber.instances[0].closed is True

    def test_cleanup_runs_even_when_the_child_fails(self, wired, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("child exploded")

        monkeypatch.setattr(cli, "run_claude", boom)
        with pytest.raises(RuntimeError):
            cli.main(["launch", "--target-model", "m"])
        assert FakeProxy.instances[0].stopped is True
        assert FakeTranscriber.instances[0].closed is True


class TestPortInUse:
    def test_bind_failure_exits_with_the_preflight_code(self, wired, monkeypatch):
        original_start = FakeProxy.start

        def failing_start(self):
            raise OSError(48, "Address already in use")

        monkeypatch.setattr(FakeProxy, "start", failing_start)
        try:
            assert cli.main(["launch", "--target-model", "m"]) == cli.EXIT_PREFLIGHT_FAILED
        finally:
            monkeypatch.setattr(FakeProxy, "start", original_start)

    def test_bind_failure_closes_the_transcriber(self, wired, monkeypatch):
        monkeypatch.setattr(
            FakeProxy, "start", lambda self: (_ for _ in ()).throw(OSError(48, "in use"))
        )
        cli.main(["launch", "--target-model", "m"])
        assert FakeTranscriber.instances[0].closed is True

    def test_bind_failure_message_suggests_another_port(self, wired, monkeypatch, capsys):
        monkeypatch.setattr(
            FakeProxy, "start", lambda self: (_ for _ in ()).throw(OSError(48, "in use"))
        )
        cli.main(["launch", "--target-model", "m"])
        assert "--proxy-port" in capsys.readouterr().err

    def test_child_is_never_spawned_when_the_bind_fails(self, wired, monkeypatch):
        monkeypatch.setattr(
            FakeProxy, "start", lambda self: (_ for _ in ()).throw(OSError(48, "in use"))
        )
        cli.main(["launch", "--target-model", "m"])
        assert "env" not in wired


class TestPreflightFailures:
    def test_unreachable_ollama_exits_two(self, wired, monkeypatch):
        def unreachable(url, client=None):
            raise PreflightError("Cannot reach the Ollama server")

        monkeypatch.setattr(cli, "check_ollama_running", unreachable)
        assert cli.main(["launch", "--target-model", "m"]) == cli.EXIT_PREFLIGHT_FAILED

    def test_text_only_vision_model_is_refused(self, wired, monkeypatch):
        monkeypatch.setattr(
            cli,
            "vision_model_status",
            lambda model, url, client=None: VisionModelStatus.NO_VISION,
        )

        def refuse(model, url, client=None):
            raise PreflightError(f"{model} does not support vision")

        monkeypatch.setattr(cli, "check_vision_model", refuse)
        assert cli.main(["launch", "--target-model", "m"]) == cli.EXIT_PREFLIGHT_FAILED

    def test_missing_model_without_consent_does_not_pull(self, wired, monkeypatch):
        monkeypatch.setattr(
            cli,
            "vision_model_status",
            lambda model, url, client=None: VisionModelStatus.MISSING,
        )
        monkeypatch.setattr(cli, "_confirm", lambda question: False)
        pulled = []
        monkeypatch.setattr(cli, "pull_model", lambda model: pulled.append(model) or 0)
        assert cli.main(["launch", "--target-model", "m"]) == cli.EXIT_PREFLIGHT_FAILED
        assert pulled == []

    def test_missing_model_with_yes_pulls_it(self, wired, monkeypatch):
        states = iter([VisionModelStatus.MISSING])
        monkeypatch.setattr(
            cli,
            "vision_model_status",
            lambda model, url, client=None: next(states, VisionModelStatus.OK),
        )
        pulled = []
        monkeypatch.setattr(cli, "pull_model", lambda model: (pulled.append(model), 0)[1])
        monkeypatch.setattr(cli, "check_vision_model", lambda *a, **k: None)
        assert cli.main(["launch", "--target-model", "m", "-y"]) == 0
        assert pulled == ["gemma3:4b"]

    def test_failed_pull_exits_two(self, wired, monkeypatch):
        monkeypatch.setattr(
            cli,
            "vision_model_status",
            lambda model, url, client=None: VisionModelStatus.MISSING,
        )
        monkeypatch.setattr(cli, "pull_model", lambda model: 1)
        assert cli.main(["launch", "--target-model", "m", "-y"]) == cli.EXIT_PREFLIGHT_FAILED


class TestInterrupt:
    def test_keyboard_interrupt_returns_130(self, wired, monkeypatch):
        def interrupted(*args, **kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(cli, "_launch", interrupted)
        assert cli.main(["launch", "--target-model", "m"]) == 130


class TestArgumentParsing:
    def test_target_model_is_required(self):
        with pytest.raises(SystemExit):
            cli.main(["launch"])

    def test_a_subcommand_is_required(self):
        with pytest.raises(SystemExit):
            cli.main([])

    def test_version_flag_exits_cleanly(self):
        with pytest.raises(SystemExit) as exc:
            cli.main(["--version"])
        assert exc.value.code == 0
