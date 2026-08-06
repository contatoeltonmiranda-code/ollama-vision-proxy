"""Tests for the claude-CLI environment and argument passthrough."""

from ollama_vision_proxy.launcher import build_claude_env, split_passthrough


class TestBuildClaudeEnv:
    def test_base_url_points_at_the_proxy_not_ollama(self):
        """The whole launcher exists to redirect this away from 11434."""
        env = build_claude_env("glm-5.2:cloud", proxy_port=11435, base_env={})
        assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:11435"

    def test_proxy_port_is_respected(self):
        env = build_claude_env("m", proxy_port=9999, base_env={})
        assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9999"

    def test_auth_token_is_ollama(self):
        env = build_claude_env("m", proxy_port=11435, base_env={})
        assert env["ANTHROPIC_AUTH_TOKEN"] == "ollama"

    def test_api_key_is_present_but_empty(self):
        """Must be set to empty, not absent, to neutralise a real inherited key."""
        env = build_claude_env("m", proxy_port=11435, base_env={"ANTHROPIC_API_KEY": "sk-real"})
        assert "ANTHROPIC_API_KEY" in env
        assert env["ANTHROPIC_API_KEY"] == ""

    def test_all_three_model_tiers_are_set(self):
        env = build_claude_env("glm-5.2:cloud", proxy_port=11435, base_env={})
        assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "glm-5.2:cloud"
        assert env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "glm-5.2:cloud"
        assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "glm-5.2:cloud"

    def test_subagent_model_is_set(self):
        env = build_claude_env("glm-5.2:cloud", proxy_port=11435, base_env={})
        assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == "glm-5.2:cloud"

    def test_ollama_performance_flags_are_set(self):
        env = build_claude_env("m", proxy_port=11435, base_env={})
        assert env["OLLAMA_FLASH_ATTENTION"] == "1"
        assert env["OLLAMA_KEEP_ALIVE"] == "-1"
        assert env["OLLAMA_KV_CACHE_TYPE"] == "q8_0"
        assert env["OLLAMA_NUM_PARALLEL"] == "1"

    def test_existing_environment_is_preserved(self):
        env = build_claude_env(
            "m", proxy_port=11435, base_env={"PATH": "/usr/bin", "HOME": "/home/x"}
        )
        assert env["PATH"] == "/usr/bin"
        assert env["HOME"] == "/home/x"

    def test_base_env_is_not_mutated(self):
        base = {"PATH": "/usr/bin"}
        build_claude_env("m", proxy_port=11435, base_env=base)
        assert base == {"PATH": "/usr/bin"}

    def test_inherited_base_url_is_overridden(self):
        env = build_claude_env(
            "m", proxy_port=11435, base_env={"ANTHROPIC_BASE_URL": "http://127.0.0.1:11434"}
        )
        assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:11435"

    def test_values_are_all_strings(self):
        """subprocess env rejects non-string values on Windows and POSIX alike."""
        env = build_claude_env("m", proxy_port=11435, base_env={})
        assert all(isinstance(k, str) and isinstance(v, str) for k, v in env.items())


class TestSigintOwnership:
    """subprocess.call kills the child on any exception, so a KeyboardInterrupt
    in the parent SIGKILLed claude before it could shut down cleanly."""

    def test_sigint_is_ignored_while_the_child_runs(self):
        import signal

        from ollama_vision_proxy.launcher import run_claude

        original = signal.getsignal(signal.SIGINT)
        seen = {}

        class FakePopen:
            def __init__(self, command, env=None):
                pass

            def wait(self):
                seen["handler"] = signal.getsignal(signal.SIGINT)
                return 7

        assert run_claude("/bin/claude", [], {}, popen=FakePopen) == 7
        assert seen["handler"] is signal.SIG_IGN
        assert signal.getsignal(signal.SIGINT) is original

    def test_handler_is_restored_even_if_the_child_raises(self):
        import signal

        from ollama_vision_proxy.launcher import run_claude

        original = signal.getsignal(signal.SIGINT)

        class Boom:
            def __init__(self, command, env=None):
                raise OSError("cannot spawn")

        try:
            run_claude("/bin/claude", [], {}, popen=Boom)
        except OSError:
            pass
        assert signal.getsignal(signal.SIGINT) is original

    def test_child_is_not_killed_on_keyboard_interrupt(self):
        from ollama_vision_proxy.launcher import run_claude

        class FakePopen:
            def __init__(self, command, env=None):
                self.raised = False
                self.killed = False

            def wait(self):
                if not self.raised:
                    self.raised = True
                    raise KeyboardInterrupt
                return 0

            def kill(self):
                self.killed = True

        created = []

        def factory(command, env=None):
            process = FakePopen(command, env)
            created.append(process)
            return process

        assert run_claude("/bin/claude", [], {}, popen=factory) == 0
        assert created[0].killed is False

    def test_env_and_command_reach_the_child(self):
        from ollama_vision_proxy.launcher import run_claude

        seen = {}

        class FakePopen:
            def __init__(self, command, env=None):
                seen["command"] = command
                seen["env"] = env

            def wait(self):
                return 0

        run_claude("/bin/claude", ["--agent", "x"], {"A": "1"}, popen=FakePopen)
        assert seen["command"] == ["/bin/claude", "--agent", "x"]
        assert seen["env"] == {"A": "1"}


class TestBuildClaudeCommand:
    def test_posix_invokes_the_binary_directly(self):
        from ollama_vision_proxy.launcher import build_claude_command

        command = build_claude_command(
            "/usr/local/bin/claude", ["--agent", "manager"], is_windows=False
        )
        assert command == ["/usr/local/bin/claude", "--agent", "manager"]

    def test_windows_exe_invokes_the_binary_directly(self):
        from ollama_vision_proxy.launcher import build_claude_command

        command = build_claude_command(r"C:\bin\claude.exe", ["-p"], is_windows=True)
        assert command == [r"C:\bin\claude.exe", "-p"]

    def test_windows_cmd_shim_is_run_through_cmd(self):
        """CreateProcess cannot execute a .cmd directly, so npm shims need cmd /c."""
        from ollama_vision_proxy.launcher import build_claude_command

        command = build_claude_command(r"C:\npm\claude.cmd", ["-p"], is_windows=True)
        assert command == ["cmd", "/c", r"C:\npm\claude.cmd", "-p"]

    def test_windows_bat_shim_is_run_through_cmd(self):
        from ollama_vision_proxy.launcher import build_claude_command

        command = build_claude_command(r"C:\npm\claude.bat", [], is_windows=True)
        assert command == ["cmd", "/c", r"C:\npm\claude.bat"]

    def test_no_args_is_fine(self):
        from ollama_vision_proxy.launcher import build_claude_command

        assert build_claude_command("/bin/claude", [], is_windows=False) == ["/bin/claude"]


class TestSplitPassthrough:
    def test_splits_on_the_first_bare_double_dash(self):
        own, rest = split_passthrough(
            ["launch", "--target-model", "m", "--", "--agent", "manager"]
        )
        assert own == ["launch", "--target-model", "m"]
        assert rest == ["--agent", "manager"]

    def test_no_separator_means_no_passthrough(self):
        own, rest = split_passthrough(["launch", "--target-model", "m"])
        assert own == ["launch", "--target-model", "m"]
        assert rest == []

    def test_trailing_separator_yields_empty_passthrough(self):
        own, rest = split_passthrough(["launch", "--target-model", "m", "--"])
        assert own == ["launch", "--target-model", "m"]
        assert rest == []

    def test_later_double_dashes_stay_in_the_passthrough(self):
        own, rest = split_passthrough(["launch", "--", "-p", "--", "hello"])
        assert own == ["launch"]
        assert rest == ["-p", "--", "hello"]

    def test_ovp_looking_flags_after_separator_are_not_claimed(self):
        own, rest = split_passthrough(["launch", "--", "--proxy-port", "1"])
        assert own == ["launch"]
        assert rest == ["--proxy-port", "1"]

    def test_empty_argv(self):
        assert split_passthrough([]) == ([], [])

    def test_input_list_is_not_mutated(self):
        argv = ["launch", "--", "-p"]
        split_passthrough(argv)
        assert argv == ["launch", "--", "-p"]
