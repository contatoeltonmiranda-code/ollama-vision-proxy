"""Spawn the claude CLI pointed at the proxy instead of at Ollama.

This replaces `ollama launch claude`. It cannot wrap it, because that subcommand
sets ANTHROPIC_BASE_URL to the Ollama port itself and would undo the redirect. So
the same environment is reproduced here with the base URL pointing at the proxy.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Performance flags that `ollama launch` sets; kept so behaviour matches.
OLLAMA_PERFORMANCE_ENV = {
    "OLLAMA_FLASH_ATTENTION": "1",
    "OLLAMA_KEEP_ALIVE": "-1",
    "OLLAMA_KV_CACHE_TYPE": "q8_0",
    "OLLAMA_NUM_PARALLEL": "1",
}

WINDOWS_SHIM_SUFFIXES = {".cmd", ".bat"}


def build_claude_env(
    target_model: str,
    proxy_port: int,
    base_env: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Environment for the claude CLI, routed through the proxy."""
    env: Dict[str, str] = dict(os.environ if base_env is None else base_env)
    env.update(
        {
            "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{proxy_port}",
            "ANTHROPIC_AUTH_TOKEN": "ollama",
            # Empty rather than absent, so an inherited real key is neutralised.
            "ANTHROPIC_API_KEY": "",
            "ANTHROPIC_DEFAULT_OPUS_MODEL": target_model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": target_model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": target_model,
            "CLAUDE_CODE_SUBAGENT_MODEL": target_model,
        }
    )
    env.update(OLLAMA_PERFORMANCE_ENV)
    return {str(key): str(value) for key, value in env.items()}


def build_claude_command(
    claude_path: str,
    claude_args: Sequence[str],
    is_windows: Optional[bool] = None,
) -> List[str]:
    """Argv for spawning claude, handling Windows .cmd/.bat shims."""
    if is_windows is None:
        is_windows = os.name == "nt"
    if is_windows and Path(claude_path).suffix.lower() in WINDOWS_SHIM_SUFFIXES:
        return ["cmd", "/c", claude_path, *claude_args]
    return [claude_path, *claude_args]


def split_passthrough(argv: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Split argv at the first bare `--`; everything after it is for claude."""
    args = list(argv)
    if "--" not in args:
        return args, []
    index = args.index("--")
    return args[:index], args[index + 1 :]


def run_claude(
    claude_path: str,
    claude_args: Sequence[str],
    env: Mapping[str, str],
) -> int:
    """Run claude in the foreground and return its exit code."""
    command = build_claude_command(claude_path, claude_args)
    logger.debug("spawning %s", command)
    try:
        return subprocess.call(command, env=dict(env))
    except KeyboardInterrupt:
        # claude shares the terminal's process group and handles Ctrl-C itself.
        return 130
