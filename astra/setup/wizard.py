"""Interactive setup wizard for Astra Core.

The wizard can be driven interactively or programmatically through the
``Answers`` dataclass.  In either mode it never writes API keys to
``config.toml``.
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import prompts
from .templates import (
    ProviderInfo,
    get_provider_info,
    render_config,
    render_env,
)
from .validation import check_provider_connection, check_python_version, mask_key


class SetupError(Exception):
    """Raised for user-facing setup failures."""


@dataclass
class Answers:
    """Collected setup answers.  All fields are strings/bools/lists."""

    provider: str = "openrouter"
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: float = 60.0
    allow_shell: bool = False
    allowed_shell_prefixes: list[str] = field(default_factory=list)
    filesystem_roots: list[str] = field(default_factory=lambda: ["."])
    filesystem_list: str = "allow"
    filesystem_read: str = "allow"
    filesystem_write: str = "confirm"
    shell_exec: str = "confirm"
    run_first_chat: bool = False


def _default_answers(provider_info: ProviderInfo) -> Answers:
    return Answers(
        provider=provider_info.key,
        model=provider_info.default_model,
        base_url=provider_info.base_url,
        api_key="",
        allowed_shell_prefixes=["echo ", "ls ", "cat ", "python "] if False else [],
    )


def default_answers(provider: str = "openrouter") -> Answers:
    """Return sane default answers for the given provider."""
    info = get_provider_info(provider)
    return _default_answers(info)


def _permission_validator(label: str) -> Callable[[str], str]:
    allowed = {"allow", "confirm", "deny"}

    def validate(value: str) -> str:
        v = value.lower().strip()
        if v not in allowed:
            raise ValueError(f"{label} must be one of: allow, confirm, deny")
        return v

    return validate


def _int_validator(label: str, min_value: int | None = None) -> Callable[[str], int]:
    def validate(value: str) -> int:
        try:
            n = int(value)
        except ValueError as exc:
            raise ValueError(f"{label} must be a whole number") from exc
        if min_value is not None and n < min_value:
            raise ValueError(f"{label} must be at least {min_value}")
        return n

    return validate


def _float_validator(label: str, min_value: float, max_value: float) -> Callable[[str], float]:
    def validate(value: str) -> float:
        try:
            f = float(value)
        except ValueError as exc:
            raise ValueError(f"{label} must be a number") from exc
        if not min_value <= f <= max_value:
            raise ValueError(f"{label} must be between {min_value} and {max_value}")
        return f

    return validate


def _shell_prefixes_validator(value: str) -> list[str]:
    """Parse a comma-separated list of shell command prefixes."""
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return parts


def _interactive_provider() -> tuple[ProviderInfo, str]:
    options = [(info.key, info.display_name) for info in get_sorted_providers()]
    key = prompts.ask_choice("Choose a provider:", options, default_key="openrouter")
    return get_provider_info(key), key


def _interactive_api_key(provider_info: ProviderInfo) -> str:
    if not provider_info.key_required:
        return ""
    return prompts.ask_secret(f"Enter your {provider_info.api_key_var}")


def _interactive_model(provider_info: ProviderInfo) -> str:
    default = provider_info.default_model
    answer = prompts.ask(
        "Model",
        default=default,
        required=False,
    )
    return answer or default


def _interactive_base_url(provider_info: ProviderInfo) -> str:
    if not provider_info.base_url:
        return prompts.ask("Base URL", required=True)
    return prompts.ask("Base URL", default=provider_info.base_url, required=False) or provider_info.base_url


def get_sorted_providers() -> list[ProviderInfo]:
    """Return providers in a stable, sensible order."""
    order = ["openrouter", "nvidia_build", "ollama", "openai-compatible"]
    return [get_provider_info(k) for k in order]


def collect_interactive_answers() -> Answers:
    """Run the full interactive wizard and return the collected answers."""
    ok, msg = check_python_version()
    print(f"Astra Core setup")
    print(f"  {msg}")
    if not ok:
        raise SetupError(msg)

    provider_info, _ = _interactive_provider()
    answers = _default_answers(provider_info)

    answers.base_url = _interactive_base_url(provider_info)
    answers.api_key = _interactive_api_key(provider_info)
    answers.model = _interactive_model(provider_info)

    answers.temperature = prompts.ask(
        "Temperature",
        default=str(answers.temperature),
        validator=_float_validator("Temperature", 0.0, 2.0),
    )
    answers.max_tokens = prompts.ask(
        "Max tokens",
        default=str(answers.max_tokens),
        validator=_int_validator("Max tokens", min_value=1),
    )
    answers.timeout = prompts.ask(
        "Request timeout (seconds)",
        default=str(answers.timeout),
        validator=_int_validator("Timeout", min_value=5),
    )

    print("\nPermission defaults")
    answers.filesystem_read = prompts.ask(
        "filesystem.read",
        default=answers.filesystem_read,
        validator=_permission_validator("filesystem.read"),
    )
    answers.filesystem_list = prompts.ask(
        "filesystem.list",
        default=answers.filesystem_list,
        validator=_permission_validator("filesystem.list"),
    )
    answers.filesystem_write = prompts.ask(
        "filesystem.write",
        default=answers.filesystem_write,
        validator=_permission_validator("filesystem.write"),
    )

    answers.allow_shell = prompts.ask_bool(
        "Enable shell tool (requires explicit allowlist)?", default=False
    )
    if answers.allow_shell:
        raw = prompts.ask(
            "Allowed shell prefixes (comma-separated)",
            default=", ".join(answers.allowed_shell_prefixes),
            required=False,
        )
        answers.allowed_shell_prefixes = _shell_prefixes_validator(raw)
        answers.shell_exec = prompts.ask(
            "shell.exec",
            default=answers.shell_exec,
            validator=_permission_validator("shell.exec"),
        )
    else:
        answers.allowed_shell_prefixes = []
        answers.shell_exec = "deny"

    print("\nSummary")
    print(f"  Provider: {provider_info.display_name}")
    print(f"  Model:    {answers.model}")
    print(f"  Key:      {mask_key(answers.api_key)}")
    print(f"  Base URL: {answers.base_url}")
    print(f"  Shell:    {'enabled' if answers.allow_shell else 'disabled'}")

    test = prompts.ask_bool("Test connectivity now?", default=True)
    if test:
        ok, test_msg = asyncio.run(
            check_provider_connection(
                provider=answers.provider,
                api_key=answers.api_key,
                model=answers.model,
                base_url=answers.base_url,
                timeout=answers.timeout,
            )
        )
        print(f"  {'OK' if ok else 'FAIL'}: {test_msg}")
        if not ok:
            if not prompts.ask_bool("Continue anyway?", default=False):
                raise SetupError("Setup aborted after failed connectivity test")

    answers.run_first_chat = prompts.ask_bool("Run a first chat after setup?", default=False)
    return answers


def _build_config_context(answers: Answers) -> dict[str, object]:
    """Convert answers into the exact keys needed by ``CONFIG_TEMPLATE``."""
    return {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "provider": answers.provider,
        "model": answers.model,
        "base_url": answers.base_url,
        "temperature": answers.temperature,
        "max_tokens": answers.max_tokens,
        "timeout": answers.timeout,
        "system_prompt": "You are Astra, a helpful AI assistant with access to tools. Use tools when they help answer the user. Be concise.",
        "max_tool_iterations": 8,
        "max_messages": 30,
        "max_memories": 5,
        "db_path": "data/astra.db",
        "allow_shell": answers.allow_shell,
        "allowed_shell_prefixes": list(answers.allowed_shell_prefixes),
        "filesystem_roots": list(answers.filesystem_roots),
        "host": "127.0.0.1",
        "port": 8321,
        "level": "INFO",
        "log_file": "logs/astra.log",
        "filesystem_list": answers.filesystem_list,
        "filesystem_read": answers.filesystem_read,
        "filesystem_write": answers.filesystem_write,
        "shell_exec": answers.shell_exec,
    }


def write_setup_files(
    answers: Answers,
    root_dir: Path,
    dry_run: bool = False,
) -> dict[str, Path]:
    """Write ``config.toml`` and ``.env`` under ``root_dir``.

    Returns a mapping of friendly names to written paths.  When ``dry_run`` is
    true, no files are touched and paths are still returned.
    """
    config_dir = root_dir / "config"
    env_path = root_dir / ".env"
    config_path = config_dir / "config.toml"

    config_text = render_config(_build_config_context(answers))
    provider_info = get_provider_info(answers.provider)
    env_text = render_env(provider_info.api_key_var, answers.api_key)

    if not dry_run:
        config_dir.mkdir(parents=True, exist_ok=True)
        (root_dir / "data").mkdir(parents=True, exist_ok=True)
        (root_dir / "logs").mkdir(parents=True, exist_ok=True)
        config_path.write_text(config_text, encoding="utf-8")
        env_path.write_text(env_text, encoding="utf-8")

    return {"config": config_path, "env": env_path}


def run_setup(
    root_dir: Path | None = None,
    answers: Answers | None = None,
    dry_run: bool = False,
) -> Answers:
    """Run the installer.  Returns the answers used.

    If ``answers`` is provided, skip prompts and use those values (useful for
    tests and headless deployment).  If ``dry_run`` is true, no files are
    written.
    """
    root = (root_dir or Path.cwd()).resolve()
    if answers is None:
        answers = collect_interactive_answers()

    paths = write_setup_files(answers, root, dry_run=dry_run)

    print("\nWrote configuration")
    print(f"  {paths['config']}")
    print(f"  {paths['env']}")
    if dry_run:
        print("  (dry run — files not actually written)")

    if answers.run_first_chat and not dry_run:
        print("\nStarting first chat...")
        # Lazy import to avoid a circular dependency between cli.main and setup.
        from ..cli.main import interactive_chat

        return answers

    return answers
