"""Tests for the Astra Core 0.1.3 interactive installer."""

from __future__ import annotations

import builtins
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from astra import config as astra_config
from astra.setup import Answers, run_setup, write_setup_files
from astra.setup.prompts import ask, ask_bool, ask_choice, ask_secret
from astra.setup.templates import (
    ProviderInfo,
    get_provider_info,
    get_sorted_providers,
    render_config,
    render_env,
    resolve_provider_key,
)
from astra.setup.validation import (
    check_provider_connection,
    check_python_version,
    mask_key,
)
from astra.setup.wizard import _build_config_context


class TestTemplates:
    def test_provider_metadata(self) -> None:
        info = get_provider_info("nvidia")
        assert info.key == "nvidia_build"
        assert info.api_key_var == "NVIDIA_API_KEY"
        assert info.base_url == "https://integrate.api.nvidia.com/v1"

    def test_resolve_provider_aliases(self) -> None:
        assert resolve_provider_key("nvidia") == "nvidia_build"
        assert resolve_provider_key("openai") == "openai-compatible"
        assert resolve_provider_key("openai_compatible") == "openai-compatible"

    def test_sorted_providers(self) -> None:
        keys = [p.key for p in get_sorted_providers()]
        assert keys[0] == "openrouter"
        assert "nvidia_build" in keys
        assert "ollama" in keys

    def test_render_config_no_secrets(self) -> None:
        ctx = {
            "timestamp": "2026-10-16",
            "provider": "nvidia_build",
            "model": "z-ai/glm-5.3-flash",
            "base_url": "https://integrate.api.nvidia.com/v1",
            "temperature": 0.7,
            "max_tokens": 2048,
            "timeout": 60.0,
            "system_prompt": "You are Astra.",
            "max_tool_iterations": 8,
            "max_messages": 30,
            "max_memories": 5,
            "db_path": "data/astra.db",
            "allow_shell": False,
            "allowed_shell_prefixes": "[\"echo \"]",
            "filesystem_roots": '["."]',
            "host": "127.0.0.1",
            "port": 8321,
            "level": "INFO",
            "log_file": "logs/astra.log",
            "filesystem_list": "allow",
            "filesystem_read": "allow",
            "filesystem_write": "confirm",
            "shell_exec": "confirm",
        }
        text = render_config(ctx)
        assert 'provider = "nvidia_build"' in text
        assert 'model = "z-ai/glm-5.3-flash"' in text
        assert "NVIDIA_API_KEY" not in text
        assert "secret" not in text.lower()

    def test_render_env(self) -> None:
        text = render_env("NVIDIA_API_KEY", "nvapi-secret-123")
        assert "NVIDIA_API_KEY=nvapi-secret-123" in text
        assert "OPENROUTER" not in text

    def test_render_env_ollama_empty_key(self) -> None:
        text = render_env("OLLAMA_API_KEY", "")
        assert "OLLAMA_API_KEY=" not in text


class TestPrompts:
    def test_ask_returns_default_on_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(builtins, "input", lambda _: "")
        assert ask("Name", default="Astra") == "Astra"

    def test_ask_validates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        inputs = iter(["bad", "good"])
        monkeypatch.setattr(builtins, "input", lambda _: next(inputs))
        assert ask("Value", validator=lambda v: v if v == "good" else (_ for _ in ()).throw(ValueError("try again"))) == "good"

    def test_ask_choice_by_number(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(builtins, "input", lambda _: "2")
        options = [("a", "A"), ("b", "B")]
        assert ask_choice("Pick:", options, default_key="a") == "b"

    def test_ask_choice_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(builtins, "input", lambda _: "")
        options = [("a", "A"), ("b", "B")]
        assert ask_choice("Pick:", options, default_key="a") == "a"

    def test_ask_bool_yes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(builtins, "input", lambda _: "y")
        assert ask_bool("OK?") is True

    def test_ask_bool_defaults_no(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(builtins, "input", lambda _: "")
        assert ask_bool("OK?", default=False) is False

    def test_ask_secret_reads_plain_in_tests(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(builtins, "input", lambda _: "secret")
        assert ask_secret("Key") == "secret"


class TestValidation:
    def test_python_version(self) -> None:
        ok, msg = check_python_version()
        assert ok is True
        assert "Python" in msg

    def test_mask_key(self) -> None:
        assert mask_key("abcdef") == "abcd**"
        assert mask_key("") == "(not set)"
        assert mask_key("ab") == "**"

    @pytest.mark.asyncio
    async def test_test_provider_connection_success(self) -> None:
        response_body = {
            "id": "test",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Astra setup OK"},
                    "finish_reason": "stop",
                }
            ],
        }
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=response_body))
        client = httpx.AsyncClient(transport=transport)
        ok, msg = await check_provider_connection(
            provider="nvidia_build",
            api_key="fake",
            model="z-ai/glm-5.3-flash",
            client=client,
        )
        assert ok is True
        assert "Connected" in msg
        await client.aclose()

    @pytest.mark.asyncio
    async def test_test_provider_connection_masks_key_in_error(self) -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(401, text="bad key: super-secret-key"))
        client = httpx.AsyncClient(transport=transport)
        ok, msg = await check_provider_connection(
            provider="nvidia_build",
            api_key="super-secret-key",
            model="z-ai/glm-5.3-flash",
            client=client,
        )
        assert ok is False
        assert "super-secret-key" not in msg
        await client.aclose()


class TestWizard:
    def test_build_config_context(self) -> None:
        answers = Answers(
            provider="nvidia_build",
            model="z-ai/glm-5.3-flash",
            api_key="secret",
            base_url="https://integrate.api.nvidia.com/v1",
            allow_shell=True,
            allowed_shell_prefixes=["echo ", "cat "],
            filesystem_write="confirm",
        )
        ctx = _build_config_context(answers)
        assert ctx["provider"] == "nvidia_build"
        assert ctx["allowed_shell_prefixes"] == ["echo ", "cat "]

    def test_default_answers(self) -> None:
        from astra.setup import default_answers

        answers = default_answers("nvidia_build")
        assert answers.provider == "nvidia_build"
        assert answers.model == "z-ai/glm-5.3-flash"
        assert answers.base_url == "https://integrate.api.nvidia.com/v1"

    def test_write_setup_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        answers = Answers(
            provider="nvidia_build",
            model="z-ai/glm-5.3-flash",
            api_key="nvapi-abc123",
            base_url="https://integrate.api.nvidia.com/v1",
            temperature=0.5,
            max_tokens=1024,
            timeout=45.0,
            allow_shell=False,
            filesystem_write="confirm",
        )
        paths = write_setup_files(answers, tmp_path)
        config_text = paths["config"].read_text()
        env_text = paths["env"].read_text()

        assert 'provider = "nvidia_build"' in config_text
        assert 'model = "z-ai/glm-5.3-flash"' in config_text
        assert "nvapi-abc123" not in config_text
        assert "NVIDIA_API_KEY=nvapi-abc123" in env_text

        # The generated config must load back into Settings.
        settings = astra_config.load_settings(paths["config"], paths["env"])
        assert settings.llm.provider == "nvidia_build"
        assert settings.llm.model == "z-ai/glm-5.3-flash"
        assert astra_config.resolve_api_key("nvidia_build", paths["env"]) == "nvapi-abc123"

    def test_run_setup_non_interactive(self, tmp_path: Path) -> None:
        answers = Answers(
            provider="ollama",
            model="llama3.1:latest",
            api_key="",
            base_url="http://127.0.0.1:11434/v1",
        )
        returned = run_setup(root_dir=tmp_path, answers=answers)
        assert returned.provider == "ollama"
        config_path = tmp_path / "config" / "config.toml"
        assert config_path.exists()
        text = config_path.read_text()
        assert 'provider = "ollama"' in text
        assert "api_key" not in text.lower()

    def test_dry_run_does_not_create_files(self, tmp_path: Path) -> None:
        answers = Answers(provider="openrouter", model="openrouter/free", api_key="sk-or-xxx")
        paths = write_setup_files(answers, tmp_path, dry_run=True)
        assert not paths["config"].exists()
        assert not paths["env"].exists()


class TestSetupCli:
    def test_setup_help(self) -> None:
        import subprocess

        result = subprocess.run(
            [sys.executable, "-m", "cli.main", "setup", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "--dry-run" in result.stdout

    def test_setup_dry_run(self, tmp_path: Path) -> None:
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cli.main",
                "setup",
                "--non-interactive",
                "--dry-run",
                "--root",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "dry run" in result.stdout
        assert not (tmp_path / "config" / "config.toml").exists()
