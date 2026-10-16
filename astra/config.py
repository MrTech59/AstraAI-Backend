"""Astra Core configuration.

Loads TOML config (``config/config.toml``) layered over built-in defaults,
with a minimal ``.env`` file and environment-variable overrides for secrets.

No credentials are read from TOML files directly; API keys come from the
environment (or .env) via ``*_API_KEY`` variable names only.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR = ROOT_DIR / "data"
CONFIG_PATH = CONFIG_DIR / "config.toml"
ENV_PATH = ROOT_DIR / ".env"


class ConfigError(Exception):
    """Raised when configuration is missing or invalid."""


@dataclass
class LLMConfig:
    provider: str = "openrouter"
    model: str = "openrouter/free"
    base_url: str = ""  # per-provider default applied by the factory
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: float = 60.0


@dataclass
class AgentConfig:
    system_prompt: str = (
        "You are Astra, a helpful AI assistant with access to tools. "
        "Use tools when they help answer the user. Be concise."
    )
    max_tool_iterations: int = 8


@dataclass
class ContextConfig:
    max_messages: int = 30
    max_memories: int = 5


@dataclass
class MemoryConfig:
    db_path: str = "data/astra.db"


@dataclass
class SecurityConfig:
    allow_shell: bool = False
    allowed_shell_prefixes: list[str] = field(default_factory=list)
    filesystem_roots: list[str] = field(default_factory=lambda: ["."])


@dataclass
class ApiConfig:
    host: str = "127.0.0.1"
    port: int = 8321


@dataclass
class McpServerConfig:
    name: str
    url: str
    enabled: bool = True


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "logs/astra.log"


@dataclass
class PermissionsConfig:
    """Per-tool permission actions: "allow" | "confirm" | "deny".

    None means "use the built-in safe default" — missing config never
    widens access.
    """

    filesystem_list: str | None = None
    filesystem_read: str | None = None
    filesystem_write: str | None = None
    shell_exec: str | None = None
    default: str | None = None


@dataclass
class Settings:
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    mcp_servers: list[McpServerConfig] = field(default_factory=list)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    permissions: PermissionsConfig = field(default_factory=PermissionsConfig)


def load_dotenv(path: Path = ENV_PATH) -> dict[str, str]:
    """Parse a minimal KEY=VALUE .env file. Returns the parsed mapping."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _env(name: str, dotenv: dict[str, str]) -> str:
    """Resolve a setting from the process environment, then .env."""
    return os.environ.get(name) or dotenv.get(name) or ""


_API_KEY_VARS = {
    "openrouter": "OPENROUTER_API_KEY",
    "ollama": "OLLAMA_API_KEY",
    "nvidia_build": "NVIDIA_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
}


def resolve_api_key(provider: str, env_path: Path = ENV_PATH) -> str:
    """Resolve a provider's API key from the environment or .env file.

    This is the single place that maps provider names to their key
    environment variables; load_settings and manual tooling share it.
    """
    dotenv = load_dotenv(env_path)
    key_var = _API_KEY_VARS.get(provider, f"{provider.upper()}_API_KEY")
    return _env(key_var, dotenv)


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def load_settings(config_path: Path = CONFIG_PATH, env_path: Path = ENV_PATH) -> Settings:
    """Load settings: defaults <- config.toml <- environment."""
    data: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("rb") as handle:
            data = tomllib.load(handle)

    dotenv = load_dotenv(env_path)
    settings = Settings()

    llm = data.get("llm", {})
    settings.llm.provider = str(llm.get("provider", settings.llm.provider))
    settings.llm.model = str(llm.get("model", settings.llm.model))
    settings.llm.base_url = str(llm.get("base_url", settings.llm.base_url))
    settings.llm.temperature = float(llm.get("temperature", settings.llm.temperature))
    settings.llm.max_tokens = int(llm.get("max_tokens", settings.llm.max_tokens))
    settings.llm.timeout = float(llm.get("timeout", settings.llm.timeout))
    # API keys are never read from TOML; environment/.env only.
    settings.llm.api_key = resolve_api_key(settings.llm.provider, env_path)

    agent = data.get("agent", {})
    settings.agent.system_prompt = str(agent.get("system_prompt", settings.agent.system_prompt))
    settings.agent.max_tool_iterations = int(
        agent.get("max_tool_iterations", settings.agent.max_tool_iterations)
    )

    context = data.get("context", {})
    settings.context.max_messages = int(context.get("max_messages", settings.context.max_messages))
    settings.context.max_memories = int(
        context.get("max_memories", settings.context.max_memories)
    )

    memory = data.get("memory", {})
    settings.memory.db_path = str(memory.get("db_path", settings.memory.db_path))

    security = data.get("security", {})
    settings.security.allow_shell = bool(
        security.get("allow_shell", settings.security.allow_shell)
    )
    settings.security.allowed_shell_prefixes = list(
        security.get("allowed_shell_prefixes", settings.security.allowed_shell_prefixes)
    )
    settings.security.filesystem_roots = list(
        security.get("filesystem_roots", settings.security.filesystem_roots)
    )

    api = data.get("api", {})
    settings.api.host = str(api.get("host", settings.api.host))
    settings.api.port = int(api.get("port", settings.api.port))

    for server in data.get("mcp", {}).get("servers", []):
        settings.mcp_servers.append(
            McpServerConfig(
                name=str(server.get("name", "unnamed")),
                url=str(server.get("url", "")),
                enabled=bool(server.get("enabled", True)),
            )
        )

    logging_ = data.get("logging", {})
    settings.logging.level = str(logging_.get("level", settings.logging.level))
    settings.logging.file = str(logging_.get("file", settings.logging.file))

    permissions = data.get("permissions", {})
    for key in ("filesystem_list", "filesystem_read", "filesystem_write", "shell_exec", "default"):
        if key in permissions:
            setattr(settings.permissions, key, str(permissions[key]))

    return settings


def memory_db_path(settings: Settings) -> Path:
    return _resolve_path(settings.memory.db_path)


def filesystem_roots(settings: Settings) -> list[Path]:
    return [_resolve_path(root) for root in settings.security.filesystem_roots]


def log_file_path(settings: Settings) -> Path:
    return _resolve_path(settings.logging.file)
