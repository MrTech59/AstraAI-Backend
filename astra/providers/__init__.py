"""Provider factory: configuration decides the active LLM backend."""

from __future__ import annotations

import httpx

from ..config import LLMConfig
from .base import ChatResponse, LLMProvider, LLMProviderError, Message, ToolCall
from .nvidia_build import NvidiaBuildProvider
from .ollama import OllamaProvider
from .openai_compatible import OpenAICompatibleProvider
from .openrouter import OpenRouterProvider

__all__ = [
    "ChatResponse",
    "LLMProvider",
    "LLMProviderError",
    "Message",
    "NvidiaBuildProvider",
    "ToolCall",
    "create_provider",
]


def create_provider(config: LLMConfig, client: httpx.AsyncClient | None = None) -> LLMProvider:
    """Instantiate the provider selected in configuration."""
    provider = config.provider.lower()
    shared = {"model": config.model, "timeout": config.timeout}
    if client is not None:
        shared["client"] = client
    if provider == "openrouter":
        return OpenRouterProvider(api_key=config.api_key, **shared)
    if provider in ("nvidia_build", "nvidia"):
        return NvidiaBuildProvider(
            api_key=config.api_key,
            base_url=config.base_url or "https://integrate.api.nvidia.com/v1",
            **shared,
        )
    if provider == "ollama":
        return OllamaProvider(base_url=config.base_url or "http://127.0.0.1:11434/v1", api_key=config.api_key, **shared)
    if provider in ("openai", "openai_compatible", "openai-compatible"):
        if not config.api_key:
            raise LLMProviderError(
                f"Provider '{provider}' requires an API key. Set "
                f"{provider.upper().replace('-', '_')}_API_KEY in your environment."
            )
        if not config.base_url:
            raise LLMProviderError(
                f"Provider '{provider}' requires a base_url in config."
            )
        return OpenAICompatibleProvider(base_url=config.base_url, api_key=config.api_key, **shared)
    raise LLMProviderError(
        f"Unknown LLM provider: {config.provider!r}. "
        "Supported: openrouter, nvidia_build, ollama, openai-compatible."
    )
