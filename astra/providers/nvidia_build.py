"""NVIDIA Build API provider (https://build.nvidia.com).

NVIDIA hosts 100+ models behind a single OpenAI-compatible endpoint:

    POST https://integrate.api.nvidia.com/v1/chat/completions
    Authorization: Bearer $NVIDIA_API_KEY

The OpenAI compatibility is isolated in this module; the agent only sees
the provider-neutral :class:`~astra.providers.base.LLMProvider` interface.
"""

from __future__ import annotations

from typing import Any

import httpx

from .base import ChatResponse, LLMProviderError, Message
from .openai_compatible import OpenAICompatibleProvider

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
# Verified live 2026-09-18 (200 OK, ~6s). deepseek-ai/deepseek-v4-flash-0731
# still appears in the catalog but its function hangs without responding.
DEFAULT_MODEL = "z-ai/glm-5.3-flash"


class NvidiaBuildProvider(OpenAICompatibleProvider):
    name = "nvidia_build"

    def __init__(
        self,
        api_key: str = "",
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
        **kwargs,
    ) -> None:
        if not api_key:
            raise LLMProviderError(
                "NVIDIA API key is missing. Set NVIDIA_API_KEY in your "
                "environment (or .env). Get a free key at https://build.nvidia.com."
            )
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout=timeout,
            **kwargs,
        )

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> ChatResponse:
        response = await self._request(messages, tools, temperature, max_tokens)

        if response.status_code == 400:
            detail = response.text[:300]
            raise LLMProviderError(
                f"NVIDIA Build API rejected the request (HTTP 400): {detail}. "
                "Check the model name and request parameters in config."
            )
        if response.status_code == 429:
            raise LLMProviderError(
                "NVIDIA Build API rate limit reached (HTTP 429). "
                "The free tier is limited to ~40 requests/minute — retry shortly."
            )
        if response.status_code >= 500:
            raise LLMProviderError(
                f"NVIDIA Build API server error (HTTP {response.status_code}). "
                "Retry later; the failure was upstream, not in Astra."
            )
        if response.status_code != 200:
            detail = response.text[:300]
            raise LLMProviderError(
                f"NVIDIA Build API error (HTTP {response.status_code}): {detail}"
            )

        return self._parse_response(response)
