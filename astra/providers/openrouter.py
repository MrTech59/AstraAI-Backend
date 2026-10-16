"""OpenRouter provider (https://openrouter.ai).

Requires ``OPENROUTER_API_KEY`` in the environment. Optional app
identification headers can be set via config later; kept minimal for 0.1.
"""

from __future__ import annotations

from .base import LLMProviderError
from .openai_compatible import OpenAICompatibleProvider


class OpenRouterProvider(OpenAICompatibleProvider):
    name = "openrouter"

    def __init__(self, api_key: str = "", model: str = "", timeout: float = 60.0, **kwargs) -> None:
        if not api_key:
            raise LLMProviderError(
                "OpenRouter API key is missing. Set OPENROUTER_API_KEY in your "
                "environment (or .env)."
            )
        super().__init__(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            model=model,
            timeout=timeout,
            default_headers={
                "HTTP-Referer": "https://github.com/astra-local/astra",
                "X-Title": "Astra Core",
            },
            **kwargs,
        )
