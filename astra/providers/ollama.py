"""Ollama provider.

Talks to a local Ollama server through its OpenAI-compatible endpoint
(default ``http://127.0.0.1:11434/v1``). No API key required.
"""

from __future__ import annotations

from .openai_compatible import OpenAICompatibleProvider


class OllamaProvider(OpenAICompatibleProvider):
    name = "ollama"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434/v1",
        model: str = "qwen3",
        timeout: float = 120.0,
        api_key: str = "",
        **kwargs,
    ) -> None:
        super().__init__(
            base_url=base_url,
            api_key=api_key,  # usually unused; kept for custom proxies
            model=model,
            timeout=timeout,
            **kwargs,
        )
