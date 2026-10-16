"""LLM provider abstraction.

Astra Core never talks to a specific vendor directly — everything goes
through :class:`LLMProvider`. Tool definitions are passed in the standard
OpenAI function-calling format so any compatible backend can serve Astra.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class LLMProviderError(Exception):
    """Raised for provider-level failures (config, network, API errors)."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    finish_reason: str = "stop"

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


Message = dict[str, Any]  # OpenAI-style message dicts


class LLMProvider(ABC):
    """Abstract chat provider."""

    name: str = "base"

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> ChatResponse:
        """Send a chat completion request and return the parsed response."""
