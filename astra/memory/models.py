"""Memory data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    role: str  # "user" | "assistant" | "system" | "tool"
    content: str
    conversation_id: str = "default"
    tool_call_id: str | None = None
    name: str | None = None  # tool name for role="tool"
    timestamp: float = 0.0
    id: int | None = None

    def to_chat_message(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_call_id:
            message["tool_call_id"] = self.tool_call_id
        if self.name:
            message["name"] = self.name
        return message


@dataclass
class Memory:
    content: str
    key: str | None = None
    id: int | None = None
    created_at: float = 0.0
