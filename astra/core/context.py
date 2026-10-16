"""Context builder.

Combines: system prompt + relevant memories + recent conversation + current
user message. History is capped by configuration — never unlimited.
"""

from __future__ import annotations

import logging
from typing import Any

from ..memory import MemoryStore

logger = logging.getLogger("astra.core")


class ContextBuilder:
    def __init__(
        self,
        memory: MemoryStore | None = None,
        system_prompt: str = "You are Astra, a helpful assistant.",
        max_messages: int = 30,
        max_memories: int = 5,
    ) -> None:
        self.memory = memory
        self.system_prompt = system_prompt
        self.max_messages = max_messages
        self.max_memories = max_memories

    def build(self, user_message: str, conversation_id: str = "default") -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]

        memory_note = self._memory_block(user_message)
        if memory_note:
            messages.append({"role": "system", "content": memory_note})

        if self.memory is not None:
            for stored in self.memory.get_recent_messages(conversation_id, self.max_messages):
                messages.append(stored.to_chat_message())

        messages.append({"role": "user", "content": user_message})
        return messages

    def _memory_block(self, user_message: str) -> str:
        if self.memory is None or self.max_memories <= 0:
            return ""
        try:
            memories = self.memory.search_memory(user_message, self.max_memories)
        except Exception as exc:  # memory must never break a turn
            logger.warning("Memory search failed: %s", exc)
            return ""
        if not memories:
            return ""
        lines = "\n".join(f"- {memory.content}" for memory in memories)
        return f"Relevant things you remember:\n{lines}"
