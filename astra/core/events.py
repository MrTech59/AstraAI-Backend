"""Lightweight event system.

UIs/API clients subscribe to agent events. Only safe status information is
emitted — never raw model chain-of-thought.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

#: agent.started, agent.thinking, tool.started, tool.finished,
#: message.delta, message.completed, agent.finished, error
EVENT_AGENT_STARTED = "agent.started"
EVENT_AGENT_THINKING = "agent.thinking"
EVENT_TOOL_STARTED = "tool.started"
EVENT_TOOL_FINISHED = "tool.finished"
EVENT_MESSAGE_DELTA = "message.delta"
EVENT_MESSAGE_COMPLETED = "message.completed"
EVENT_AGENT_FINISHED = "agent.finished"
EVENT_ERROR = "error"
EVENT_PERMISSION_REQUESTED = "permission.requested"
EVENT_PERMISSION_APPROVED = "permission.approved"
EVENT_PERMISSION_DENIED = "permission.denied"


@dataclass
class Event:
    type: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "data": self.data, "timestamp": self.timestamp}


EventCallback = Callable[[Event], Awaitable[None]]


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[EventCallback] = []

    def subscribe(self, callback: EventCallback) -> None:
        self._subscribers.append(callback)

    def unsubscribe(self, callback: EventCallback) -> None:
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    async def emit(self, event: Event) -> None:
        for callback in list(self._subscribers):
            await callback(event)
