"""Agent loop.

Orchestrates a single turn:
1. Build context
2. Send to LLM
3. If tool calls requested, execute each via the registry and re-prompt
4. Stop after max_tool_iterations

Emits events so UI/API layers can show status safely.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..memory import MemoryStore, Message
from ..permissions import PermissionManager
from ..providers.base import ChatResponse, LLMProvider
from ..tools import ToolRegistry, ToolResult
from .context import ContextBuilder
from .events import (
    EVENT_AGENT_FINISHED,
    EVENT_AGENT_STARTED,
    EVENT_AGENT_THINKING,
    EVENT_ERROR,
    EVENT_MESSAGE_COMPLETED,
    EVENT_MESSAGE_DELTA,
    EVENT_TOOL_FINISHED,
    EVENT_TOOL_STARTED,
    Event,
    EventBus,
)

logger = logging.getLogger("astra.core")


class AgentLoop:
    def __init__(
        self,
        provider: LLMProvider,
        tools: ToolRegistry,
        context: ContextBuilder,
        max_iterations: int = 8,
        event_bus: EventBus | None = None,
        memory: MemoryStore | None = None,
        permissions: PermissionManager | None = None,
    ) -> None:
        self.provider = provider
        self.tools = tools
        self.context = context
        self.max_iterations = max_iterations
        self.events = event_bus or EventBus()
        self.memory = memory
        #: Security boundary: every model-generated tool call passes through
        #: this before ToolRegistry execution. Defaults to fail-closed
        #: (non-interactive) if the caller does not supply one.
        self.permissions = permissions or PermissionManager(registry=tools, event_bus=self.events)

    def _persist(self, message: Message) -> None:
        if self.memory is None:
            return
        try:
            self.memory.save_message(message)
        except Exception as exc:
            logger.warning("Failed to persist message: %s", exc)

    async def run(self, user_message: str, conversation_id: str = "default") -> str:
        await self.events.emit(Event(EVENT_AGENT_STARTED, {"conversation_id": conversation_id}))
        self._persist(Message(role="user", content=user_message, conversation_id=conversation_id))
        messages = self.context.build(user_message, conversation_id)
        final_response = ""

        for iteration in range(self.max_iterations):
            await self.events.emit(Event(EVENT_AGENT_THINKING, {"iteration": iteration}))
            logger.info("Agent iteration %s", iteration)

            try:
                response = await self.provider.chat(
                    messages,
                    tools=self.tools.definitions(),
                )
            except Exception as exc:
                logger.exception("LLM request failed")
                await self.events.emit(Event(EVENT_ERROR, {"error": str(exc)}))
                return self._error_reply(exc)

            if not response.has_tool_calls:
                final_response = response.content
                await self.events.emit(
                    Event(EVENT_MESSAGE_COMPLETED, {"content": final_response})
                )
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments,
                            },
                        }
                        for call in response.tool_calls
                    ],
                }
            )

            for call in response.tool_calls:
                await self.events.emit(
                    Event(EVENT_TOOL_STARTED, {"name": call.name, "arguments": call.arguments})
                )
                decision = await self.permissions.authorize(call.name, call.arguments)
                if decision.allowed:
                    result = await self.tools.execute(call.name, decision.arguments)
                else:
                    # Structured, model-consumable denial; execution skipped.
                    result = ToolResult.error(decision.error_text())
                    logger.info("Tool blocked by permission layer: %s (%s)",
                                call.name, decision.error.get("error"))
                await self.events.emit(
                    Event(
                        EVENT_TOOL_FINISHED,
                        {"name": call.name, "result": result.content, "error": result.is_error},
                    )
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": result.content,
                    }
                )
                self._persist(
                    Message(
                        role="tool",
                        content=result.content,
                        conversation_id=conversation_id,
                        tool_call_id=call.id,
                        name=call.name,
                    )
                )
        else:
            logger.warning("Agent reached max tool iterations (%s)", self.max_iterations)
            final_response = (
                "I reached the maximum number of tool iterations without finishing. "
                "If you need more steps, break your request into smaller pieces."
            )
            await self.events.emit(
                Event(EVENT_MESSAGE_COMPLETED, {"content": final_response})
            )

        self._persist(
            Message(role="assistant", content=final_response, conversation_id=conversation_id)
        )
        await self.events.emit(Event(EVENT_AGENT_FINISHED, {"conversation_id": conversation_id}))
        return final_response

    def _error_reply(self, exc: Exception) -> str:
        return (
            f"Astra error: {type(exc).__name__}: {exc}\n"
            "Check your provider configuration and API key."
        )
