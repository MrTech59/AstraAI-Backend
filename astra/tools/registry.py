"""Tool registry: the single place the agent discovers and executes tools."""

from __future__ import annotations

import logging
from typing import Any

from .base import Tool, ToolResult

logger = logging.getLogger("astra.tools")


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            logger.warning("Overwriting registered tool: %s", tool.name)
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def definitions(self) -> list[dict[str, Any]]:
        """Tool definitions in OpenAI format for the LLM request."""
        return [tool.definition() for tool in self._tools.values()]

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.error(f"Unknown tool: {name}")
        result = await tool.execute(arguments)
        if result.is_error:
            logger.info("Tool failed: %s -> %s", name, result.content[:200])
        else:
            logger.info("Tool executed: %s", name)
        return result
