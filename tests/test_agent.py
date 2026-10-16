"""Agent loop tests."""

from __future__ import annotations

from typing import Any

import pytest

from astra.core import AgentLoop, ContextBuilder
from astra.core.events import EventBus
from astra.providers.base import ChatResponse, LLMProvider, ToolCall
from astra.tools import ToolRegistry, make_filesystem_tools


class FakeProvider(LLMProvider):
    name = "fake"

    def __init__(self, responses: list[ChatResponse]) -> None:
        self._responses = list(responses)
        self._index = 0

    async def chat(self, messages: list[dict[str, Any]], tools=None, temperature=0.7, max_tokens=2048) -> ChatResponse:
        response = self._responses[self._index]
        self._index += 1
        return response


class TestAgentLoop:
    @pytest.mark.asyncio
    async def test_normal_reply(self, tmp_path) -> None:
        provider = FakeProvider([ChatResponse(content="hi there")])
        tools = ToolRegistry()
        for tool in make_filesystem_tools(roots=[tmp_path]):
            tools.register(tool)
        context = ContextBuilder(system_prompt="Be helpful.")
        agent = AgentLoop(provider, tools, context, max_iterations=3)
        reply = await agent.run("hello")
        assert reply == "hi there"

    @pytest.mark.asyncio
    async def test_tool_call_then_reply(self, tmp_path) -> None:
        provider = FakeProvider([
            ChatResponse(
                tool_calls=[ToolCall(id="c1", name="filesystem.list", arguments={"path": "."})]
            ),
            ChatResponse(content="done"),
        ])
        tools = ToolRegistry()
        for tool in make_filesystem_tools(roots=[tmp_path]):
            tools.register(tool)
        context = ContextBuilder(system_prompt="Be helpful.")
        agent = AgentLoop(provider, tools, context, max_iterations=3)
        reply = await agent.run("list files")
        assert reply == "done"

    @pytest.mark.asyncio
    async def test_max_iterations(self, tmp_path) -> None:
        provider = FakeProvider([
            ChatResponse(
                tool_calls=[ToolCall(id=f"c{i}", name="filesystem.list", arguments={"path": "."})]
            )
            for i in range(10)
        ])
        tools = ToolRegistry()
        for tool in make_filesystem_tools(roots=[tmp_path]):
            tools.register(tool)
        context = ContextBuilder(system_prompt="Be helpful.")
        agent = AgentLoop(provider, tools, context, max_iterations=2)
        reply = await agent.run("list files")
        assert "maximum" in reply.lower()

    @pytest.mark.asyncio
    async def test_tool_failure_returned(self, tmp_path) -> None:
        provider = FakeProvider([
            ChatResponse(
                tool_calls=[ToolCall(id="c1", name="filesystem.read", arguments={"path": "missing.txt"})]
            ),
            ChatResponse(content="not found"),
        ])
        tools = ToolRegistry()
        for tool in make_filesystem_tools(roots=[tmp_path]):
            tools.register(tool)
        context = ContextBuilder(system_prompt="Be helpful.")
        agent = AgentLoop(provider, tools, context, max_iterations=3)
        reply = await agent.run("read missing")
        assert "not found" in reply.lower() or reply == "not found"
