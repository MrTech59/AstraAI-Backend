"""End-to-end smoke test: scripted provider drives the full agent flow.

Simulates:
1. list files -> tool call -> final answer
2. write file -> tool call -> final answer
3. read file -> tool call -> final answer

No network or API key required.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astra.core import AgentLoop, ContextBuilder
from astra.memory import MemoryStore
from astra.permissions import AutoApproveProvider, PermissionManager
from astra.providers.base import ChatResponse, LLMProvider, ToolCall
from astra.security import ShellPolicy
from astra.tools import ToolRegistry, make_filesystem_tools, make_shell_tool


class ScriptedProvider(LLMProvider):
    name = "scripted"

    def __init__(self, script: list[ChatResponse]) -> None:
        self._script = list(script)
        self._idx = 0

    async def chat(self, messages, tools=None, temperature=0.7, max_tokens=2048) -> ChatResponse:
        resp = self._script[self._idx]
        self._idx += 1
        return resp


async def main() -> int:
    tmp = Path(__file__).resolve().parent / "_smoke_tmp"
    tmp.mkdir(exist_ok=True)

    provider = ScriptedProvider([
        # turn 1: list
        ChatResponse(tool_calls=[ToolCall(id="1", name="filesystem.list", arguments={"path": "."})]),
        ChatResponse(content="I found a few files here."),
        # turn 2: write
        ChatResponse(tool_calls=[ToolCall(id="2", name="filesystem.write", arguments={"path": "test.txt", "content": "Hello from Astra"})]),
        ChatResponse(content="Done, created test.txt."),
        # turn 3: read
        ChatResponse(tool_calls=[ToolCall(id="3", name="filesystem.read", arguments={"path": "test.txt"})]),
        ChatResponse(content="It contains: Hello from Astra"),
    ])

    tools = ToolRegistry()
    for tool in make_filesystem_tools(roots=[tmp]):
        tools.register(tool)
    tools.register(make_shell_tool(ShellPolicy(allow_shell=False)))

    memory = MemoryStore(tmp / "smoke.db")
    context = ContextBuilder(memory=memory, system_prompt="You are Astra.")
    approval = AutoApproveProvider()
    permissions = PermissionManager(approval=approval, registry=tools)
    agent = AgentLoop(provider, tools, context, max_iterations=5, memory=memory, permissions=permissions)

    r1 = await agent.run("What files are here?", conversation_id="smoke")
    print("Turn 1:", r1)
    assert r1 == "I found a few files here."

    r2 = await agent.run('Create test.txt containing "Hello from Astra"', conversation_id="smoke")
    print("Turn 2:", r2)
    assert (tmp / "test.txt").read_text() == "Hello from Astra"

    r3 = await agent.run("Read test.txt", conversation_id="smoke")
    print("Turn 3:", r3)
    assert "Hello from Astra" in r3

    approved_tools = [r.tool_name for r in approval.requests]
    print("Approval requests seen:", approved_tools)
    assert "filesystem.write" in approved_tools  # write went through the gate

    history = memory.get_recent_messages("smoke", 20)
    print(f"History: {len(history)} messages persisted")
    assert len(history) >= 6

    memory.close()
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
