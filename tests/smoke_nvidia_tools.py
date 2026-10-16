"""Optional LIVE NVIDIA Build API tool-loop integration test.

Manual only (excluded from pytest via collect_ignore in conftest):

    python tests/smoke_nvidia_tools.py

Verifies the real authenticated model can drive the full loop:

    prompt -> NVIDIA -> tool call -> permission layer -> approval ->
    ToolRegistry -> filesystem tool -> result -> NVIDIA -> final reply

Uses a temporary workspace under the system temp dir; never touches the
real project files. Requires NVIDIA_API_KEY in the environment or .env.
The key is never printed.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astra.config import load_settings, resolve_api_key
from astra.core import AgentLoop, ContextBuilder
from astra.memory import MemoryStore
from astra.permissions import AutoApproveProvider, PermissionManager
from astra.providers import LLMProviderError, create_provider
from astra.providers.nvidia_build import DEFAULT_MODEL
from astra.tools import ToolRegistry, make_filesystem_tools


async def main() -> int:
    settings = load_settings()
    settings.llm.provider = "nvidia_build"
    settings.llm.model = DEFAULT_MODEL
    settings.llm.api_key = resolve_api_key(settings.llm.provider)
    settings.llm.timeout = 180.0

    if not settings.llm.api_key:
        print("SKIP: NVIDIA_API_KEY is not set in the environment or .env.")
        return 2

    workspace = Path(tempfile.mkdtemp(prefix="astra_toolloop_"))
    registry = ToolRegistry()
    for tool in make_filesystem_tools(roots=[workspace]):
        registry.register(tool)

    approval = AutoApproveProvider()
    permissions = PermissionManager(approval=approval, registry=registry)
    memory = MemoryStore(workspace / "memory.db")
    provider = create_provider(settings.llm)
    agent = AgentLoop(
        provider,
        registry,
        ContextBuilder(memory=memory, system_prompt=(
            "You are Astra, a test assistant. When the user asks you to create "
            "or read files, use the filesystem tools. Always use tools for file "
            "operations rather than describing them."
        )),
        max_iterations=settings.agent.max_tool_iterations,
        memory=memory,
        permissions=permissions,
    )

    target = workspace / "astra_tool_test.txt"
    try:
        print(f"Workspace: {workspace}")
        print("Step 1: asking model to create a file (expects filesystem.write tool call)")
        t0 = time.monotonic()
        try:
            reply = await agent.run(
                f"Use the filesystem.write tool to create a file named {target.name} "
                "containing exactly: Astra NVIDIA tool-loop test",
                conversation_id="toolloop",
            )
        except LLMProviderError as exc:
            print(f"FAIL (provider): {exc}")
            return 1
        elapsed = time.monotonic() - t0

        if not target.exists():
            print(f"FAIL: file was not created. Model reply: {reply[:300]!r}")
            print("Approval requests seen:", [r.tool_name for r in approval.requests])
            return 1
        content = target.read_text(encoding="utf-8")
        if "Astra NVIDIA tool-loop test" not in content:
            print(f"FAIL: unexpected file content: {content!r}")
            return 1
        print(f"Step 1 OK in {elapsed:.1f}s — file created with expected content")
        print(f"  model reply: {reply[:200]!r}")
        print("  approval requests:", [r.tool_name for r in approval.requests])

        print("Step 2: asking model to read the file back (expects filesystem.read)")
        approval.requests.clear()
        reply2 = await agent.run(
            f"Use the filesystem.read tool to read {target.name} and tell me its contents.",
            conversation_id="toolloop",
        )
        if "Astra NVIDIA tool-loop test" not in reply2:
            print(f"FAIL: read-back reply missing content: {reply2[:300]!r}")
            return 1
        print(f"Step 2 OK — model read the file: {reply2[:200]!r}")
        print("  approval requests (read should be auto-allowed):", [r.tool_name for r in approval.requests])

        print("SMOKE OK — live NVIDIA tool loop verified")
        return 0
    finally:
        memory.close()
        for child in sorted(workspace.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            else:
                child.rmdir()
        workspace.rmdir()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
