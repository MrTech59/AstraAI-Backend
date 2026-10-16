"""Permission layer tests: policy, manager, approvals, and agent integration."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from astra.core import AgentLoop, ContextBuilder
from astra.memory import MemoryStore
from astra.core.events import (
    EVENT_PERMISSION_APPROVED,
    EVENT_PERMISSION_DENIED,
    EVENT_PERMISSION_REQUESTED,
    EventBus,
)
from astra.permissions import (
    ApprovalProvider,
    AutoApproveProvider,
    NonInteractiveApprovalProvider,
    PermissionAction,
    PermissionManager,
    PermissionOutcome,
    PermissionPolicy,
    sanitize_arguments,
)
from astra.providers.base import ChatResponse, LLMProvider, ToolCall
from astra.security import ShellPolicy
from astra.tools import ToolRegistry, make_filesystem_tools, make_shell_tool
from astra.tools.base import Tool, ToolPermission, ToolResult


@pytest.fixture
def tmp_root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def registry(tmp_root: Path) -> ToolRegistry:
    reg = ToolRegistry()
    for tool in make_filesystem_tools(roots=[tmp_root]):
        reg.register(tool)
    return reg


def manager(reg: ToolRegistry, approval: ApprovalProvider | None = None,
            policy: PermissionPolicy | None = None) -> PermissionManager:
    return PermissionManager(policy=policy, approval=approval or NonInteractiveApprovalProvider(), registry=reg)


class TestPolicyDefaults:
    @pytest.mark.asyncio
    async def test_list_auto_allowed(self, registry):
        d = await manager(registry).authorize("filesystem.list", {"path": "."})
        assert d.allowed

    @pytest.mark.asyncio
    async def test_read_auto_allowed(self, registry, tmp_root):
        (tmp_root / "a.txt").write_text("x")
        d = await manager(registry).authorize("filesystem.read", {"path": "a.txt"})
        assert d.allowed

    @pytest.mark.asyncio
    async def test_write_requires_confirmation(self, registry):
        d = await manager(registry).authorize("filesystem.write",
                                              {"path": "b.txt", "content": "y"})
        assert not d.allowed
        assert d.error["error"] == "permission_denied"

    @pytest.mark.asyncio
    async def test_shell_requires_confirmation(self):
        reg = ToolRegistry()
        reg.register(make_shell_tool(ShellPolicy(allow_shell=True, allowed_prefixes=["echo "])))
        d = await manager(reg).authorize("shell.exec", {"command": "echo hi"})
        assert not d.allowed
        assert d.error["error"] == "permission_denied"

    def test_missing_config_uses_safe_defaults(self):
        policy = PermissionPolicy.from_config(None)
        assert policy.action_for("filesystem.list", ToolPermission.READ) == PermissionAction.ALLOW
        assert policy.action_for("filesystem.write", ToolPermission.WRITE) == PermissionAction.CONFIRM
        assert policy.action_for("shell.exec", ToolPermission.EXECUTE) == PermissionAction.CONFIRM

    def test_config_overrides(self):
        class Cfg:
            filesystem_write = "deny"
            default = None
        policy = PermissionPolicy.from_config(Cfg())
        assert policy.action_for("filesystem.write", ToolPermission.WRITE) == PermissionAction.DENY
        # untouched keys keep defaults
        assert policy.action_for("filesystem.read", ToolPermission.READ) == PermissionAction.ALLOW

    def test_invalid_config_value_falls_back(self):
        class Cfg:
            filesystem_write = "bogus"
        policy = PermissionPolicy.from_config(Cfg())
        assert policy.action_for("filesystem.write", ToolPermission.WRITE) == PermissionAction.CONFIRM


class TestApprovals:
    @pytest.mark.asyncio
    async def test_write_approved_executes_once(self, registry, tmp_root):
        approval = AutoApproveProvider()
        m = manager(registry, approval)
        d = await m.authorize("filesystem.write", {"path": "w.txt", "content": "hello"})
        assert d.allowed
        result = await registry.execute("filesystem.write", d.arguments)
        assert not result.is_error
        assert (tmp_root / "w.txt").read_text() == "hello"
        # exactly one approval request
        assert len(approval.requests) == 1
        assert approval.requests[0].tool_name == "filesystem.write"

    @pytest.mark.asyncio
    async def test_write_denied_does_not_execute(self, registry, tmp_root):
        class Deny(ApprovalProvider):
            interactive = True
            async def approve(self, request):
                return False
        m = manager(registry, Deny())
        d = await m.authorize("filesystem.write", {"path": "w.txt", "content": "hello"})
        assert not d.allowed
        assert not (tmp_root / "w.txt").exists()

    @pytest.mark.asyncio
    async def test_non_interactive_fails_closed(self, registry):
        m = manager(registry, NonInteractiveApprovalProvider())
        d = await m.authorize("filesystem.write", {"path": "w.txt", "content": "x"})
        assert not d.allowed
        assert d.error["error"] == "permission_denied"
        assert "no interactive approval" in d.error["detail"]

    @pytest.mark.asyncio
    async def test_each_tool_call_asked_independently(self, registry):
        approval = AutoApproveProvider()
        m = manager(registry, approval)
        for i in range(3):
            d = await m.authorize("filesystem.write", {"path": f"f{i}.txt", "content": "x"})
            assert d.allowed
        assert len(approval.requests) == 3


class TestShellDefenceInDepth:
    @pytest.mark.asyncio
    async def test_shell_disabled_cannot_be_overridden(self, tmp_root):
        # shell tool not even registered when disabled
        reg = ToolRegistry()
        for t in make_filesystem_tools(roots=[tmp_root]):
            reg.register(t)
        m = manager(reg, AutoApproveProvider())
        d = await m.authorize("shell.exec", {"command": "echo hi"})
        assert not d.allowed
        assert d.error["error"] == "unknown_tool"

    @pytest.mark.asyncio
    async def test_shell_allowlist_cannot_be_overridden(self):
        reg = ToolRegistry()
        reg.register(make_shell_tool(ShellPolicy(allow_shell=True, allowed_prefixes=["echo "])))
        m = manager(reg, AutoApproveProvider())  # approval says yes; allowlist says no
        d = await m.authorize("shell.exec", {"command": "rm -rf /"})
        assert not d.allowed
        assert d.error["error"] == "invalid_arguments"

    @pytest.mark.asyncio
    async def test_shell_allowed_and_approved_executes(self):
        reg = ToolRegistry()
        reg.register(make_shell_tool(ShellPolicy(allow_shell=True, allowed_prefixes=["echo "])))
        m = manager(reg, AutoApproveProvider())
        d = await m.authorize("shell.exec", {"command": "echo safe"})
        assert d.allowed
        result = await reg.execute("shell.exec", d.arguments)
        assert "safe" in result.content


class TestRejections:
    @pytest.mark.asyncio
    async def test_unknown_tool(self, registry):
        d = await manager(registry).authorize("evil.tool", {})
        assert not d.allowed
        assert d.error["error"] == "unknown_tool"

    @pytest.mark.asyncio
    async def test_malformed_arguments(self, registry):
        d = await manager(registry).authorize("filesystem.read", {"path": 123})
        assert not d.allowed
        assert d.error["error"] == "invalid_arguments"

    @pytest.mark.asyncio
    async def test_traversal_blocked_even_with_approval(self, registry):
        m = manager(registry, AutoApproveProvider())  # user would say yes
        d = await m.authorize("filesystem.read", {"path": "../../secrets.txt"})
        assert not d.allowed
        assert d.error["error"] == "invalid_arguments"

    @pytest.mark.asyncio
    async def test_traversal_write_blocked_before_prompt(self, registry, tmp_root):
        approval = AutoApproveProvider()
        m = manager(registry, approval)
        d = await m.authorize("filesystem.write", {"path": "../escape.txt", "content": "x"})
        assert not d.allowed
        assert approval.requests == []  # never asked
        assert not (tmp_root.parent / "escape.txt").exists()


class TestSecretRedaction:
    @pytest.mark.asyncio
    async def test_secret_values_redacted_in_request(self, tmp_root):
        reg = ToolRegistry()
        reg.register(Tool(
            name="network.call",
            description="Call an API",
            parameters={"type": "object", "properties": {}},
            permission=ToolPermission.NETWORK,
            execute_fn=lambda a: None,
        ))
        captured = []
        class Capture(ApprovalProvider):
            interactive = True
            async def approve(self, request):
                captured.append(request)
                return False
        m = manager(reg, Capture())
        d = await m.authorize("network.call", {
            "url": "https://example.com",
            "api_key": "SUPERSECRET123",
            "access_token": "TOKENTOKENTOKEN",
            "password": "hunter2",
        })
        assert not d.allowed
        req = captured[0]
        assert req.arguments["api_key"] == "***REDACTED***"
        assert req.arguments["access_token"] == "***REDACTED***"
        assert req.arguments["password"] == "***REDACTED***"
        assert req.arguments["url"] == "https://example.com"
        assert "SUPERSECRET123" not in str(req.arguments)
        assert "TOKENTOKENTOKEN" not in str(req.arguments)

    def test_sanitize_strips_approval_forgery(self):
        clean = sanitize_arguments({"path": "x", "approved": True, "user_approved": "yes", "api_token": "tok"})
        assert "approved" not in clean
        assert "user_approved" not in clean
        assert clean["api_token"] == "***REDACTED***"
        assert clean["path"] == "x"

    @pytest.mark.asyncio
    async def test_model_cannot_self_approve(self, registry, tmp_root):
        """A tool call carrying fake approval fields gains nothing."""
        m = manager(registry, AutoApproveProvider())
        d = await m.authorize("filesystem.write", {
            "path": "s.txt", "content": "x", "approved": True, "permission_granted": True,
        })
        # AutoApprove allows — but the approval keys must not reach execution.
        assert d.allowed
        assert "approved" not in d.arguments
        assert "permission_granted" not in d.arguments
        # And crucially: with a fail-closed provider, fake fields do nothing.
        m2 = manager(registry, NonInteractiveApprovalProvider())
        d2 = await m2.authorize("filesystem.write", {
            "path": "s.txt", "content": "x", "approved": True,
        })
        assert not d2.allowed


class TestOrdering:
    @pytest.mark.asyncio
    async def test_permission_check_immediately_before_execution(self, tmp_root):
        events: list[str] = []
        reg = ToolRegistry()
        reg.register(Tool(
            name="tracked.write",
            description="tracked",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            permission=ToolPermission.WRITE,
            execute_fn=lambda a: (events.append("executed"), ToolResult.ok("done"))[1],
        ))
        class OrderingApprove(ApprovalProvider):
            interactive = True
            async def approve(self, request):
                events.append("asked")
                assert "executed" not in events  # not run before the ask
                return True
        m = PermissionManager(approval=OrderingApprove(), registry=reg)
        d = await m.authorize("tracked.write", {"path": "x"})
        assert d.allowed
        await reg.execute("tracked.write", d.arguments)
        assert events == ["asked", "executed"]

    @pytest.mark.asyncio
    async def test_events_emitted(self, registry):
        bus = EventBus()
        seen: list[str] = []
        bus.subscribe(lambda e: seen.append(e.type) or asyncio.sleep(0))
        m = PermissionManager(approval=AutoApproveProvider(), registry=registry, event_bus=bus)
        await m.authorize("filesystem.write", {"path": "e.txt", "content": "x"})
        assert EVENT_PERMISSION_REQUESTED in seen
        assert EVENT_PERMISSION_APPROVED in seen

        m2 = PermissionManager(approval=NonInteractiveApprovalProvider(), registry=registry, event_bus=bus)
        await m2.authorize("filesystem.write", {"path": "e.txt", "content": "x"})
        assert EVENT_PERMISSION_DENIED in seen


class TestAgentLoopIntegration:
    class ScriptedProvider(LLMProvider):
        name = "scripted"
        def __init__(self, responses):
            self._responses = list(responses)
            self._idx = 0
        async def chat(self, messages, tools=None, temperature=0.7, max_tokens=2048):
            resp = self._responses[self._idx]
            self._idx += 1
            return resp

    @pytest.mark.asyncio
    async def test_denied_tool_produces_structured_result(self, tmp_root):
        provider = self.ScriptedProvider([
            ChatResponse(tool_calls=[ToolCall(id="1", name="filesystem.write",
                                              arguments={"path": "n.txt", "content": "x"})]),
            ChatResponse(content="understood, it was denied"),
        ])
        reg = ToolRegistry()
        for t in make_filesystem_tools(roots=[tmp_root]):
            reg.register(t)
        # fail-closed by default (no approval provider)
        memory = MemoryStore(tmp_root / "t.db")
        agent = AgentLoop(provider, reg, ContextBuilder(memory=memory, system_prompt="t"), memory=memory)
        reply = await agent.run("write a file")
        assert reply == "understood, it was denied"
        assert not (tmp_root / "n.txt").exists()
        # the tool result fed back to the model was the structured denial
        last_tool_msg = [m for m in agent.context.memory.get_recent_messages("default", 10)
                         if m.role == "tool"][-1]
        payload = json.loads(last_tool_msg.content)
        assert payload["ok"] is False
        assert payload["error"] == "permission_denied"
        assert payload["tool"] == "filesystem.write"

    @pytest.mark.asyncio
    async def test_approved_tool_executes_in_loop(self, tmp_root):
        provider = self.ScriptedProvider([
            ChatResponse(tool_calls=[ToolCall(id="1", name="filesystem.write",
                                              arguments={"path": "y.txt", "content": "z"})]),
            ChatResponse(content="file created"),
        ])
        reg = ToolRegistry()
        for t in make_filesystem_tools(roots=[tmp_root]):
            reg.register(t)
        perms = PermissionManager(approval=AutoApproveProvider(), registry=reg)
        agent = AgentLoop(provider, reg, ContextBuilder(system_prompt="t"), permissions=perms)
        reply = await agent.run("write a file")
        assert reply == "file created"
        assert (tmp_root / "y.txt").read_text() == "z"

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_evaluated_independently(self, tmp_root):
        provider = self.ScriptedProvider([
            ChatResponse(tool_calls=[
                ToolCall(id="1", name="filesystem.list", arguments={"path": "."}),
                ToolCall(id="2", name="filesystem.write", arguments={"path": "m.txt", "content": "q"}),
            ]),
            ChatResponse(content="done"),
        ])
        reg = ToolRegistry()
        for t in make_filesystem_tools(roots=[tmp_root]):
            reg.register(t)
        approval = AutoApproveProvider()
        perms = PermissionManager(approval=approval, registry=reg)
        agent = AgentLoop(provider, reg, ContextBuilder(system_prompt="t"), permissions=perms)
        reply = await agent.run("list and write")
        assert reply == "done"
        # list auto-allowed (no request), write asked exactly once
        assert [r.tool_name for r in approval.requests] == ["filesystem.write"]
        assert (tmp_root / "m.txt").exists()


class TestRegistryUnchanged:
    @pytest.mark.asyncio
    async def test_direct_registry_execution_still_works(self, tmp_root):
        """Trusted callers using the registry directly are unaffected."""
        reg = ToolRegistry()
        for t in make_filesystem_tools(roots=[tmp_root]):
            reg.register(t)
        result = await reg.execute("filesystem.write", {"path": "direct.txt", "content": "d"})
        assert not result.is_error
        assert (tmp_root / "direct.txt").exists()
