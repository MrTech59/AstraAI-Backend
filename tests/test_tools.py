"""Tool tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from astra.security import SecurityError, ShellPolicy, resolve_allowed_path
from astra.tools import ToolRegistry, make_filesystem_tools, make_shell_tool
from astra.tools.base import ToolPermission


@pytest.fixture
def tmp_root(tmp_path: Path) -> Path:
    return tmp_path


class TestFilesystemTools:
    @pytest.mark.asyncio
    async def test_read(self, tmp_root: Path) -> None:
        (tmp_root / "a.txt").write_text("hello", encoding="utf-8")
        tools = make_filesystem_tools(roots=[tmp_root])
        read_tool = next(t for t in tools if t.name == "filesystem.read")
        result = await read_tool.execute({"path": "a.txt"})
        assert result.content == "hello"
        assert not result.is_error

    @pytest.mark.asyncio
    async def test_read_outside_root(self, tmp_root: Path) -> None:
        tools = make_filesystem_tools(roots=[tmp_root])
        read_tool = next(t for t in tools if t.name == "filesystem.read")
        result = await read_tool.execute({"path": "../outside.txt"})
        assert result.is_error
        assert "outside allowed roots" in result.content or "outside allowed" in result.content

    @pytest.mark.asyncio
    async def test_write(self, tmp_root: Path) -> None:
        tools = make_filesystem_tools(roots=[tmp_root])
        write_tool = next(t for t in tools if t.name == "filesystem.write")
        result = await write_tool.execute({"path": "b.txt", "content": "world"})
        assert not result.is_error
        assert (tmp_root / "b.txt").read_text(encoding="utf-8") == "world"

    @pytest.mark.asyncio
    async def test_list(self, tmp_root: Path) -> None:
        (tmp_root / "a.txt").touch()
        (tmp_root / "subdir").mkdir()
        tools = make_filesystem_tools(roots=[tmp_root])
        list_tool = next(t for t in tools if t.name == "filesystem.list")
        result = await list_tool.execute({"path": "."})
        assert "a.txt" in result.content
        assert "subdir/" in result.content


class TestShellTool:
    @pytest.mark.asyncio
    async def test_disabled_by_default(self) -> None:
        policy = ShellPolicy(allow_shell=False)
        tool = make_shell_tool(policy)
        result = await tool.execute({"command": "echo hi"})
        assert result.is_error
        assert "disabled" in result.content

    @pytest.mark.asyncio
    async def test_allowed_prefix(self) -> None:
        policy = ShellPolicy(allow_shell=True, allowed_prefixes=["echo "])
        tool = make_shell_tool(policy)
        result = await tool.execute({"command": "echo hello"})
        assert not result.is_error
        assert "hello" in result.content

    @pytest.mark.asyncio
    async def test_unallowed_prefix(self) -> None:
        policy = ShellPolicy(allow_shell=True, allowed_prefixes=["echo "])
        tool = make_shell_tool(policy)
        result = await tool.execute({"command": "rm -rf /"})
        assert result.is_error
        assert "allowed prefix" in result.content

    @pytest.mark.asyncio
    async def test_no_prefixes_configured(self) -> None:
        policy = ShellPolicy(allow_shell=True, allowed_prefixes=[])
        tool = make_shell_tool(policy)
        result = await tool.execute({"command": "echo hi"})
        assert result.is_error
        assert "no allowed_shell_prefixes" in result.content


class TestToolRegistry:
    def test_register_and_execute(self) -> None:
        registry = ToolRegistry()
        tools = make_filesystem_tools(roots=[Path(".")])
        for tool in tools:
            registry.register(tool)
        assert len(registry) == 3
        assert "filesystem.read" in registry.names()

    def test_definitions(self) -> None:
        registry = ToolRegistry()
        for tool in make_filesystem_tools(roots=[Path(".")]):
            registry.register(tool)
        defs = registry.definitions()
        assert all(d["type"] == "function" for d in defs)

    @pytest.mark.asyncio
    async def test_execute_unknown(self) -> None:
        registry = ToolRegistry()
        result = await registry.execute("nope", {})
        assert result.is_error
        assert "Unknown tool" in result.content


class TestResolveAllowedPath:
    def test_inside_root(self, tmp_root: Path) -> None:
        resolved = resolve_allowed_path("a.txt", [tmp_root])
        assert resolved == tmp_root / "a.txt"

    def test_traversal_blocked(self, tmp_root: Path) -> None:
        with pytest.raises(SecurityError, match="outside allowed"):
            resolve_allowed_path("../outside.txt", [tmp_root])
