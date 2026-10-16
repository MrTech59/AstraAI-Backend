"""Lightweight MCP client abstraction.

0.1 scope:
* Connect to an MCP server via HTTP+JSON
* Discover tools via a /tools endpoint
* Convert discovered tools into Astra tools for the agent

Full protocol adoption (JSON-RPC, resources, prompts) is a future step.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from ..tools.base import Tool, ToolPermission, ToolResult


class MCPClientError(Exception):
    """Raised when MCP communication fails."""


class MCPClient:
    """HTTP MCP client stub suitable for simple MCP bridges."""

    def __init__(self, name: str, url: str, client: httpx.AsyncClient | None = None) -> None:
        self.name = name
        self.url = url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=30.0)

    async def list_tools(self) -> list[Tool]:
        endpoint = f"{self.url}/mcp/tools"
        try:
            response = await self._client.get(endpoint)
        except httpx.TransportError as exc:
            raise MCPClientError(f"Cannot reach MCP server {self.name!r}: {exc}") from exc

        if response.status_code != 200:
            detail = response.text[:300]
            raise MCPClientError(
                f"MCP server {self.name!r} returned HTTP {response.status_code}: {detail}"
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise MCPClientError(f"MCP server {self.name!r} returned invalid JSON.") from exc

        tools: list[Tool] = []
        for raw in data.get("tools", data.get("functions", [])):
            name = str(raw.get("name", ""))
            if not name:
                continue
            parameters = raw.get("parameters") or raw.get("input_schema") or {}
            description = str(raw.get("description", ""))
            tool = Tool(
                name=f"{self.name}.{name}",
                description=description,
                parameters=parameters,
                permission=ToolPermission.NETWORK,
                execute_fn=await self._build_executor(name),
            )
            tools.append(tool)
        return tools

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        endpoint = f"{self.url}/mcp/call"
        try:
            response = await self._client.post(
                endpoint,
                json={"name": tool_name, "arguments": arguments},
            )
        except httpx.TransportError as exc:
            return ToolResult.error(f"MCP call failed: {exc}")
        if response.status_code != 200:
            return ToolResult.error(f"MCP call failed (HTTP {response.status_code}): {response.text[:300]}")
        try:
            data = response.json()
        except ValueError:
            return ToolResult.error("MCP call returned invalid JSON.")
        if data.get("error"):
            return ToolResult.error(str(data["error"]))
        return ToolResult.ok(str(data.get("result", data.get("content", ""))))

    def _build_executor(self, tool_name: str):
        async def executor(arguments: dict[str, Any]) -> ToolResult:
            return await self.call_tool(tool_name, arguments)

        return executor


async def connect_mcp_servers(configured: list[Any], client: httpx.AsyncClient | None = None) -> list[Tool]:
    """Given a list of config dicts (or objects), connect to all enabled MCP servers and return discovered tools."""
    tools: list[Tool] = []
    for item in configured:
        if not getattr(item, "enabled", item.get("enabled")):
            continue
        name = str(getattr(item, "name", item.get("name", "")))
        url = str(getattr(item, "url", item.get("url", "")))
        if not name or not url:
            continue
        mcp = MCPClient(name=name, url=url, client=client)
        try:
            server_tools = await mcp.list_tools()
            tools.extend(server_tools)
        except MCPClientError as exc:
            from ..logging import logging as _logging
            _logging.getLogger("astra.mcp").warning("MCP %s: %s", name, exc)
    return tools
