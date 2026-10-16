"""Filesystem tools: read, write, list.

All paths are confined to configured roots (see astra.security). Validation
runs both at execution time and via ``validate_fn`` so the permission layer
can reject traversal/invalid calls *before* asking the user.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..security import resolve_allowed_path
from .base import Tool, ToolPermission, ToolResult

MAX_READ_BYTES = 64 * 1024
MAX_LIST_ENTRIES = 200

_PATH_PROPERTY = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "File or directory path"},
    },
    "required": ["path"],
}


def _require_path(arguments: dict[str, Any]) -> str:
    path = arguments.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Argument 'path' must be a non-empty string.")
    return path


def make_filesystem_tools(roots: list[Path]) -> list[Tool]:
    """Create the three filesystem tools bound to ``roots``."""

    async def read_file(arguments: dict[str, Any]) -> ToolResult:
        path = resolve_allowed_path(_require_path(arguments), roots)
        if not path.is_file():
            return ToolResult.error(f"Not a file: {path}")
        if path.stat().st_size > MAX_READ_BYTES:
            return ToolResult.error(f"File too large to read ({path.stat().st_size} bytes).")
        return ToolResult.ok(path.read_text(encoding="utf-8", errors="replace"))

    async def write_file(arguments: dict[str, Any]) -> ToolResult:
        path = resolve_allowed_path(_require_path(arguments), roots)
        content = arguments.get("content")
        if not isinstance(content, str):
            return ToolResult.error("Argument 'content' must be a string.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return ToolResult.ok(f"Wrote {len(content)} characters to {path}.")

    async def list_dir(arguments: dict[str, Any]) -> ToolResult:
        path = resolve_allowed_path(_require_path(arguments), roots)
        if not path.is_dir():
            return ToolResult.error(f"Not a directory: {path}")
        lines = []
        for index, entry in enumerate(sorted(path.iterdir())):
            if index >= MAX_LIST_ENTRIES:
                lines.append(f"... ({MAX_LIST_ENTRIES} entry limit reached)")
                break
            suffix = "/" if entry.is_dir() else ""
            lines.append(entry.name + suffix)
        return ToolResult.ok("\n".join(lines) if lines else "(empty directory)")

    # --- permission-layer validators (run BEFORE any confirmation prompt) ---

    async def validate_read(arguments: dict[str, Any]) -> str | None:
        try:
            path = resolve_allowed_path(_require_path(arguments), roots)
        except Exception as exc:
            return str(exc)
        if not path.is_file():
            return f"Not a file: {path}"
        return None

    async def validate_write(arguments: dict[str, Any]) -> str | None:
        try:
            resolve_allowed_path(_require_path(arguments), roots)
        except Exception as exc:
            return str(exc)
        if not isinstance(arguments.get("content"), str):
            return "Argument 'content' must be a string."
        return None

    async def validate_list(arguments: dict[str, Any]) -> str | None:
        try:
            path = resolve_allowed_path(_require_path(arguments), roots)
        except Exception as exc:
            return str(exc)
        if not path.is_dir():
            return f"Not a directory: {path}"
        return None

    return [
        Tool(
            name="filesystem.list",
            description="List files and directories at a path.",
            parameters=_PATH_PROPERTY,
            permission=ToolPermission.READ,
            execute_fn=list_dir,
            validate_fn=validate_list,
            describe_fn=lambda a: f"List directory {a.get('path')!r}",
        ),
        Tool(
            name="filesystem.read",
            description="Read a text file. Returns the file contents.",
            parameters=_PATH_PROPERTY,
            permission=ToolPermission.READ,
            execute_fn=read_file,
            validate_fn=validate_read,
            describe_fn=lambda a: f"Read file {a.get('path')!r}",
        ),
        Tool(
            name="filesystem.write",
            description="Write text content to a file, creating parent directories.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "Text content to write"},
                },
                "required": ["path", "content"],
            },
            permission=ToolPermission.WRITE,
            execute_fn=write_file,
            validate_fn=validate_write,
            describe_fn=lambda a: (
                f"Write {len(a.get('content', ''))} characters to {a.get('path')!r}"
                if isinstance(a.get("content"), str)
                else f"Write file {a.get('path')!r}"
            ),
        ),
    ]
