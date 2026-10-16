"""Tool abstraction.

A tool is a named, schema-described capability the agent can invoke:

* ``name`` — unique identifier (e.g. ``filesystem.read``)
* ``description`` — shown to the model
* ``parameters`` — JSON Schema for arguments
* ``permission`` — coarse sensitivity class used by the security layer
* ``execute()`` — async implementation returning a :class:`ToolResult`
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolPermission(str, Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"


@dataclass
class ToolResult:
    content: str
    is_error: bool = False

    @classmethod
    def ok(cls, content: str) -> "ToolResult":
        return cls(content=content, is_error=False)

    @classmethod
    def error(cls, content: str) -> "ToolResult":
        return cls(content=content, is_error=True)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    permission: ToolPermission
    execute_fn: Any = field(repr=False)
    validate_fn: Any = field(default=None, repr=False)
    describe_fn: Any = field(default=None, repr=False)

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            return await self.execute_fn(arguments)
        except Exception as exc:  # tools must never crash the agent loop
            return ToolResult.error(f"{type(exc).__name__}: {exc}")

    async def validate(self, arguments: dict[str, Any]) -> str | None:
        """Pre-execution check. Returns an error message, or None if valid.

        Used by the permission layer so confirmation can never launder an
        invalid call (path traversal, shell allowlist violation, ...).
        """
        if self.validate_fn is None:
            return None
        try:
            return await self.validate_fn(arguments)
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"

    def action_description(self, arguments: dict[str, Any]) -> str:
        """Short human-readable description of what this call would do."""
        if self.describe_fn is not None:
            try:
                return str(self.describe_fn(arguments))
            except Exception:
                pass
        return self.description

    def definition(self) -> dict[str, Any]:
        """OpenAI function-calling tool definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
