"""Shell execution tool, gated by a configurable command policy.

Disabled by default. When enabled, only commands starting with one of the
configured prefixes are executed. Model output is never executed directly —
it always passes through :class:`ShellPolicy`.
"""

from __future__ import annotations

import asyncio

from ..security import ShellPolicy
from .base import Tool, ToolPermission, ToolResult

DEFAULT_TIMEOUT_SEC = 30


def make_shell_tool(policy: ShellPolicy, timeout: int = DEFAULT_TIMEOUT_SEC) -> Tool:
    async def exec_command(arguments: dict[str, Any]) -> ToolResult:
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return ToolResult.error("Argument 'command' must be a non-empty string.")
        policy.check(command)  # raises SecurityError -> caught by Tool.execute

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except OSError as exc:
            return ToolResult.error(f"Failed to start command: {exc}")

        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return ToolResult.error(f"Command timed out after {timeout}s.")

        output = stdout.decode("utf-8", errors="replace")[:4000]
        text = output.strip() or "(no output)"
        if process.returncode != 0:
            return ToolResult.ok(f"exit code {process.returncode}\n{text}")
        return ToolResult.ok(text)

    async def validate_exec(arguments: dict[str, Any]) -> str | None:
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return "Argument 'command' must be a non-empty string."
        try:
            policy.check(command)  # raises SecurityError if not permitted
        except Exception as exc:
            return str(exc)
        return None

    return Tool(
        name="shell.exec",
        description=(
            "Execute an allowed shell command and return its output. "
            "Only commands matching the configured allowlist work."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
            },
            "required": ["command"],
        },
        permission=ToolPermission.EXECUTE,
        execute_fn=exec_command,
        validate_fn=validate_exec,
        describe_fn=lambda a: f"Run shell command: {a.get('command')!r}",
    )
