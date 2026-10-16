"""Approval providers: where a CONFIRM decision gets its yes/no answer.

The abstract interface is deliberately tiny so future providers (Slack
button, FastAPI pending-request endpoint, MCP client flow) can plug in
without touching the permission manager or the agent loop.

0.1.2 ships:
* :class:`CliApprovalProvider`    — interactive [y/N] prompt (default DENY)
* :class:`NonInteractiveApprovalProvider` — always denies; fail-closed
* :class:`AutoApproveProvider`    — test/smoke automation only
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .manager import PermissionRequest


class ApprovalProvider(ABC):
    #: True if a human can actually answer; False => fail-closed denial.
    interactive: bool = False

    @abstractmethod
    async def approve(self, request: "PermissionRequest") -> bool:
        """Return True to allow execution, False to deny."""


class NonInteractiveApprovalProvider(ApprovalProvider):
    """Fail-closed: used by API/background contexts where no one can answer."""

    interactive = False

    async def approve(self, request: "PermissionRequest") -> bool:
        return False


class AutoApproveProvider(ApprovalProvider):
    """Approves everything. For tests and scripted smoke runs only — never
    wire this into the CLI or API."""

    interactive = True

    def __init__(self) -> None:
        self.requests: list[PermissionRequest] = []

    async def approve(self, request: "PermissionRequest") -> bool:
        self.requests.append(request)
        return True


class CliApprovalProvider(ApprovalProvider):
    """Interactive stdin prompt. Default answer is DENY; only 'y'/'yes' allow.

    EOF, Ctrl-C, closed stdin, or any other input all deny — never hangs,
    never auto-approves.
    """

    interactive = True

    async def approve(self, request: "PermissionRequest") -> bool:
        print()
        print("Permission required")
        print(f"  Tool:   {request.tool_name}")
        print(f"  Risk:   {request.category.value.upper()}")
        print(f"  Action: {request.description}")
        if request.arguments:
            rendered = ", ".join(f"{k}={v!r}" for k, v in request.arguments.items())
            print(f"  Args:   {rendered}")
        try:
            answer = await asyncio.to_thread(input, "Allow this action? [y/N]: ")
        except (EOFError, OSError):
            print("(no interactive input — denied)")
            return False
        return answer.strip().lower() in {"y", "yes"}
