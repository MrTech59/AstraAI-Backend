"""PermissionManager: the security boundary between model requests and tool execution.

Authority flow (provider-independent):

    model tool call
        → PermissionManager.authorize()
            1. tool known?
            2. arguments valid? (tool-level validation, e.g. path confinement)
            3. policy action for this tool (allow / confirm / deny)
            4. if confirm → ApprovalProvider decides
        → ALLOWED → caller executes via ToolRegistry
        → DENIED  → structured result returned to the agent/model

The model can never grant itself permission: any approval-like keys in the
arguments (``approved``, ``confirm``, ...) are stripped before execution and
are never read as authorization signals.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..core.events import (
    EVENT_PERMISSION_DENIED,
    EVENT_PERMISSION_REQUESTED,
    EVENT_PERMISSION_APPROVED,
    Event,
    EventBus,
)
from ..tools.base import ToolPermission
from ..tools.registry import ToolRegistry
from .approval import ApprovalProvider, NonInteractiveApprovalProvider
from .policy import PermissionAction, PermissionPolicy

logger = logging.getLogger("astra.permissions")

#: Argument keys the model must never use as an authorization signal.
_APPROVAL_KEYS = {
    "approved", "approve", "approved_by_user", "user_approved",
    "confirm", "confirmed", "confirmation", "authorization", "authorized",
    "permission", "permission_granted", "granted", "allow", "allowed",
}

#: Substrings marking argument values that must never be shown in prompts/logs.
_SECRET_MARKERS = ("api_key", "apikey", "token", "password", "secret", "authorization")

_REDACTED = "***REDACTED***"


class PermissionOutcome(str, Enum):
    ALLOWED = "allowed"
    DENIED = "denied"


@dataclass
class PermissionRequest:
    """Everything an approval interface needs to show the user, sanitized."""

    tool_name: str
    category: ToolPermission
    description: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class PermissionDecision:
    outcome: PermissionOutcome
    #: Structured, model-consumable error payload (deny only).
    error: dict[str, Any] = field(default_factory=dict)
    #: Sanitized arguments the caller may pass to execution (never the raw
    #: model arguments — approval keys are stripped).
    arguments: dict[str, Any] = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return self.outcome == PermissionOutcome.ALLOWED

    def error_text(self) -> str:
        return json.dumps(self.error, ensure_ascii=False)


def sanitize_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Strip approval-forgery keys and redact secret-looking values."""
    clean: dict[str, Any] = {}
    for key, value in arguments.items():
        if str(key).strip().lower() in _APPROVAL_KEYS:
            logger.warning("Stripped approval-forgery argument key %r from tool call", key)
            continue
        key_text = str(key).lower()
        if any(marker in key_text for marker in _SECRET_MARKERS):
            clean[key] = _REDACTED
        else:
            clean[key] = value
    return clean


class PermissionManager:
    def __init__(
        self,
        policy: PermissionPolicy | None = None,
        approval: ApprovalProvider | None = None,
        registry: ToolRegistry | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self.policy = policy or PermissionPolicy.defaults()
        #: Fail closed: no approval provider means deny all confirmations.
        self.approval = approval or NonInteractiveApprovalProvider()
        self.registry = registry
        self.events = event_bus or EventBus()

    async def authorize(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
    ) -> PermissionDecision:
        """Decide whether a tool call may execute. Never executes anything."""
        raw = arguments if isinstance(arguments, dict) else {}
        sanitized = sanitize_arguments(raw)

        # 1. Tool must exist.
        tool = self.registry.get(tool_name) if self.registry is not None else None
        if tool is None:
            return self._deny(tool_name, "unknown_tool", "Unknown tool", sanitized)

        # 2. Arguments must pass tool-level validation (path confinement,
        #    shell allowlist, schema sanity). This runs BEFORE any prompt,
        #    so a confirmation can never launder an invalid call.
        validation_error = await tool.validate(sanitized)
        if validation_error is not None:
            return self._deny(tool_name, "invalid_arguments", validation_error, sanitized)

        # 3. Policy decides allow/confirm/deny.
        action = self.policy.action_for(tool_name, tool.permission)
        if action == PermissionAction.DENY:
            return self._deny(tool_name, "permission_denied", "Tool is denied by policy", sanitized)
        if action == PermissionAction.ALLOW:
            return PermissionDecision(
                outcome=PermissionOutcome.ALLOWED,
                arguments=sanitized,
            )

        # 4. CONFIRM: ask the approval provider. Fail closed on any error.
        request = PermissionRequest(
            tool_name=tool_name,
            category=tool.permission,
            description=tool.action_description(sanitized),
            arguments={k: v for k, v in sanitized.items() if isinstance(v, (str, int, float, bool))},
        )
        await self.events.emit(Event(EVENT_PERMISSION_REQUESTED, {"tool": tool_name}))
        logger.info("Permission requested: %s", tool_name)
        try:
            approved = await self.approval.approve(request)
        except Exception as exc:
            logger.warning("Approval provider failed for %s: %s", tool_name, exc)
            approved = False

        if approved:
            await self.events.emit(Event(EVENT_PERMISSION_APPROVED, {"tool": tool_name}))
            logger.info("Permission approved: %s", tool_name)
            return PermissionDecision(outcome=PermissionOutcome.ALLOWED, arguments=sanitized)

        await self.events.emit(Event(EVENT_PERMISSION_DENIED, {"tool": tool_name}))
        logger.info("Permission denied: %s", tool_name)
        return self._deny(
            tool_name,
            "permission_denied",
            "User did not approve this action" if self.approval.interactive
            else "Confirmation required but no interactive approval is available",
            sanitized,
        )

    def _deny(
        self,
        tool_name: str,
        error_code: str,
        detail: str,
        sanitized: dict[str, Any],
    ) -> PermissionDecision:
        logger.info("Tool blocked: %s (%s)", tool_name, error_code)
        return PermissionDecision(
            outcome=PermissionOutcome.DENIED,
            error={"ok": False, "error": error_code, "tool": tool_name, "detail": detail},
            arguments=sanitized,
        )
