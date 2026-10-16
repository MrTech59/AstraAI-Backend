"""Permission policy: which tools need confirmation, which are auto-allowed.

Policy is a per-tool-name mapping configured through ``[permissions]`` in
TOML, layered over safe built-in defaults. Missing/invalid configuration
never widens access — it falls back to the safe default for that tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..tools.base import ToolPermission


class PermissionAction(str, Enum):
    ALLOW = "allow"          # execute without asking
    CONFIRM = "confirm"      # ask an approval provider first
    DENY = "deny"            # never execute


#: Safe defaults when configuration is missing or partial.
DEFAULT_ACTIONS: dict[str, PermissionAction] = {
    "filesystem.list": PermissionAction.ALLOW,
    "filesystem.read": PermissionAction.ALLOW,
    "filesystem.write": PermissionAction.CONFIRM,
    "shell.exec": PermissionAction.CONFIRM,
}

#: Fallback per risk class when a tool has no explicit policy entry.
_PERMISSION_FALLBACK: dict[ToolPermission, PermissionAction] = {
    ToolPermission.READ: PermissionAction.ALLOW,
    ToolPermission.WRITE: PermissionAction.CONFIRM,
    ToolPermission.EXECUTE: PermissionAction.CONFIRM,
    ToolPermission.NETWORK: PermissionAction.CONFIRM,
}


def _parse_action(value: object, fallback: PermissionAction) -> PermissionAction:
    try:
        return PermissionAction(str(value).strip().lower())
    except ValueError:
        return fallback


@dataclass
class PermissionPolicy:
    actions: dict[str, PermissionAction] = field(default_factory=dict)
    default: PermissionAction = PermissionAction.CONFIRM

    def action_for(self, tool_name: str, permission: ToolPermission) -> PermissionAction:
        if tool_name in self.actions:
            return self.actions[tool_name]
        return _PERMISSION_FALLBACK.get(permission, self.default)

    @classmethod
    def defaults(cls) -> "PermissionPolicy":
        return cls(actions=dict(DEFAULT_ACTIONS))

    @classmethod
    def from_config(cls, config: object | None) -> "PermissionPolicy":
        """Build a policy from a PermissionsConfig (or None).

        Only known tool keys are read; anything else keeps its safe default.
        """
        policy = cls.defaults()
        if config is None:
            return policy
        policy.default = _parse_action(getattr(config, "default", "confirm"), PermissionAction.CONFIRM)
        for attr, tool_name in (
            ("filesystem_list", "filesystem.list"),
            ("filesystem_read", "filesystem.read"),
            ("filesystem_write", "filesystem.write"),
            ("shell_exec", "shell.exec"),
        ):
            value = getattr(config, attr, None)
            if value is not None:
                policy.actions[tool_name] = _parse_action(value, DEFAULT_ACTIONS[tool_name])
        return policy
