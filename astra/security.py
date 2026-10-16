"""Security primitives for Astra Core tools.

Two policies live here:

* path policy — every filesystem access is resolved and confined to a set of
  allowed roots (blocks traversal like ``../../etc``).
* shell policy — shell execution is gated behind an explicit enable flag plus
  an allowlist of command prefixes. Disabled by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


class SecurityError(Exception):
    """Raised when a tool request violates a security policy."""


def resolve_allowed_path(raw_path: str, roots: list[Path]) -> Path:
    """Resolve ``raw_path`` (absolute or relative to the first root) and
    verify it stays inside one of the allowed roots.

    Raises SecurityError on traversal or missing roots.
    """
    if not roots:
        raise SecurityError("No filesystem roots are configured.")

    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = roots[0] / candidate

    resolved = candidate.resolve()
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
            return resolved
        except ValueError:
            continue
    raise SecurityError(f"Path is outside allowed roots: {raw_path!r}")


@dataclass
class ShellPolicy:
    allow_shell: bool = False
    allowed_prefixes: list[str] = field(default_factory=list)

    def check(self, command: str) -> None:
        """Raise SecurityError if ``command`` is not permitted."""
        if not self.allow_shell:
            raise SecurityError(
                "Shell execution is disabled. "
                "Set security.allow_shell = true in config to enable it."
            )
        if not self.allowed_prefixes:
            raise SecurityError(
                "Shell execution is enabled but no allowed_shell_prefixes are configured."
            )
        stripped = command.strip()
        for prefix in self.allowed_prefixes:
            if stripped.startswith(prefix):
                return
        raise SecurityError(
            f"Command is not in the allowed prefix list: {stripped[:64]!r}"
        )
