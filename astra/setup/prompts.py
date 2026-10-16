"""Input prompts for the interactive setup wizard.

All prompt helpers are synchronous and read from ``sys.stdin`` so they can be
monkeypatched easily in tests.  Secrets are read with ``getpass`` when a TTY is
available, falling back to plain ``input`` for CI/test redirection.
"""

from __future__ import annotations

import getpass
import sys
from typing import Callable


class PromptCancelled(Exception):
    """Raised when the user aborts a prompt (e.g. Ctrl-C)."""


def _input_fn(prompt: str) -> str:
    """Read a line, handling EOF/Ctrl-C gracefully."""
    try:
        return input(prompt)
    except EOFError:
        raise PromptCancelled("EOF received")
    except KeyboardInterrupt:
        raise PromptCancelled("Setup cancelled")


def ask(
    question: str,
    default: str = "",
    validator: Callable[[str], str] | None = None,
    required: bool = True,
) -> str:
    """Ask for a string value with an optional default.

    ``validator`` receives the raw input and must return a normalized value or
    raise ``ValueError`` with a message for the user.
    """
    suffix = f" [{default}]" if default else ""
    while True:
        answer = _input_fn(f"{question}{suffix}: ").strip()
        if not answer:
            if required and not default:
                print("  This field is required.")
                continue
            answer = default
        if validator is None:
            return answer
        try:
            return validator(answer)
        except ValueError as exc:
            print(f"  {exc}")


def ask_choice(question: str, options: list[tuple[str, str]], default_key: str = "") -> str:
    """Present a numbered list and return the selected option key."""
    print(question)
    for i, (key, label) in enumerate(options, start=1):
        marker = " (default)" if key == default_key else ""
        print(f"  {i}. {label}{marker}")
    default_num = next((i for i, (k, _) in enumerate(options, start=1) if k == default_key), 1)
    while True:
        raw = _input_fn(f"Select [{default_num}]: ").strip()
        if not raw:
            return default_key
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1][0]
        # Allow typing the key directly.
        keys = {k for k, _ in options}
        aliases = {k.lower(): k for k, _ in options}
        if raw in keys:
            return raw
        if raw.lower() in aliases:
            return aliases[raw.lower()]
        print(f"  Enter a number 1-{len(options)} or one of {', '.join(keys)}.")


def ask_bool(question: str, default: bool = False) -> bool:
    """Ask a yes/no question.  Only 'y'/'yes' approve."""
    suffix = " [Y/n]" if default else " [y/N]"
    while True:
        raw = _input_fn(f"{question}{suffix}: ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("  Please answer 'y' or 'n'.")


def ask_secret(question: str) -> str:
    """Ask for a secret without echoing it when possible."""
    try:
        if sys.stdin.isatty():
            value = getpass.getpass(f"{question}: ")
        else:
            value = _input_fn(f"{question}: ")
    except EOFError:
        raise PromptCancelled("EOF received")
    return value.strip()
