"""Validation helpers for the Astra installer."""

from __future__ import annotations

import platform
import sys
from typing import Any

import httpx

from ..providers.base import LLMProviderError
from ..providers import create_provider


MIN_PYTHON_VERSION = (3, 11)


def check_python_version() -> tuple[bool, str]:
    """Return (ok, message) for the running Python interpreter."""
    version = sys.version_info[:2]
    if version >= MIN_PYTHON_VERSION:
        return True, f"Python {platform.python_version()} OK"
    return False, f"Python {platform.python_version()} is too old; Astra needs {'.'.join(map(str, MIN_PYTHON_VERSION))}+"


def mask_key(key: str, visible: int = 4) -> str:
    """Return a display-safe representation of an API key.

    >>> mask_key("nvapi-abc123xyz")
    'nvapi-***********'
    """
    if not key:
        return "(not set)"
    if len(key) <= visible:
        return "*" * len(key)
    prefix = key[:visible]
    return f"{prefix}{'*' * (len(key) - visible)}"


def _clean_detail(detail: str) -> str:
    """Strip newlines and truncate provider error details for display."""
    detail = detail.replace("\n", " ").strip()
    if len(detail) > 160:
        detail = detail[:157] + "..."
    return detail


async def check_provider_connection(
    provider: str,
    api_key: str,
    model: str,
    base_url: str = "",
    timeout: float = 30.0,
    client: httpx.AsyncClient | None = None,
) -> tuple[bool, str]:
    """Ping the chosen provider with a minimal chat completion.

    Returns (ok, message).  The key is never included in the message.
    """
    from ..config import LLMConfig

    config = LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
    )
    close_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=timeout)
    try:
        llm = create_provider(config, client=client)
        response = await llm.chat(
            messages=[{"role": "user", "content": "Reply with exactly: Astra setup OK"}],
            max_tokens=32,
            temperature=0.0,
        )
        content = (response.content or "").strip()
        if content:
            return True, f"Connected — model replied with {len(content)} characters"
        return False, "Connected but model returned empty content"
    except LLMProviderError as exc:
        detail = str(exc).replace(api_key, "***")
        return False, f"Provider error: {_clean_detail(detail)}"
    except httpx.TimeoutException:
        return False, f"Connection timed out after {timeout}s"
    except Exception as exc:
        return False, f"Connection failed: {type(exc).__name__}: {_clean_detail(str(exc))}"
    finally:
        if close_client:
            await client.aclose()
