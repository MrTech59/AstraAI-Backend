"""Optional real NVIDIA Build API smoke test.

NOT part of the pytest suite (filename has no ``test_`` prefix and the
module is ignored by pytest via collect_ignore in conftest).

Run manually, only when you want to verify against the live API:

    python tests/smoke_nvidia.py

Requires NVIDIA_API_KEY in the environment (or .env). The key is never
printed. Makes exactly one chat completion request.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astra.config import load_settings, resolve_api_key
from astra.providers import LLMProviderError, create_provider
from astra.providers.nvidia_build import DEFAULT_MODEL


async def main() -> int:
    settings = load_settings()
    settings.llm.provider = "nvidia_build"
    settings.llm.model = DEFAULT_MODEL
    # load_settings resolved the key for the *configured* provider; this
    # script forces nvidia_build, so re-resolve through the same mechanism.
    settings.llm.api_key = resolve_api_key(settings.llm.provider)

    if not settings.llm.api_key:
        print("SKIP: NVIDIA_API_KEY is not set in the environment or .env.")
        return 2

    provider = create_provider(settings.llm)
    print(f"Requesting {settings.llm.model} from NVIDIA Build API ...")
    try:
        response = await provider.chat(
            [{"role": "user", "content": "Reply with exactly: NVIDIA OK"}],
            max_tokens=256,  # reasoning models spend budget on reasoning_content first
        )
    except LLMProviderError as exc:
        print(f"FAIL: {exc}")
        return 1

    print(f"Response: {response.content!r}")
    if response.content.strip():
        print("SMOKE OK")
        return 0
    print("FAIL: empty response content")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
