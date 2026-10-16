"""Provider tests."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from astra.config import LLMConfig
from astra.providers import LLMProviderError, create_provider
from astra.providers.openai_compatible import OpenAICompatibleProvider


def _make_response(content: str = "", tool_calls: list[dict[str, Any]] | None = None) -> httpx.Response:
    choice: dict[str, Any] = {
        "message": {"role": "assistant", "content": content, "tool_calls": tool_calls},
        "finish_reason": "stop",
    }
    return httpx.Response(
        200,
        json={"model": "test-model", "choices": [choice]},
    )


def _make_tool_call(name: str, arguments: dict[str, Any], call_id: str = "call_1") -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class TestOpenAICompatibleProvider:
    @pytest.mark.asyncio
    async def test_chat_returns_content(self) -> None:
        transport = httpx.MockTransport(lambda request: _make_response("hello"))
        provider = OpenAICompatibleProvider(base_url="http://test", model="m", client=httpx.AsyncClient(transport=transport))
        response = await provider.chat([{"role": "user", "content": "hi"}])
        assert response.content == "hello"
        assert not response.has_tool_calls

    @pytest.mark.asyncio
    async def test_chat_tool_call(self) -> None:
        call = _make_tool_call("filesystem.read", {"path": "x.txt"})
        transport = httpx.MockTransport(lambda request: _make_response(tool_calls=[call]))
        provider = OpenAICompatibleProvider(base_url="http://test", model="m", client=httpx.AsyncClient(transport=transport))
        response = await provider.chat([{"role": "user", "content": "hi"}])
        assert response.has_tool_calls
        assert response.tool_calls[0].name == "filesystem.read"
        assert response.tool_calls[0].arguments == {"path": "x.txt"}

    @pytest.mark.asyncio
    async def test_chat_401(self) -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(401, text="unauthorized"))
        provider = OpenAICompatibleProvider(base_url="http://test", model="m", client=httpx.AsyncClient(transport=transport))
        with pytest.raises(LLMProviderError, match="authentication failed"):
            await provider.chat([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_chat_500(self) -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(500, text="boom"))
        provider = OpenAICompatibleProvider(base_url="http://test", model="m", client=httpx.AsyncClient(transport=transport))
        with pytest.raises(LLMProviderError, match="boom"):
            await provider.chat([{"role": "user", "content": "hi"}])


class TestCreateProvider:
    def test_openrouter_requires_key(self) -> None:
        config = LLMConfig(provider="openrouter", model="x")
        with pytest.raises(LLMProviderError, match="API key is missing"):
            create_provider(config)

    def test_ollama_no_key_ok(self) -> None:
        config = LLMConfig(provider="ollama", model="qwen3")
        provider = create_provider(config)
        assert provider.name == "ollama"

    def test_unknown_provider(self) -> None:
        config = LLMConfig(provider="bogus")
        with pytest.raises(LLMProviderError, match="Unknown LLM provider"):
            create_provider(config)
