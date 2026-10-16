"""NVIDIA Build API provider tests — fully mocked, no network, no key."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from astra.config import LLMConfig, load_settings
from astra.providers import LLMProviderError, create_provider
from astra.providers.nvidia_build import DEFAULT_BASE_URL, NvidiaBuildProvider

API_KEY = "test-nvidia-key"  # fake key for unit tests only


def _make_provider(handler) -> NvidiaBuildProvider:
    transport = httpx.MockTransport(handler)
    return NvidiaBuildProvider(
        api_key=API_KEY,
        model="z-ai/glm-5.3-flash",
        client=httpx.AsyncClient(transport=transport),
    )


def _ok_response(content: str = "", tool_calls: list[dict[str, Any]] | None = None) -> httpx.Response:
    choice: dict[str, Any] = {
        "message": {"role": "assistant", "content": content, "tool_calls": tool_calls},
        "finish_reason": "stop",
    }
    return httpx.Response(200, json={"model": "z-ai/glm-5.3-flash", "choices": [choice]})


class TestSuccessfulResponses:
    @pytest.mark.asyncio
    async def test_normal_response(self) -> None:
        provider = _make_provider(lambda request: _ok_response("hello from nvidia"))
        response = await provider.chat([{"role": "user", "content": "hi"}])
        assert response.content == "hello from nvidia"
        assert response.model == "z-ai/glm-5.3-flash"
        assert not response.has_tool_calls

    @pytest.mark.asyncio
    async def test_tool_call_response(self) -> None:
        call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "filesystem.list", "arguments": json.dumps({"path": "."})},
        }
        provider = _make_provider(lambda request: _ok_response(tool_calls=[call]))
        response = await provider.chat([{"role": "user", "content": "list files"}])
        assert response.has_tool_calls
        assert response.tool_calls[0].name == "filesystem.list"
        assert response.tool_calls[0].arguments == {"path": "."}

    @pytest.mark.asyncio
    async def test_sends_bearer_auth_and_tools(self) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["auth"] = request.headers.get("authorization")
            captured["payload"] = json.loads(request.content)
            return _ok_response("ok")

        provider = _make_provider(handler)
        tools = [{"type": "function", "function": {"name": "filesystem.list", "parameters": {}}}]
        await provider.chat([{"role": "user", "content": "hi"}], tools=tools)
        assert captured["auth"] == f"Bearer {API_KEY}"
        sent = captured["payload"]["tools"]
        assert sent[0]["function"]["name"] == "filesystem_list"  # provider-safe encoding
        assert provider._decode_tool_name(sent[0]["function"]["name"]) == "filesystem.list"
        assert captured["payload"]["tool_choice"] == "auto"
        assert captured["payload"]["model"] == "z-ai/glm-5.3-flash"

    @pytest.mark.asyncio
    async def test_decodes_tool_names_in_response(self) -> None:
        call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "filesystem_list", "arguments": json.dumps({"path": "."})},
        }
        provider = _make_provider(lambda request: _ok_response(tool_calls=[call]))
        provider._safe_tool_names["filesystem_list"] = "filesystem.list"
        response = await provider.chat(
            [{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "filesystem.list", "parameters": {}}}],
        )
        assert response.tool_calls[0].name == "filesystem.list"

    @pytest.mark.asyncio
    async def test_key_never_appears_in_error_messages(self) -> None:
        provider = _make_provider(lambda request: httpx.Response(401, text="unauthorized"))
        with pytest.raises(LLMProviderError) as exc_info:
            await provider.chat([{"role": "user", "content": "hi"}])
        assert API_KEY not in str(exc_info.value)


class TestErrors:
    @pytest.mark.asyncio
    async def test_missing_key(self) -> None:
        with pytest.raises(LLMProviderError, match="NVIDIA_API_KEY"):
            NvidiaBuildProvider(api_key="")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [401, 403])
    async def test_auth_failure(self, status: int) -> None:
        provider = _make_provider(lambda request: httpx.Response(status, text="denied"))
        with pytest.raises(LLMProviderError, match="authentication failed"):
            await provider.chat([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_400(self) -> None:
        provider = _make_provider(lambda request: httpx.Response(400, text="invalid model"))
        with pytest.raises(LLMProviderError, match="HTTP 400"):
            await provider.chat([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_429(self) -> None:
        provider = _make_provider(lambda request: httpx.Response(429, text="slow down"))
        with pytest.raises(LLMProviderError, match="rate limit"):
            await provider.chat([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [500, 503])
    async def test_server_error(self, status: int) -> None:
        provider = _make_provider(lambda request: httpx.Response(status, text="upstream boom"))
        with pytest.raises(LLMProviderError, match="server error"):
            await provider.chat([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_timeout(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("timed out", request=request)

        provider = _make_provider(handler)
        with pytest.raises(LLMProviderError, match="Could not reach"):
            await provider.chat([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_connection_failure(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        provider = _make_provider(handler)
        with pytest.raises(LLMProviderError, match="Could not reach"):
            await provider.chat([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_invalid_json(self) -> None:
        provider = _make_provider(lambda request: httpx.Response(200, text="<html>not json</html>"))
        with pytest.raises(LLMProviderError, match="invalid JSON"):
            await provider.chat([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_missing_choices(self) -> None:
        provider = _make_provider(lambda request: httpx.Response(200, json={"model": "x"}))
        with pytest.raises(LLMProviderError, match="missing choices"):
            await provider.chat([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_empty_choices(self) -> None:
        provider = _make_provider(lambda request: httpx.Response(200, json={"choices": []}))
        with pytest.raises(LLMProviderError, match="missing choices"):
            await provider.chat([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_malformed_tool_call_tolerated(self) -> None:
        call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "filesystem.read", "arguments": "{not-json"},
        }
        provider = _make_provider(lambda request: _ok_response(tool_calls=[call]))
        response = await provider.chat([{"role": "user", "content": "read"}])
        assert response.has_tool_calls
        assert response.tool_calls[0].arguments == {}

    @pytest.mark.asyncio
    async def test_null_content_tolerated(self) -> None:
        choice = {"message": {"role": "assistant", "content": None}, "finish_reason": "stop"}
        provider = _make_provider(
            lambda request: httpx.Response(200, json={"model": "m", "choices": [choice]})
        )
        response = await provider.chat([{"role": "user", "content": "hi"}])
        assert response.content == ""
        assert not response.has_tool_calls


class TestFactory:
    def test_factory_selects_nvidia(self) -> None:
        config = LLMConfig(provider="nvidia_build", model="z-ai/glm-5.3-flash", api_key=API_KEY)
        provider = create_provider(config)
        assert isinstance(provider, NvidiaBuildProvider)
        assert provider.name == "nvidia_build"
        assert provider.base_url == DEFAULT_BASE_URL

    def test_factory_accepts_nvidia_alias(self) -> None:
        config = LLMConfig(provider="nvidia", api_key=API_KEY)
        provider = create_provider(config)
        assert isinstance(provider, NvidiaBuildProvider)

    def test_factory_nvidia_missing_key(self) -> None:
        config = LLMConfig(provider="nvidia_build", api_key="")
        with pytest.raises(LLMProviderError, match="NVIDIA_API_KEY"):
            create_provider(config)

    def test_config_key_resolution(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("NVIDIA_API_KEY", API_KEY)
        config_file = tmp_path / "config.toml"
        config_file.write_text('[llm]\nprovider = "nvidia_build"\n', encoding="utf-8")
        settings = load_settings(config_path=config_file, env_path=tmp_path / "missing.env")
        assert settings.llm.api_key == API_KEY
