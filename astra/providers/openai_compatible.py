"""OpenAI-compatible chat provider.

Works with any API implementing ``POST {base_url}/chat/completions``
(OpenRouter, Ollama, and other OpenAI-compatible endpoints).
"""

from __future__ import annotations

from typing import Any

import httpx

from .base import ChatResponse, LLMProvider, LLMProviderError, Message, ToolCall


class OpenAICompatibleProvider(LLMProvider):
    name = "openai-compatible"

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        model: str = "",
        timeout: float = 60.0,
        default_headers: dict[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._default_headers = dict(default_headers or {})
        # Tests can inject a client (e.g. with httpx.MockTransport).
        self._client = client or httpx.AsyncClient(timeout=timeout)
        #: Some backends (e.g. NVIDIA Build) only accept [a-zA-Z0-9_-] in
        #: tool names. We send a safe encoding and map it back on receive,
        #: so the rest of Astra keeps its dotted names (filesystem.write).
        self._safe_tool_names: dict[str, str] = {}  # safe -> original

    def _encode_tool_name(self, name: str) -> str:
        safe = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in name)
        if safe != name:
            candidate = safe
            while candidate in self._safe_tool_names and self._safe_tool_names[candidate] != name:
                candidate += "_"
            self._safe_tool_names[candidate] = name
            return candidate
        self._safe_tool_names.setdefault(safe, name)
        return safe

    def _decode_tool_name(self, safe: str) -> str:
        return self._safe_tool_names.get(safe, safe)

    def _encode_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        encoded = []
        for tool in tools:
            tool = dict(tool)
            function = dict(tool.get("function") or {})
            if function.get("name"):
                function["name"] = self._encode_tool_name(str(function["name"]))
            tool["function"] = function
            encoded.append(tool)
        return encoded

    def _encode_messages(self, messages: list[Message]) -> list[Message]:
        """Encode tool names embedded in conversation history (assistant
        tool_calls and tool-role name fields) for this request."""
        encoded: list[Message] = []
        for message in messages:
            message = dict(message)
            if message.get("role") == "tool" and message.get("name"):
                message["name"] = self._encode_tool_name(str(message["name"]))
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                if function.get("name"):
                    function["name"] = self._encode_tool_name(str(function["name"]))
            encoded.append(message)
        return encoded

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        headers.update(self._default_headers)
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _url(self) -> str:
        return f"{self.base_url}/chat/completions"

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> ChatResponse:
        response = await self._request(messages, tools, temperature, max_tokens)
        if response.status_code != 200:
            detail = response.text[:300]
            raise LLMProviderError(
                f"LLM API error (HTTP {response.status_code}): {detail}"
            )
        return self._parse_response(response)

    async def _request(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> httpx.Response:
        """Send the chat completion request and return the raw response.

        Subclasses may override to add provider-specific status handling
        while reusing :meth:`_parse_response`. Only auth/not-found statuses
        are raised here; other non-200 statuses are the caller's concern.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._encode_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = self._encode_tools(tools)
            payload["tool_choice"] = "auto"

        try:
            response = await self._client.post(
                self._url(), headers=self._headers(), json=payload
            )
        except httpx.TransportError as exc:
            raise LLMProviderError(
                f"Could not reach LLM API at {self.base_url}: {exc}"
            ) from exc

        if response.status_code in (401, 403):
            raise LLMProviderError(
                f"LLM API authentication failed (HTTP {response.status_code}). "
                "Check your API key environment variable."
            )
        if response.status_code == 404:
            detail = response.text[:300]
            raise LLMProviderError(
                f"LLM API endpoint or model not found (HTTP 404): {detail}. "
                "Check the base_url and model in config."
            )

        return response

    def _parse_response(self, response: httpx.Response) -> ChatResponse:
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMProviderError("LLM API returned invalid JSON.") from exc

        try:
            choice = data["choices"][0]
            message = choice.get("message") or {}
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError("LLM API response is missing choices.") from exc

        tool_calls = []
        for raw in message.get("tool_calls") or []:
            function = raw.get("function") or {}
            arguments: dict[str, Any] = {}
            raw_args = function.get("arguments")
            if isinstance(raw_args, str) and raw_args.strip():
                import json

                try:
                    parsed = json.loads(raw_args)
                    arguments = parsed if isinstance(parsed, dict) else {}
                except ValueError:
                    arguments = {}
            elif isinstance(raw_args, dict):
                arguments = raw_args
            tool_calls.append(
                ToolCall(
                    id=str(raw.get("id", "")),
                    name=self._decode_tool_name(str(function.get("name", ""))),
                    arguments=arguments,
                )
            )

        return ChatResponse(
            content=message.get("content") or "",
            tool_calls=tool_calls,
            model=data.get("model", self.model) or self.model,
            finish_reason=choice.get("finish_reason") or "stop",
        )
