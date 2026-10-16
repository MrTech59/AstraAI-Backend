"""Tests for the Astra voice gateway (WebSocket protocol + signaling)."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from astra.voice_gateway import (
    ConnectionManager,
    MessageType,
    VoiceGateway,
    VoiceGatewayError,
    WsMessage,
    chat_completed,
    chat_request,
    signal_answer,
    signal_ice,
    signal_offer,
)
from astra.voice_gateway.protocol import error


class TestProtocol:
    def test_parse_chat_request(self) -> None:
        msg = WsMessage.from_dict(
            {"type": "chat.request", "data": {"message": "hello", "conversation_id": "c1"}}
        )
        assert msg.is_chat_request()
        assert msg.data["message"] == "hello"

    def test_signal_types(self) -> None:
        offer = signal_offer("sdp", "call1")
        assert offer.type == MessageType.SIGNAL_OFFER
        assert offer.is_signal()

        answer = signal_answer("sdp", "call1")
        assert answer.type == MessageType.SIGNAL_ANSWER

        ice = signal_ice("candidate", "mid", 0, "call1")
        assert ice.type == MessageType.SIGNAL_ICE

    def test_error_message(self) -> None:
        msg = error("boom")
        assert msg.type == MessageType.ERROR
        assert msg.data["error"] == "boom"


class TestConnectionManager:
    @pytest.mark.asyncio
    async def test_send(self) -> None:
        manager = ConnectionManager()
        ws = AsyncMock()
        await manager.send(ws, chat_completed("hi", "c1"))
        ws.send_json.assert_awaited_once_with(
            {"type": MessageType.CHAT_COMPLETED, "data": {"content": "hi", "conversation_id": "c1"}}
        )

    @pytest.mark.asyncio
    async def test_send_error(self) -> None:
        manager = ConnectionManager()
        ws = AsyncMock()
        await manager.send_error(ws, "bad")
        ws.send_json.assert_awaited_once()
        payload = ws.send_json.await_args[0][0]
        assert payload["type"] == MessageType.ERROR


class TestVoiceGateway:
    def test_availability_without_aiortc(self) -> None:
        gateway = VoiceGateway()
        # In the test environment aiortc is not installed.
        assert gateway.is_available() is False

    @pytest.mark.asyncio
    async def test_offer_fails_when_unavailable(self) -> None:
        gateway = VoiceGateway()
        with pytest.raises(VoiceGatewayError, match="Voice gateway is not available"):
            await gateway.handle_offer("call1", "v=0\r\n")


class TestApiWebSocket:
    def test_chat_over_websocket(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "chat.request", "data": {"message": "Say OK"}})
            response = ws.receive_json()
            assert response["type"] in {MessageType.CHAT_COMPLETED, MessageType.EVENT, MessageType.ERROR}

    def test_ping_pong(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "ping"})
            response = ws.receive_json()
            assert response["type"] == MessageType.PONG

    def test_signal_without_aiortc_returns_error(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            ws.send_json(
                {
                    "type": "signal.offer",
                    "data": {"sdp": "v=0\r\n", "call_id": "test-call"},
                }
            )
            response = ws.receive_json()
            assert response["type"] == MessageType.ERROR
