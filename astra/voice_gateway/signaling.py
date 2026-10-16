"""WebSocket connection and signaling management for voice calls."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import WebSocket

from .protocol import MessageType, WsMessage, error

logger = logging.getLogger("astra.voice_gateway.signaling")


class ConnectionManager:
    """Track active WebSocket clients and route signaling messages."""

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()

    def add(self, websocket: WebSocket) -> None:
        self._connections.add(websocket)

    def remove(self, websocket: WebSocket) -> None:
        self._connections.discard(websocket)

    async def send(self, websocket: WebSocket, message: WsMessage) -> None:
        try:
            await websocket.send_json(message.to_dict())
        except Exception as exc:
            logger.debug("Failed to send websocket message: %s", exc)

    async def send_error(self, websocket: WebSocket, text: str) -> None:
        await self.send(websocket, error(text))

    async def ping(self, websocket: WebSocket) -> None:
        await self.send(websocket, WsMessage(MessageType.PONG))


class SignalingError(Exception):
    """Raised when signaling state is invalid."""
