"""WebSocket message protocol for the Astra mobile app.

The same WebSocket connection carries both chat messages and WebRTC
signaling.  All frames are JSON objects with a ``type`` and a ``data``
payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class MessageType:
    """Protocol message types."""

    PING = "ping"
    PONG = "pong"

    # Chat
    CHAT_REQUEST = "chat.request"
    CHAT_DELTA = "chat.delta"
    CHAT_COMPLETED = "chat.completed"

    # WebRTC signaling
    SIGNAL_OFFER = "signal.offer"
    SIGNAL_ANSWER = "signal.answer"
    SIGNAL_ICE = "signal.ice"
    SIGNAL_CALL_START = "signal.call.start"
    SIGNAL_CALL_END = "signal.call.end"

    # Generic events / errors
    EVENT = "event"
    ERROR = "error"


@dataclass
class WsMessage:
    """A parsed WebSocket message."""

    type: str
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WsMessage":
        return cls(
            type=str(payload.get("type", "")).strip(),
            data=payload.get("data") or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "data": self.data}

    def is_chat_request(self) -> bool:
        return self.type == MessageType.CHAT_REQUEST

    def is_signal(self) -> bool:
        return self.type.startswith("signal.")


def chat_request(message: str, conversation_id: str = "") -> WsMessage:
    return WsMessage(
        MessageType.CHAT_REQUEST,
        {"message": message, "conversation_id": conversation_id},
    )


def chat_delta(content: str, conversation_id: str = "") -> WsMessage:
    return WsMessage(
        MessageType.CHAT_DELTA,
        {"content": content, "conversation_id": conversation_id},
    )


def chat_completed(content: str, conversation_id: str = "") -> WsMessage:
    return WsMessage(
        MessageType.CHAT_COMPLETED,
        {"content": content, "conversation_id": conversation_id},
    )


def signal_offer(sdp: str, call_id: str) -> WsMessage:
    return WsMessage(MessageType.SIGNAL_OFFER, {"sdp": sdp, "call_id": call_id})


def signal_answer(sdp: str, call_id: str) -> WsMessage:
    return WsMessage(MessageType.SIGNAL_ANSWER, {"sdp": sdp, "call_id": call_id})


def signal_ice(
    candidate: str,
    sdp_mid: str,
    sdp_m_line_index: int,
    call_id: str,
) -> WsMessage:
    return WsMessage(
        MessageType.SIGNAL_ICE,
        {
            "candidate": candidate,
            "sdpMid": sdp_mid,
            "sdpMLineIndex": sdp_m_line_index,
            "call_id": call_id,
        },
    )


def error(message: str) -> WsMessage:
    return WsMessage(MessageType.ERROR, {"error": message})
