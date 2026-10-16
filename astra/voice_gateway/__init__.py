"""Astra voice gateway: WebRTC signaling and real-time voice calls."""

from .gateway import VoiceGateway, VoiceGatewayError
from .protocol import (
    MessageType,
    WsMessage,
    chat_completed,
    chat_delta,
    chat_request,
    error,
    signal_answer,
    signal_ice,
    signal_offer,
)
from .signaling import ConnectionManager, SignalingError

__all__ = [
    "ConnectionManager",
    "MessageType",
    "SignalingError",
    "VoiceGateway",
    "VoiceGatewayError",
    "WsMessage",
    "chat_completed",
    "chat_delta",
    "chat_request",
    "error",
    "signal_answer",
    "signal_ice",
    "signal_offer",
]
