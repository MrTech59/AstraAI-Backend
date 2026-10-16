"""Voice gateway: WebRTC peer connections and audio pipeline.

This module is intentionally optional.  If ``aiortc`` is not installed the
signaling endpoints still work, but the server will reject offers with a
clear "voice gateway not available" error.  That lets the text-chat path
run on a lightweight Pi while voice calls require a server with ``aiortc``
installed.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("astra.voice_gateway.gateway")

# Optional aiortc import.  Failures are deferred until a call is attempted.
try:
    from aiortc import RTCPeerConnection, RTCSessionDescription

    _AIORTC_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    _AIORTC_AVAILABLE = False
    RTCPeerConnection = None  # type: ignore[assignment,misc]
    RTCSessionDescription = None  # type: ignore[assignment,misc]


class VoiceGatewayError(Exception):
    """Raised for voice-gateway failures."""


class VoiceGateway:
    """Manages WebRTC peer connections for real-time voice calls.

    Currently this is a scaffold: it validates offers/answers and tracks
    calls, but the audio processing pipeline (STT → LLM → TTS) is not yet
    wired.  Future work will add a MediaStreamTrack that feeds incoming audio
    to a speech-to-text service and plays synthesized replies back.
    """

    def __init__(self) -> None:
        self._calls: dict[str, Any] = {}

    def _ensure_available(self) -> None:
        if not _AIORTC_AVAILABLE:
            raise VoiceGatewayError(
                "Voice gateway is not available. Install aiortc to enable WebRTC calls."
            )

    async def handle_offer(self, call_id: str, sdp: str, type_: str = "offer") -> str:
        """Process an SDP offer and return an SDP answer."""
        self._ensure_available()
        if call_id in self._calls:
            raise VoiceGatewayError(f"Call {call_id} already exists")

        pc = RTCPeerConnection()
        self._calls[call_id] = {"pc": pc}

        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=sdp, type=type_)
        )
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        logger.info("Voice call %s accepted (scaffold mode)", call_id)
        return pc.localDescription.sdp

    async def handle_ice(
        self,
        call_id: str,
        candidate: str,
        sdp_mid: str | None = None,
        sdp_m_line_index: int | None = None,
    ) -> None:
        """Add a remote ICE candidate to a call."""
        self._ensure_available()
        call = self._calls.get(call_id)
        if call is None:
            raise VoiceGatewayError(f"Unknown call {call_id}")

        pc = call["pc"]
        await pc.addIceCandidate(
            {
                "candidate": candidate,
                "sdpMid": sdp_mid,
                "sdpMLineIndex": sdp_m_line_index,
            }
        )

    async def end_call(self, call_id: str) -> None:
        """Close a voice call and clean up."""
        call = self._calls.pop(call_id, None)
        if call is not None and _AIORTC_AVAILABLE:
            pc = call["pc"]
            await pc.close()
        logger.info("Voice call %s ended", call_id)

    def is_available(self) -> bool:
        return _AIORTC_AVAILABLE
