"""Astra Core HTTP API.

Endpoints:
    GET  /health
    GET  /api/info
    POST /api/chat
    GET  /api/memory
    GET  /ws (WebSocket placeholder)
"""

from __future__ import annotations

import logging
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import astra
from astra.config import filesystem_roots, load_settings, memory_db_path
from astra.core import AgentLoop, ContextBuilder, Event, EventBus
from astra.core.events import (
    EVENT_AGENT_FINISHED,
    EVENT_AGENT_STARTED,
    EVENT_AGENT_THINKING,
    EVENT_ERROR,
    EVENT_MESSAGE_COMPLETED,
    EVENT_MESSAGE_DELTA,
    EVENT_TOOL_FINISHED,
    EVENT_TOOL_STARTED,
)
from astra.logging import setup_logging
from astra.memory import MemoryStore
from astra.mcp.client import connect_mcp_servers
from astra.providers import create_provider
from astra.security import ShellPolicy
from astra.tools import ToolRegistry, make_filesystem_tools, make_shell_tool
from astra.voice_gateway import (
    ConnectionManager,
    MessageType,
    VoiceGateway,
    WsMessage,
    chat_completed,
    chat_delta,
    error,
    signal_answer,
    signal_ice,
)

logger = logging.getLogger("astra.api")

# Global references wired up at startup.
_settings = None
_tools: ToolRegistry | None = None
_agent: AgentLoop | None = None
_memory: MemoryStore | None = None
_voice_gateway: VoiceGateway | None = None
_connection_manager: ConnectionManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _settings, _tools, _agent, _memory
    _settings = load_settings()
    setup_logging(_settings.logging.level)
    logger.info("Astra API %s starting", astra.__version__)

    provider = create_provider(_settings.llm)
    _tools = ToolRegistry()
    roots = filesystem_roots(_settings) or [Path.cwd()]
    for tool in make_filesystem_tools(roots=roots):
        _tools.register(tool)
    if _settings.security.allow_shell:
        policy = ShellPolicy(
            allow_shell=True,
            allowed_prefixes=_settings.security.allowed_shell_prefixes,
        )
        _tools.register(make_shell_tool(policy))

    _memory = MemoryStore(memory_db_path(_settings))
    context = ContextBuilder(
        memory=_memory,
        system_prompt=_settings.agent.system_prompt,
        max_messages=_settings.context.max_messages,
        max_memories=_settings.context.max_memories,
    )
    _agent = AgentLoop(
        provider=provider,
        tools=_tools,
        context=context,
        max_iterations=_settings.agent.max_tool_iterations,
        memory=_memory,
    )
    global _voice_gateway, _connection_manager
    _voice_gateway = VoiceGateway()
    _connection_manager = ConnectionManager()

    try:
        mcp_tools = await connect_mcp_servers(_settings.mcp_servers)
        for tool in mcp_tools:
            _tools.register(tool)
    except Exception as exc:
        logger.warning("MCP setup failed: %s", exc)

    logger.info("Provider: %s", _settings.llm.provider)
    logger.info("Model: %s", _settings.llm.model)
    logger.info("Loaded %s tools", len(_tools))
    yield
    if _memory is not None:
        _memory.close()


app = FastAPI(title="Astra Core API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- request/response models ---

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class ChatResponse(BaseModel):
    response: str
    conversation_id: str


class InfoResponse(BaseModel):
    version: str
    provider: str
    model: str


# --- endpoints ---

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/info")
async def info() -> InfoResponse:
    return InfoResponse(
        version=astra.__version__,
        provider=_settings.llm.provider,
        model=_settings.llm.model,
    )


@app.post("/api/chat")
async def chat(req: ChatRequest) -> ChatResponse:
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized.")
    try:
        reply = await _agent.run(req.message, conversation_id=req.conversation_id)
    except Exception as exc:
        logger.exception("Chat endpoint error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ChatResponse(response=reply, conversation_id=req.conversation_id)


@app.get("/api/memory")
async def memory(conversation_id: str = "default", limit: int = 30) -> dict[str, Any]:
    if _memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized.")
    recent = _memory.get_recent_messages(conversation_id, limit)
    return {
        "conversation_id": conversation_id,
        "messages": [
            {"role": m.role, "content": m.content, "timestamp": m.timestamp}
            for m in recent
        ],
    }


# --- WebSocket ---

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    if _agent is None or _connection_manager is None:
        await websocket.close(code=1011, reason="Agent not initialized")
        return

    await websocket.accept()
    _connection_manager.add(websocket)

    async def send_event(event: Event) -> None:
        try:
            await _connection_manager.send(websocket, WsMessage(MessageType.EVENT, event.to_dict()))
        except Exception:
            pass

    _agent.events.subscribe(send_event)

    async def handle_chat(message: WsMessage) -> None:
        text = message.data.get("message", "")
        conversation_id = message.data.get("conversation_id") or str(uuid.uuid4())
        if not text:
            await _connection_manager.send_error(websocket, "Empty message")
            return
        # Stream deltas if the provider supports them; otherwise just the final reply.
        accumulated = ""
        try:
            # TODO: wire streaming responses from the provider
            reply = await _agent.run(text, conversation_id=conversation_id)
            await _connection_manager.send(websocket, chat_completed(reply, conversation_id))
        except Exception as exc:
            logger.exception("WebSocket chat error")
            await _connection_manager.send_error(websocket, str(exc))

    async def handle_signal(message: WsMessage) -> None:
        if _voice_gateway is None:
            await _connection_manager.send_error(websocket, "Voice gateway unavailable")
            return
        call_id = message.data.get("call_id") or str(uuid.uuid4())
        try:
            if message.type == MessageType.SIGNAL_OFFER:
                answer_sdp = await _voice_gateway.handle_offer(
                    call_id, message.data.get("sdp", "")
                )
                await _connection_manager.send(
                    websocket, signal_answer(answer_sdp, call_id)
                )
            elif message.type == MessageType.SIGNAL_ICE:
                await _voice_gateway.handle_ice(
                    call_id,
                    message.data.get("candidate", ""),
                    sdp_mid=message.data.get("sdpMid"),
                    sdp_m_line_index=message.data.get("sdpMLineIndex"),
                )
            elif message.type == MessageType.SIGNAL_CALL_END:
                await _voice_gateway.end_call(call_id)
            else:
                await _connection_manager.send_error(
                    websocket, f"Unknown signal type: {message.type}"
                )
        except Exception as exc:
            logger.exception("WebSocket signal error")
            await _connection_manager.send_error(websocket, str(exc))

    try:
        while True:
            payload = await websocket.receive_json()
            message = WsMessage.from_dict(payload)

            if message.type == MessageType.PING:
                await _connection_manager.send(websocket, WsMessage(MessageType.PONG))
            elif message.is_chat_request():
                await handle_chat(message)
            elif message.is_signal():
                await handle_signal(message)
            else:
                await _connection_manager.send_error(
                    websocket, f"Unknown message type: {message.type}"
                )
    except WebSocketDisconnect:
        pass
    finally:
        _agent.events.unsubscribe(send_event)
        _connection_manager.remove(websocket)


# --- run entry point ---

if __name__ == "__main__":
    _settings = load_settings()
    import uvicorn

    uvicorn.run(
        "api.server:app",
        host=_settings.api.host,
        port=_settings.api.port,
        log_level=_settings.logging.level.lower(),
    )
