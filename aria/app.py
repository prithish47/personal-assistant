"""Application composition root and secure FastAPI transport adapters."""

from __future__ import annotations

import base64
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Header, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from aria.agents.computer_use_agent import ComputerUseAgent
from aria.agents.general_agent import GeneralAgent
from aria.agents.memory_agent import MemoryAgent
from aria.agents.research_agent import ResearchAgent
from aria.api.rate_limit import RateLimiter
from aria.api.schemas import HealthResponse
from aria.api.security import authorize_websocket, require_api_key
from aria.core.audit import AuditLogger
from aria.core.config import Settings
from aria.core.errors import AriaError, AuthorizationError
from aria.core.logging import configure_logging
from aria.memory.repository import MemoryRepository
from aria.memory.service import MemoryService
from aria.orchestration.orchestrator import JanusOrchestrator
from aria.planning.planner import Planner
from aria.plugins.computer_use import ComputerUsePlugin
from aria.plugins.desktop import DesktopPlugin
from aria.plugins.manager import PluginManager
from aria.providers.base import LLMProvider
from aria.providers.factory import create_provider
from aria.routing.router import AgentRouter
from aria.tools.executor import ToolExecutor
from aria.tools.permissions import PermissionManager
from aria.tools.registry import ToolRegistry
from aria.voice.stt import SpeechToText, WhisperSTT
from aria.voice.tts import Pyttsx3TTS, TextToSpeech, split_into_speech_chunks

logger = logging.getLogger(__name__)


@dataclass
class Services:
    """Explicit runtime dependencies available at the transport boundary."""

    provider: LLMProvider
    tools: ToolRegistry
    executor: ToolExecutor
    plugins: PluginManager
    orchestrator: JanusOrchestrator
    rate_limiter: RateLimiter
    stt: SpeechToText
    tts: TextToSpeech


def create_app(settings: Settings) -> FastAPI:
    """Build an application with no import-time I/O or mutable global services."""

    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            tools = ToolRegistry()
            plugins = PluginManager(tools)
            plugins.load(DesktopPlugin(settings))
            computer_use_plugin = ComputerUsePlugin(settings)
            plugins.load(computer_use_plugin)
            plugins.discover()
            provider = create_provider(settings, client)
            memory_repository = MemoryRepository(settings.memory_persist_directory)
            await memory_repository.initialize()
            planner = Planner(provider, tools)
            executor = ToolExecutor(tools, PermissionManager(), AuditLogger(settings.data_dir / "audit.jsonl"))
            memory = MemoryService(memory_repository, provider)
            agents = [
                GeneralAgent(provider),
                ResearchAgent(planner, executor, provider, max_iterations=settings.research_agent_max_iterations),
                MemoryAgent(memory, provider),
                ComputerUseAgent(
                    executor,
                    computer_use_plugin.screenshot_tool,
                    tools,
                    provider,
                    max_steps=settings.computer_use_max_steps,
                ),
            ]
            router = AgentRouter(agents, provider, default_agent_name=GeneralAgent.name, tools=tools)
            app.state.services = Services(
                provider=provider,
                tools=tools,
                executor=executor,
                plugins=plugins,
                orchestrator=JanusOrchestrator(router, memory),
                rate_limiter=RateLimiter(settings.rate_limit_per_minute),
                # Construction here is cheap for both -- WhisperSTT defers its actual model
                # load to the first transcription request (see aria/voice/stt.py), and
                # Pyttsx3TTS only stores config -- so a machine with neither optional voice
                # dependency installed still starts up and serves text chat normally; a real
                # failure only surfaces as a `voice_error` event on an actual voice turn.
                stt=WhisperSTT(settings.voice_stt_model, settings.voice_stt_device, settings.voice_stt_compute_type),
                tts=Pyttsx3TTS(settings.voice_tts_voice_id, settings.voice_tts_rate),
            )
            yield

    app = FastAPI(title="ARIA", version="0.1.0", lifespan=lifespan)
    allowed_origins = [str(origin).rstrip("/") for origin in settings.allowed_origins]
    if allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type"],
        )

    @app.exception_handler(AriaError)
    async def aria_error_handler(_: Request, error: AriaError) -> Response:
        status = 401 if isinstance(error, AuthorizationError) else 400
        return JSONResponse(
            status_code=status, content={"error": {"code": error.__class__.__name__, "message": str(error)}}
        )

    def protected(authorization: str | None = Header(default=None)) -> None:
        """FastAPI dependency that closes over immutable application settings."""

        require_api_key(settings, authorization)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        ready = await app.state.services.provider.healthcheck()
        return HealthResponse(status="ok" if ready else "degraded", provider_ready=ready)

    @app.get("/v1/tools", dependencies=[Depends(protected)])
    async def list_tools() -> list[dict[str, object]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "permission_level": tool.permission_level.value,
                "schema": tool.schema(),
            }
            for tool in app.state.services.tools.list()
        ]

    @app.websocket("/v1/ws")
    async def agent_websocket(websocket: WebSocket) -> None:
        async def run_turn(conversation_id: str, goal: str, approval_token: Any) -> str:
            """Run one turn through the unchanged orchestrator, forwarding every event as-is.

            Used by both the typed-chat path and the voice path below, so a
            voice-transcribed goal is handled by JanusOrchestrator identically
            to typed input -- same routing, memory, and tool permission rules.
            """

            response = ""
            async for event in app.state.services.orchestrator.run("local-user", conversation_id, goal, approval_token):
                if event.get("type") == "token":
                    response += str(event.get("content", ""))
                await websocket.send_json(event)
            return response

        async def speak(text: str) -> None:
            """Synthesize and stream the response as audio chunks.

            Runs only after `run_turn` has already sent the complete text
            response, so any failure here becomes a `voice_error` event, never
            a failure of the underlying chat turn.
            """

            await websocket.send_json({"type": "voice_status", "data": {"state": "speaking"}})
            try:
                chunks = split_into_speech_chunks(text, settings.voice_tts_max_chunk_chars)
                started = time.monotonic()
                for index, chunk in enumerate(chunks):
                    audio = await app.state.services.tts.synthesize(chunk)
                    await websocket.send_json(
                        {
                            "type": "audio_chunk",
                            "data": {
                                "audio_base64": base64.b64encode(audio).decode("ascii"),
                                "mime_type": "audio/wav",
                                "index": index,
                                "final": index == len(chunks) - 1,
                            },
                        }
                    )
                logger.info(
                    "voice tts_ms=%d chunks=%d response_chars=%d",
                    round((time.monotonic() - started) * 1000),
                    len(chunks),
                    len(text),
                )
            except Exception as error:
                logger.warning("voice tts_failed error=%s", error.__class__.__name__)
                await websocket.send_json(
                    {"type": "voice_error", "data": {"code": "tts_failed", "message": "Speech synthesis failed"}}
                )
            finally:
                await websocket.send_json({"type": "voice_status", "data": {"state": "idle"}})

        async def handle_voice_input(payload: dict[str, Any]) -> None:
            """Decode and transcribe recorded audio, then hand the text to `run_turn`.

            From this point on a voice turn and a typed turn are
            indistinguishable to the orchestrator: only the transcript text
            (never audio bytes) crosses into `run_turn`.
            """

            conversation_id = str(payload.get("conversation_id", "default"))
            approval_token = payload.get("approval_token")
            audio_field = payload.get("audio_base64")
            if not isinstance(audio_field, str) or not audio_field:
                await websocket.send_json(
                    {"type": "voice_error", "data": {"code": "invalid_audio", "message": "audio_base64 is required"}}
                )
                return
            try:
                audio_bytes = base64.b64decode(audio_field, validate=True)
            except ValueError:
                await websocket.send_json(
                    {
                        "type": "voice_error",
                        "data": {"code": "invalid_audio", "message": "audio_base64 is not valid base64"},
                    }
                )
                return
            if len(audio_bytes) > settings.voice_max_audio_bytes:
                await websocket.send_json(
                    {
                        "type": "voice_error",
                        "data": {"code": "audio_too_large", "message": "Recording exceeds the size limit"},
                    }
                )
                return

            await websocket.send_json({"type": "voice_status", "data": {"state": "transcribing"}})
            started = time.monotonic()
            try:
                transcript = await app.state.services.stt.transcribe(audio_bytes)
            except Exception as error:
                logger.warning("voice stt_failed error=%s", error.__class__.__name__)
                await websocket.send_json(
                    {"type": "voice_error", "data": {"code": "stt_failed", "message": "Speech recognition failed"}}
                )
                return
            goal = transcript.strip()
            logger.info(
                "voice stt_ms=%d audio_bytes=%d transcript_chars=%d",
                round((time.monotonic() - started) * 1000),
                len(audio_bytes),
                len(goal),
            )
            if not goal:
                await websocket.send_json(
                    {"type": "voice_error", "data": {"code": "empty_transcription", "message": "No speech detected"}}
                )
                return
            await websocket.send_json({"type": "transcription", "data": {"text": goal}})

            response = await run_turn(conversation_id, goal, approval_token)

            if response.strip():
                await speak(response)

        try:
            await authorize_websocket(websocket, settings)
            await websocket.accept()
            while True:
                identity = websocket.client.host if websocket.client else "unknown"
                await app.state.services.rate_limiter.check(identity)
                payload = await websocket.receive_json()
                if payload.get("type") == "voice_input":
                    await handle_voice_input(payload)
                    continue
                goal = str(payload.get("content", "")).strip()
                if not goal:
                    await websocket.send_json(
                        {"type": "error", "code": "invalid_request", "message": "content is required"}
                    )
                    continue
                await run_turn(str(payload.get("conversation_id", "default")), goal, payload.get("approval_token"))
        except AuthorizationError:
            await websocket.close(code=1008)
        except WebSocketDisconnect:
            return

    return app
