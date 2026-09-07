"""End-to-end tests for the voice extension to the `/v1/ws` protocol.

These go through the real `create_app()` composition root (like
test_api.py) so the actual WebSocket handler code in aria/app.py is what
runs -- not a reimplementation of it. `app.state.services.stt`/`.tts`/
`.orchestrator` are swapped for deterministic fakes right after the app's
real lifespan has constructed them, exactly the way a test overrides one
field of a plain dataclass; no Ollama, no real microphone, no real STT/TTS
engine, and no ChromaDB persistence beyond a per-test tmp_path directory.
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from pydantic import BaseModel, SecretStr

from aria.agents.computer_use_agent import ComputerUseAgent
from aria.agents.general_agent import GeneralAgent
from aria.agents.memory_agent import MemoryAgent
from aria.agents.research_agent import ResearchAgent
from aria.app import create_app
from aria.core.config import Settings
from aria.domain.models import ModelResponse, PermissionLevel, ToolCall
from aria.memory.models import MemoryKind
from aria.memory.repository import MemoryRepository
from aria.memory.service import MemoryService
from aria.orchestration.orchestrator import JanusOrchestrator
from aria.planning.planner import Planner
from aria.routing.router import AgentRouter
from aria.tools.base import Tool, ToolContext
from aria.tools.executor import ToolExecutor
from aria.tools.permissions import PermissionManager
from aria.tools.registry import ToolRegistry
from tests.fakes import FakeProvider


class _EchoInput(BaseModel):
    value: str


class _EchoTool(Tool[_EchoInput]):
    """A harmless READ tool that stands in for a real computer-use action."""

    name = "test.echo"
    description = "Echo a value."
    permission_level = PermissionLevel.READ
    input_model = _EchoInput

    async def execute(self, context: ToolContext, arguments: _EchoInput) -> dict[str, str]:
        return {"value": arguments.value}


class _ScreenshotInput(BaseModel):
    pass


class _FakeScreenshotTool(Tool[_ScreenshotInput]):
    name = "computer.screenshot"
    description = "Fake screenshot capture."
    permission_level = PermissionLevel.READ
    input_model = _ScreenshotInput

    def __init__(self) -> None:
        self._last_capture: bytes | None = None

    @property
    def last_capture(self) -> bytes | None:
        return self._last_capture

    async def execute(self, context: ToolContext, arguments: _ScreenshotInput) -> dict[str, object]:
        self._last_capture = b"fake-jpeg-bytes"
        return {"width": 100, "height": 100, "format": "jpeg"}


class _FakeSTT:
    """A deterministic SpeechToText double: returns fixed text, or raises on command."""

    def __init__(self, text: str = "", error: Exception | None = None) -> None:
        self._text = text
        self._error = error
        self.calls: list[bytes] = []

    async def transcribe(self, audio: bytes) -> str:
        self.calls.append(audio)
        if self._error is not None:
            raise self._error
        return self._text


class _FakeTTS:
    """A deterministic TextToSpeech double: returns fixed bytes, or raises on command."""

    def __init__(self, audio: bytes = b"RIFF-fake-wav", error: Exception | None = None) -> None:
        self._audio = audio
        self._error = error
        self.calls: list[str] = []

    async def synthesize(self, text: str) -> bytes:
        self.calls.append(text)
        if self._error is not None:
            raise self._error
        return self._audio


async def _build_orchestrator(
    tmp_path: Path, provider: FakeProvider, tools: ToolRegistry | None = None
) -> tuple[JanusOrchestrator, MemoryRepository]:
    tools = tools if tools is not None else ToolRegistry()
    repository = MemoryRepository(tmp_path / "chroma")
    await repository.initialize()
    memory = MemoryService(repository, provider)
    screenshot_tool = _FakeScreenshotTool()
    tools.register(screenshot_tool)
    executor = ToolExecutor(tools, PermissionManager())
    agents = [
        GeneralAgent(provider),
        ResearchAgent(Planner(provider, tools), executor, provider),
        MemoryAgent(memory, provider),
        ComputerUseAgent(executor, screenshot_tool, tools, provider),
    ]
    router = AgentRouter(agents, provider, default_agent_name="general", tools=tools)
    return JanusOrchestrator(router, memory), repository


def _drain_until(ws: Any, is_terminal: Callable[[dict[str, Any]], bool]) -> list[dict[str, Any]]:
    """Collect events from the socket until one satisfies `is_terminal`.

    Typed loosely (`Any` values) deliberately: these are raw JSON wire
    events, and asserting on their nested shape is the whole point of these
    tests -- `dict[str, object]` (as other test files use) would need a
    `# type: ignore[index]` on nearly every assertion here.
    """

    events: list[dict[str, Any]] = []
    while True:
        event = ws.receive_json()
        events.append(event)
        if is_terminal(event):
            return events


def _run(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run a setup coroutine from a sync test function (no event loop is running here)."""

    return asyncio.run(coro)


_API_KEY = "a" * 40


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    return Settings(api_key=SecretStr(_API_KEY), memory_persist_directory=tmp_path / "chroma", **overrides)


def test_legacy_text_chat_message_is_unaffected_by_voice_changes(tmp_path: Path) -> None:
    """A typed message with no `type` field behaves exactly as it did before voice existed."""

    settings = _settings(tmp_path)
    app = create_app(settings)
    with TestClient(app) as client:
        provider = FakeProvider(
            response_content='{"agent": "general", "reason": "chit-chat", "confidence": 0.9}',
            stream_tokens=["Hello", " there"],
        )
        app.state.services.orchestrator, _ = _run(_build_orchestrator(tmp_path, provider))

        with client.websocket_connect(f"/v1/ws?token={_API_KEY}") as ws:
            ws.send_json({"content": "say hello", "conversation_id": "c1", "approval_token": None})
            events = _drain_until(ws, lambda e: e["type"] == "complete")

    event_types = [event["type"] for event in events]
    assert event_types == ["routing_decision", "agent_started", "token", "token", "complete"]
    assert not any(t.startswith("voice") or t in ("transcription", "audio_chunk") for t in event_types)


def test_voice_input_transcribes_runs_full_turn_and_speaks_the_reply(tmp_path: Path) -> None:
    """audio -> transcription -> the unchanged orchestrator pipeline -> spoken reply, in order.

    The transcript is an explicit "remember" request, so this also proves
    STEP 7's example end to end: routing takes the same deterministic,
    zero-model-call path a typed "Remember that I prefer C++" already takes
    (see test_router.py's `test_explicit_remember_request_...`), and the
    fact lands in the same ChromaDB-backed memory typed input would use.
    """

    settings = _settings(tmp_path)
    app = create_app(settings)
    with TestClient(app) as client:
        provider = FakeProvider()
        orchestrator, repository = _run(_build_orchestrator(tmp_path, provider))
        app.state.services.orchestrator = orchestrator
        fake_stt = _FakeSTT(text="Remember that I prefer C++")
        fake_tts = _FakeTTS(audio=b"RIFF-reply-audio")
        app.state.services.stt = fake_stt
        app.state.services.tts = fake_tts

        with client.websocket_connect(f"/v1/ws?token={_API_KEY}") as ws:
            ws.send_json(
                {
                    "type": "voice_input",
                    "conversation_id": "c1",
                    "approval_token": None,
                    "audio_base64": base64.b64encode(b"fake-recorded-webm-bytes").decode("ascii"),
                    "mime_type": "audio/webm",
                }
            )
            events = _drain_until(ws, lambda e: e["type"] == "voice_status" and e["data"]["state"] == "idle")

    event_types = [event["type"] for event in events]
    assert event_types == [
        "voice_status",
        "transcription",
        "routing_decision",
        "agent_started",
        "token",
        "complete",
        "voice_status",
        "audio_chunk",
        "voice_status",
    ]
    assert events[0]["data"]["state"] == "transcribing"
    assert events[1]["data"]["text"] == "Remember that I prefer C++"
    assert events[2]["data"]["agent"] == "memory"
    assert events[2]["data"]["confidence"] == 1.0
    assert events[6]["data"]["state"] == "speaking"
    assert events[7]["data"]["audio_base64"] == base64.b64encode(b"RIFF-reply-audio").decode("ascii")
    assert events[7]["data"]["final"] is True
    assert fake_stt.calls == [b"fake-recorded-webm-bytes"]
    assert provider.complete_calls == []  # the deterministic remember-bypass never calls the model

    memories = _run(repository.find("local-user", MemoryKind.EPISODIC))
    assert len(memories) == 1
    assert "Remember that I prefer C++" in memories[0].content


def test_voice_command_can_route_to_computer_use_agent(tmp_path: Path) -> None:
    """A voice-transcribed desktop command reaches ComputerUseAgent through normal routing."""

    settings = _settings(tmp_path)
    app = create_app(settings)
    with TestClient(app) as client:
        tools = ToolRegistry()
        tools.register(_EchoTool())
        provider = FakeProvider(
            stream_tokens=["Opened", " it."],
            complete_responses=[
                ModelResponse(content='{"agent": "computer_use", "reason": "desktop task", "confidence": 0.9}'),
                ModelResponse(tool_calls=[ToolCall(name="test.echo", arguments={"value": "notepad"})]),
                ModelResponse(),
            ],
        )
        orchestrator, _ = _run(_build_orchestrator(tmp_path, provider, tools))
        app.state.services.orchestrator = orchestrator
        app.state.services.stt = _FakeSTT(text="Open Notepad")
        app.state.services.tts = _FakeTTS()

        with client.websocket_connect(f"/v1/ws?token={_API_KEY}") as ws:
            ws.send_json(
                {
                    "type": "voice_input",
                    "conversation_id": "c1",
                    "approval_token": "approved",
                    "audio_base64": base64.b64encode(b"open-notepad-audio").decode("ascii"),
                }
            )
            events = _drain_until(ws, lambda e: e["type"] == "voice_status" and e["data"]["state"] == "idle")

    routing = next(e for e in events if e["type"] == "routing_decision")
    assert routing["data"]["agent"] == "computer_use"
    assert any(e["type"] == "tool_call" for e in events)
    assert any(e["type"] == "tool_result" for e in events)


def test_invalid_audio_payload_is_rejected_without_calling_stt(tmp_path: Path) -> None:
    """A missing audio_base64 field fails cleanly and never reaches the STT engine."""

    settings = _settings(tmp_path)
    app = create_app(settings)
    with TestClient(app) as client:
        fake_stt = _FakeSTT(text="should never run")
        app.state.services.stt = fake_stt

        with client.websocket_connect(f"/v1/ws?token={_API_KEY}") as ws:
            ws.send_json({"type": "voice_input", "conversation_id": "c1", "approval_token": None})
            events = _drain_until(ws, lambda e: e["type"] == "voice_error")

    assert events == [{"type": "voice_error", "data": {"code": "invalid_audio", "message": "audio_base64 is required"}}]
    assert fake_stt.calls == []


def test_oversized_audio_is_rejected_without_calling_stt(tmp_path: Path) -> None:
    """Audio larger than the configured cap is rejected before it ever reaches the STT engine."""

    settings = _settings(tmp_path, voice_max_audio_bytes=1_000)
    app = create_app(settings)
    with TestClient(app) as client:
        fake_stt = _FakeSTT(text="should never run")
        app.state.services.stt = fake_stt

        with client.websocket_connect(f"/v1/ws?token={_API_KEY}") as ws:
            ws.send_json(
                {
                    "type": "voice_input",
                    "conversation_id": "c1",
                    "approval_token": None,
                    "audio_base64": base64.b64encode(b"x" * 2_000).decode("ascii"),
                }
            )
            events = _drain_until(ws, lambda e: e["type"] == "voice_error")

    assert [e["type"] for e in events] == ["voice_error"]
    assert events[0]["data"]["code"] == "audio_too_large"
    assert fake_stt.calls == []


def test_stt_failure_emits_voice_error_and_never_reaches_the_orchestrator(tmp_path: Path) -> None:
    """An STT engine failure is reported cleanly and the turn never reaches routing."""

    settings = _settings(tmp_path)
    app = create_app(settings)
    with TestClient(app) as client:
        app.state.services.stt = _FakeSTT(error=RuntimeError("model crashed"))

        with client.websocket_connect(f"/v1/ws?token={_API_KEY}") as ws:
            ws.send_json(
                {
                    "type": "voice_input",
                    "conversation_id": "c1",
                    "approval_token": None,
                    "audio_base64": base64.b64encode(b"some-audio-bytes").decode("ascii"),
                }
            )
            events = _drain_until(ws, lambda e: e["type"] == "voice_error")

    assert [e["type"] for e in events] == ["voice_status", "voice_error"]
    assert events[1]["data"]["code"] == "stt_failed"
    assert "model crashed" not in str(events)  # the raw exception text is not leaked to the client


def test_empty_transcription_emits_voice_error(tmp_path: Path) -> None:
    """Silence/unrecognizable speech (empty transcript) is reported cleanly, not routed onward."""

    settings = _settings(tmp_path)
    app = create_app(settings)
    with TestClient(app) as client:
        app.state.services.stt = _FakeSTT(text="   ")

        with client.websocket_connect(f"/v1/ws?token={_API_KEY}") as ws:
            ws.send_json(
                {
                    "type": "voice_input",
                    "conversation_id": "c1",
                    "approval_token": None,
                    "audio_base64": base64.b64encode(b"silence").decode("ascii"),
                }
            )
            events = _drain_until(ws, lambda e: e["type"] == "voice_error")

    assert [e["type"] for e in events] == ["voice_status", "voice_error"]
    assert events[1]["data"]["code"] == "empty_transcription"


def test_tts_failure_does_not_fail_the_already_completed_chat_turn(tmp_path: Path) -> None:
    """A TTS crash never undoes or blocks the text response that was already delivered."""

    settings = _settings(tmp_path)
    app = create_app(settings)
    with TestClient(app) as client:
        provider = FakeProvider(
            response_content='{"agent": "general", "reason": "chit-chat", "confidence": 0.9}',
            stream_tokens=["All", " good."],
        )
        orchestrator, _ = _run(_build_orchestrator(tmp_path, provider))
        app.state.services.orchestrator = orchestrator
        app.state.services.stt = _FakeSTT(text="How are you")
        app.state.services.tts = _FakeTTS(error=RuntimeError("engine exploded"))

        with client.websocket_connect(f"/v1/ws?token={_API_KEY}") as ws:
            ws.send_json(
                {
                    "type": "voice_input",
                    "conversation_id": "c1",
                    "approval_token": None,
                    "audio_base64": base64.b64encode(b"how-are-you-audio").decode("ascii"),
                }
            )
            events = _drain_until(ws, lambda e: e["type"] == "voice_status" and e["data"]["state"] == "idle")

    assert any(e["type"] == "complete" for e in events)
    token_text = "".join(str(e["content"]) for e in events if e["type"] == "token")
    assert token_text == "All good."
    voice_errors = [e for e in events if e["type"] == "voice_error"]
    assert len(voice_errors) == 1
    assert voice_errors[0]["data"]["code"] == "tts_failed"
    assert not any(e["type"] == "audio_chunk" for e in events)
