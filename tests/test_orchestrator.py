"""Tests for JanusOrchestrator: routing, agent execution, event propagation, and memory.

FakeProvider drives both the routing classification and each agent's model
calls, so no Ollama server or model is required.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from aria.agents.computer_use_agent import ComputerUseAgent
from aria.agents.general_agent import GeneralAgent
from aria.agents.memory_agent import MemoryAgent
from aria.agents.research_agent import ResearchAgent
from aria.domain.models import ChatMessage, ModelResponse, PermissionLevel, ToolCall
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


class EchoInput(BaseModel):
    """Validated test input."""

    value: str


class EchoTool(Tool[EchoInput]):
    """A read-only test capability that never touches the operating system."""

    name = "test.echo"
    description = "Echo a value."
    permission_level = PermissionLevel.READ
    input_model = EchoInput

    async def execute(self, context: ToolContext, arguments: EchoInput) -> dict[str, str]:
        """Return the validated value."""

        return {"value": arguments.value}


class ScreenshotInput(BaseModel):
    """Screenshot takes no parameters."""


class FakeScreenshotTool(Tool[ScreenshotInput]):
    """Stands in for the real, pyautogui/Pillow-backed ScreenshotTool in tests."""

    name = "computer.screenshot"
    description = "Fake screenshot capture."
    permission_level = PermissionLevel.READ
    input_model = ScreenshotInput

    def __init__(self) -> None:
        self._last_capture: bytes | None = None

    @property
    def last_capture(self) -> bytes | None:
        """The most recently captured (fake) screenshot bytes."""

        return self._last_capture

    async def execute(self, context: ToolContext, arguments: ScreenshotInput) -> dict[str, object]:
        """Hand back fixed bytes, exactly like a successful real capture would."""

        self._last_capture = b"fake-jpeg-bytes"
        return {"width": 100, "height": 100, "format": "jpeg"}


class FailingProvider(FakeProvider):
    """A provider whose classification call always fails, to exercise the error path."""

    async def complete(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, object]] | None = None,
        temperature: float | None = None,
    ) -> ModelResponse:
        """Simulate an unreachable or misbehaving model."""

        raise RuntimeError("boom")


async def _build(
    tmp_path: Path, provider: FakeProvider, tools: ToolRegistry | None = None
) -> tuple[JanusOrchestrator, MemoryRepository]:
    tools = tools if tools is not None else ToolRegistry()
    repository = MemoryRepository(tmp_path / "chroma")
    await repository.initialize()
    memory = MemoryService(repository, provider)
    screenshot_tool = FakeScreenshotTool()
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


@pytest.mark.asyncio
async def test_orchestrator_invokes_selected_agent_and_streams_tokens(tmp_path: Path) -> None:
    """The orchestrator routes to the classified agent and forwards its streamed tokens live."""

    provider = FakeProvider(
        response_content='{"agent": "general", "reason": "chit-chat", "confidence": 0.9}',
        stream_tokens=["Hello", ", ", "world"],
    )
    orchestrator, _ = await _build(tmp_path, provider)

    events = [event async for event in orchestrator.run("u", "c1", "say hello", None)]

    event_types = [event["type"] for event in events]
    assert event_types[0] == "routing_decision"
    assert events[0]["data"]["agent"] == "general"  # type: ignore[index]
    assert "agent_started" in event_types
    assert event_types[-1] == "complete"
    token_text = "".join(str(event["content"]) for event in events if event["type"] == "token")
    assert token_text == "Hello, world"


@pytest.mark.asyncio
async def test_orchestrator_propagates_tool_events_from_research_agent(tmp_path: Path) -> None:
    """Tool call/result events from an agent are forwarded through the orchestrator unchanged."""

    tools = ToolRegistry()
    tools.register(EchoTool())
    # The router's classification call and the research agent's own planning calls all go
    # through the same FakeProvider.complete(), in order: routing decision first, then one
    # research round with a tool call, then an empty round that ends the tool loop.
    provider = FakeProvider(
        stream_tokens=["done"],
        complete_responses=[
            ModelResponse(content='{"agent": "research", "reason": "needs a tool", "confidence": 0.9}'),
            ModelResponse(tool_calls=[ToolCall(name="test.echo", arguments={"value": "hi"})]),
            ModelResponse(),
        ],
    )
    orchestrator, _ = await _build(tmp_path, provider, tools)

    events = [event async for event in orchestrator.run("u", "c1", "look this up", None)]

    event_types = [event["type"] for event in events]
    assert "tool_call" in event_types
    assert "tool_result" in event_types
    tool_result = next(event for event in events if event["type"] == "tool_result")
    assert tool_result["data"]["status"] == "succeeded"  # type: ignore[index]


@pytest.mark.asyncio
async def test_orchestrator_event_order_for_computer_use_turn(tmp_path: Path) -> None:
    """screenshot -> mouse_click -> screenshot -> final response, wrapped correctly end to end."""

    tools = ToolRegistry()
    click_tool = EchoTool()  # any READ/harmless tool stands in for a click here
    tools.register(click_tool)
    provider = FakeProvider(
        stream_tokens=["Clicked", " it."],
        complete_responses=[
            ModelResponse(content='{"agent": "computer_use", "reason": "desktop task", "confidence": 0.9}'),
            ModelResponse(tool_calls=[ToolCall(name="computer.screenshot", arguments={})]),
            ModelResponse(tool_calls=[ToolCall(name="test.echo", arguments={"value": "click"})]),
            ModelResponse(tool_calls=[ToolCall(name="computer.screenshot", arguments={})]),
            ModelResponse(),
        ],
    )
    orchestrator, _ = await _build(tmp_path, provider, tools)

    events = [event async for event in orchestrator.run("u", "c1", "click the button", "approved")]

    event_types = [event["type"] for event in events]
    assert event_types == [
        "routing_decision",
        "agent_started",
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "token",
        "token",
        "complete",
    ]
    assert events[0]["data"]["agent"] == "computer_use"  # type: ignore[index]


@pytest.mark.asyncio
async def test_orchestrator_persists_memory_after_completed_turn(tmp_path: Path) -> None:
    """A completed turn is persisted as one episodic memory, regardless of which agent ran."""

    provider = FakeProvider(
        response_content='{"agent": "general", "reason": "chit-chat", "confidence": 0.9}',
        stream_tokens=["Hi", " there"],
    )
    orchestrator, repository = await _build(tmp_path, provider)

    _ = [event async for event in orchestrator.run("u", "c1", "say hi", None)]

    stored = await repository.find("u", MemoryKind.EPISODIC)
    assert len(stored) == 1
    assert "say hi" in stored[0].content
    assert "Hi there" in stored[0].content


@pytest.mark.asyncio
async def test_orchestrator_emits_error_event_instead_of_raising(tmp_path: Path) -> None:
    """An internal failure becomes a public error event, not an unhandled exception."""

    orchestrator, _ = await _build(tmp_path, FailingProvider())

    events = [event async for event in orchestrator.run("u", "c1", "say hi", None)]

    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "RuntimeError"
