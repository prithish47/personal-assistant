"""Tests for AgentRouter's classification behavior.

The router asks the provider for a JSON classification; FakeProvider lets us
pin that response deterministically, so no Ollama server is required.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from aria.agents.computer_use_agent import ComputerUseAgent
from aria.agents.general_agent import GeneralAgent
from aria.agents.memory_agent import MemoryAgent
from aria.agents.research_agent import ResearchAgent
from aria.domain.models import PermissionLevel, ToolCall
from aria.memory.repository import MemoryRepository
from aria.memory.service import MemoryService
from aria.planning.planner import Planner
from aria.routing.models import RoutingDecision
from aria.routing.router import AgentRouter
from aria.tools.base import Tool, ToolContext
from aria.tools.executor import ToolExecutor
from aria.tools.permissions import PermissionManager
from aria.tools.registry import ToolRegistry
from tests.fakes import FakeProvider


class _EchoInput(BaseModel):
    """Validated test input."""

    value: str


class _EchoTool(Tool[_EchoInput]):
    """A read-only test capability so the router has a non-empty tool schema to offer."""

    name = "test.echo"
    description = "Echo a value."
    permission_level = PermissionLevel.READ
    input_model = _EchoInput

    async def execute(self, context: ToolContext, arguments: _EchoInput) -> dict[str, str]:
        """Return the validated value."""

        return {"value": arguments.value}


class _FakeScreenshotInput(BaseModel):
    """Screenshot takes no parameters."""


class _FakeScreenshotTool(Tool[_FakeScreenshotInput]):
    """Stands in for the real, pyautogui/Pillow-backed ScreenshotTool in tests."""

    name = "computer.screenshot"
    description = "Fake screenshot capture."
    permission_level = PermissionLevel.READ
    input_model = _FakeScreenshotInput

    @property
    def last_capture(self) -> bytes | None:
        """Never populated -- the router tests never execute this tool."""

        return None

    async def execute(self, context: ToolContext, arguments: _FakeScreenshotInput) -> dict[str, object]:
        """Unused by these tests; present only so ComputerUseAgent can be constructed."""

        return {}


async def _build_router(tmp_path: Path, provider: FakeProvider, tools: ToolRegistry | None = None) -> AgentRouter:
    repository = MemoryRepository(tmp_path / "chroma")
    await repository.initialize()
    memory = MemoryService(repository, provider)
    tools = tools if tools is not None else ToolRegistry()
    screenshot_tool = _FakeScreenshotTool()
    tools.register(screenshot_tool)
    executor = ToolExecutor(tools, PermissionManager())
    agents = [
        GeneralAgent(provider),
        ResearchAgent(Planner(provider, tools), executor, provider),
        MemoryAgent(memory, provider),
        ComputerUseAgent(executor, screenshot_tool, tools, provider),
    ]
    return AgentRouter(agents, provider, default_agent_name="general", tools=tools)


@pytest.mark.asyncio
async def test_router_selects_general_agent(tmp_path: Path) -> None:
    """A classification response naming 'general' routes to GeneralAgent."""

    provider = FakeProvider(response_content='{"agent": "general", "reason": "small talk", "confidence": 0.9}')
    router = await _build_router(tmp_path, provider)

    decision = await router.route("How are you today?")

    assert decision.agent_name == "general"
    assert router.resolve(decision).name == "general"


@pytest.mark.asyncio
async def test_router_selects_research_agent(tmp_path: Path) -> None:
    """A classification response naming 'research' routes to ResearchAgent."""

    provider = FakeProvider(
        response_content='{"agent": "research", "reason": "needs a tool lookup", "confidence": 0.85}'
    )
    router = await _build_router(tmp_path, provider)

    decision = await router.route("Open my notes application for me")

    assert decision.agent_name == "research"
    assert router.resolve(decision).name == "research"


@pytest.mark.asyncio
async def test_router_selects_memory_agent(tmp_path: Path) -> None:
    """A classification response naming 'memory' routes to MemoryAgent."""

    provider = FakeProvider(response_content='{"agent": "memory", "reason": "recall request", "confidence": 0.95}')
    router = await _build_router(tmp_path, provider)

    decision = await router.route("What did I tell you about my dog?")

    assert decision.agent_name == "memory"
    assert router.resolve(decision).name == "memory"


@pytest.mark.asyncio
async def test_router_selects_computer_use_agent(tmp_path: Path) -> None:
    """A classification response naming 'computer_use' routes to ComputerUseAgent."""

    provider = FakeProvider(
        response_content='{"agent": "computer_use", "reason": "desktop interaction", "confidence": 0.9}'
    )
    router = await _build_router(tmp_path, provider)

    decision = await router.route("Open Notepad and type hello")

    assert decision.agent_name == "computer_use"
    assert router.resolve(decision).name == "computer_use"


@pytest.mark.asyncio
async def test_router_returns_structured_decision(tmp_path: Path) -> None:
    """Routing always returns a RoutingDecision with the documented fields."""

    provider = FakeProvider(response_content='{"agent": "general", "reason": "chit-chat", "confidence": 0.7}')
    router = await _build_router(tmp_path, provider)

    decision = await router.route("Tell me a joke")

    assert isinstance(decision, RoutingDecision)
    assert decision.agent_name == "general"
    assert decision.reason == "chit-chat"
    assert decision.confidence == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_router_falls_back_to_default_on_unparseable_response(tmp_path: Path) -> None:
    """A malformed classification response never crashes routing; it falls back safely."""

    provider = FakeProvider(response_content="not json at all")
    router = await _build_router(tmp_path, provider)

    decision = await router.route("anything")

    assert decision.agent_name == "general"
    assert decision.confidence == 0.0
    assert decision.preliminary_tool_calls == []


@pytest.mark.asyncio
async def test_router_requests_low_temperature_for_classification(tmp_path: Path) -> None:
    """The classification call asks for near-zero temperature, for reliable JSON."""

    provider = FakeProvider(response_content='{"agent": "general", "reason": "chit-chat", "confidence": 0.9}')
    router = await _build_router(tmp_path, provider)

    await router.route("Tell me a joke")

    assert provider.complete_temperatures == [0.0]


@pytest.mark.asyncio
async def test_explicit_remember_request_performs_zero_reasoning_model_calls(tmp_path: Path) -> None:
    """An explicit 'remember X' goal is routed deterministically, with no LLM call at all."""

    provider = FakeProvider(response_content="this should never be read")
    router = await _build_router(tmp_path, provider)

    decision = await router.route("Remember that I like tea")

    assert provider.complete_calls == []
    assert decision.agent_name == "memory"
    assert decision.confidence == 1.0
    assert decision.preliminary_tool_calls == []


@pytest.mark.asyncio
async def test_router_returns_preliminary_tool_call_for_research(tmp_path: Path) -> None:
    """A single classification call can also produce research's first tool call."""

    tools = ToolRegistry()
    tools.register(_EchoTool())
    provider = FakeProvider(
        response_content='{"agent": "research", "reason": "needs a lookup", "confidence": 0.9}',
        tool_calls=[ToolCall(name="test.echo", arguments={"value": "hi"})],
    )
    router = await _build_router(tmp_path, provider, tools)

    decision = await router.route("look this up for me")

    assert decision.agent_name == "research"
    assert len(decision.preliminary_tool_calls) == 1
    assert decision.preliminary_tool_calls[0].name == "test.echo"
    assert decision.preliminary_tool_calls[0].arguments == {"value": "hi"}
    # The router offered the tool schema in that same call -- proving the fusion, not a guess.
    assert provider.complete_tools[0] is not None
    assert any(schema["name"] == "test.echo" for schema in provider.complete_tools[0])


@pytest.mark.asyncio
async def test_router_returns_preliminary_tool_call_for_computer_use(tmp_path: Path) -> None:
    """A single classification call can also produce computer_use's first action (e.g. open an app)."""

    tools = ToolRegistry()
    tools.register(_EchoTool())
    provider = FakeProvider(
        response_content='{"agent": "computer_use", "reason": "open an application", "confidence": 0.9}',
        tool_calls=[ToolCall(name="test.echo", arguments={"value": "notepad"})],
    )
    router = await _build_router(tmp_path, provider, tools)

    decision = await router.route("open notepad")

    assert decision.agent_name == "computer_use"
    assert len(decision.preliminary_tool_calls) == 1
    assert decision.preliminary_tool_calls[0].name == "test.echo"


@pytest.mark.asyncio
async def test_preliminary_tool_calls_discarded_for_non_research_agent(tmp_path: Path) -> None:
    """Tool calls returned alongside a non-research classification are never retained."""

    tools = ToolRegistry()
    tools.register(_EchoTool())
    provider = FakeProvider(
        response_content='{"agent": "general", "reason": "just chatting", "confidence": 0.9}',
        tool_calls=[ToolCall(name="test.echo", arguments={"value": "hi"})],
    )
    router = await _build_router(tmp_path, provider, tools)

    decision = await router.route("how are you?")

    assert decision.agent_name == "general"
    assert decision.preliminary_tool_calls == []


def test_router_requires_default_agent_to_be_registered() -> None:
    """An unregistered default agent name is rejected at construction, not discovered later."""

    with pytest.raises(ValueError):
        AgentRouter([], object(), default_agent_name="general", tools=ToolRegistry())  # type: ignore[arg-type]
