"""Tests for ComputerUseAgent's bounded perceive-decide-act loop.

No Ollama, no real screen, no real pyautogui call is required: fake tools
stand in for ScreenshotTool/mouse/keyboard exactly the way EchoTool stands
in for a real tool elsewhere in this test suite, and FakeProvider's
`complete_responses` queue drives the decision sequence.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from aria.agents.base import AgentContext, AgentEventType
from aria.agents.computer_use_agent import ComputerUseAgent
from aria.domain.models import ModelResponse, PermissionLevel, ToolCall
from aria.tools.base import Tool, ToolContext
from aria.tools.executor import ToolExecutor
from aria.tools.permissions import PermissionManager
from aria.tools.registry import ToolRegistry
from tests.fakes import FakeProvider


class _ScreenshotInput(BaseModel):
    """Screenshot takes no parameters."""


class _FakeScreenshotTool(Tool[_ScreenshotInput]):
    """Stands in for the real, pyautogui/Pillow-backed ScreenshotTool in tests."""

    name = "computer.screenshot"
    description = "Fake screenshot capture."
    permission_level = PermissionLevel.READ
    input_model = _ScreenshotInput

    def __init__(self, capture_bytes: bytes = b"fake-jpeg-bytes") -> None:
        self._capture_bytes = capture_bytes
        self._last_capture: bytes | None = None
        self.capture_count = 0

    @property
    def last_capture(self) -> bytes | None:
        """The most recently captured (fake) screenshot bytes."""

        return self._last_capture

    async def execute(self, context: ToolContext, arguments: _ScreenshotInput) -> dict[str, object]:
        """Record that a capture happened and hand back fixed bytes."""

        self.capture_count += 1
        self._last_capture = self._capture_bytes
        return {"width": 100, "height": 100, "format": "jpeg"}


class _ClickInput(BaseModel):
    """A screen position to click."""

    x: int
    y: int


class _FakeClickTool(Tool[_ClickInput]):
    """A USER_CONFIRM-gated fake click, so permission enforcement can be tested for real."""

    name = "computer.mouse_click"
    description = "Fake click."
    permission_level = PermissionLevel.USER_CONFIRM
    input_model = _ClickInput

    def __init__(self) -> None:
        self.clicks: list[tuple[int, int]] = []

    async def execute(self, context: ToolContext, arguments: _ClickInput) -> dict[str, object]:
        """Record the click -- this must never run without approval."""

        self.clicks.append((arguments.x, arguments.y))
        return {"x": arguments.x, "y": arguments.y}


def _context(goal: str = "do something on screen", approval_token: str | None = "approved") -> AgentContext:
    return AgentContext(user_id="u", conversation_id="c", goal=goal, approval_token=approval_token)


def _build_agent(
    provider: FakeProvider,
    tools: ToolRegistry,
    screenshot_tool: _FakeScreenshotTool,
    max_steps: int = 8,
) -> ComputerUseAgent:
    return ComputerUseAgent(
        ToolExecutor(tools, PermissionManager()), screenshot_tool, tools, provider, max_steps=max_steps
    )


@pytest.mark.asyncio
async def test_screenshot_reaches_provider_as_image_bytes() -> None:
    """A screenshot the agent decides to take is attached to the next decision call's message."""

    tools = ToolRegistry()
    screenshot_tool = _FakeScreenshotTool(capture_bytes=b"fake-png-bytes")
    tools.register(screenshot_tool)
    provider = FakeProvider(
        complete_responses=[
            ModelResponse(tool_calls=[ToolCall(name="computer.screenshot", arguments={})]),
            ModelResponse(),
        ],
        stream_tokens=["done"],
    )
    agent = _build_agent(provider, tools, screenshot_tool)

    events = [event async for event in agent.run(_context())]

    assert len(provider.complete_calls) == 2
    second_call_messages = provider.complete_calls[1]
    assert second_call_messages[-1].images == [b"fake-png-bytes"]
    tool_results = [event for event in events if event.type is AgentEventType.TOOL_RESULT]
    assert tool_results[0].data["status"] == "succeeded"


@pytest.mark.asyncio
async def test_invalid_action_is_rejected_without_crashing() -> None:
    """A malformed action fails validation and becomes a FAILED observation, not a crash."""

    tools = ToolRegistry()
    click_tool = _FakeClickTool()
    tools.register(click_tool)
    screenshot_tool = _FakeScreenshotTool()
    tools.register(screenshot_tool)
    provider = FakeProvider(
        complete_responses=[
            ModelResponse(tool_calls=[ToolCall(name="computer.mouse_click", arguments={"x": "not-a-number", "y": 10})]),
            ModelResponse(),
        ],
        stream_tokens=["done"],
    )
    agent = _build_agent(provider, tools, screenshot_tool)

    events = [event async for event in agent.run(_context())]

    assert click_tool.clicks == []
    tool_results = [event for event in events if event.type is AgentEventType.TOOL_RESULT]
    assert tool_results[0].data["status"] == "failed"


@pytest.mark.asyncio
async def test_valid_action_goes_through_tool_executor() -> None:
    """A well-formed click actually reaches the fake tool's execute(), through ToolExecutor."""

    tools = ToolRegistry()
    click_tool = _FakeClickTool()
    tools.register(click_tool)
    screenshot_tool = _FakeScreenshotTool()
    tools.register(screenshot_tool)
    provider = FakeProvider(
        complete_responses=[
            ModelResponse(tool_calls=[ToolCall(name="computer.mouse_click", arguments={"x": 640, "y": 420})]),
            ModelResponse(),
        ],
        stream_tokens=["Clicked it."],
    )
    agent = _build_agent(provider, tools, screenshot_tool)

    events = [event async for event in agent.run(_context(approval_token="approved"))]

    assert click_tool.clicks == [(640, 420)]
    tool_results = [event for event in events if event.type is AgentEventType.TOOL_RESULT]
    assert tool_results[0].data["status"] == "succeeded"
    tokens = "".join(str(event.data["content"]) for event in events if event.type is AgentEventType.TOKEN)
    assert tokens == "Clicked it."


@pytest.mark.asyncio
async def test_permission_denial_prevents_execution() -> None:
    """A click without an approval token is denied, and the tool's side effect never runs."""

    tools = ToolRegistry()
    click_tool = _FakeClickTool()
    tools.register(click_tool)
    screenshot_tool = _FakeScreenshotTool()
    tools.register(screenshot_tool)
    provider = FakeProvider(
        complete_responses=[
            ModelResponse(tool_calls=[ToolCall(name="computer.mouse_click", arguments={"x": 640, "y": 420})]),
            ModelResponse(),
        ],
        stream_tokens=["Cannot do that without approval."],
    )
    agent = _build_agent(provider, tools, screenshot_tool)

    events = [event async for event in agent.run(_context(approval_token=None))]

    assert click_tool.clicks == []
    tool_results = [event for event in events if event.type is AgentEventType.TOOL_RESULT]
    assert tool_results[0].data["status"] == "denied"
    assert tool_results[0].data["result"] is None


@pytest.mark.asyncio
async def test_multi_step_flow_requests_verification_screenshot() -> None:
    """screenshot -> click -> a second (verification) screenshot -> done, in that exact order."""

    tools = ToolRegistry()
    click_tool = _FakeClickTool()
    tools.register(click_tool)
    screenshot_tool = _FakeScreenshotTool()
    tools.register(screenshot_tool)
    provider = FakeProvider(
        complete_responses=[
            ModelResponse(tool_calls=[ToolCall(name="computer.screenshot", arguments={})]),
            ModelResponse(tool_calls=[ToolCall(name="computer.mouse_click", arguments={"x": 640, "y": 420})]),
            ModelResponse(tool_calls=[ToolCall(name="computer.screenshot", arguments={})]),
            ModelResponse(),
        ],
        stream_tokens=["Clicked", " it."],
    )
    agent = _build_agent(provider, tools, screenshot_tool)

    events = [event async for event in agent.run(_context())]

    event_types = [event.type for event in events]
    assert event_types == [
        AgentEventType.TOOL_CALL,
        AgentEventType.TOOL_RESULT,
        AgentEventType.TOOL_CALL,
        AgentEventType.TOOL_RESULT,
        AgentEventType.TOOL_CALL,
        AgentEventType.TOOL_RESULT,
        AgentEventType.TOKEN,
        AgentEventType.TOKEN,
    ]
    assert screenshot_tool.capture_count == 2
    assert click_tool.clicks == [(640, 420)]


@pytest.mark.asyncio
async def test_max_steps_limit_prevents_infinite_loop() -> None:
    """A model that always wants another screenshot is still hard-capped."""

    tools = ToolRegistry()
    screenshot_tool = _FakeScreenshotTool()
    tools.register(screenshot_tool)
    # No complete_responses queue: this FakeProvider always returns the same tool call,
    # simulating a model that never decides the task is complete.
    provider = FakeProvider(
        tool_calls=[ToolCall(name="computer.screenshot", arguments={})],
        stream_tokens=["giving up"],
    )
    agent = _build_agent(provider, tools, screenshot_tool, max_steps=3)

    events = [event async for event in agent.run(_context())]

    tool_calls = [event for event in events if event.type is AgentEventType.TOOL_CALL]
    assert len(tool_calls) == 3
    tokens = "".join(str(event.data["content"]) for event in events if event.type is AgentEventType.TOKEN)
    assert tokens == "giving up"


@pytest.mark.asyncio
async def test_no_screenshot_needed_for_a_task_the_model_can_do_blindly() -> None:
    """The model can act (and finish) without ever requesting a screenshot."""

    tools = ToolRegistry()
    click_tool = _FakeClickTool()
    tools.register(click_tool)
    screenshot_tool = _FakeScreenshotTool()
    tools.register(screenshot_tool)
    provider = FakeProvider(
        complete_responses=[
            ModelResponse(tool_calls=[ToolCall(name="computer.mouse_click", arguments={"x": 10, "y": 10})]),
            ModelResponse(),
        ],
        stream_tokens=["done"],
    )
    agent = _build_agent(provider, tools, screenshot_tool)

    _ = [event async for event in agent.run(_context())]

    assert screenshot_tool.capture_count == 0
    assert click_tool.clicks == [(10, 10)]


@pytest.mark.asyncio
async def test_preliminary_tool_call_skips_first_decision_call() -> None:
    """A router-supplied first action runs directly -- no separate first decision call."""

    tools = ToolRegistry()
    click_tool = _FakeClickTool()
    tools.register(click_tool)
    screenshot_tool = _FakeScreenshotTool()
    tools.register(screenshot_tool)
    # Only one queued response: if a first decision call were made, it would consume this
    # and the loop would never see a "done" response for the second (real) decision call.
    provider = FakeProvider(complete_responses=[ModelResponse()], stream_tokens=["Clicked", " it."])
    agent = ComputerUseAgent(ToolExecutor(tools, PermissionManager()), screenshot_tool, tools, provider, max_steps=8)
    context = AgentContext(
        user_id="u",
        conversation_id="c",
        goal="click at 640,420",
        approval_token="approved",
        preliminary_tool_calls=[ToolCall(name="computer.mouse_click", arguments={"x": 640, "y": 420})],
    )

    events = [event async for event in agent.run(context)]

    assert click_tool.clicks == [(640, 420)]
    tool_calls = [event for event in events if event.type is AgentEventType.TOOL_CALL]
    assert len(tool_calls) == 1
    # Exactly one model call: the "is that enough?" round on step 2, not step 1.
    assert len(provider.complete_calls) == 1
    tokens = "".join(str(event.data["content"]) for event in events if event.type is AgentEventType.TOKEN)
    assert tokens == "Clicked it."


def test_max_steps_must_be_positive() -> None:
    """A misconfigured non-positive limit is rejected up front, not discovered mid-loop."""

    tools = ToolRegistry()
    screenshot_tool = _FakeScreenshotTool()
    provider = FakeProvider()

    with pytest.raises(ValueError):
        _build_agent(provider, tools, screenshot_tool, max_steps=0)
