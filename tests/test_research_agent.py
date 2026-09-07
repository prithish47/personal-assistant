"""Tests for ResearchAgent's bounded observe-reason-act loop over tools.

No Ollama or model server is required. FakeProvider's `complete_responses`
queue drives successive planning rounds (e.g. "call a tool" then "no more
tools needed"), and `complete_calls` records what each round's messages
looked like so tests can confirm prior tool observations actually reached
the next round.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from aria.agents.base import AgentContext, AgentEventType
from aria.agents.research_agent import ResearchAgent
from aria.domain.models import ModelResponse, PermissionLevel, ToolCall
from aria.planning.planner import Planner
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


class FailingTool(Tool[EchoInput]):
    """A read-only tool that always raises, to exercise the failure-recovery path."""

    name = "test.failing"
    description = "Always fails."
    permission_level = PermissionLevel.READ
    input_model = EchoInput

    def __init__(self) -> None:
        self.executed = False

    async def execute(self, context: ToolContext, arguments: EchoInput) -> dict[str, str]:
        """Mark that execution was attempted, then fail."""

        self.executed = True
        raise RuntimeError("boom")


class ProtectedTool(Tool[EchoInput]):
    """A tool that requires explicit user approval, to exercise the permission-denial path."""

    name = "test.protected"
    description = "Requires approval."
    permission_level = PermissionLevel.USER_CONFIRM
    input_model = EchoInput

    def __init__(self) -> None:
        self.executed = False

    async def execute(self, context: ToolContext, arguments: EchoInput) -> dict[str, str]:
        """Mark that execution happened -- this must never be reached without approval."""

        self.executed = True
        return {"value": arguments.value}


def _context(goal: str = "look this up") -> AgentContext:
    return AgentContext(user_id="u", conversation_id="c", goal=goal, approval_token=None)


@pytest.mark.asyncio
async def test_single_tool_call_then_final_answer() -> None:
    """One planning round that asks for a tool, then a second round that stops, then tokens."""

    tools = ToolRegistry()
    tools.register(EchoTool())
    provider = FakeProvider(
        complete_responses=[
            ModelResponse(tool_calls=[ToolCall(name="test.echo", arguments={"value": "hi"})]),
            ModelResponse(),
        ],
        stream_tokens=["Found", " it"],
    )
    agent = ResearchAgent(Planner(provider, tools), ToolExecutor(tools, PermissionManager()), provider)

    events = [event async for event in agent.run(_context())]

    tool_calls = [event for event in events if event.type is AgentEventType.TOOL_CALL]
    tool_results = [event for event in events if event.type is AgentEventType.TOOL_RESULT]
    tokens = [event for event in events if event.type is AgentEventType.TOKEN]
    assert len(tool_calls) == 1
    assert len(tool_results) == 1
    assert tool_results[0].data["status"] == "succeeded"
    # Tokens arrive as separate events, not one pre-joined blob -- proves no buffering.
    assert [event.data["content"] for event in tokens] == ["Found", " it"]


@pytest.mark.asyncio
async def test_multiple_sequential_tool_calls() -> None:
    """The agent can call a second, different tool after observing the first result."""

    tools = ToolRegistry()
    tools.register(EchoTool())
    provider = FakeProvider(
        complete_responses=[
            ModelResponse(tool_calls=[ToolCall(name="test.echo", arguments={"value": "first"})]),
            ModelResponse(tool_calls=[ToolCall(name="test.echo", arguments={"value": "second"})]),
            ModelResponse(),
        ],
        stream_tokens=["done"],
    )
    agent = ResearchAgent(Planner(provider, tools), ToolExecutor(tools, PermissionManager()), provider)

    events = [event async for event in agent.run(_context())]

    tool_calls = [event for event in events if event.type is AgentEventType.TOOL_CALL]
    tool_results = [event for event in events if event.type is AgentEventType.TOOL_RESULT]
    assert [call.data["arguments"] for call in tool_calls] == [{"value": "first"}, {"value": "second"}]
    assert [result.data["result"] for result in tool_results] == [{"value": "first"}, {"value": "second"}]


@pytest.mark.asyncio
async def test_tool_result_is_available_to_next_reasoning_step() -> None:
    """The next planning call's prompt actually contains the previous tool's observation."""

    tools = ToolRegistry()
    tools.register(EchoTool())
    provider = FakeProvider(
        complete_responses=[
            ModelResponse(tool_calls=[ToolCall(name="test.echo", arguments={"value": "first"})]),
            ModelResponse(),
        ],
        stream_tokens=["done"],
    )
    agent = ResearchAgent(Planner(provider, tools), ToolExecutor(tools, PermissionManager()), provider)

    _ = [event async for event in agent.run(_context())]

    assert len(provider.complete_calls) == 2
    second_round_prompt = provider.complete_calls[1][-1].content
    assert "first" in second_round_prompt
    assert "succeeded" in second_round_prompt


@pytest.mark.asyncio
async def test_failed_tool_call_does_not_crash_agent() -> None:
    """A tool that raises becomes a FAILED observation, not an unhandled exception."""

    tools = ToolRegistry()
    failing_tool = FailingTool()
    tools.register(failing_tool)
    provider = FakeProvider(
        complete_responses=[
            ModelResponse(tool_calls=[ToolCall(name="test.failing", arguments={"value": "x"})]),
            ModelResponse(),
        ],
        stream_tokens=["Sorry, that failed."],
    )
    agent = ResearchAgent(Planner(provider, tools), ToolExecutor(tools, PermissionManager()), provider)

    events = [event async for event in agent.run(_context())]

    assert failing_tool.executed is True
    tool_results = [event for event in events if event.type is AgentEventType.TOOL_RESULT]
    assert tool_results[0].data["status"] == "failed"
    tokens = "".join(str(event.data["content"]) for event in events if event.type is AgentEventType.TOKEN)
    assert tokens == "Sorry, that failed."


@pytest.mark.asyncio
async def test_denied_tool_call_does_not_bypass_permissions() -> None:
    """A tool requiring approval is denied -- and its side effect never actually runs."""

    tools = ToolRegistry()
    protected_tool = ProtectedTool()
    tools.register(protected_tool)
    provider = FakeProvider(
        complete_responses=[
            ModelResponse(tool_calls=[ToolCall(name="test.protected", arguments={"value": "x"})]),
            ModelResponse(),
        ],
        stream_tokens=["Cannot do that without approval."],
    )
    agent = ResearchAgent(Planner(provider, tools), ToolExecutor(tools, PermissionManager()), provider)

    events = [event async for event in agent.run(_context())]  # approval_token=None in _context()

    assert protected_tool.executed is False
    tool_results = [event for event in events if event.type is AgentEventType.TOOL_RESULT]
    assert tool_results[0].data["status"] == "denied"
    assert tool_results[0].data["result"] is None


@pytest.mark.asyncio
async def test_iteration_limit_prevents_infinite_loop() -> None:
    """A model that always wants another tool call is still hard-capped."""

    tools = ToolRegistry()
    tools.register(EchoTool())
    # No complete_responses queue: this FakeProvider returns the same tool call forever,
    # simulating a model that never decides it has enough information.
    provider = FakeProvider(
        tool_calls=[ToolCall(name="test.echo", arguments={"value": "again"})],
        stream_tokens=["giving up"],
    )
    agent = ResearchAgent(
        Planner(provider, tools), ToolExecutor(tools, PermissionManager()), provider, max_iterations=3
    )

    events = [event async for event in agent.run(_context())]

    tool_calls = [event for event in events if event.type is AgentEventType.TOOL_CALL]
    tool_results = [event for event in events if event.type is AgentEventType.TOOL_RESULT]
    assert len(tool_calls) == 3
    assert len(tool_results) == 3
    # The loop still terminates and reaches the final answer instead of hanging forever.
    tokens = "".join(str(event.data["content"]) for event in events if event.type is AgentEventType.TOKEN)
    assert tokens == "giving up"


@pytest.mark.asyncio
async def test_tool_events_stream_in_correct_order() -> None:
    """Each tool_call is immediately followed by its own tool_result, before any tokens."""

    tools = ToolRegistry()
    tools.register(EchoTool())
    provider = FakeProvider(
        complete_responses=[
            ModelResponse(tool_calls=[ToolCall(name="test.echo", arguments={"value": "a"})]),
            ModelResponse(tool_calls=[ToolCall(name="test.echo", arguments={"value": "b"})]),
            ModelResponse(),
        ],
        stream_tokens=["ok"],
    )
    agent = ResearchAgent(Planner(provider, tools), ToolExecutor(tools, PermissionManager()), provider)

    events = [event async for event in agent.run(_context())]

    event_types = [event.type for event in events]
    assert event_types == [
        AgentEventType.TOOL_CALL,
        AgentEventType.TOOL_RESULT,
        AgentEventType.TOOL_CALL,
        AgentEventType.TOOL_RESULT,
        AgentEventType.TOKEN,
    ]


@pytest.mark.asyncio
async def test_preliminary_tool_call_skips_first_planner_call() -> None:
    """A router-supplied first tool call is executed directly, without asking the planner."""

    tools = ToolRegistry()
    tools.register(EchoTool())
    # Only one queued response: if the planner were asked on iteration 1, this would be
    # consumed there and the loop would never see the "stop" response for iteration 2.
    provider = FakeProvider(
        complete_responses=[ModelResponse()],
        stream_tokens=["Found", " it"],
    )
    agent = ResearchAgent(Planner(provider, tools), ToolExecutor(tools, PermissionManager()), provider)
    context = AgentContext(
        user_id="u",
        conversation_id="c",
        goal="look this up",
        approval_token=None,
        preliminary_tool_calls=[ToolCall(name="test.echo", arguments={"value": "hi"})],
    )

    events = [event async for event in agent.run(context)]

    tool_calls = [event for event in events if event.type is AgentEventType.TOOL_CALL]
    tool_results = [event for event in events if event.type is AgentEventType.TOOL_RESULT]
    assert len(tool_calls) == 1
    assert tool_calls[0].data == {"name": "test.echo", "arguments": {"value": "hi"}}
    assert tool_results[0].data["status"] == "succeeded"
    # Exactly one planner call -- the "is that enough?" round on iteration 2, not iteration 1.
    assert len(provider.complete_calls) == 1
    tokens = "".join(str(event.data["content"]) for event in events if event.type is AgentEventType.TOKEN)
    assert tokens == "Found it"


@pytest.mark.asyncio
async def test_preliminary_tool_call_then_further_iterations_continue_normally() -> None:
    """After the preliminary tool call, the agent can still decide it needs another one."""

    tools = ToolRegistry()
    tools.register(EchoTool())
    provider = FakeProvider(
        complete_responses=[
            # Iteration 2: planner decides a second, different tool call is needed.
            ModelResponse(tool_calls=[ToolCall(name="test.echo", arguments={"value": "second"})]),
            # Iteration 3: planner decides it now has enough information.
            ModelResponse(),
        ],
        stream_tokens=["done"],
    )
    agent = ResearchAgent(Planner(provider, tools), ToolExecutor(tools, PermissionManager()), provider)
    context = AgentContext(
        user_id="u",
        conversation_id="c",
        goal="look this up",
        approval_token=None,
        preliminary_tool_calls=[ToolCall(name="test.echo", arguments={"value": "first"})],
    )

    events = [event async for event in agent.run(context)]

    tool_calls = [event for event in events if event.type is AgentEventType.TOOL_CALL]
    assert [call.data["arguments"] for call in tool_calls] == [{"value": "first"}, {"value": "second"}]
    assert len(provider.complete_calls) == 2
    # The second round's prompt must include the first (preliminary) tool's observation.
    assert "first" in provider.complete_calls[0][-1].content


@pytest.mark.asyncio
async def test_empty_preliminary_tool_calls_preserve_existing_behavior() -> None:
    """No preliminary tool calls means the loop behaves exactly as it did before this change."""

    tools = ToolRegistry()
    tools.register(EchoTool())
    provider = FakeProvider(
        complete_responses=[
            ModelResponse(tool_calls=[ToolCall(name="test.echo", arguments={"value": "hi"})]),
            ModelResponse(),
        ],
        stream_tokens=["Found", " it"],
    )
    agent = ResearchAgent(Planner(provider, tools), ToolExecutor(tools, PermissionManager()), provider)
    context = AgentContext(user_id="u", conversation_id="c", goal="look this up", approval_token=None)

    events = [event async for event in agent.run(context)]

    tool_calls = [event for event in events if event.type is AgentEventType.TOOL_CALL]
    assert len(tool_calls) == 1
    assert len(provider.complete_calls) == 2  # planner asked on iteration 1 this time, unlike the fused case
    tokens = "".join(str(event.data["content"]) for event in events if event.type is AgentEventType.TOKEN)
    assert tokens == "Found it"


@pytest.mark.asyncio
async def test_max_iterations_must_be_positive() -> None:
    """A misconfigured non-positive limit is rejected up front, not discovered mid-loop."""

    tools = ToolRegistry()
    provider = FakeProvider()

    with pytest.raises(ValueError):
        ResearchAgent(Planner(provider, tools), ToolExecutor(tools, PermissionManager()), provider, max_iterations=0)
