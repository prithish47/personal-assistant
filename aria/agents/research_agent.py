"""Handles research requests via a bounded observe-reason-act loop over tools.

Each iteration reuses the existing Planner (unchanged) to decide the next
tool call given everything observed so far, then the existing ToolExecutor
(unchanged) to run it — so permission checks and audit logging apply on
every single call, exactly as they did before this agent was made
iterative. This class only decides how many rounds to run and what context
to feed back between them.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from aria.agents.base import Agent, AgentContext, AgentEvent, AgentEventType
from aria.domain.models import ChatMessage, MessageRole, ToolCall
from aria.planning.planner import Planner
from aria.providers.base import LLMProvider
from aria.tools.base import ToolContext
from aria.tools.executor import ToolExecutor

DEFAULT_MAX_ITERATIONS = 5


class ResearchAgent(Agent):
    """Answers information-seeking requests via a bounded plan -> execute -> observe loop."""

    name = "research"
    description = "Information and research requests that may benefit from one or more registered tools."

    def __init__(
        self,
        planner: Planner,
        executor: ToolExecutor,
        provider: LLMProvider,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        self._planner = planner
        self._executor = executor
        self._provider = provider
        self._max_iterations = max_iterations

    async def run(self, context: AgentContext) -> AsyncIterator[AgentEvent]:
        """Call tools for up to max_iterations rounds, then stream a final answer.

        If the router already produced an obvious first tool call
        (`context.preliminary_tool_calls`), iteration 1 executes those
        directly instead of asking the planner the same question again;
        every later iteration is unaffected and works exactly as before.
        """

        observations: list[str] = []
        iterations_used = 0
        limit_reached = False
        pending_tool_calls: list[ToolCall] | None = (
            context.preliminary_tool_calls if context.preliminary_tool_calls else None
        )

        while True:
            if iterations_used >= self._max_iterations:
                limit_reached = True
                break

            if pending_tool_calls is not None:
                tool_calls = pending_tool_calls
                pending_tool_calls = None  # only ever supplied for iteration 1
            else:
                plan = await self._planner.create_plan(self._build_prompt(context.goal, observations))
                tool_calls = [step.tool_call for step in plan.steps if step.tool_call is not None]
            if not tool_calls:
                break

            for tool_call in tool_calls:
                if iterations_used >= self._max_iterations:
                    limit_reached = True
                    break

                yield AgentEvent(AgentEventType.TOOL_CALL, {"name": tool_call.name, "arguments": tool_call.arguments})
                result = await self._executor.execute(
                    tool_call.name,
                    tool_call.arguments,
                    ToolContext(
                        user_id=context.user_id,
                        conversation_id=context.conversation_id,
                        approval_token=context.approval_token,
                    ),
                )
                observations.append(f"{result.tool_name} ({result.status.value}): {result.result or result.error}")
                yield AgentEvent(AgentEventType.TOOL_RESULT, result.model_dump(mode="json"))
                iterations_used += 1

            if limit_reached:
                break

        async for event in self._stream_final_answer(context, observations, limit_reached):
            yield event

    @staticmethod
    def _build_prompt(goal: str, observations: list[str]) -> str:
        """Feed prior tool observations back as context for the next planning call."""

        if not observations:
            return goal
        return (
            f"Original request: {goal}\n\n"
            f"Tool observations so far:\n{chr(10).join(observations)}\n\n"
            "If another tool call is still needed to fully answer, request it. "
            "Otherwise, request no tool -- you already have enough information to answer."
        )

    async def _stream_final_answer(
        self, context: AgentContext, observations: list[str], limit_reached: bool
    ) -> AsyncIterator[AgentEvent]:
        memory_context = "\n".join(f"- {memory.content}" for memory in context.memories)
        limit_note = (
            "\nNote: the maximum number of tool calls was reached; answer using the information gathered so far."
            if limit_reached
            else ""
        )
        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=(
                    "You are ARIA's research agent. Use the tool outcomes below when relevant, "
                    f"and answer clearly and concisely.{limit_note}\n"
                    f"Relevant user memory:\n{memory_context or 'None'}\n"
                    f"Tool outcomes:\n{chr(10).join(observations) or 'None'}"
                ),
            ),
            ChatMessage(role=MessageRole.USER, content=context.goal),
        ]
        async for token in self._provider.stream(messages):
            yield AgentEvent(AgentEventType.TOKEN, {"content": token})
