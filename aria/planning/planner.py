"""Planning layer that chooses direct answers or structured tool steps."""

from __future__ import annotations

from aria.domain.models import ChatMessage, MessageRole
from aria.planning.models import PlanStep, TaskPlan
from aria.providers.base import LLMProvider
from aria.tools.registry import ToolRegistry

# Tool selection is a structured-output decision, not conversational generation --
# a low temperature keeps it reliable. Final answers never pass this and keep using
# the model's normally configured temperature.
_PLANNING_TEMPERATURE = 0.0


class Planner:
    """Converts a user goal into an inspectable plan without persona routing."""

    def __init__(self, provider: LLMProvider, tools: ToolRegistry) -> None:
        self._provider = provider
        self._tools = tools

    async def create_plan(self, goal: str) -> TaskPlan:
        """Ask the model whether the next safe action is a tool call or a response."""

        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=(
                    "You are ARIA's planner. Select a registered tool only when it is necessary. Never invent tools."
                ),
            ),
            ChatMessage(role=MessageRole.USER, content=goal),
        ]
        response = await self._provider.complete(messages, self._tools.schemas(), temperature=_PLANNING_TEMPERATURE)
        if response.tool_calls:
            return TaskPlan(
                goal=goal, steps=[PlanStep(title=f"Run {call.name}", tool_call=call) for call in response.tool_calls]
            )
        return TaskPlan(goal=goal, steps=[PlanStep(title="Generate response")])
