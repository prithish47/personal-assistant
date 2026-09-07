"""The explicit result of a routing decision."""

from __future__ import annotations

from pydantic import BaseModel, Field

from aria.domain.models import ToolCall


class RoutingDecision(BaseModel):
    """Which registered agent should handle a request, and why.

    `preliminary_tool_calls` is populated only when `agent_name == "research"`
    and the router's single classification call also produced an obvious
    first tool call -- letting ResearchAgent skip its own first planning
    call. It is always empty for every other agent.
    """

    agent_name: str
    reason: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    preliminary_tool_calls: list[ToolCall] = Field(default_factory=list)
