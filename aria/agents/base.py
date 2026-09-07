"""Common agent abstraction: every agent turns one turn into a stream of events."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum

from aria.domain.models import ToolCall
from aria.memory.models import MemoryRecord


class AgentEventType(StrEnum):
    """The kinds of events an agent may emit while handling a turn."""

    TOKEN = "token"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


@dataclass(frozen=True)
class AgentEvent:
    """One unit of agent-produced, transport-agnostic output."""

    type: AgentEventType
    data: dict[str, object]


@dataclass(frozen=True)
class AgentContext:
    """Everything an agent needs to handle one turn, without knowing about transport."""

    user_id: str
    conversation_id: str
    goal: str
    approval_token: str | None
    memories: list[MemoryRecord] = field(default_factory=list)
    preliminary_tool_calls: list[ToolCall] = field(default_factory=list)


class Agent(ABC):
    """A capability with one clear responsibility, selected by the router."""

    name: str
    description: str

    @abstractmethod
    def run(self, context: AgentContext) -> AsyncIterator[AgentEvent]:
        """Handle one turn and yield a stream of transport-agnostic events."""
