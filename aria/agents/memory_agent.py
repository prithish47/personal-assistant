"""Handles explicit remember requests and long-term memory recall."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

from aria.agents.base import Agent, AgentContext, AgentEvent, AgentEventType
from aria.domain.models import ChatMessage, MessageRole
from aria.memory.models import MemoryKind, MemoryRecord
from aria.memory.service import MemoryService
from aria.providers.base import LLMProvider

_REMEMBER_PATTERN = re.compile(
    r"^(please\s+)?(remember|note down|don't forget)\s+(that\s+)?(?P<content>.+)$", re.IGNORECASE
)


def detect_remember_request(goal: str) -> str | None:
    """Return the fact to remember if `goal` is an explicit remember request, else None.

    This is the single definition of "what counts as an explicit remember
    request" -- both AgentRouter (to skip the classification call entirely
    for this case) and MemoryAgent (to actually store the fact) call this
    same function, so the two can never drift apart.
    """

    match = _REMEMBER_PATTERN.match(goal.strip())
    return match.group("content").strip().rstrip(".") if match else None


class MemoryAgent(Agent):
    """Stores explicit facts on request, and answers recall/search requests from memory."""

    name = "memory"
    description = "Requests to remember something, or to recall/search previously remembered information."

    def __init__(self, memory: MemoryService, provider: LLMProvider) -> None:
        self._memory = memory
        self._provider = provider

    async def run(self, context: AgentContext) -> AsyncIterator[AgentEvent]:
        """Store an explicit fact, or answer a recall request using only remembered content."""

        content = detect_remember_request(context.goal)
        if content is not None:
            await self._memory.remember(
                MemoryRecord(user_id=context.user_id, kind=MemoryKind.SEMANTIC, content=content, importance=0.8)
            )
            yield AgentEvent(AgentEventType.TOKEN, {"content": f"Got it — I'll remember that {content}."})
            return

        recalled = context.memories or await self._memory.retrieve(context.user_id, context.goal, limit=8)
        memory_context = "\n".join(f"- {memory.content}" for memory in recalled)
        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=(
                    "You are ARIA's memory agent. Answer using only the remembered information below, "
                    "and say so plainly if nothing relevant was remembered.\n"
                    f"Remembered information:\n{memory_context or 'None'}"
                ),
            ),
            ChatMessage(role=MessageRole.USER, content=context.goal),
        ]
        async for token in self._provider.stream(messages):
            yield AgentEvent(AgentEventType.TOKEN, {"content": token})
