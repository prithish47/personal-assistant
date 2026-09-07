"""Handles everyday conversation, explanations, and reasoning."""

from __future__ import annotations

from collections.abc import AsyncIterator

from aria.agents.base import Agent, AgentContext, AgentEvent, AgentEventType
from aria.domain.models import ChatMessage, MessageRole
from aria.providers.base import LLMProvider


class GeneralAgent(Agent):
    """Answers general requests directly from the model, using memory only as context."""

    name = "general"
    description = "Everyday conversation, explanations, and reasoning that need no tools or memory search."

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    async def run(self, context: AgentContext) -> AsyncIterator[AgentEvent]:
        """Stream a direct answer, with any already-retrieved memory as background context."""

        memory_context = "\n".join(f"- {memory.content}" for memory in context.memories)
        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=(
                    "You are ARIA, a helpful local assistant. Answer clearly and concisely. "
                    f"Relevant user memory:\n{memory_context or 'None'}"
                ),
            ),
            ChatMessage(role=MessageRole.USER, content=context.goal),
        ]
        async for token in self._provider.stream(messages):
            yield AgentEvent(AgentEventType.TOKEN, {"content": token})
