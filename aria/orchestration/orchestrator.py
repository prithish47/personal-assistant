"""Transport-independent orchestration: route a request to an agent and stream results.

Callers (the WebSocket handler, a future CLI, tests) only need to iterate the
event dicts this yields; none of them need to know about agents, routing, or
memory. This is deliberately the only place that sequences those concerns.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from aria.agents.base import AgentContext, AgentEventType
from aria.memory.models import MemoryKind, MemoryRecord
from aria.memory.service import MemoryService
from aria.routing.router import AgentRouter


class JanusOrchestrator:
    """Owns one full conversation turn: memory, routing, agent execution, and persistence."""

    def __init__(self, router: AgentRouter, memory: MemoryService) -> None:
        self._router = router
        self._memory = memory

    async def run(
        self, user_id: str, conversation_id: str, goal: str, approval_token: str | None
    ) -> AsyncIterator[dict[str, object]]:
        """Handle one user turn end-to-end as a stream of transport-agnostic events."""

        try:
            memories = await self._memory.retrieve(user_id, goal)
            decision = await self._router.route(goal)
            yield {
                "type": "routing_decision",
                "data": {"agent": decision.agent_name, "reason": decision.reason, "confidence": decision.confidence},
            }
            agent = self._router.resolve(decision)
            yield {"type": "agent_started", "data": {"agent": agent.name}}

            context = AgentContext(
                user_id=user_id,
                conversation_id=conversation_id,
                goal=goal,
                approval_token=approval_token,
                memories=memories,
                # Already filtered to research-only by AgentRouter._parse; safe to pass
                # through unconditionally for every agent.
                preliminary_tool_calls=decision.preliminary_tool_calls,
            )
            response = ""
            async for event in agent.run(context):
                if event.type is AgentEventType.TOKEN:
                    content = str(event.data["content"])
                    response += content
                    yield {"type": "token", "content": content}
                else:
                    yield {"type": event.type.value, "data": event.data}

            if goal.strip() and response.strip():
                await self._memory.remember(
                    MemoryRecord(
                        user_id=user_id,
                        kind=MemoryKind.EPISODIC,
                        content=f"User: {goal}\nAssistant: {response}",
                    )
                )
            yield {"type": "complete", "data": {"agent": agent.name, "conversation_id": conversation_id}}
        except Exception as error:
            yield {"type": "error", "code": error.__class__.__name__, "message": str(error)}
