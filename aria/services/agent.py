"""Goal execution service joining planning, permissions, memory, and response generation."""

from __future__ import annotations

from collections.abc import AsyncIterator

from aria.domain.models import ChatMessage, MessageRole, ToolExecutionRecord
from aria.memory.service import MemoryService
from aria.planning.models import PlanStepStatus
from aria.planning.planner import Planner
from aria.providers.base import LLMProvider
from aria.tools.base import ToolContext
from aria.tools.executor import ToolExecutor


class AgentService:
    """Executes an inspectable plan and yields public events, never hidden reasoning."""

    def __init__(self, planner: Planner, executor: ToolExecutor, memory: MemoryService, provider: LLMProvider) -> None:
        self._planner = planner
        self._executor = executor
        self._memory = memory
        self._provider = provider

    async def run(
        self, user_id: str, conversation_id: str, goal: str, approval_token: str | None
    ) -> AsyncIterator[dict[str, object]]:
        """Create, execute, and report a plan as a stream of UI-safe events."""

        memories = await self._memory.retrieve(user_id, goal)
        plan = await self._planner.create_plan(goal)
        yield {"type": "plan", "data": plan.model_dump(mode="json")}
        results: list[ToolExecutionRecord] = []
        for step in plan.steps:
            if not step.tool_call:
                continue
            step.status = PlanStepStatus.RUNNING
            yield {"type": "plan_step", "data": step.model_dump(mode="json")}
            result = await self._executor.execute(
                step.tool_call.name,
                step.tool_call.arguments,
                ToolContext(user_id=user_id, conversation_id=conversation_id, approval_token=approval_token),
            )
            results.append(result)
            step.status = PlanStepStatus.COMPLETED if result.status.value == "succeeded" else PlanStepStatus.BLOCKED
            yield {"type": "tool_result", "data": result.model_dump(mode="json")}
        context = "\n".join(f"- {memory.content}" for memory in memories)
        outcomes = "\n".join(f"{result.tool_name}: {result.result or result.error}" for result in results)
        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=(
                    "You are ARIA. Answer clearly and concisely. "
                    f"Relevant user memory:\n{context or 'None'}\n"
                    f"Tool outcomes:\n{outcomes or 'None'}"
                ),
            ),
            ChatMessage(role=MessageRole.USER, content=goal),
        ]
        async for token in self._provider.stream(messages):
            yield {"type": "token", "content": token}
        yield {"type": "complete", "plan_id": str(plan.id)}
