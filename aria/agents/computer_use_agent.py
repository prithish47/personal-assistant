"""Handles desktop interaction via a bounded perceive-decide-act loop.

Each step asks the model for exactly one next action -- which may itself be
`computer.screenshot` if the model judges it needs to see the screen first,
or any other registered tool, or no tool at all if the task is already done.
This is a deliberate choice over a hardcoded "screenshot before every step"
rule: letting the model decide when it needs to look keeps the common case
(e.g. "open Notepad") screenshot-free, while still allowing it to verify
after any action it judges risky enough to check.

If the router already produced an obvious first action (see AgentRouter's
fusion of the classification call with a first tool call), iteration 1 uses
that directly instead of asking a second, separate question -- one fewer
model round trip for cases like "open Notepad" that don't need to see the
screen before their first action.

Every action -- screenshot included -- runs through the same ToolExecutor
used everywhere else, so permission checks and audit logging apply
uniformly; this class only decides how many steps to run and what image (if
any) to attach to the next decision call.

A per-step trace is logged via the standard `logging` module (safe metadata
only -- see `_log_step`) so real Ollama/Qwen-VL runs can be diagnosed
without ever writing screenshot bytes or typed text to a log file.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from aria.agents.base import Agent, AgentContext, AgentEvent, AgentEventType
from aria.domain.models import ChatMessage, MessageRole, PermissionLevel, ToolCall, ToolExecutionRecord, ToolStatus
from aria.providers.base import LLMProvider
from aria.tools.base import ToolContext, redact_arguments
from aria.tools.computer_use import ScreenshotCapture
from aria.tools.executor import ToolExecutor
from aria.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 8

# Structured-output (action-decision) calls want reliable, low-variance output; the
# final summary is unaffected and keeps using the model's normally configured temperature.
_DECISION_TEMPERATURE = 0.0


class ComputerUseAgent(Agent):
    """Interacts with the user's desktop via screenshots and controlled computer actions."""

    name = "computer_use"
    description = (
        "Interacting with the user's desktop using screenshots and controlled computer actions "
        "(mouse, keyboard, opening applications) -- not information research, not memory."
    )

    def __init__(
        self,
        executor: ToolExecutor,
        screenshot_tool: ScreenshotCapture,
        tools: ToolRegistry,
        provider: LLMProvider,
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self._executor = executor
        self._screenshot_tool = screenshot_tool
        self._tools = tools
        self._provider = provider
        self._max_steps = max_steps

    async def run(self, context: AgentContext) -> AsyncIterator[AgentEvent]:
        """Perceive, decide, and act for up to max_steps rounds, then stream a summary."""

        observations: list[str] = []
        steps_used = 0
        limit_reached = False
        current_image: bytes | None = None
        # A router-supplied first action (see AgentRouter._parse) skips this agent's own
        # first decision call entirely; None for every step after that, same as before.
        pending_decision: ToolCall | None = (
            context.preliminary_tool_calls[0] if context.preliminary_tool_calls else None
        )
        tool_context = ToolContext(
            user_id=context.user_id, conversation_id=context.conversation_id, approval_token=context.approval_token
        )

        while True:
            if steps_used >= self._max_steps:
                limit_reached = True
                break

            image_attached = current_image is not None
            image_byte_size = len(current_image) if current_image is not None else None

            if pending_decision is not None:
                decision: ToolCall | None = pending_decision
                pending_decision = None
            else:
                decision = await self._decide_next_action(context.goal, observations, current_image)
            if decision is None:
                break

            tool = self._tools.find(decision.name)
            safe_arguments = (
                redact_arguments(decision.arguments, tool.sensitive_fields) if tool is not None else decision.arguments
            )
            yield AgentEvent(AgentEventType.TOOL_CALL, {"name": decision.name, "arguments": safe_arguments})

            result = await self._executor.execute(decision.name, decision.arguments, tool_context)
            observations.append(f"{result.tool_name} ({result.status.value}): {result.result or result.error}")
            yield AgentEvent(AgentEventType.TOOL_RESULT, result.model_dump(mode="json"))

            is_screenshot_capture = (
                decision.name == self._screenshot_tool.name and result.status is ToolStatus.SUCCEEDED
            )
            self._log_step(
                step_number=steps_used + 1,
                tool_name=decision.name,
                image_attached=image_attached,
                image_byte_size=image_byte_size,
                result=result,
                permission_required=tool.permission_level is not PermissionLevel.READ if tool is not None else None,
                screenshot_captured=is_screenshot_capture,
            )

            current_image = self._screenshot_tool.last_capture if is_screenshot_capture else None
            steps_used += 1

        logger.info(
            "computer_use finished steps_used=%d terminated_by=%s",
            steps_used,
            "max_steps" if limit_reached else "model_done",
        )

        async for event in self._stream_final_answer(context, observations, limit_reached):
            yield event

    @staticmethod
    def _log_step(
        *,
        step_number: int,
        tool_name: str,
        image_attached: bool,
        image_byte_size: int | None,
        result: ToolExecutionRecord,
        permission_required: bool | None,
        screenshot_captured: bool,
    ) -> None:
        """Log safe, structured metadata for one step -- never arguments, results, or image bytes."""

        logger.info(
            "computer_use step=%d tool=%s image_attached=%s image_bytes=%s status=%s duration_ms=%s "
            "permission_required=%s permission_denied=%s screenshot_captured=%s",
            step_number,
            tool_name,
            image_attached,
            image_byte_size,
            result.status.value,
            result.duration_ms,
            permission_required,
            result.status is ToolStatus.DENIED,
            screenshot_captured,
        )

    async def _decide_next_action(self, goal: str, observations: list[str], image: bytes | None) -> ToolCall | None:
        """Ask the model for exactly one next action, or None if the task is already complete."""

        system_content = (
            "You are ARIA's computer-use agent. Decide the single next action needed to accomplish "
            "the task using the available tools -- call computer.screenshot first if you need to see "
            "the screen before deciding -- or call no tool if the task is already complete.\n"
            f"Task: {goal}\n"
            f"Actions so far:\n{chr(10).join(observations) or 'None'}"
        )
        user_content = "Here is the current screenshot." if image is not None else "No screenshot captured yet."
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_content),
            ChatMessage(role=MessageRole.USER, content=user_content, images=[image] if image is not None else None),
        ]
        response = await self._provider.complete(
            messages, tools=self._tools.schemas(), temperature=_DECISION_TEMPERATURE
        )
        return response.tool_calls[0] if response.tool_calls else None

    async def _stream_final_answer(
        self, context: AgentContext, observations: list[str], limit_reached: bool
    ) -> AsyncIterator[AgentEvent]:
        memory_context = "\n".join(f"- {memory.content}" for memory in context.memories)
        limit_note = (
            "\nNote: the maximum number of computer-use steps was reached; summarize what was done so far."
            if limit_reached
            else ""
        )
        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=(
                    "You are ARIA's computer-use agent. Summarize what was done on the desktop, clearly "
                    f"and concisely.{limit_note}\n"
                    f"Relevant user memory:\n{memory_context or 'None'}\n"
                    f"Actions taken:\n{chr(10).join(observations) or 'None'}"
                ),
            ),
            ChatMessage(role=MessageRole.USER, content=context.goal),
        ]
        async for token in self._provider.stream(messages):
            yield AgentEvent(AgentEventType.TOKEN, {"content": token})
