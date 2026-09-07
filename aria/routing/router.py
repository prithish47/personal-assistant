"""Lightweight LLM-backed router that maps a request to one registered agent.

This is intentionally the only place that decides *which* agent runs, and
now also the only place that decides whether a research request's first
tool call is already obvious enough to skip a second, separate model call.
Adding a fourth agent later means registering it here; no other component
needs to change, since the classification prompt and the fallback are both
built from the registered agent list rather than hardcoded.
"""

from __future__ import annotations

import json
import logging

from aria.agents.base import Agent
from aria.agents.memory_agent import detect_remember_request
from aria.domain.models import ChatMessage, MessageRole, ModelResponse
from aria.providers.base import LLMProvider
from aria.routing.models import RoutingDecision
from aria.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Structured-output calls (classification, tool selection) want reliable, low-variance
# JSON -- not conversational variety. Final answers are unaffected: they never pass an
# explicit temperature, so they keep using the model's normally configured temperature.
_ROUTING_TEMPERATURE = 0.0

_RESEARCH_AGENT_NAME = "research"
_COMPUTER_USE_AGENT_NAME = "computer_use"
_MEMORY_AGENT_NAME = "memory"

# Agents whose first tool call the router's own classification call may pre-decide (see
# `_parse`). Both are agents with their own bounded tool-decision loop already prepared to
# accept a "preliminary" first action (ResearchAgent, ComputerUseAgent) -- fusing saves one
# full model round trip whenever an obvious first action is already clear from the request.
_FUSABLE_AGENT_NAMES = frozenset({_RESEARCH_AGENT_NAME, _COMPUTER_USE_AGENT_NAME})

_SYSTEM_PROMPT_TEMPLATE = (
    "You are JANUS's routing classifier. Choose exactly one agent to handle the user's message.\n"
    "Available agents:\n{catalog}\n"
    'Respond with ONLY a JSON object, no other text: {{"agent": "<name>", "reason": "<short reason>", '
    '"confidence": <0-1>}}.\n'
    "If you choose research or computer_use, and an obvious first tool call is possible without "
    "needing to see the screen first (e.g. opening an application), you may also call one of the "
    "available tools in this same response. Never guess screen coordinates without having seen a "
    "screenshot -- if the task needs one, leave the tool call to computer_use's own next step."
)


class AgentRouter:
    """Classifies a goal into one of the registered agents via a single model call."""

    def __init__(
        self, agents: list[Agent], provider: LLMProvider, default_agent_name: str, tools: ToolRegistry
    ) -> None:
        self._agents = {agent.name: agent for agent in agents}
        if default_agent_name not in self._agents:
            raise ValueError(f"Unknown default agent: {default_agent_name}")
        self._provider = provider
        self._default_agent_name = default_agent_name
        self._tools = tools

    async def route(self, goal: str) -> RoutingDecision:
        """Ask the model which registered agent should handle this goal.

        Skips the model call entirely for an explicit "remember X" request --
        that pattern is already unambiguous, so asking the model to confirm it
        would just be a second opinion on a question already answered for free.
        """

        if _MEMORY_AGENT_NAME in self._agents and detect_remember_request(goal) is not None:
            return RoutingDecision(
                agent_name=_MEMORY_AGENT_NAME,
                reason="Deterministic: explicit remember phrasing",
                confidence=1.0,
            )

        catalog = "\n".join(f"- {name}: {agent.description}" for name, agent in self._agents.items())
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=_SYSTEM_PROMPT_TEMPLATE.format(catalog=catalog)),
            ChatMessage(role=MessageRole.USER, content=goal),
        ]
        response = await self._provider.complete(
            messages, tools=self._tools.schemas(), temperature=_ROUTING_TEMPERATURE
        )
        return self._parse(response)

    def resolve(self, decision: RoutingDecision) -> Agent:
        """Return the concrete agent instance chosen by a routing decision."""

        return self._agents[decision.agent_name]

    def _parse(self, response: ModelResponse) -> RoutingDecision:
        text = response.content.strip()
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        try:
            payload = json.loads(text)
            agent_name = str(payload["agent"])
            if agent_name not in self._agents:
                raise ValueError(f"Unknown agent: {agent_name}")
            return RoutingDecision(
                agent_name=agent_name,
                reason=str(payload.get("reason") or "Model classification"),
                confidence=float(payload.get("confidence", 1.0)),
                # Only agents with their own tool-execution loop accept a preliminary
                # call -- for every other agent, any tool calls the model returned
                # alongside its classification are discarded here, not merely left
                # unused downstream.
                preliminary_tool_calls=response.tool_calls if agent_name in _FUSABLE_AGENT_NAMES else [],
            )
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as error:
            logger.warning("Routing classification failed, falling back to '%s': %s", self._default_agent_name, error)
            return RoutingDecision(
                agent_name=self._default_agent_name,
                reason=f"Fallback: could not parse routing response ({error})",
                confidence=0.0,
            )
