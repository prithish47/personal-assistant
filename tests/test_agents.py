"""Tests for MemoryAgent: explicit remember requests and recall from memory.

No Ollama or model server is required; FakeProvider drives the streamed
answer. ResearchAgent's tests live in test_research_agent.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aria.agents.base import AgentContext, AgentEventType
from aria.agents.memory_agent import MemoryAgent
from aria.memory.models import MemoryKind, MemoryRecord
from aria.memory.repository import MemoryRepository
from aria.memory.service import MemoryService
from tests.fakes import FakeProvider


@pytest.mark.asyncio
async def test_memory_agent_stores_explicit_remember_request(tmp_path: Path) -> None:
    """An explicit 'remember X' request is stored via MemoryService, not just echoed back."""

    repository = MemoryRepository(tmp_path / "chroma")
    await repository.initialize()
    provider = FakeProvider()
    memory = MemoryService(repository, provider)
    agent = MemoryAgent(memory, provider)
    context = AgentContext(
        user_id="u", conversation_id="c", goal="remember that my favorite color is blue", approval_token=None
    )

    events = [event async for event in agent.run(context)]

    stored = await repository.find("u", MemoryKind.SEMANTIC)
    assert len(stored) == 1
    assert "favorite color is blue" in stored[0].content
    assert any("remember" in str(event.data["content"]).lower() for event in events)


@pytest.mark.asyncio
async def test_memory_agent_recalls_using_provided_memories(tmp_path: Path) -> None:
    """A recall request answers from already-retrieved memory context via MemoryService."""

    repository = MemoryRepository(tmp_path / "chroma")
    await repository.initialize()
    provider = FakeProvider(stream_tokens=["Your favorite color is blue."])
    memory = MemoryService(repository, provider)
    agent = MemoryAgent(memory, provider)
    recalled = MemoryRecord(user_id="u", kind=MemoryKind.SEMANTIC, content="favorite color is blue")
    context = AgentContext(
        user_id="u",
        conversation_id="c",
        goal="what is my favorite color?",
        approval_token=None,
        memories=[recalled],
    )

    events = [event async for event in agent.run(context)]

    token_text = "".join(str(event.data["content"]) for event in events if event.type is AgentEventType.TOKEN)
    assert token_text == "Your favorite color is blue."
