"""Tests for local memory lifecycle guarantees."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from aria.domain.models import ChatMessage, ModelResponse
from aria.memory.models import MemoryKind, MemoryRecord
from aria.memory.repository import MemoryRepository
from aria.memory.service import MemoryService
from aria.providers.base import LLMProvider


class FakeProvider(LLMProvider):
    """Deterministic embedding provider used without a model server."""

    async def complete(
        self, messages: list[ChatMessage], tools: list[dict[str, object]] | None = None
    ) -> ModelResponse:
        """Return a minimal response unused by memory tests."""

        return ModelResponse()

    async def stream(
        self, messages: list[ChatMessage], tools: list[dict[str, object]] | None = None
    ) -> AsyncIterator[str]:
        """Yield no tokens."""

        if False:
            yield ""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return fixed-size deterministic vectors."""

        return [[float(len(text)), 1.0] for text in texts]

    async def healthcheck(self) -> bool:
        """Report test-provider readiness."""

        return True


@pytest.mark.asyncio
async def test_memory_deduplicates_and_deletes(tmp_path: Path) -> None:
    """Exact duplicate memories update one record and remain user-deletable."""

    repository = MemoryRepository(tmp_path / "aria.sqlite3")
    await repository.initialize()
    memory = MemoryService(repository, FakeProvider())
    first = await memory.remember(
        MemoryRecord(user_id="u", kind=MemoryKind.USER_PROFILE, content="Prefers concise answers")
    )
    second = await memory.remember(
        MemoryRecord(user_id="u", kind=MemoryKind.USER_PROFILE, content="prefers concise answers")
    )

    records = await repository.find("u")

    assert first.id == second.id
    assert len(records) == 1
    assert await repository.delete(first.id, "u")
    assert await repository.find("u") == []
