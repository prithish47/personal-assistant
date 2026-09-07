"""Tests for local memory lifecycle guarantees backed by ChromaDB.

None of these tests require Ollama, Llama, or nomic-embed-text: embeddings
come from `FakeProvider`, and ChromaDB's `PersistentClient` only ever
receives embeddings we hand it explicitly, so it never calls out to a model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aria.memory.models import MemoryKind, MemoryRecord
from aria.memory.repository import MemoryRepository
from aria.memory.service import MemoryService
from tests.fakes import FakeProvider


@pytest.mark.asyncio
async def test_memory_deduplicates_and_deletes(tmp_path: Path) -> None:
    """Exact duplicate memories update one record and remain user-deletable."""

    repository = MemoryRepository(tmp_path / "chroma")
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


@pytest.mark.asyncio
async def test_semantic_retrieval_ranks_by_meaning(tmp_path: Path) -> None:
    """The closest embedding, not just the newest record, is ranked first."""

    hiking = "I love hiking in the mountains"
    python = "My favorite programming language is Python"
    provider = FakeProvider(
        embeddings={
            hiking: [1.0, 0.0, 0.0],
            python: [0.0, 1.0, 0.0],
            "tell me about hiking": [0.9, 0.1, 0.0],
        }
    )
    repository = MemoryRepository(tmp_path / "chroma")
    await repository.initialize()
    memory = MemoryService(repository, provider)
    await memory.remember(MemoryRecord(user_id="u", kind=MemoryKind.SEMANTIC, content=hiking))
    await memory.remember(MemoryRecord(user_id="u", kind=MemoryKind.SEMANTIC, content=python))

    results = await memory.retrieve("u", "tell me about hiking", limit=1)

    assert [record.content for record in results] == [hiking]


@pytest.mark.asyncio
async def test_retrieve_respects_top_k_limit(tmp_path: Path) -> None:
    """Retrieval never returns more than the requested number of memories."""

    repository = MemoryRepository(tmp_path / "chroma")
    await repository.initialize()
    memory = MemoryService(repository, FakeProvider())
    for index in range(10):
        await memory.remember(MemoryRecord(user_id="u", kind=MemoryKind.SEMANTIC, content=f"fact number {index}"))

    results = await memory.retrieve("u", "fact", limit=3)

    assert len(results) == 3


@pytest.mark.asyncio
async def test_user_isolation(tmp_path: Path) -> None:
    """One user's memories are never visible to another user."""

    repository = MemoryRepository(tmp_path / "chroma")
    await repository.initialize()
    memory = MemoryService(repository, FakeProvider())
    await memory.remember(MemoryRecord(user_id="alice", kind=MemoryKind.SEMANTIC, content="alice's secret"))
    await memory.remember(MemoryRecord(user_id="bob", kind=MemoryKind.SEMANTIC, content="bob's secret"))

    alice_results = await memory.retrieve("alice", "secret", limit=10)
    bob_records = await repository.find("bob")

    assert [record.content for record in alice_results] == ["alice's secret"]
    assert [record.content for record in bob_records] == ["bob's secret"]


@pytest.mark.asyncio
async def test_memory_persists_to_disk_across_repository_instances(tmp_path: Path) -> None:
    """Memories survive a process restart because Chroma persists to disk."""

    persist_directory = tmp_path / "chroma"
    first_repository = MemoryRepository(persist_directory)
    await first_repository.initialize()
    memory = MemoryService(first_repository, FakeProvider())
    saved = await memory.remember(MemoryRecord(user_id="u", kind=MemoryKind.USER_PROFILE, content="Lives in Chennai"))

    second_repository = MemoryRepository(persist_directory)
    await second_repository.initialize()
    records = await second_repository.find("u")

    assert [record.id for record in records] == [saved.id]
    assert records[0].content == "Lives in Chennai"
