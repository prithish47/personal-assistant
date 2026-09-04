"""Memory policy: deduplication, semantic retrieval, retention, and review."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from aria.memory.models import MemoryKind, MemoryRecord
from aria.memory.repository import MemoryRepository
from aria.providers.base import LLMProvider


class MemoryService:
    """Coordinates local durable memory without leaking storage details upward."""

    def __init__(self, repository: MemoryRepository, provider: LLMProvider) -> None:
        self._repository = repository
        self._provider = provider

    async def remember(self, record: MemoryRecord) -> MemoryRecord:
        """Embed, deduplicate exact content, and persist a memory record."""

        existing = await self._repository.find(record.user_id, record.kind)
        normalized = " ".join(record.content.lower().split())
        for candidate in existing:
            if " ".join(candidate.content.lower().split()) == normalized:
                candidate.updated_at = datetime.now(UTC)
                candidate.importance = max(candidate.importance, record.importance)
                return await self._repository.save(candidate)
        record.embedding = (await self._provider.embed([record.content]))[0]
        if record.kind is MemoryKind.WORKING:
            record.expires_at = datetime.now(UTC) + timedelta(hours=24)
        return await self._repository.save(record)

    async def retrieve(self, user_id: str, query: str, limit: int = 8) -> list[MemoryRecord]:
        """Return importance-, recency-, and cosine-ranked memories for a query."""

        query_vector = (await self._provider.embed([query]))[0]
        now = datetime.now(UTC)
        scored: list[tuple[float, MemoryRecord]] = []
        for record in await self._repository.find(user_id):
            similarity = self._cosine(query_vector, record.embedding or [])
            age_days = max((now - record.updated_at).total_seconds() / 86_400, 0.0)
            decay = 1.0 if record.pinned else math.exp(-age_days / 180)
            score = similarity * 0.65 + record.importance * 0.25 + decay * 0.10
            scored.append((score, record))
        return [record for _, record in sorted(scored, key=lambda item: item[0], reverse=True)[:limit]]

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if not left or len(left) != len(right):
            return 0.0
        denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(sum(value * value for value in right))
        return sum(a * b for a, b in zip(left, right, strict=True)) / denominator if denominator else 0.0
