"""Memory records and categories with explicit lifecycle metadata."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class MemoryKind(StrEnum):
    """The purpose of a stored memory determines its retention and retrieval use."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    USER_PROFILE = "user_profile"
    WORKING = "working"
    TASK = "task"


class MemoryRecord(BaseModel):
    """A durable, user-reviewable unit of agent memory."""

    id: UUID = Field(default_factory=uuid4)
    user_id: str
    kind: MemoryKind
    content: str = Field(min_length=1, max_length=20_000)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    pinned: bool = False
    metadata: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_accessed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None
    embedding: list[float] | None = None
