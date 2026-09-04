"""SQLite-backed local memory repository with no database global state."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from aria.memory.models import MemoryKind, MemoryRecord


class MemoryRepository:
    """Persists memory records locally and exposes explicit CRUD operations."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path

    async def initialize(self) -> None:
        """Create the local schema once during application startup."""

        await asyncio.to_thread(self._initialize_sync)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_sync(self) -> None:
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY, user_id TEXT NOT NULL, kind TEXT NOT NULL, content TEXT NOT NULL,
                importance REAL NOT NULL, pinned INTEGER NOT NULL, metadata TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL, last_accessed_at TEXT NOT NULL,
                expires_at TEXT, embedding TEXT
            )""")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_memories_user_kind ON memories(user_id, kind)")

    async def save(self, record: MemoryRecord) -> MemoryRecord:
        """Insert or replace a validated memory record."""

        await asyncio.to_thread(self._save_sync, record)
        return record

    def _save_sync(self, record: MemoryRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(record.id),
                    record.user_id,
                    record.kind.value,
                    record.content,
                    record.importance,
                    int(record.pinned),
                    json.dumps(record.metadata),
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                    record.last_accessed_at.isoformat(),
                    record.expires_at.isoformat() if record.expires_at else None,
                    json.dumps(record.embedding) if record.embedding else None,
                ),
            )

    async def find(self, user_id: str, kind: MemoryKind | None = None) -> list[MemoryRecord]:
        """Return non-expired memory for the user, newest first."""

        return await asyncio.to_thread(self._find_sync, user_id, kind)

    def _find_sync(self, user_id: str, kind: MemoryKind | None) -> list[MemoryRecord]:
        query = "SELECT * FROM memories WHERE user_id = ? AND (expires_at IS NULL OR expires_at > ?)"
        params: list[str] = [user_id, datetime.now(UTC).isoformat()]
        if kind:
            query += " AND kind = ?"
            params.append(kind.value)
        query += " ORDER BY pinned DESC, importance DESC, updated_at DESC"
        with self._connect() as connection:
            return [self._record(row) for row in connection.execute(query, params).fetchall()]

    async def delete(self, memory_id: UUID, user_id: str) -> bool:
        """Delete exactly one user-owned memory record."""

        return await asyncio.to_thread(self._delete_sync, memory_id, user_id)

    def _delete_sync(self, memory_id: UUID, user_id: str) -> bool:
        with self._connect() as connection:
            return (
                connection.execute(
                    "DELETE FROM memories WHERE id = ? AND user_id = ?", (str(memory_id), user_id)
                ).rowcount
                > 0
            )

    @staticmethod
    def _record(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=UUID(row["id"]),
            user_id=row["user_id"],
            kind=MemoryKind(row["kind"]),
            content=row["content"],
            importance=row["importance"],
            pinned=bool(row["pinned"]),
            metadata=json.loads(row["metadata"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            last_accessed_at=datetime.fromisoformat(row["last_accessed_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
            embedding=json.loads(row["embedding"]) if row["embedding"] else None,
        )
