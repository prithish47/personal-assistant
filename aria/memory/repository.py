"""ChromaDB-backed local memory repository with no database global state."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import chromadb
from chromadb.config import Settings as ChromaSettings

from aria.memory.models import MemoryKind, MemoryRecord

_COLLECTION_NAME = "memories"


class MemoryRepository:
    """Persists memory records in a local, on-disk ChromaDB collection.

    Embeddings are always supplied by the caller (see `MemoryService`), so
    Chroma's own embedding function is never invoked and no model is ever
    downloaded or called over the network by this class.
    """

    def __init__(self, persist_directory: Path) -> None:
        self._persist_directory = persist_directory
        self._client: Any = None
        self._collection: Any = None

    async def initialize(self) -> None:
        """Open (or create) the local persistent collection once at startup."""

        await asyncio.to_thread(self._initialize_sync)

    def _initialize_sync(self) -> None:
        self._persist_directory.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self._persist_directory),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    async def save(self, record: MemoryRecord) -> MemoryRecord:
        """Insert or replace a validated memory record."""

        await asyncio.to_thread(self._save_sync, record)
        return record

    def _save_sync(self, record: MemoryRecord) -> None:
        self._collection.upsert(
            ids=[str(record.id)],
            embeddings=[record.embedding] if record.embedding is not None else None,
            documents=[record.content],
            metadatas=[self._to_metadata(record)],
        )

    async def find(self, user_id: str, kind: MemoryKind | None = None) -> list[MemoryRecord]:
        """Return non-expired memory for the user, pinned/important/recent first."""

        return await asyncio.to_thread(self._find_sync, user_id, kind)

    def _find_sync(self, user_id: str, kind: MemoryKind | None) -> list[MemoryRecord]:
        where: dict[str, Any] = {"user_id": user_id}
        if kind:
            where = {"$and": [{"user_id": user_id}, {"kind": kind.value}]}
        result = self._collection.get(where=where, include=["documents", "metadatas", "embeddings"])
        records = [record for record in self._records_from_get(result) if not self._is_expired(record)]
        records.sort(key=lambda item: (item.pinned, item.importance, item.updated_at), reverse=True)
        return records

    async def query_similar(
        self, user_id: str, query_embedding: list[float], limit: int
    ) -> list[tuple[MemoryRecord, float]]:
        """Return the nearest memories to a query embedding, most similar first."""

        return await asyncio.to_thread(self._query_similar_sync, user_id, query_embedding, limit)

    def _query_similar_sync(
        self, user_id: str, query_embedding: list[float], limit: int
    ) -> list[tuple[MemoryRecord, float]]:
        if limit <= 0 or self._collection.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where={"user_id": user_id},
            include=["documents", "metadatas", "embeddings", "distances"],
        )
        scored: list[tuple[MemoryRecord, float]] = []
        for record, distance in zip(self._records_from_query(result), result["distances"][0], strict=True):
            if self._is_expired(record):
                continue
            similarity = max(0.0, 1.0 - distance)
            scored.append((record, similarity))
        return scored

    async def delete(self, memory_id: UUID, user_id: str) -> bool:
        """Delete exactly one user-owned memory record."""

        return await asyncio.to_thread(self._delete_sync, memory_id, user_id)

    def _delete_sync(self, memory_id: UUID, user_id: str) -> bool:
        existing = self._collection.get(ids=[str(memory_id)], include=["metadatas"])
        if not existing["ids"] or existing["metadatas"][0].get("user_id") != user_id:
            return False
        self._collection.delete(ids=[str(memory_id)])
        return True

    @staticmethod
    def _to_metadata(record: MemoryRecord) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "user_id": record.user_id,
            "kind": record.kind.value,
            "importance": record.importance,
            "pinned": record.pinned,
            "metadata_json": json.dumps(record.metadata),
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
            "last_accessed_at": record.last_accessed_at.isoformat(),
        }
        if record.expires_at is not None:
            metadata["expires_at"] = record.expires_at.isoformat()
        return metadata

    @staticmethod
    def _is_expired(record: MemoryRecord) -> bool:
        return record.expires_at is not None and record.expires_at <= datetime.now(UTC)

    @classmethod
    def _record_from_parts(
        cls, memory_id: str, document: str, metadata: dict[str, Any], embedding: list[float] | None
    ) -> MemoryRecord:
        expires_at = metadata.get("expires_at")
        return MemoryRecord(
            id=UUID(memory_id),
            user_id=metadata["user_id"],
            kind=MemoryKind(metadata["kind"]),
            content=document,
            importance=metadata["importance"],
            pinned=bool(metadata["pinned"]),
            metadata=json.loads(metadata.get("metadata_json", "{}")),
            created_at=datetime.fromisoformat(metadata["created_at"]),
            updated_at=datetime.fromisoformat(metadata["updated_at"]),
            last_accessed_at=datetime.fromisoformat(metadata["last_accessed_at"]),
            expires_at=datetime.fromisoformat(expires_at) if expires_at else None,
            embedding=list(embedding) if embedding is not None else None,
        )

    @classmethod
    def _records_from_get(cls, result: dict[str, Any]) -> list[MemoryRecord]:
        embeddings = result.get("embeddings")
        if embeddings is None:
            embeddings = [None] * len(result["ids"])
        return [
            cls._record_from_parts(memory_id, document, metadata, embedding)
            for memory_id, document, metadata, embedding in zip(
                result["ids"], result["documents"], result["metadatas"], embeddings, strict=True
            )
        ]

    @classmethod
    def _records_from_query(cls, result: dict[str, Any]) -> list[MemoryRecord]:
        ids, documents, metadatas = result["ids"][0], result["documents"][0], result["metadatas"][0]
        embeddings = result.get("embeddings")
        embeddings = embeddings[0] if embeddings is not None else [None] * len(ids)
        return [
            cls._record_from_parts(memory_id, document, metadata, embedding)
            for memory_id, document, metadata, embedding in zip(ids, documents, metadatas, embeddings, strict=True)
        ]
