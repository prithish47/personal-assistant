"""Append-only local audit trail for security-relevant agent actions."""

from __future__ import annotations

import asyncio
from pathlib import Path

from aria.domain.models import ToolExecutionRecord


class AuditLogger:
    """Writes structured tool outcomes to a user-owned append-only JSONL file."""

    def __init__(self, path: Path) -> None:
        self._path = path

    async def record(self, execution: ToolExecutionRecord) -> None:
        """Persist an execution outcome without blocking the event loop."""

        await asyncio.to_thread(self._record_sync, execution)

    def _record_sync(self, execution: ToolExecutionRecord) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as audit_file:
            audit_file.write(execution.model_dump_json() + "\n")
