"""Provider-neutral domain models shared across ARIA's layers."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class MessageRole(StrEnum):
    """Roles accepted by every model provider."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ChatMessage(BaseModel):
    """A provider-neutral conversation message."""

    role: MessageRole
    content: str
    name: str | None = None


class ToolCall(BaseModel):
    """A structured model request to invoke a registered tool."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ModelResponse(BaseModel):
    """A completed provider response."""

    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    input_tokens: int | None = None
    output_tokens: int | None = None


class PermissionLevel(StrEnum):
    """Increasing levels of impact used by tool policy."""

    READ = "read"
    USER_CONFIRM = "user_confirm"
    ELEVATED = "elevated"


class ToolStatus(StrEnum):
    """Lifecycle state for visible tool executions."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    DENIED = "denied"
    FAILED = "failed"


class ToolExecutionRecord(BaseModel):
    """Auditable outcome of one tool invocation."""

    id: UUID = Field(default_factory=uuid4)
    tool_name: str
    arguments: dict[str, Any]
    status: ToolStatus = ToolStatus.PENDING
    result: dict[str, Any] | None = None
    error: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    duration_ms: int | None = None
