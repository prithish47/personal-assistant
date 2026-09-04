"""Public API contracts, isolated from domain models to permit safe evolution."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """A request to plan and execute one user goal."""

    conversation_id: str = Field(min_length=1, max_length=128)
    content: str = Field(min_length=1, max_length=20_000)
    approval_token: str | None = None


class HealthResponse(BaseModel):
    """Public readiness signal."""

    status: str
    provider_ready: bool
