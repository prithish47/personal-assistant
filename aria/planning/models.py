"""Inspectable task plans, intentionally distinct from private model reasoning."""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from aria.domain.models import ToolCall


class PlanStepStatus(StrEnum):
    """Public execution state for a plan step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"


class PlanStep(BaseModel):
    """A user-visible action or answer step."""

    id: UUID = Field(default_factory=uuid4)
    title: str
    status: PlanStepStatus = PlanStepStatus.PENDING
    tool_call: ToolCall | None = None


class TaskPlan(BaseModel):
    """An execution graph representation; v1 uses a safe ordered subset."""

    id: UUID = Field(default_factory=uuid4)
    goal: str
    steps: list[PlanStep]
