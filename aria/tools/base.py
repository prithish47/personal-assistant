"""Strongly typed, shell-free tool contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from aria.domain.models import PermissionLevel

InputT = TypeVar("InputT", bound=BaseModel)


class ToolContext(BaseModel):
    """Trusted context supplied by the executor, never the language model."""

    user_id: str
    conversation_id: str
    approval_token: str | None = None


class Tool(ABC, Generic[InputT]):
    """A capability with validated input and explicit permission requirements."""

    name: str
    description: str
    permission_level: PermissionLevel
    input_model: type[InputT]

    def schema(self) -> dict[str, object]:
        """Return an LLM-compatible JSON schema derived from the input model."""

        return self.input_model.model_json_schema()

    @abstractmethod
    async def execute(self, context: ToolContext, arguments: InputT) -> dict[str, Any]:
        """Execute a validated capability without invoking a shell."""
