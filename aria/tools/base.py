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
    # Argument field names whose values must never reach a log, the audit trail, or a
    # tool_call/tool_result event verbatim (e.g. free-form typed text). Empty by default;
    # a tool with genuinely sensitive input overrides this. See `redact_arguments`.
    sensitive_fields: frozenset[str] = frozenset()

    def schema(self) -> dict[str, object]:
        """Return an LLM-compatible JSON schema derived from the input model."""

        return self.input_model.model_json_schema()

    @abstractmethod
    async def execute(self, context: ToolContext, arguments: InputT) -> dict[str, Any]:
        """Execute a validated capability without invoking a shell."""


def redact_arguments(arguments: dict[str, Any], sensitive_fields: frozenset[str]) -> dict[str, Any]:
    """Replace sensitive argument values before they reach any log, audit record, or event.

    Keeps a length hint (e.g. `<redacted:12 chars>`) rather than a flat
    placeholder, so logs stay useful for diagnosing *that* something was
    typed and roughly how much, without ever exposing its content.
    """

    if not sensitive_fields:
        return arguments
    return {
        key: (f"<redacted:{len(str(value))} chars>" if key in sensitive_fields else value)
        for key, value in arguments.items()
    }
