"""The only contract business logic uses to interact with language models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from aria.domain.models import ChatMessage, ModelResponse


class LLMProvider(ABC):
    """Provider-neutral chat, streaming, embedding, and health interface."""

    @abstractmethod
    async def complete(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, object]] | None = None,
        temperature: float | None = None,
    ) -> ModelResponse:
        """Return a complete response, optionally containing structured tool calls.

        `temperature`, when given, overrides the configured default for this
        call only -- structured-output callers (routing, planning) request a
        low value for reliable JSON; conversational callers leave it unset.
        """

    @abstractmethod
    def stream(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, object]] | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Yield response text deltas."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Create vector embeddings for the supplied text."""

    @abstractmethod
    async def healthcheck(self) -> bool:
        """Return whether the configured provider is ready."""


def tool_schema(name: str, description: str, parameters: dict[str, object]) -> dict[str, object]:
    """Build the provider-neutral JSON tool schema."""

    return {"name": name, "description": description, "parameters": parameters}
