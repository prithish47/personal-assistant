"""Deterministic provider fakes so tests never require Ollama or a model server."""

from __future__ import annotations

from collections.abc import AsyncIterator

from aria.domain.models import ChatMessage, ModelResponse, ToolCall
from aria.providers.base import LLMProvider


class FakeProvider(LLMProvider):
    """A fully deterministic, in-memory stand-in for a real LLM provider.

    `embeddings` lets a test pin the exact vector returned for a given input
    string, so semantic ranking can be asserted precisely; any text not in
    that mapping falls back to a length-based vector, matching prior test
    behavior for cases that only care about dedup/CRUD, not ranking.

    `complete_responses`, when given, is a queue consumed one response per
    `complete()` call (the last entry repeats once exhausted) — this is what
    lets a test drive a multi-round reasoning loop: e.g. "call a tool" then
    "no more tools needed". Every call's messages are recorded in
    `complete_calls` so a test can assert what context a later round saw
    (e.g. that a prior tool's observation is present).
    """

    def __init__(
        self,
        embeddings: dict[str, list[float]] | None = None,
        response_content: str = "",
        stream_tokens: list[str] | None = None,
        tool_calls: list[ToolCall] | None = None,
        complete_responses: list[ModelResponse] | None = None,
    ) -> None:
        self._embeddings = embeddings or {}
        self._response_content = response_content
        self._stream_tokens = stream_tokens or []
        self._tool_calls = tool_calls or []
        self._complete_responses = complete_responses
        self.complete_calls: list[list[ChatMessage]] = []
        self.complete_temperatures: list[float | None] = []
        self.complete_tools: list[list[dict[str, object]] | None] = []

    async def complete(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, object]] | None = None,
        temperature: float | None = None,
    ) -> ModelResponse:
        """Return the next queued response, or a fixed one, instead of calling a model."""

        self.complete_calls.append(messages)
        self.complete_temperatures.append(temperature)
        self.complete_tools.append(tools)
        if self._complete_responses is not None:
            index = min(len(self.complete_calls) - 1, len(self._complete_responses) - 1)
            return self._complete_responses[index]
        return ModelResponse(content=self._response_content, tool_calls=self._tool_calls)

    async def stream(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, object]] | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Yield preconfigured tokens instead of streaming from a model."""

        for token in self._stream_tokens:
            yield token

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return pinned or deterministic fallback vectors, never calling a model."""

        return [self._embeddings.get(text, [float(len(text)), 1.0]) for text in texts]

    async def healthcheck(self) -> bool:
        """Always report ready; there is no real backend to check."""

        return True
