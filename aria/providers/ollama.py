"""Ollama implementation of the provider contract."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from aria.core.config import Settings
from aria.core.errors import ProviderError
from aria.domain.models import ChatMessage, ModelResponse, ToolCall
from aria.providers.base import LLMProvider


class OllamaProvider(LLMProvider):
    """Local Ollama provider with no Ollama types outside this adapter."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self._base_url = str(settings.ollama_base_url).rstrip("/")
        self._model = settings.model
        self._embedding_model = settings.embedding_model
        self._temperature = settings.model_temperature
        self._context_window = settings.context_window
        self._client = client

    def _messages(self, messages: list[ChatMessage]) -> list[dict[str, str]]:
        return [{"role": message.role.value, "content": message.content} for message in messages]

    async def complete(
        self, messages: list[ChatMessage], tools: list[dict[str, object]] | None = None
    ) -> ModelResponse:
        payload: dict[str, object] = {
            "model": self._model,
            "messages": self._messages(messages),
            "stream": False,
            "options": {"temperature": self._temperature, "num_ctx": self._context_window},
        }
        if tools:
            payload["tools"] = [{"type": "function", "function": tool} for tool in tools]
        try:
            response = await self._client.post(f"{self._base_url}/api/chat", json=payload)
            response.raise_for_status()
            message = response.json()["message"]
            calls = [
                ToolCall(name=item["function"]["name"], arguments=item["function"].get("arguments", {}))
                for item in message.get("tool_calls", [])
            ]
            return ModelResponse(content=message.get("content", ""), tool_calls=calls)
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
            raise ProviderError(f"Ollama completion failed: {error}") from error

    async def stream(
        self, messages: list[ChatMessage], tools: list[dict[str, object]] | None = None
    ) -> AsyncIterator[str]:
        payload: dict[str, object] = {
            "model": self._model,
            "messages": self._messages(messages),
            "stream": True,
            "options": {"temperature": self._temperature, "num_ctx": self._context_window},
        }
        if tools:
            payload["tools"] = [{"type": "function", "function": tool} for tool in tools]
        try:
            async with self._client.stream("POST", f"{self._base_url}/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        content = json.loads(line).get("message", {}).get("content", "")
                        if content:
                            yield content
        except (httpx.HTTPError, ValueError) as error:
            raise ProviderError(f"Ollama stream failed: {error}") from error

    async def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            vectors: list[list[float]] = []
            for text in texts:
                response = await self._client.post(
                    f"{self._base_url}/api/embeddings", json={"model": self._embedding_model, "prompt": text}
                )
                response.raise_for_status()
                vectors.append(response.json()["embedding"])
            return vectors
        except (httpx.HTTPError, KeyError, TypeError) as error:
            raise ProviderError(f"Ollama embedding failed: {error}") from error

    async def healthcheck(self) -> bool:
        try:
            return (await self._client.get(f"{self._base_url}/api/tags")).is_success
        except httpx.HTTPError:
            return False
