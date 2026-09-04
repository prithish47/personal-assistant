"""Provider composition; domain services receive only the LLMProvider interface."""

from __future__ import annotations

import httpx

from aria.core.config import Settings
from aria.providers.base import LLMProvider
from aria.providers.ollama import OllamaProvider


def create_provider(settings: Settings, client: httpx.AsyncClient) -> LLMProvider:
    """Construct the configured provider without exposing it to business logic."""

    if settings.provider == "ollama":
        return OllamaProvider(settings, client)
    raise NotImplementedError(f"Provider '{settings.provider}' is not configured in this deployment")
