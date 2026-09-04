"""Centralized, validated application configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings loaded once by the composition root, never by domain code."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: Literal["development", "test", "production"] = "development"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    api_key: SecretStr = Field(min_length=32)
    allowed_origins: list[AnyHttpUrl] = Field(default_factory=list)
    data_dir: Path = Path("data")
    memory_database: Path = Path("data/aria.sqlite3")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    rate_limit_per_minute: int = Field(default=60, ge=1, le=10_000)
    provider: Literal["ollama", "gemini", "openai", "anthropic"] = "ollama"
    model: str = "llama3.1:8b"
    embedding_model: str = "nomic-embed-text"
    ollama_base_url: AnyHttpUrl = AnyHttpUrl("http://127.0.0.1:11434")
    model_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    context_window: int = Field(default=8192, ge=1024, le=1_000_000)
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    application_allowlist: dict[str, Path] = Field(default_factory=dict)


@lru_cache
def get_settings() -> Settings:
    """Return validated settings for command-line entry points only."""

    return Settings()  # type: ignore[call-arg]
