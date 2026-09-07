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
    memory_persist_directory: Path = Path("data/chroma")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    rate_limit_per_minute: int = Field(default=60, ge=1, le=10_000)
    provider: Literal["ollama", "gemini", "openai", "anthropic"] = "ollama"
    # Single reasoning + vision model for all agents. This tag is a best guess (Ollama's
    # published name for Qwen 2.5-VL 7B at the time this was written) and has NOT been
    # verified against a live Ollama instance from this environment -- confirm it with
    # `ollama list` / `ollama pull qwen2.5vl:7b` on the target machine, and override via
    # the MODEL environment variable (or .env) if it differs.
    model: str = Field(
        default="qwen2.5vl:7b",
        description="Ollama tag for the single Qwen 2.5-VL reasoning+vision model; verify on the target machine.",
    )
    embedding_model: str = "nomic-embed-text"
    ollama_base_url: AnyHttpUrl = AnyHttpUrl("http://127.0.0.1:11434")
    model_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    context_window: int = Field(default=8192, ge=1024, le=1_000_000)
    research_agent_max_iterations: int = Field(default=5, ge=1, le=20)
    computer_use_max_steps: int = Field(default=8, ge=1, le=30)
    screenshot_max_dimension: int = Field(default=1280, ge=256, le=4096)
    screenshot_jpeg_quality: int = Field(default=70, ge=10, le=100)
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    application_allowlist: dict[str, Path] = Field(default_factory=dict)
    # Local speech-to-text (faster-whisper). The model is downloaded once (by faster-whisper
    # itself, from Hugging Face) and cached on disk; inference afterward is fully offline.
    voice_stt_model: str = "tiny.en"
    voice_stt_device: Literal["cpu", "cuda"] = "cpu"
    voice_stt_compute_type: str = "int8"
    # Local text-to-speech (pyttsx3, backed by the OS's native voice engine -- SAPI5 on
    # Windows). `voice_tts_voice_id` is an engine-reported voice id; None uses the OS default.
    voice_tts_voice_id: str | None = None
    voice_tts_rate: int = Field(default=175, ge=80, le=400)
    voice_tts_max_chunk_chars: int = Field(default=280, ge=40, le=2000)
    # Safety cap on one recorded utterance's payload size (~90s of compressed webm/opus audio).
    voice_max_audio_bytes: int = Field(default=8_000_000, ge=1_000, le=50_000_000)


@lru_cache
def get_settings() -> Settings:
    """Return validated settings for command-line entry points only."""

    return Settings()  # type: ignore[call-arg]
