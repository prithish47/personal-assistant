"""Local speech-to-text: turns recorded audio into plain text for the existing pipeline.

Voice input is not a new capability -- the text this produces re-enters
JANUS through the exact same `JanusOrchestrator.run()` call typed input
uses (see aria/app.py's WebSocket handler), so every existing routing,
memory, tool-permission, and redaction guarantee applies to it unchanged.
"""

from __future__ import annotations

import asyncio
import threading
from io import BytesIO
from typing import Any, Protocol


class SpeechToText(Protocol):
    """What the WebSocket transport needs from any STT engine."""

    async def transcribe(self, audio: bytes) -> str:
        """Return the transcribed text for one complete utterance (may be empty)."""


class WhisperSTT:
    """Local, offline transcription via faster-whisper (a CTranslate2 Whisper build).

    The underlying model is loaded lazily -- on the first real transcription
    request, not at construction -- and then held and reused for every
    request after that. This means constructing this class at application
    startup never pays the model download/load cost, only the first actual
    voice turn does, and every later turn in the process's lifetime reuses
    that same in-memory model rather than reloading it.

    Inference is blocking, CPU-bound work, so it always runs in a worker
    thread via `asyncio.to_thread` and never blocks the event loop that also
    serves every other WebSocket connection.
    """

    def __init__(self, model_size: str, device: str, compute_type: str) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model: Any = None
        self._load_lock = threading.Lock()

    async def transcribe(self, audio: bytes) -> str:
        """Transcribe one complete utterance; empty audio yields an empty string."""

        if not audio:
            return ""
        return await asyncio.to_thread(self._transcribe_sync, audio)

    def _get_model(self) -> Any:
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    from faster_whisper import WhisperModel

                    self._model = WhisperModel(self._model_size, device=self._device, compute_type=self._compute_type)
        return self._model

    def _transcribe_sync(self, audio: bytes) -> str:
        segments, _info = self._get_model().transcribe(BytesIO(audio), beam_size=1)
        return "".join(segment.text for segment in segments).strip()
