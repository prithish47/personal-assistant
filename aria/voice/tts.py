"""Local text-to-speech: turns a completed agent response into playable audio.

Only ever invoked by the WebSocket handler after a turn's full text
response has already been sent to the client (see aria/app.py), so a
synthesis failure here can never turn an otherwise-successful chat turn
into a failed one -- see `test_tts_failure_does_not_fail_the_chat_turn`.
"""

from __future__ import annotations

import asyncio
import re
import tempfile
from pathlib import Path
from typing import Protocol

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def split_into_speech_chunks(text: str, max_chunk_chars: int) -> list[str]:
    """Group sentences into a small number of natural, speakable chunks.

    Splitting is sentence-aligned, not per-token or per-word, so a long
    response becomes a handful of chunks -- not "dozens of tiny unnatural
    speech requests". A single sentence longer than `max_chunk_chars` is
    still returned whole, as its own chunk, rather than cut mid-sentence.
    """

    sentences = [s.strip() for s in _SENTENCE_BOUNDARY.split(text.strip()) if s.strip()]
    if not sentences:
        return []
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if current and len(candidate) > max_chunk_chars:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


class TextToSpeech(Protocol):
    """What the WebSocket transport needs from any TTS engine."""

    async def synthesize(self, text: str) -> bytes:
        """Return complete WAV audio bytes for one chunk of text."""


class Pyttsx3TTS:
    """Local, offline synthesis via the OS-native voice engine (SAPI5 on Windows), through pyttsx3.

    pyttsx3 has no in-memory API: each call writes a temporary WAV file and
    reads it back, and that file is always removed afterward (the
    `TemporaryDirectory` is cleaned up on every exit path) -- no generated
    audio is ever persisted to disk beyond the single request that needs it.

    A fresh engine is constructed for every call rather than reused, because
    pyttsx3's Windows (SAPI5/COM) driver is thread-affine: reusing one engine
    object across the different worker threads `asyncio.to_thread` may run
    this on is unreliable, whereas creating it fresh inside whichever thread
    actually runs the call is not. Synthesis blocks the calling thread, so it
    always runs via `asyncio.to_thread`.
    """

    def __init__(self, voice_id: str | None, rate: int) -> None:
        self._voice_id = voice_id
        self._rate = rate

    async def synthesize(self, text: str) -> bytes:
        """Synthesize one chunk of text to complete WAV bytes."""

        return await asyncio.to_thread(self._synthesize_sync, text)

    def _synthesize_sync(self, text: str) -> bytes:
        try:
            import pythoncom

            pythoncom.CoInitialize()
        except ImportError:
            pythoncom = None  # not on Windows, or pywin32 unavailable; COM init is not needed

        import pyttsx3

        engine = pyttsx3.init()
        try:
            engine.setProperty("rate", self._rate)
            if self._voice_id:
                engine.setProperty("voice", self._voice_id)
            with tempfile.TemporaryDirectory() as tmp_dir:
                path = Path(tmp_dir) / "utterance.wav"
                engine.save_to_file(text, str(path))
                engine.runAndWait()
                return path.read_bytes() if path.exists() else b""
        finally:
            engine.stop()
            if pythoncom is not None:
                pythoncom.CoUninitialize()
