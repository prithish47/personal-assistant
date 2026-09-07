"""Tests for Pyttsx3TTS and the sentence-chunking helper it's paired with.

Like test_voice_stt.py, these inject a fake `pyttsx3` module into
`sys.modules` before construction -- no real SAPI5 engine, no real audio
device, no persisted files left behind by the test.
"""

from __future__ import annotations

import asyncio
import os
import sys
import types
from pathlib import Path

import pytest

from aria.voice.tts import Pyttsx3TTS, split_into_speech_chunks


class _FakeEngine:
    """Stands in for a pyttsx3 engine: records properties, writes fixed bytes to the given path."""

    def __init__(self, write_bytes: bytes = b"RIFF-fake-wav-bytes", fail: bool = False) -> None:
        self.properties: dict[str, object] = {}
        self.saved_paths: list[str] = []
        self.stopped = False
        self._write_bytes = write_bytes
        self._fail = fail

    def setProperty(self, name: str, value: object) -> None:  # noqa: N802 (matches pyttsx3's real API)
        self.properties[name] = value

    def save_to_file(self, text: str, path: str) -> None:  # noqa: N802
        if self._fail:
            raise RuntimeError("synthesis engine crashed")
        self.saved_paths.append(path)
        Path(path).write_bytes(self._write_bytes)

    def runAndWait(self) -> None:  # noqa: N802
        pass

    def stop(self) -> None:
        self.stopped = True


def _install_fake_module(engine: _FakeEngine) -> None:
    fake_module = types.ModuleType("pyttsx3")
    fake_module.init = lambda: engine  # type: ignore[attr-defined]
    sys.modules["pyttsx3"] = fake_module


@pytest.mark.asyncio
async def test_synthesize_returns_the_engines_written_bytes() -> None:
    """A successful call returns exactly what the engine wrote to its temp file."""

    engine = _FakeEngine(write_bytes=b"RIFF-hello-wav")
    _install_fake_module(engine)
    tts = Pyttsx3TTS(voice_id=None, rate=175)

    audio = await tts.synthesize("Hello there.")

    assert audio == b"RIFF-hello-wav"
    assert engine.stopped is True


@pytest.mark.asyncio
async def test_synthesize_applies_configured_rate_and_voice() -> None:
    """The configured rate and voice id are set on the engine before synthesis."""

    engine = _FakeEngine()
    _install_fake_module(engine)
    tts = Pyttsx3TTS(voice_id="com.apple.some-voice", rate=210)

    await tts.synthesize("Testing configuration.")

    assert engine.properties["rate"] == 210
    assert engine.properties["voice"] == "com.apple.some-voice"


@pytest.mark.asyncio
async def test_synthesize_does_not_set_voice_when_none() -> None:
    """A None voice id leaves the engine's default voice untouched."""

    engine = _FakeEngine()
    _install_fake_module(engine)
    tts = Pyttsx3TTS(voice_id=None, rate=175)

    await tts.synthesize("Default voice.")

    assert "voice" not in engine.properties


@pytest.mark.asyncio
async def test_temp_file_does_not_survive_the_call() -> None:
    """The temporary WAV file is cleaned up -- nothing persists past one synthesize() call."""

    engine = _FakeEngine()
    _install_fake_module(engine)
    tts = Pyttsx3TTS(voice_id=None, rate=175)

    await tts.synthesize("Ephemeral audio.")

    assert len(engine.saved_paths) == 1
    saved_path_still_exists = await asyncio.to_thread(os.path.exists, engine.saved_paths[0])
    assert not saved_path_still_exists


@pytest.mark.asyncio
async def test_synthesize_failure_propagates_as_an_exception() -> None:
    """An engine failure raises rather than silently returning empty/garbage audio.

    The WebSocket transport (aria/app.py) is what turns this into a clean
    `voice_error` event without failing the chat turn that already completed.
    """

    engine = _FakeEngine(fail=True)
    _install_fake_module(engine)
    tts = Pyttsx3TTS(voice_id=None, rate=175)

    with pytest.raises(RuntimeError, match="synthesis engine crashed"):
        await tts.synthesize("This will fail.")

    assert engine.stopped is True  # the engine is still cleaned up even on failure


# --- split_into_speech_chunks ---


def test_split_empty_text_returns_no_chunks() -> None:
    assert split_into_speech_chunks("", max_chunk_chars=280) == []
    assert split_into_speech_chunks("   ", max_chunk_chars=280) == []


def test_split_single_short_sentence_is_one_chunk() -> None:
    assert split_into_speech_chunks("Notepad is now open.", max_chunk_chars=280) == ["Notepad is now open."]


def test_split_merges_short_sentences_into_one_chunk() -> None:
    text = "I opened Notepad. I typed your note. Anything else?"

    chunks = split_into_speech_chunks(text, max_chunk_chars=280)

    assert chunks == ["I opened Notepad. I typed your note. Anything else?"]


def test_split_starts_a_new_chunk_once_the_limit_is_exceeded() -> None:
    text = "First sentence here. Second sentence here. Third sentence here."

    chunks = split_into_speech_chunks(text, max_chunk_chars=40)

    assert len(chunks) > 1
    assert all(len(chunk) <= 40 or " " not in chunk for chunk in chunks)
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


def test_split_keeps_an_overlong_single_sentence_as_its_own_chunk() -> None:
    long_sentence = "This is one single sentence that is deliberately much longer than the configured limit."

    chunks = split_into_speech_chunks(long_sentence, max_chunk_chars=20)

    assert chunks == [long_sentence]
