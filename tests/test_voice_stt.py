"""Tests for WhisperSTT: lazy model loading, threading, and failure handling.

The real `faster_whisper` package is installed in this environment, but
these tests never load a real model or touch a real audio file -- they
inject a fake `faster_whisper` module into `sys.modules` before construction,
exactly the way `ScreenshotCapture` test fakes stand in for pyautogui
elsewhere in this suite. No microphone, no real Whisper weights, no network.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterable
from dataclasses import dataclass

import pytest

from aria.voice.stt import WhisperSTT


@dataclass
class _FakeSegment:
    text: str


class _FakeWhisperModel:
    """Stands in for faster_whisper.WhisperModel -- records construction args, returns fixed text."""

    instances: list[_FakeWhisperModel] = []

    def __init__(self, model_size: str, device: str, compute_type: str) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.transcribe_calls = 0
        _FakeWhisperModel.instances.append(self)

    def transcribe(self, audio: object, beam_size: int) -> tuple[Iterable[_FakeSegment], object]:
        self.transcribe_calls += 1
        return [_FakeSegment(" Open "), _FakeSegment("Notepad. ")], object()


class _FailingWhisperModel(_FakeWhisperModel):
    def transcribe(self, audio: object, beam_size: int) -> tuple[Iterable[_FakeSegment], object]:
        raise RuntimeError("decode failed")


@pytest.fixture(autouse=True)
def _reset_fake_instances() -> None:
    _FakeWhisperModel.instances = []


def _install_fake_module(model_cls: type[_FakeWhisperModel]) -> None:
    fake_module = types.ModuleType("faster_whisper")
    fake_module.WhisperModel = model_cls  # type: ignore[attr-defined]
    sys.modules["faster_whisper"] = fake_module


@pytest.mark.asyncio
async def test_transcription_success_joins_segments_and_strips_whitespace() -> None:
    """A successful transcription concatenates segment text and trims the result."""

    _install_fake_module(_FakeWhisperModel)
    stt = WhisperSTT(model_size="tiny.en", device="cpu", compute_type="int8")

    text = await stt.transcribe(b"fake-webm-bytes")

    assert text == "Open Notepad."
    assert _FakeWhisperModel.instances[0].model_size == "tiny.en"
    assert _FakeWhisperModel.instances[0].device == "cpu"
    assert _FakeWhisperModel.instances[0].compute_type == "int8"


@pytest.mark.asyncio
async def test_model_is_constructed_once_and_reused_across_calls() -> None:
    """The (expensive) model is loaded on first use and never reloaded for later utterances."""

    _install_fake_module(_FakeWhisperModel)
    stt = WhisperSTT(model_size="tiny.en", device="cpu", compute_type="int8")

    await stt.transcribe(b"first-utterance")
    await stt.transcribe(b"second-utterance")

    assert len(_FakeWhisperModel.instances) == 1
    assert _FakeWhisperModel.instances[0].transcribe_calls == 2


@pytest.mark.asyncio
async def test_constructing_the_wrapper_never_loads_the_model() -> None:
    """Construction alone (e.g. at app startup) must not pay the model load cost."""

    _install_fake_module(_FakeWhisperModel)
    WhisperSTT(model_size="tiny.en", device="cpu", compute_type="int8")

    assert _FakeWhisperModel.instances == []


@pytest.mark.asyncio
async def test_empty_audio_returns_empty_string_without_touching_the_model() -> None:
    """No audio bytes at all is handled as an empty transcription, not an engine call."""

    _install_fake_module(_FakeWhisperModel)
    stt = WhisperSTT(model_size="tiny.en", device="cpu", compute_type="int8")

    text = await stt.transcribe(b"")

    assert text == ""
    assert _FakeWhisperModel.instances == []


@pytest.mark.asyncio
async def test_transcription_failure_propagates_as_an_exception() -> None:
    """An engine failure (e.g. undecodable audio) raises rather than silently returning text.

    The WebSocket transport (aria/app.py) is what turns this into a clean
    `voice_error` event -- this test only proves the wrapper does not
    swallow or misreport a real failure.
    """

    _install_fake_module(_FailingWhisperModel)
    stt = WhisperSTT(model_size="tiny.en", device="cpu", compute_type="int8")

    with pytest.raises(RuntimeError, match="decode failed"):
        await stt.transcribe(b"corrupt-audio")
