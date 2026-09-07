"""Tests for the vision-preparation groundwork: no ComputerUseAgent yet, just the plumbing.

Confirms images can flow through ChatMessage and the provider interface
without any real Ollama server or vision model -- FakeProvider records
whatever it was given, exactly like it does for text-only messages.
"""

from __future__ import annotations

import base64

import httpx
import pytest
from pydantic import SecretStr

from aria.core.config import Settings
from aria.domain.models import ChatMessage, MessageRole
from aria.providers.ollama import OllamaProvider
from tests.fakes import FakeProvider


def test_chat_message_images_field_defaults_to_none() -> None:
    """A normal text-only message never carries an image unless one is explicitly attached."""

    message = ChatMessage(role=MessageRole.USER, content="hello")

    assert message.images is None


@pytest.mark.asyncio
async def test_fake_provider_records_images_attached_to_a_message() -> None:
    """An image attached to a ChatMessage reaches the provider call untouched."""

    provider = FakeProvider(response_content="I see a desktop.")
    screenshot_bytes = b"\x89PNG-fake-bytes-for-testing"
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="Describe the screen."),
        ChatMessage(role=MessageRole.USER, content="What's on screen?", images=[screenshot_bytes]),
    ]

    response = await provider.complete(messages)

    assert response.content == "I see a desktop."
    assert provider.complete_calls[0][-1].images == [screenshot_bytes]


def test_ollama_provider_base64_encodes_images_only_when_present() -> None:
    """OllamaProvider attaches base64 images for vision messages, and omits the key otherwise.

    This never calls the network -- `_messages` is a pure formatting step --
    so it needs no live Ollama server, matching every other test here.
    """

    settings = Settings(api_key=SecretStr("a" * 40))
    provider = OllamaProvider(settings, httpx.AsyncClient())
    screenshot_bytes = b"\x89PNG-fake-bytes-for-testing"

    text_only = provider._messages([ChatMessage(role=MessageRole.USER, content="hello")])
    with_image = provider._messages(
        [ChatMessage(role=MessageRole.USER, content="what's on screen?", images=[screenshot_bytes])]
    )

    assert "images" not in text_only[0]
    assert with_image[0]["images"] == [base64.b64encode(screenshot_bytes).decode("ascii")]
