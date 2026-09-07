"""Computer-use tools: screenshot capture and controlled mouse/keyboard actions.

Every action here goes through the same `Tool` contract as every other tool
in the system -- validated input, an explicit permission level, and nothing
that bypasses `ToolExecutor`. Mouse/keyboard control uses `pyautogui`; screen
capture uses Pillow's `ImageGrab`. Neither is called anywhere except inside
these `execute()` methods.
"""

from __future__ import annotations

import asyncio
import io
from typing import Any, Literal, Protocol

import pyautogui
from PIL import ImageGrab
from pydantic import BaseModel, Field, field_validator

from aria.domain.models import PermissionLevel
from aria.tools.base import Tool, ToolContext

_VALID_KEYS = frozenset(pyautogui.KEYBOARD_KEYS)


def _validate_known_key(key: str) -> str:
    if key not in _VALID_KEYS:
        raise ValueError(f"Unknown key: {key!r}")
    return key


class ScreenshotCapture(Protocol):
    """The minimal interface ComputerUseAgent needs from a screenshot tool.

    Structural, not nominal: the real `ScreenshotTool` below satisfies this
    without inheriting from it, and so can a lightweight test fake -- neither
    needs to import the other.
    """

    name: str

    @property
    def last_capture(self) -> bytes | None: ...


class ScreenshotInput(BaseModel):
    """Screenshot takes no parameters."""


class ScreenshotTool(Tool[ScreenshotInput]):
    """Captures the current desktop as a resized, compressed JPEG.

    The raw bytes are never part of the auditable `result` (that would mean
    persisting screen contents to the audit JSONL forever) -- they live only
    in `last_capture`, an in-memory, single-slot handoff that ComputerUseAgent
    reads immediately after this tool runs and that is overwritten on the
    next capture. Nothing here ever touches ChromaDB or any other durable
    store, so screenshots stay ephemeral by construction.
    """

    name = "computer.screenshot"
    description = "Capture the current desktop screen to see its current state."
    permission_level = PermissionLevel.READ
    input_model = ScreenshotInput

    def __init__(self, max_dimension: int = 1280, jpeg_quality: int = 70) -> None:
        self._max_dimension = max_dimension
        self._jpeg_quality = jpeg_quality
        self._last_capture: bytes | None = None

    @property
    def last_capture(self) -> bytes | None:
        """The most recently captured screenshot's compressed bytes, if any."""

        return self._last_capture

    async def execute(self, context: ToolContext, arguments: ScreenshotInput) -> dict[str, Any]:
        """Capture, downscale, and JPEG-compress the desktop off the event loop."""

        image_bytes, width, height = await asyncio.to_thread(self._capture_and_compress)
        self._last_capture = image_bytes
        return {"width": width, "height": height, "format": "jpeg"}

    def _capture_and_compress(self) -> tuple[bytes, int, int]:
        with ImageGrab.grab() as screenshot:
            image = screenshot.convert("RGB")
        image.thumbnail((self._max_dimension, self._max_dimension))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=self._jpeg_quality)
        return buffer.getvalue(), image.width, image.height


class MouseMoveInput(BaseModel):
    """A screen position to move the cursor to, without clicking."""

    x: int = Field(ge=0, le=10_000)
    y: int = Field(ge=0, le=10_000)


class MouseMoveTool(Tool[MouseMoveInput]):
    """Moves the cursor with no click -- a pure, side-effect-free observation aid."""

    name = "computer.mouse_move"
    description = "Move the mouse cursor to a screen position without clicking."
    permission_level = PermissionLevel.READ
    input_model = MouseMoveInput

    async def execute(self, context: ToolContext, arguments: MouseMoveInput) -> dict[str, Any]:
        await asyncio.to_thread(pyautogui.moveTo, arguments.x, arguments.y)
        return {"x": arguments.x, "y": arguments.y}


class MouseClickInput(BaseModel):
    """A screen position and button to click."""

    x: int = Field(ge=0, le=10_000)
    y: int = Field(ge=0, le=10_000)
    button: Literal["left", "right", "middle"] = "left"


class MouseClickTool(Tool[MouseClickInput]):
    """Clicks the mouse at a screen position -- requires user approval."""

    name = "computer.mouse_click"
    description = "Click the mouse at a screen position."
    permission_level = PermissionLevel.USER_CONFIRM
    input_model = MouseClickInput

    async def execute(self, context: ToolContext, arguments: MouseClickInput) -> dict[str, Any]:
        await asyncio.to_thread(pyautogui.click, arguments.x, arguments.y, button=arguments.button)
        return {"x": arguments.x, "y": arguments.y, "button": arguments.button}


class KeyboardTypeInput(BaseModel):
    """Text to type at the current input focus."""

    text: str = Field(min_length=1, max_length=10_000)


class KeyboardTypeTool(Tool[KeyboardTypeInput]):
    """Types text at whatever currently has input focus -- requires user approval.

    `text` is marked sensitive: it never appears verbatim in the audit log or
    in a tool_call/tool_result event, only as a redacted length hint (see
    `aria.tools.base.redact_arguments`). It is still the real, unredacted
    text that actually gets typed -- only the auditable record is affected.
    """

    name = "computer.keyboard_type"
    description = "Type text at the current input focus."
    permission_level = PermissionLevel.USER_CONFIRM
    input_model = KeyboardTypeInput
    sensitive_fields = frozenset({"text"})

    async def execute(self, context: ToolContext, arguments: KeyboardTypeInput) -> dict[str, Any]:
        await asyncio.to_thread(pyautogui.write, arguments.text)
        return {"characters_typed": len(arguments.text)}


class KeyboardPressInput(BaseModel):
    """A single named key, validated against pyautogui's known key names."""

    key: str = Field(min_length=1, max_length=32)

    @field_validator("key")
    @classmethod
    def _check_key(cls, value: str) -> str:
        return _validate_known_key(value)


class KeyboardPressTool(Tool[KeyboardPressInput]):
    """Presses one named key (e.g. 'enter', 'esc', 'tab') -- requires user approval."""

    name = "computer.keyboard_press"
    description = "Press a single named key, e.g. 'enter', 'esc', 'tab', 'backspace'."
    permission_level = PermissionLevel.USER_CONFIRM
    input_model = KeyboardPressInput

    async def execute(self, context: ToolContext, arguments: KeyboardPressInput) -> dict[str, Any]:
        await asyncio.to_thread(pyautogui.press, arguments.key)
        return {"key": arguments.key}


class HotkeyInput(BaseModel):
    """A key combination pressed together, e.g. ['ctrl', 's']."""

    keys: list[str] = Field(min_length=2, max_length=5)

    @field_validator("keys")
    @classmethod
    def _check_keys(cls, value: list[str]) -> list[str]:
        return [_validate_known_key(key) for key in value]


class HotkeyTool(Tool[HotkeyInput]):
    """Presses a key combination together (e.g. ctrl+s) -- requires user approval."""

    name = "computer.hotkey"
    description = "Press a combination of keys together, e.g. ['ctrl', 's']."
    permission_level = PermissionLevel.USER_CONFIRM
    input_model = HotkeyInput

    async def execute(self, context: ToolContext, arguments: HotkeyInput) -> dict[str, Any]:
        await asyncio.to_thread(pyautogui.hotkey, *arguments.keys)
        return {"keys": arguments.keys}


class ScrollInput(BaseModel):
    """A bounded scroll amount; positive scrolls up, negative scrolls down."""

    amount: int = Field(ge=-50, le=50)


class ScrollTool(Tool[ScrollInput]):
    """Scrolls the window under the cursor -- requires user approval."""

    name = "computer.scroll"
    description = "Scroll the window under the cursor; positive scrolls up, negative scrolls down."
    permission_level = PermissionLevel.USER_CONFIRM
    input_model = ScrollInput

    async def execute(self, context: ToolContext, arguments: ScrollInput) -> dict[str, Any]:
        await asyncio.to_thread(pyautogui.scroll, arguments.amount)
        return {"amount": arguments.amount}
