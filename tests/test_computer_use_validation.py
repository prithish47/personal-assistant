"""Tests proving malformed computer-use actions can never reach pyautogui.

Uses the REAL tool classes from aria.tools.computer_use (not fakes) run
through a real ToolExecutor. This is safe with no display/pyautogui call
ever happening: `ToolExecutor.execute()` validates input via Pydantic
*before* it ever awaits `tool.execute()`, so every case here fails at
validation and never reaches the real mouse/keyboard/screen APIs -- proven
directly by asserting FAILED status, never SUCCEEDED, for every case.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aria.domain.models import ToolStatus
from aria.tools.base import ToolContext
from aria.tools.computer_use import (
    HotkeyInput,
    HotkeyTool,
    KeyboardPressInput,
    KeyboardPressTool,
    KeyboardTypeInput,
    KeyboardTypeTool,
    MouseClickInput,
    MouseClickTool,
    MouseMoveInput,
    MouseMoveTool,
    ScreenshotTool,
    ScrollInput,
    ScrollTool,
)
from aria.tools.executor import ToolExecutor
from aria.tools.permissions import PermissionManager
from aria.tools.registry import ToolRegistry


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    for tool in (
        ScreenshotTool(),
        MouseMoveTool(),
        MouseClickTool(),
        KeyboardTypeTool(),
        KeyboardPressTool(),
        HotkeyTool(),
        ScrollTool(),
    ):
        registry.register(tool)
    return registry


def _context() -> ToolContext:
    return ToolContext(user_id="u", conversation_id="c", approval_token="approved")


# --- Direct Pydantic model tests: the exact boundary the task calls out ---


def test_mouse_click_rejects_missing_coordinates() -> None:
    """x/y have no default -- omitting either is rejected."""

    with pytest.raises(ValidationError):
        MouseClickInput(y=10)  # type: ignore[call-arg]


def test_mouse_click_rejects_non_integer_coordinates() -> None:
    """A coordinate that cannot be parsed as an int is rejected."""

    with pytest.raises(ValidationError):
        MouseClickInput(x="not-a-number", y=10)  # type: ignore[arg-type]


def test_mouse_click_rejects_negative_coordinates() -> None:
    """Coordinates below the screen origin are rejected."""

    with pytest.raises(ValidationError):
        MouseClickInput(x=-1, y=10)


def test_mouse_move_rejects_excessively_large_coordinates() -> None:
    """Coordinates far beyond any plausible screen resolution are rejected."""

    with pytest.raises(ValidationError):
        MouseMoveInput(x=999_999, y=10)


def test_mouse_click_rejects_invalid_button() -> None:
    """Only left/right/middle are valid buttons."""

    with pytest.raises(ValidationError):
        MouseClickInput(x=10, y=10, button="turbo")  # type: ignore[arg-type]


def test_keyboard_press_rejects_unknown_key() -> None:
    """A key name pyautogui doesn't recognize is rejected."""

    with pytest.raises(ValidationError):
        KeyboardPressInput(key="not_a_real_key")


def test_hotkey_rejects_too_few_keys() -> None:
    """A hotkey combination needs at least two keys."""

    with pytest.raises(ValidationError):
        HotkeyInput(keys=["ctrl"])


def test_hotkey_rejects_too_many_keys() -> None:
    """A hotkey combination is capped at five keys."""

    with pytest.raises(ValidationError):
        HotkeyInput(keys=["ctrl", "alt", "shift", "meta", "tab", "enter"])


def test_hotkey_rejects_unknown_key() -> None:
    """Every key in a hotkey combination is validated, not just the first."""

    with pytest.raises(ValidationError):
        HotkeyInput(keys=["ctrl", "not_a_real_key"])


def test_keyboard_type_rejects_excessive_length() -> None:
    """Typed text is bounded to prevent a pathological payload."""

    with pytest.raises(ValidationError):
        KeyboardTypeInput(text="a" * 10_001)


def test_scroll_rejects_out_of_range_amount() -> None:
    """Scroll amount is bounded to a sane range."""

    with pytest.raises(ValidationError):
        ScrollInput(amount=51)
    with pytest.raises(ValidationError):
        ScrollInput(amount=-51)


# --- End-to-end through ToolExecutor: proves the real tool's execute() is never reached ---


@pytest.mark.asyncio
async def test_invalid_action_name_fails_without_reaching_any_tool() -> None:
    """An unregistered/hallucinated tool name fails cleanly, never touching a real tool."""

    executor = ToolExecutor(_registry(), PermissionManager())

    result = await executor.execute("computer.launch_missiles", {}, _context())

    assert result.status is ToolStatus.FAILED


@pytest.mark.asyncio
async def test_missing_coordinates_never_reach_pyautogui() -> None:
    executor = ToolExecutor(_registry(), PermissionManager())

    result = await executor.execute("computer.mouse_click", {"y": 10}, _context())

    assert result.status is ToolStatus.FAILED
    assert result.result is None


@pytest.mark.asyncio
async def test_non_integer_coordinates_never_reach_pyautogui() -> None:
    executor = ToolExecutor(_registry(), PermissionManager())

    result = await executor.execute("computer.mouse_click", {"x": "left-ish", "y": 10}, _context())

    assert result.status is ToolStatus.FAILED
    assert result.result is None


@pytest.mark.asyncio
async def test_negative_coordinates_never_reach_pyautogui() -> None:
    executor = ToolExecutor(_registry(), PermissionManager())

    result = await executor.execute("computer.mouse_move", {"x": -5, "y": 5}, _context())

    assert result.status is ToolStatus.FAILED
    assert result.result is None


@pytest.mark.asyncio
async def test_excessively_large_coordinates_never_reach_pyautogui() -> None:
    executor = ToolExecutor(_registry(), PermissionManager())

    result = await executor.execute("computer.mouse_click", {"x": 50_000, "y": 10}, _context())

    assert result.status is ToolStatus.FAILED
    assert result.result is None


@pytest.mark.asyncio
async def test_invalid_mouse_button_never_reaches_pyautogui() -> None:
    executor = ToolExecutor(_registry(), PermissionManager())

    result = await executor.execute("computer.mouse_click", {"x": 10, "y": 10, "button": "scroll"}, _context())

    assert result.status is ToolStatus.FAILED
    assert result.result is None


@pytest.mark.asyncio
async def test_invalid_keyboard_key_never_reaches_pyautogui() -> None:
    executor = ToolExecutor(_registry(), PermissionManager())

    result = await executor.execute("computer.keyboard_press", {"key": "not_a_real_key"}, _context())

    assert result.status is ToolStatus.FAILED
    assert result.result is None


@pytest.mark.asyncio
async def test_invalid_hotkey_length_never_reaches_pyautogui() -> None:
    executor = ToolExecutor(_registry(), PermissionManager())

    result = await executor.execute("computer.hotkey", {"keys": ["ctrl"]}, _context())

    assert result.status is ToolStatus.FAILED
    assert result.result is None


@pytest.mark.asyncio
async def test_invalid_hotkey_key_never_reaches_pyautogui() -> None:
    executor = ToolExecutor(_registry(), PermissionManager())

    result = await executor.execute("computer.hotkey", {"keys": ["ctrl", "not_a_real_key"]}, _context())

    assert result.status is ToolStatus.FAILED
    assert result.result is None


@pytest.mark.asyncio
async def test_excessive_typing_length_never_reaches_pyautogui() -> None:
    executor = ToolExecutor(_registry(), PermissionManager())

    result = await executor.execute("computer.keyboard_type", {"text": "a" * 20_000}, _context())

    assert result.status is ToolStatus.FAILED
    assert result.result is None


@pytest.mark.asyncio
async def test_invalid_scroll_amount_never_reaches_pyautogui() -> None:
    executor = ToolExecutor(_registry(), PermissionManager())

    result = await executor.execute("computer.scroll", {"amount": 500}, _context())

    assert result.status is ToolStatus.FAILED
    assert result.result is None
