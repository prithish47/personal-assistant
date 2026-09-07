"""Tests for the observability/privacy pass: safe trace logging, redaction, and
screenshot ephemerality/failure handling.

No Ollama, no real screen, no real pyautogui call anywhere here -- fake
tools stand in exactly as they do in test_computer_use_agent.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from pydantic import BaseModel

from aria.agents.base import AgentContext, AgentEventType
from aria.agents.computer_use_agent import ComputerUseAgent
from aria.core.audit import AuditLogger
from aria.domain.models import ModelResponse, PermissionLevel, ToolCall, ToolStatus
from aria.tools.base import Tool, ToolContext, redact_arguments
from aria.tools.computer_use import KeyboardTypeTool
from aria.tools.executor import ToolExecutor
from aria.tools.permissions import PermissionManager
from aria.tools.registry import ToolRegistry
from tests.fakes import FakeProvider


class _ScreenshotInput(BaseModel):
    """Screenshot takes no parameters."""


class _FakeScreenshotTool(Tool[_ScreenshotInput]):
    """Stands in for the real, pyautogui/Pillow-backed ScreenshotTool in tests."""

    name = "computer.screenshot"
    description = "Fake screenshot capture."
    permission_level = PermissionLevel.READ
    input_model = _ScreenshotInput

    def __init__(self, capture_bytes: bytes = b"fake-jpeg-bytes") -> None:
        self._capture_bytes = capture_bytes
        self._last_capture: bytes | None = None

    @property
    def last_capture(self) -> bytes | None:
        """The most recently captured (fake) screenshot bytes."""

        return self._last_capture

    async def execute(self, context: ToolContext, arguments: _ScreenshotInput) -> dict[str, object]:
        """Hand back fixed bytes, exactly like a successful real capture would."""

        self._last_capture = self._capture_bytes
        return {"width": 100, "height": 100, "format": "jpeg"}


class _FailingScreenshotTool(Tool[_ScreenshotInput]):
    """Simulates a real ImageGrab failure (e.g. no accessible display)."""

    name = "computer.screenshot"
    description = "Always fails to capture."
    permission_level = PermissionLevel.READ
    input_model = _ScreenshotInput

    @property
    def last_capture(self) -> bytes | None:
        """Never populated -- capture never succeeds."""

        return None

    async def execute(self, context: ToolContext, arguments: _ScreenshotInput) -> dict[str, object]:
        """Simulate a screen-capture failure."""

        raise RuntimeError("display not available")


class _SensitiveInput(BaseModel):
    """A tool input carrying a sensitive free-form field, for testing redaction."""

    text: str


class _FakeSensitiveTool(Tool[_SensitiveInput]):
    """Declares a sensitive field exactly like the real KeyboardTypeTool, without pyautogui."""

    name = "test.sensitive_type"
    description = "Fake tool with a sensitive field."
    permission_level = PermissionLevel.USER_CONFIRM
    input_model = _SensitiveInput
    sensitive_fields = frozenset({"text"})

    async def execute(self, context: ToolContext, arguments: _SensitiveInput) -> dict[str, object]:
        """Never actually types anything -- purely for testing the redaction mechanism."""

        return {"characters_typed": len(arguments.text)}


def _context(approval_token: str | None = "approved") -> AgentContext:
    return AgentContext(user_id="u", conversation_id="c", goal="do something", approval_token=approval_token)


# --- redact_arguments() itself ---


def test_redact_arguments_replaces_sensitive_values_with_length_hint() -> None:
    """A sensitive value becomes a length hint, never the content itself."""

    redacted = redact_arguments({"text": "hunter2", "other": 5}, frozenset({"text"}))

    assert redacted == {"text": "<redacted:7 chars>", "other": 5}


def test_redact_arguments_is_noop_without_sensitive_fields() -> None:
    """A tool with no sensitive fields is returned unchanged (same dict values)."""

    arguments = {"x": 10, "y": 20}

    assert redact_arguments(arguments, frozenset()) == arguments


def test_real_keyboard_type_tool_declares_text_as_sensitive() -> None:
    """The actual production tool -- not just a test double -- marks its text field sensitive."""

    assert KeyboardTypeTool.sensitive_fields == frozenset({"text"})


# --- End-to-end: nothing sensitive reaches the audit log or a tool_call/tool_result event ---


@pytest.mark.asyncio
async def test_sensitive_argument_never_appears_in_audit_log(tmp_path: Path) -> None:
    """A tool with a sensitive field never writes its real value to the audit JSONL."""

    registry = ToolRegistry()
    registry.register(_FakeSensitiveTool())
    audit_path = tmp_path / "audit.jsonl"
    executor = ToolExecutor(registry, PermissionManager(), AuditLogger(audit_path))
    secret = "my-super-secret-password-123"

    result = await executor.execute(
        "test.sensitive_type", {"text": secret}, ToolContext(user_id="u", conversation_id="c", approval_token="ok")
    )

    assert result.status is ToolStatus.SUCCEEDED
    assert secret not in str(result.arguments)
    audit_content = audit_path.read_text(encoding="utf-8")
    assert secret not in audit_content
    assert "redacted" in audit_content


@pytest.mark.asyncio
async def test_tool_call_event_redacts_sensitive_arguments_before_execution() -> None:
    """The TOOL_CALL event (emitted before ToolExecutor even runs) is also redacted."""

    tools = ToolRegistry()
    tools.register(_FakeSensitiveTool())
    screenshot_tool = _FakeScreenshotTool()
    tools.register(screenshot_tool)
    secret = "hunter2-actual-password"
    provider = FakeProvider(
        complete_responses=[
            ModelResponse(tool_calls=[ToolCall(name="test.sensitive_type", arguments={"text": secret})]),
            ModelResponse(),
        ],
        stream_tokens=["done"],
    )
    agent = ComputerUseAgent(ToolExecutor(tools, PermissionManager()), screenshot_tool, tools, provider)

    events = [event async for event in agent.run(_context())]

    tool_call = next(event for event in events if event.type is AgentEventType.TOOL_CALL)
    tool_result = next(event for event in events if event.type is AgentEventType.TOOL_RESULT)
    assert secret not in str(tool_call.data)
    assert secret not in str(tool_result.data)


# --- Screenshot ephemerality / failure handling ---


@pytest.mark.asyncio
async def test_failed_screenshot_capture_does_not_propagate_stale_image() -> None:
    """A screenshot failure is a normal FAILED result -- no crash, no stale image reused."""

    tools = ToolRegistry()
    failing_tool = _FailingScreenshotTool()
    tools.register(failing_tool)
    provider = FakeProvider(
        complete_responses=[
            ModelResponse(tool_calls=[ToolCall(name="computer.screenshot", arguments={})]),
            ModelResponse(),
        ],
        stream_tokens=["Could not see the screen."],
    )
    agent = ComputerUseAgent(ToolExecutor(tools, PermissionManager()), failing_tool, tools, provider)

    events = [event async for event in agent.run(_context())]

    tool_result = next(event for event in events if event.type is AgentEventType.TOOL_RESULT)
    assert tool_result.data["status"] == "failed"
    assert len(provider.complete_calls) == 2
    assert provider.complete_calls[1][-1].images is None


# --- Structured trace logging: safe metadata present, nothing sensitive ---


@pytest.mark.asyncio
async def test_step_trace_logs_safe_metadata_without_bytes_or_text(caplog: pytest.LogCaptureFixture) -> None:
    """The per-step trace log carries the documented safe fields and nothing sensitive."""

    caplog.set_level(logging.INFO, logger="aria.agents.computer_use_agent")
    tools = ToolRegistry()
    screenshot_tool = _FakeScreenshotTool(capture_bytes=b"\x89PNG-fake-bytes-marker")
    tools.register(screenshot_tool)
    tools.register(_FakeSensitiveTool())
    secret = "super-secret-content"
    provider = FakeProvider(
        complete_responses=[
            ModelResponse(tool_calls=[ToolCall(name="computer.screenshot", arguments={})]),
            ModelResponse(tool_calls=[ToolCall(name="test.sensitive_type", arguments={"text": secret})]),
            ModelResponse(),
        ],
        stream_tokens=["done"],
    )
    agent = ComputerUseAgent(ToolExecutor(tools, PermissionManager()), screenshot_tool, tools, provider)

    _ = [event async for event in agent.run(_context())]

    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "step=1" in log_text
    assert "step=2" in log_text
    assert "terminated_by=model_done" in log_text
    assert "screenshot_captured=True" in log_text
    assert "permission_denied=False" in log_text
    assert secret not in log_text
    assert "fake-bytes-marker" not in log_text


@pytest.mark.asyncio
async def test_step_trace_logs_max_steps_termination(caplog: pytest.LogCaptureFixture) -> None:
    """The termination-reason line distinguishes max_steps from the model deciding it's done."""

    caplog.set_level(logging.INFO, logger="aria.agents.computer_use_agent")
    tools = ToolRegistry()
    screenshot_tool = _FakeScreenshotTool()
    tools.register(screenshot_tool)
    provider = FakeProvider(
        tool_calls=[ToolCall(name="computer.screenshot", arguments={})],
        stream_tokens=["giving up"],
    )
    agent = ComputerUseAgent(ToolExecutor(tools, PermissionManager()), screenshot_tool, tools, provider, max_steps=2)

    _ = [event async for event in agent.run(_context())]

    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "terminated_by=max_steps" in log_text
