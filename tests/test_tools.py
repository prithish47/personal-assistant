"""Tests for the shell-free tool execution boundary."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from aria.domain.models import PermissionLevel, ToolStatus
from aria.tools.base import Tool, ToolContext
from aria.tools.executor import ToolExecutor
from aria.tools.permissions import PermissionManager
from aria.tools.registry import ToolRegistry


class EchoInput(BaseModel):
    """Validated test input."""

    value: str


class EchoTool(Tool[EchoInput]):
    """A test capability that never touches the operating system."""

    name = "test.echo"
    description = "Echo a value."
    permission_level = PermissionLevel.USER_CONFIRM
    input_model = EchoInput

    async def execute(self, context: ToolContext, arguments: EchoInput) -> dict[str, str]:
        """Return the validated value."""

        return {"value": arguments.value}


@pytest.fixture
def executor() -> ToolExecutor:
    """Create an isolated executor with a single test tool."""

    registry = ToolRegistry()
    registry.register(EchoTool())
    return ToolExecutor(registry, PermissionManager())


@pytest.mark.asyncio
async def test_user_confirmation_is_required(executor: ToolExecutor) -> None:
    """User-impacting tools must not run without an approval token."""

    result = await executor.execute("test.echo", {"value": "hello"}, ToolContext(user_id="u", conversation_id="c"))

    assert result.status is ToolStatus.DENIED
    assert result.result is None


@pytest.mark.asyncio
async def test_valid_approved_tool_executes(executor: ToolExecutor) -> None:
    """The executor runs only validated and approved structured calls."""

    result = await executor.execute(
        "test.echo",
        {"value": "hello"},
        ToolContext(user_id="u", conversation_id="c", approval_token="approved"),
    )

    assert result.status is ToolStatus.SUCCEEDED
    assert result.result == {"value": "hello"}
