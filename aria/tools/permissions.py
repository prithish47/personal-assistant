"""Explicit approval policy for tool execution."""

from __future__ import annotations

from aria.core.errors import PermissionDeniedError
from aria.domain.models import PermissionLevel
from aria.tools.base import ToolContext


class PermissionManager:
    """Enforces a default-deny policy for user-impacting actions."""

    async def require(self, level: PermissionLevel, context: ToolContext) -> None:
        """Require an approval token for anything beyond read-only access."""

        if level is PermissionLevel.READ:
            return
        if not context.approval_token:
            raise PermissionDeniedError("This tool requires explicit user approval")
        if level is PermissionLevel.ELEVATED:
            raise PermissionDeniedError("Elevated tools are disabled by policy")
