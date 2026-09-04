"""Windows desktop tools that operate exclusively on configured allowlists."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from aria.core.errors import ToolValidationError
from aria.domain.models import PermissionLevel
from aria.tools.base import Tool, ToolContext


class OpenApplicationInput(BaseModel):
    """The symbolic identifier of an approved application."""

    application_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")


class OpenApplicationTool(Tool[OpenApplicationInput]):
    """Open a configured Windows application without invoking a shell."""

    name = "desktop.open_application"
    description = "Open an application explicitly configured by the user."
    permission_level = PermissionLevel.USER_CONFIRM
    input_model = OpenApplicationInput

    def __init__(self, applications: dict[str, Path]) -> None:
        self._applications = {name.lower(): path.resolve() for name, path in applications.items()}

    async def execute(self, context: ToolContext, arguments: OpenApplicationInput) -> dict[str, str]:
        """Open one allowlisted executable through Windows' native launcher."""

        path = self._applications.get(arguments.application_id.lower())
        if path is None or not path.is_file():
            raise ToolValidationError("Application is not configured or no longer exists")
        os.startfile(str(path))
        return {"application_id": arguments.application_id, "status": "opened"}
