"""Built-in desktop capability plugin."""

from __future__ import annotations

from aria.core.config import Settings
from aria.plugins.base import Plugin
from aria.tools.desktop import OpenApplicationTool
from aria.tools.registry import ToolRegistry


class DesktopPlugin(Plugin):
    """Registers constrained Windows desktop capabilities."""

    id = "desktop"
    name = "Desktop"
    version = "0.1.0"
    description = "Allowlisted Windows desktop actions."

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def register(self, tools: ToolRegistry) -> None:
        """Register desktop tools from user-owned configuration."""

        tools.register(OpenApplicationTool(self._settings.application_allowlist))
