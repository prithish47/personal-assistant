"""Computer-use capability plugin: screenshot, mouse, and keyboard tools."""

from __future__ import annotations

from aria.core.config import Settings
from aria.plugins.base import Plugin
from aria.tools.computer_use import (
    HotkeyTool,
    KeyboardPressTool,
    KeyboardTypeTool,
    MouseClickTool,
    MouseMoveTool,
    ScreenshotTool,
    ScrollTool,
)
from aria.tools.registry import ToolRegistry


class ComputerUsePlugin(Plugin):
    """Registers screenshot and input-control tools; open_application is DesktopPlugin's."""

    id = "computer_use"
    name = "Computer Use"
    version = "0.1.0"
    description = "Screenshot capture and controlled mouse/keyboard actions."

    def __init__(self, settings: Settings) -> None:
        # Held so ComputerUseAgent can read the last captured screenshot's bytes directly,
        # without a second capture and without putting image data in the audited tool result.
        self.screenshot_tool = ScreenshotTool(
            max_dimension=settings.screenshot_max_dimension, jpeg_quality=settings.screenshot_jpeg_quality
        )

    def register(self, tools: ToolRegistry) -> None:
        """Register screenshot and input-control tools from user-owned configuration."""

        tools.register(self.screenshot_tool)
        tools.register(MouseMoveTool())
        tools.register(MouseClickTool())
        tools.register(KeyboardTypeTool())
        tools.register(KeyboardPressTool())
        tools.register(HotkeyTool())
        tools.register(ScrollTool())
