"""Plugin discovery and lifecycle management."""

from __future__ import annotations

from importlib.metadata import entry_points

from aria.plugins.base import Plugin
from aria.tools.registry import ToolRegistry


class PluginManager:
    """Loads enabled plugins deterministically during application composition."""

    def __init__(self, tools: ToolRegistry) -> None:
        self._tools = tools
        self._plugins: dict[str, Plugin] = {}

    def load(self, plugin: Plugin) -> None:
        """Register one uniquely identified plugin."""

        if plugin.id in self._plugins:
            raise ValueError(f"Plugin already loaded: {plugin.id}")
        plugin.register(self._tools)
        self._plugins[plugin.id] = plugin

    def list(self) -> list[Plugin]:
        """Return loaded plugins for management interfaces."""

        return list(self._plugins.values())

    def discover(self) -> None:
        """Load third-party plugins registered under ARIA's entry-point group."""

        for entry_point in entry_points(group="aria.plugins"):
            plugin_class = entry_point.load()
            self.load(plugin_class())
