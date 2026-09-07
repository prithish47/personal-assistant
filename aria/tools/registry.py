"""Registry for installed tools with duplicate-name protection."""

from __future__ import annotations

from typing import Any

from aria.tools.base import Tool


class ToolRegistry:
    """Owns the set of capabilities visible to planning and execution."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool[Any]] = {}

    def register(self, tool: Tool[Any]) -> None:
        """Register one uniquely named tool."""

        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool[Any]:
        """Return a registered tool or raise a clear lookup error."""

        try:
            return self._tools[name]
        except KeyError as error:
            raise KeyError(f"Unknown tool: {name}") from error

    def find(self, name: str) -> Tool[Any] | None:
        """Return a registered tool, or None if unknown -- for callers that inspect before executing."""

        return self._tools.get(name)

    def schemas(self) -> list[dict[str, object]]:
        """Expose only declarative schemas to the model layer."""

        return [
            {"name": tool.name, "description": tool.description, "parameters": tool.schema()}
            for tool in self._tools.values()
        ]

    def list(self) -> list[Tool[Any]]:
        """Return installed tools for the API and plugin manager."""

        return list(self._tools.values())
