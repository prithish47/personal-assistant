"""Plugin contract: plugins only register capabilities into explicit registries."""

from __future__ import annotations

from abc import ABC, abstractmethod

from aria.tools.registry import ToolRegistry


class Plugin(ABC):
    """An installable, declarative unit of ARIA capability."""

    id: str
    name: str
    version: str
    description: str

    @abstractmethod
    def register(self, tools: ToolRegistry) -> None:
        """Register plugin capabilities during application startup."""
