"""ASGI entry point for local development and production process managers."""

from __future__ import annotations

import uvicorn

from aria.app import create_app
from aria.core.config import get_settings


def run() -> None:
    """Start ARIA using validated environment configuration."""

    settings = get_settings()
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    run()
