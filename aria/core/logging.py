"""Structured logging configuration for services and local development."""

from __future__ import annotations

import logging


def configure_logging(level: str) -> None:
    """Configure a predictable process-wide logging format."""

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
