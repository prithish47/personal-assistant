"""Tests for authentication at the public API boundary."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import SecretStr

from aria.app import create_app
from aria.core.config import Settings


def test_tools_endpoint_requires_bearer_token(tmp_path: Path) -> None:
    """The tool catalog is not exposed to unauthenticated callers."""

    settings = Settings(api_key=SecretStr("a" * 40), memory_persist_directory=tmp_path / "chroma")
    with TestClient(create_app(settings)) as client:
        assert client.get("/v1/tools").status_code == 401
        response = client.get("/v1/tools", headers={"Authorization": f"Bearer {'a' * 40}"})

    assert response.status_code == 200
    assert response.json()[0]["name"] == "desktop.open_application"
