"""Authentication and origin checks for HTTP and WebSocket API boundaries."""

from __future__ import annotations

from fastapi import Header, WebSocket
from fastapi.security.utils import get_authorization_scheme_param

from aria.core.config import Settings
from aria.core.errors import AuthorizationError


def require_api_key(settings: Settings, authorization: str | None = Header(default=None)) -> None:
    """Require a bearer token for every HTTP endpoint except health."""

    scheme, token = get_authorization_scheme_param(authorization)
    if scheme.lower() != "bearer" or token != settings.api_key.get_secret_value():
        raise AuthorizationError("Invalid API credentials")


async def authorize_websocket(websocket: WebSocket, settings: Settings) -> None:
    """Reject untrusted origins and WebSocket clients without the configured token."""

    origin = websocket.headers.get("origin")
    allowed = {str(item).rstrip("/") for item in settings.allowed_origins}
    if allowed and origin not in allowed:
        raise AuthorizationError("Origin is not allowed")
    if websocket.query_params.get("token") != settings.api_key.get_secret_value():
        raise AuthorizationError("Invalid API credentials")
