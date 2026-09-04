"""Application composition root and secure FastAPI transport adapters."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
from fastapi import Depends, FastAPI, Header, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from aria.api.rate_limit import RateLimiter
from aria.api.schemas import HealthResponse
from aria.api.security import authorize_websocket, require_api_key
from aria.core.audit import AuditLogger
from aria.core.config import Settings
from aria.core.errors import AriaError, AuthorizationError
from aria.core.logging import configure_logging
from aria.memory.repository import MemoryRepository
from aria.memory.service import MemoryService
from aria.planning.planner import Planner
from aria.plugins.desktop import DesktopPlugin
from aria.plugins.manager import PluginManager
from aria.providers.base import LLMProvider
from aria.providers.factory import create_provider
from aria.services.agent import AgentService
from aria.tools.executor import ToolExecutor
from aria.tools.permissions import PermissionManager
from aria.tools.registry import ToolRegistry


@dataclass
class Services:
    """Explicit runtime dependencies available at the transport boundary."""

    provider: LLMProvider
    planner: Planner
    tools: ToolRegistry
    executor: ToolExecutor
    plugins: PluginManager
    agent: AgentService
    rate_limiter: RateLimiter


def create_app(settings: Settings) -> FastAPI:
    """Build an application with no import-time I/O or mutable global services."""

    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            tools = ToolRegistry()
            plugins = PluginManager(tools)
            plugins.load(DesktopPlugin(settings))
            plugins.discover()
            provider = create_provider(settings, client)
            memory_repository = MemoryRepository(settings.memory_database)
            await memory_repository.initialize()
            planner = Planner(provider, tools)
            executor = ToolExecutor(tools, PermissionManager(), AuditLogger(settings.data_dir / "audit.jsonl"))
            memory = MemoryService(memory_repository, provider)
            app.state.services = Services(
                provider=provider,
                planner=planner,
                tools=tools,
                executor=executor,
                plugins=plugins,
                agent=AgentService(planner, executor, memory, provider),
                rate_limiter=RateLimiter(settings.rate_limit_per_minute),
            )
            yield

    app = FastAPI(title="ARIA", version="0.1.0", lifespan=lifespan)
    allowed_origins = [str(origin).rstrip("/") for origin in settings.allowed_origins]
    if allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type"],
        )

    @app.exception_handler(AriaError)
    async def aria_error_handler(_: Request, error: AriaError) -> Response:
        status = 401 if isinstance(error, AuthorizationError) else 400
        return JSONResponse(
            status_code=status, content={"error": {"code": error.__class__.__name__, "message": str(error)}}
        )

    def protected(authorization: str | None = Header(default=None)) -> None:
        """FastAPI dependency that closes over immutable application settings."""

        require_api_key(settings, authorization)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        ready = await app.state.services.provider.healthcheck()
        return HealthResponse(status="ok" if ready else "degraded", provider_ready=ready)

    @app.get("/v1/tools", dependencies=[Depends(protected)])
    async def list_tools() -> list[dict[str, object]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "permission_level": tool.permission_level.value,
                "schema": tool.schema(),
            }
            for tool in app.state.services.tools.list()
        ]

    @app.websocket("/v1/ws")
    async def agent_websocket(websocket: WebSocket) -> None:
        try:
            await authorize_websocket(websocket, settings)
            await websocket.accept()
            while True:
                identity = websocket.client.host if websocket.client else "unknown"
                await app.state.services.rate_limiter.check(identity)
                payload = await websocket.receive_json()
                goal = str(payload.get("content", "")).strip()
                if not goal:
                    await websocket.send_json(
                        {"type": "error", "code": "invalid_request", "message": "content is required"}
                    )
                    continue
                async for event in app.state.services.agent.run(
                    "local-user", str(payload.get("conversation_id", "default")), goal, payload.get("approval_token")
                ):
                    await websocket.send_json(event)
        except AuthorizationError:
            await websocket.close(code=1008)
        except WebSocketDisconnect:
            return

    return app
