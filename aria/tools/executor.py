"""Validation, permission enforcement, timing, and audit capture for tools."""

from __future__ import annotations

from datetime import UTC, datetime
from time import perf_counter

from pydantic import ValidationError

from aria.core.audit import AuditLogger
from aria.core.errors import PermissionDeniedError, ToolValidationError
from aria.domain.models import ToolExecutionRecord, ToolStatus
from aria.tools.base import ToolContext, redact_arguments
from aria.tools.permissions import PermissionManager
from aria.tools.registry import ToolRegistry


class ToolExecutor:
    """The sole path from an approved plan to an operating-system capability."""

    def __init__(
        self, registry: ToolRegistry, permissions: PermissionManager, audit: AuditLogger | None = None
    ) -> None:
        self._registry = registry
        self._permissions = permissions
        self._audit = audit

    async def execute(self, name: str, raw_arguments: dict[str, object], context: ToolContext) -> ToolExecutionRecord:
        """Run one known tool and return an audit-ready execution record."""

        record = ToolExecutionRecord(tool_name=name, arguments=raw_arguments, status=ToolStatus.RUNNING)
        started = perf_counter()
        try:
            tool = self._registry.get(name)
            # Redact before validation even runs, so a sensitive value never reaches the
            # audited record regardless of whether the input turns out to be valid.
            record.arguments = redact_arguments(raw_arguments, tool.sensitive_fields)
            arguments = tool.input_model.model_validate(raw_arguments)
            await self._permissions.require(tool.permission_level, context)
            record.result = await tool.execute(context, arguments)
            record.status = ToolStatus.SUCCEEDED
        except PermissionDeniedError as error:
            record.status, record.error = ToolStatus.DENIED, str(error)
        except (KeyError, ValidationError, ValueError) as error:
            record.status, record.error = ToolStatus.FAILED, str(ToolValidationError(str(error)))
        except Exception:
            record.status, record.error = ToolStatus.FAILED, "Tool execution failed"
        finally:
            record.completed_at = datetime.now(UTC)
            record.duration_ms = round((perf_counter() - started) * 1000)
            if self._audit:
                await self._audit.record(record)
        return record
