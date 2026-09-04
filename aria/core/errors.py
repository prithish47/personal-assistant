"""Stable domain errors translated to structured API responses at the edge."""


class AriaError(Exception):
    """Base exception for expected ARIA failures."""


class AuthorizationError(AriaError):
    """The caller did not authenticate or lacks a required capability."""


class PermissionDeniedError(AriaError):
    """A tool call was valid but not approved."""


class ToolValidationError(AriaError):
    """Tool input does not satisfy its schema or application allowlist."""


class ProviderError(AriaError):
    """An LLM provider failed or returned an unsupported response."""
