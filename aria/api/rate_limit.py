"""Small in-process fixed-window limiter suitable for a single desktop instance."""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from time import monotonic

from aria.core.errors import AuthorizationError


class RateLimiter:
    """Limit requests per client identity without external shared infrastructure."""

    def __init__(self, maximum: int, window_seconds: float = 60.0) -> None:
        self._maximum = maximum
        self._window_seconds = window_seconds
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check(self, identity: str) -> None:
        """Raise when a client exceeds its configured request budget."""

        now = monotonic()
        async with self._lock:
            events = self._events[identity]
            while events and now - events[0] > self._window_seconds:
                events.popleft()
            if len(events) >= self._maximum:
                raise AuthorizationError("Rate limit exceeded")
            events.append(now)
