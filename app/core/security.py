"""Sec 9: rate limiting scaffold (per-session-id token bucket) and error-surface helpers.

Not enforced hard for a portfolio deploy, but the shape is here so it's a config
flag away from being real — see RateLimitMiddleware.enabled.
"""

from __future__ import annotations

import time
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


class _TokenBucket:
    def __init__(self, capacity: int, refill_per_sec: float) -> None:
        self.capacity = capacity
        self.tokens = float(capacity)
        self.refill_per_sec = refill_per_sec
        self.updated_at = time.monotonic()

    def consume(self, amount: float = 1.0) -> bool:
        now = time.monotonic()
        elapsed = now - self.updated_at
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_sec)
        self.updated_at = now
        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Keyed by client host today; swap the key function for a real session id once auth lands."""

    def __init__(self, app, *, enabled: bool = False, capacity: int = 20, refill_per_sec: float = 0.5) -> None:
        super().__init__(app)
        self.enabled = enabled
        self.capacity = capacity
        self.refill_per_sec = refill_per_sec
        self._buckets: dict[str, _TokenBucket] = defaultdict(
            lambda: _TokenBucket(capacity, refill_per_sec)
        )

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self.enabled:
            return await call_next(request)
        key = request.client.host if request.client else "unknown"
        if not self._buckets[key].consume():
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
        return await call_next(request)


def safe_error_response(request_id: str) -> dict[str, str]:
    """Generic message for the client; full detail belongs in structured logs only (Sec 9)."""
    return {"detail": "An internal error occurred.", "request_id": request_id}
