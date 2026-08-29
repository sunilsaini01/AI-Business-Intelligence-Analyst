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

from app.core.config import get_settings


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


class LoginRateLimiter:
    """Phase 15, Objective 2 — brute-force protection specifically for
    POST /auth/login, deliberately separate from RateLimitMiddleware above
    (which is IP-only, off by default, and applies to every route). Login
    is unauthenticated by definition, so the bucket key has to be
    something available before credentials are checked: client IP +
    the submitted email, so a single IP can't lock out every OTHER
    account by burning through login attempts against them (attacker's IP
    stays capped per-victim, not globally), and a single leaked/guessed
    email can't be brute-forced from many source IPs faster than the
    per-IP-per-email bucket refills. Every attempt consumes a token,
    success or failure — a successful login must not reset or bypass the
    bucket, or an attacker could interleave correct-password probes with
    throwaway successes to dodge the limit.

    Reuses `_TokenBucket` (the same primitive RateLimitMiddleware already
    uses) rather than a new mechanism — in-memory, per-process, same
    documented tradeoff as that middleware already carries for a
    portfolio-scale deploy (no Redis/shared store): see docs/security.md.
    """

    def __init__(self, *, max_attempts: int, window_seconds: int) -> None:
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        refill_per_sec = max_attempts / window_seconds
        self._buckets: dict[str, _TokenBucket] = defaultdict(
            lambda: _TokenBucket(max_attempts, refill_per_sec)
        )

    @staticmethod
    def key_for(client_ip: str, email: str) -> str:
        return f"{client_ip}:{email.strip().lower()}"

    def allow(self, client_ip: str, email: str) -> bool:
        """Consumes one token and returns whether this attempt is allowed.
        Always consumes — there is no "peek without counting" mode, by
        design (see the class docstring on why success must still count)."""
        return self._buckets[self.key_for(client_ip, email)].consume()


_login_rate_limiter: LoginRateLimiter | None = None


def get_login_rate_limiter() -> LoginRateLimiter:
    """Process-wide singleton, same lazy-cached-global pattern as
    app/core/llm.py::get_llm_client — built once from current settings, so
    every request shares the same bucket state (that's the whole point:
    counting attempts across requests). Tests that need a fresh limiter
    reset this module global directly (see tests/security/
    test_login_rate_limit.py) rather than calling this twice."""
    global _login_rate_limiter
    if _login_rate_limiter is None:
        settings = get_settings()
        _login_rate_limiter = LoginRateLimiter(
            max_attempts=settings.login_rate_limit_max_attempts,
            window_seconds=settings.login_rate_limit_window_seconds,
        )
    return _login_rate_limiter


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
