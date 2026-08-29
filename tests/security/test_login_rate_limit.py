"""Phase 15, Objective 2 — brute-force protection on POST /auth/login.
Deterministic: no live LLM call anywhere in this file. Uses a small,
monkeypatched threshold (never the 5/300s production default) so these
tests run in milliseconds instead of needing real elapsed time.
"""

from __future__ import annotations

import uuid

import pytest

import app.core.security as security_module
from app.core.config import get_settings
from app.core.security import LoginRateLimiter

_PASSWORD = "correct-horse-battery-staple"


@pytest.fixture
def small_limit(monkeypatch):
    """3 attempts per 60s, and a genuinely fresh limiter (not whatever
    process-wide singleton earlier tests may have already populated)."""
    monkeypatch.setenv("LOGIN_RATE_LIMIT_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("LOGIN_RATE_LIMIT_WINDOW_SECONDS", "60")
    get_settings.cache_clear()
    monkeypatch.setattr(security_module, "_login_rate_limiter", None)
    yield
    monkeypatch.setattr(security_module, "_login_rate_limiter", None)
    get_settings.cache_clear()


async def _register(client, email: str) -> None:
    resp = await client.post("/api/v1/auth/register", json={"email": email, "password": _PASSWORD})
    assert resp.status_code == 201


# --- unit-level: the limiter's own keying/consumption logic ----------------


def test_normal_login_pattern_stays_under_the_limit():
    limiter = LoginRateLimiter(max_attempts=3, window_seconds=60)
    email = "user@example.com"
    assert limiter.allow("1.2.3.4", email) is True
    assert limiter.allow("1.2.3.4", email) is True


def test_repeated_attempts_are_rate_limited_after_the_threshold():
    limiter = LoginRateLimiter(max_attempts=3, window_seconds=60)
    email = "user@example.com"
    assert limiter.allow("1.2.3.4", email) is True
    assert limiter.allow("1.2.3.4", email) is True
    assert limiter.allow("1.2.3.4", email) is True
    assert limiter.allow("1.2.3.4", email) is False  # 4th attempt, over the limit of 3


def test_a_successful_attempt_still_consumes_a_token_not_a_free_pass():
    """The token bucket doesn't distinguish success from failure at all —
    `allow()` is called once per attempt regardless of outcome (the route
    handler checks it before touching the DB), so this is really testing
    that consumption is unconditional, which is what makes a success
    unable to "reset" or bypass the limit."""
    limiter = LoginRateLimiter(max_attempts=3, window_seconds=60)
    email = "user@example.com"
    assert limiter.allow("1.2.3.4", email) is True
    assert limiter.allow("1.2.3.4", email) is True
    assert limiter.allow("1.2.3.4", email) is True  # the "successful" 3rd attempt, hypothetically
    assert limiter.allow("1.2.3.4", email) is False  # still blocked right after — no reset happened


def test_different_emails_from_the_same_ip_do_not_share_a_bucket():
    limiter = LoginRateLimiter(max_attempts=1, window_seconds=60)
    assert limiter.allow("1.2.3.4", "victim-a@example.com") is True
    assert limiter.allow("1.2.3.4", "victim-a@example.com") is False  # A is now exhausted
    assert limiter.allow("1.2.3.4", "victim-b@example.com") is True  # B is unaffected


def test_different_ips_for_the_same_email_do_not_share_a_bucket():
    limiter = LoginRateLimiter(max_attempts=1, window_seconds=60)
    assert limiter.allow("1.2.3.4", "user@example.com") is True
    assert limiter.allow("1.2.3.4", "user@example.com") is False  # this IP is now exhausted for this email
    assert limiter.allow("9.9.9.9", "user@example.com") is True  # a different IP is not


def test_key_for_normalizes_email_case_and_whitespace():
    """Login itself lowercases/strips email (app/services/auth_service.py)
    — the limiter's key must match that normalization, or "User@x.com" and
    "user@x.com" would get separate buckets, defeating the point."""
    assert LoginRateLimiter.key_for("1.2.3.4", "  User@Example.com  ") == LoginRateLimiter.key_for(
        "1.2.3.4", "user@example.com"
    )


# --- HTTP-level: the real /auth/login route ---------------------------------


@pytest.mark.asyncio
async def test_login_succeeds_normally_when_under_the_limit(unauthenticated_client, small_limit):
    email = f"normal-{uuid.uuid4()}@example.com"
    await _register(unauthenticated_client, email)
    resp = await unauthenticated_client.post("/api/v1/auth/login", json={"email": email, "password": _PASSWORD})
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_429_is_returned_after_the_configured_threshold(unauthenticated_client, small_limit):
    email = f"bruteforce-{uuid.uuid4()}@example.com"
    await _register(unauthenticated_client, email)

    # 3 wrong-password attempts (the configured max) each get a normal 401 —
    # the limit itself hasn't kicked in yet on these.
    for _ in range(3):
        resp = await unauthenticated_client.post(
            "/api/v1/auth/login", json={"email": email, "password": "not-the-password"}
        )
        assert resp.status_code == 401

    # The 4th attempt is rejected by the limiter before credentials are
    # even checked — 429, not another 401.
    resp = await unauthenticated_client.post(
        "/api/v1/auth/login", json={"email": email, "password": "not-the-password"}
    )
    assert resp.status_code == 429


@pytest.mark.asyncio
async def test_429_response_includes_a_retry_after_header(unauthenticated_client, small_limit):
    email = f"retry-after-{uuid.uuid4()}@example.com"
    await _register(unauthenticated_client, email)
    for _ in range(3):
        await unauthenticated_client.post("/api/v1/auth/login", json={"email": email, "password": "wrong"})
    resp = await unauthenticated_client.post("/api/v1/auth/login", json={"email": email, "password": "wrong"})
    assert resp.status_code == 429
    assert resp.headers.get("retry-after") == "60"


@pytest.mark.asyncio
async def test_a_successful_login_does_not_bypass_future_rate_limiting(unauthenticated_client, small_limit):
    email = f"success-then-limit-{uuid.uuid4()}@example.com"
    await _register(unauthenticated_client, email)

    # 2 failed attempts, then 1 successful one — 3 total, exactly the limit.
    await unauthenticated_client.post("/api/v1/auth/login", json={"email": email, "password": "wrong"})
    await unauthenticated_client.post("/api/v1/auth/login", json={"email": email, "password": "wrong"})
    ok_resp = await unauthenticated_client.post("/api/v1/auth/login", json={"email": email, "password": _PASSWORD})
    assert ok_resp.status_code == 200

    # The very next attempt — even a CORRECT one — is still blocked: the
    # earlier success did not reset the bucket.
    next_resp = await unauthenticated_client.post(
        "/api/v1/auth/login", json={"email": email, "password": _PASSWORD}
    )
    assert next_resp.status_code == 429


@pytest.mark.asyncio
async def test_rate_limiting_one_email_does_not_affect_a_different_email(unauthenticated_client, small_limit):
    victim_email = f"victim-{uuid.uuid4()}@example.com"
    other_email = f"other-{uuid.uuid4()}@example.com"
    await _register(unauthenticated_client, victim_email)
    await _register(unauthenticated_client, other_email)

    for _ in range(3):
        await unauthenticated_client.post("/api/v1/auth/login", json={"email": victim_email, "password": "wrong"})
    blocked = await unauthenticated_client.post(
        "/api/v1/auth/login", json={"email": victim_email, "password": "wrong"}
    )
    assert blocked.status_code == 429

    unaffected = await unauthenticated_client.post(
        "/api/v1/auth/login", json={"email": other_email, "password": _PASSWORD}
    )
    assert unaffected.status_code == 200


@pytest.mark.asyncio
async def test_rate_limit_never_reveals_whether_the_email_is_registered(unauthenticated_client, small_limit):
    """The 429 body/message must be identical for a real vs. a made-up
    email — otherwise the rate limiter itself becomes an account-
    enumeration oracle even though /auth/login's own 401 already avoids
    that (Phase 14)."""
    registered = f"real-{uuid.uuid4()}@example.com"
    await _register(unauthenticated_client, registered)
    unregistered = f"nobody-{uuid.uuid4()}@example.com"

    for _ in range(3):
        await unauthenticated_client.post("/api/v1/auth/login", json={"email": registered, "password": "wrong"})
    resp_real = await unauthenticated_client.post(
        "/api/v1/auth/login", json={"email": registered, "password": "wrong"}
    )

    for _ in range(3):
        await unauthenticated_client.post("/api/v1/auth/login", json={"email": unregistered, "password": "wrong"})
    resp_fake = await unauthenticated_client.post(
        "/api/v1/auth/login", json={"email": unregistered, "password": "wrong"}
    )

    assert resp_real.status_code == resp_fake.status_code == 429
    assert resp_real.json() == resp_fake.json()


@pytest.mark.asyncio
async def test_429_body_reveals_no_internal_rate_limit_state(unauthenticated_client, small_limit):
    email = f"opaque-{uuid.uuid4()}@example.com"
    await _register(unauthenticated_client, email)
    for _ in range(3):
        await unauthenticated_client.post("/api/v1/auth/login", json={"email": email, "password": "wrong"})
    resp = await unauthenticated_client.post("/api/v1/auth/login", json={"email": email, "password": "wrong"})
    assert resp.status_code == 429
    body = resp.json()
    assert set(body.keys()) == {"detail"}
    assert body["detail"] == "Too many login attempts. Please try again later."
    # no token counts, timestamps, bucket keys, or the submitted email echoed back
    for leak_marker in (email, "token", "bucket", "capacity", "refill"):
        assert leak_marker.lower() not in str(body).lower()


@pytest.mark.asyncio
async def test_login_rate_limiting_does_not_affect_normal_authenticated_requests(client, small_limit):
    """`client` (tests/conftest.py) is already logged in — repeated calls
    to an authenticated, non-login endpoint must never be affected by the
    login limiter (it's wired ONLY into POST /auth/login)."""
    for _ in range(6):  # well past the 3-attempt login threshold
        resp = await client.post("/api/v1/analyze", json={"question": "How many customers?"})
        assert resp.status_code == 202


@pytest.mark.asyncio
async def test_login_rate_limiting_can_be_disabled_via_settings(unauthenticated_client, monkeypatch):
    monkeypatch.setenv("LOGIN_RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("LOGIN_RATE_LIMIT_MAX_ATTEMPTS", "1")
    get_settings.cache_clear()
    monkeypatch.setattr(security_module, "_login_rate_limiter", None)
    try:
        email = f"disabled-{uuid.uuid4()}@example.com"
        await _register(unauthenticated_client, email)
        for _ in range(5):
            resp = await unauthenticated_client.post(
                "/api/v1/auth/login", json={"email": email, "password": "wrong"}
            )
            assert resp.status_code == 401  # never 429 — the limiter is off
    finally:
        monkeypatch.setattr(security_module, "_login_rate_limiter", None)
        get_settings.cache_clear()
