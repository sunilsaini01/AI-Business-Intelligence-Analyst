"""Shared fixtures. Integration/security/API tests assume a live Postgres
reachable via .env (docker compose up postgres, or a local instance) with
`alembic upgrade head` and `scripts/seed_database.py` already run — see
README quick start.
"""

from __future__ import annotations

import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app.db.database import close_analytics_pool, engine
from app.main import app

_TEST_PASSWORD = "correct-horse-battery-staple"


@pytest.fixture(autouse=True)
async def _reset_db_connections():
    """The asyncpg pool and SQLAlchemy engine are module-level singletons
    bound to whichever event loop first created them (fine in production,
    one persistent loop). pytest-asyncio gives each test its own event loop,
    so a pool created in test N's loop is dead by test N+1 ("Event loop is
    closed"). Tear both down after every test so the next one lazily
    recreates them under its own loop.
    """
    yield
    await close_analytics_pool()
    await engine.dispose()


async def _register_and_login(unauthed: AsyncClient) -> str:
    email = f"test-{uuid.uuid4()}@example.com"
    await unauthed.post("/api/v1/auth/register", json={"email": email, "password": _TEST_PASSWORD})
    resp = await unauthed.post("/api/v1/auth/login", json={"email": email, "password": _TEST_PASSWORD})
    return resp.json()["access_token"]


@pytest.fixture
async def unauthenticated_client():
    """No Authorization header at all — for the explicit 401 tests
    (tests/security/test_authorization.py). Every protected route requires
    a token; this is what "no token" looks like."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def client(unauthenticated_client: AsyncClient):
    """Phase 14: authenticated as a fresh, unique user, registered and
    logged in for real through the actual endpoints (not a bypass) — every
    test written against this fixture before auth existed keeps passing
    unchanged, just now implicitly "logged in as some user" rather than
    anonymous. Tests that care about a SPECIFIC identity (ownership, cross-
    user isolation) use `second_user_client` alongside this one instead of
    relying on which user `client` happens to be."""
    token = await _register_and_login(unauthenticated_client)
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test", headers={"Authorization": f"Bearer {token}"}
    ) as ac:
        yield ac


@pytest.fixture
async def second_user_client(unauthenticated_client: AsyncClient):
    """A second, distinct authenticated user (its own registered email) —
    for ownership/403 tests where `client`'s identity being "some user" isn't
    enough; the test needs two DIFFERENT, known-distinct users."""
    token = await _register_and_login(unauthenticated_client)
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test", headers={"Authorization": f"Bearer {token}"}
    ) as ac:
        yield ac
