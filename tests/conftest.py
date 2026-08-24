"""Shared fixtures. Integration/security/API tests assume a live Postgres
reachable via .env (docker compose up postgres, or a local instance) with
`alembic upgrade head` and `scripts/seed_database.py` already run — see
README quick start.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.db.database import close_analytics_pool, engine
from app.main import app


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


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
