"""Two separate connections on purpose (Sec 2, Sec 4):

- `app` role via SQLAlchemy async engine — app schema, read/write, your code only.
- `readonly_analyst` role via a dedicated asyncpg pool — analytics schema, SELECT
  only, execute-only path for LLM-generated SQL. Never routed through the ORM:
  the query builder is for *your* code, not for executing arbitrary text.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import asyncpg
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings

settings = get_settings()

# --- App engine (SQLAlchemy, app schema) ---
engine = create_async_engine(settings.database_url, pool_size=10, max_overflow=5, echo=False)
async_session_factory = async_sessionmaker(engine, expire_on_commit=False)


async def get_db_session() -> AsyncIterator[AsyncSession]:
    async with async_session_factory() as session:
        yield session


# --- Analytical pool (raw asyncpg, readonly_analyst role, analytics schema) ---
_analytics_pool: asyncpg.Pool | None = None


async def init_analytics_pool() -> None:
    global _analytics_pool
    if _analytics_pool is None:
        _analytics_pool = await asyncpg.create_pool(
            dsn=settings.analytics_database_url,
            min_size=2,
            max_size=5,
        )


async def close_analytics_pool() -> None:
    global _analytics_pool
    if _analytics_pool is not None:
        await _analytics_pool.close()
        _analytics_pool = None


@asynccontextmanager
async def analytics_readonly_connection() -> AsyncIterator[asyncpg.Connection]:
    """READ ONLY transaction + statement_timeout, per connection (Sec 4, Layer 5).

    Used exclusively by app/tools/database_tools.py to execute validated,
    allow-listed SELECTs. Nothing else should reach for this pool.
    """
    if _analytics_pool is None:
        await init_analytics_pool()
    assert _analytics_pool is not None
    async with _analytics_pool.acquire() as conn:
        async with conn.transaction(readonly=True):
            await conn.execute(f"SET LOCAL statement_timeout = '{settings.sql_statement_timeout_ms}ms'")
            yield conn


async def db_healthy() -> bool:
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
