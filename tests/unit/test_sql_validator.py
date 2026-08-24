"""Parse-layer checks only (no DB) — schema allow-list checks that need a live
DB introspection live in tests/security/test_sql_injection.py instead.
"""

from __future__ import annotations

import pytest

from app.tools.database_tools import validate_sql


@pytest.mark.asyncio
async def test_rejects_multi_statement():
    result = await validate_sql("SELECT 1; SELECT 2;")
    assert not result.ok
    assert "one statement" in result.rejection_reason


@pytest.mark.asyncio
async def test_rejects_non_select_root():
    result = await validate_sql("UPDATE analytics.customers SET status = 'churned'")
    assert not result.ok


@pytest.mark.asyncio
async def test_rejects_unparseable_sql():
    result = await validate_sql("SELEKT * FORM nowhere")
    assert not result.ok
