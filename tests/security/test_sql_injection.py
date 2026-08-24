"""Blueprint Sec 10 MUST-HAVE: the SQL safety layer is unverified, not built,
without these four cases passing. Requires a live, migrated Postgres (Layer 3
re-introspects the schema live — see docs/security.md).
"""

from __future__ import annotations

import pytest

from app.tools.database_tools import validate_sql


@pytest.mark.asyncio
async def test_stacked_query_rejected():
    result = await validate_sql("SELECT 1; DROP TABLE analytics.orders;")
    assert not result.ok


@pytest.mark.asyncio
async def test_comment_obfuscated_keyword_rejected():
    result = await validate_sql("SEL/**/ECT * FROM analytics.customers")
    assert not result.ok


@pytest.mark.asyncio
async def test_write_disguised_as_cte_rejected():
    result = await validate_sql(
        "WITH deleted AS (DELETE FROM analytics.orders RETURNING *) SELECT * FROM deleted"
    )
    assert not result.ok


@pytest.mark.asyncio
async def test_table_outside_allowlist_rejected():
    result = await validate_sql("SELECT * FROM app.users")
    assert not result.ok


@pytest.mark.asyncio
async def test_valid_select_accepted_with_limit_injected():
    result = await validate_sql("SELECT customer_id, segment FROM analytics.customers")
    assert result.ok
    assert "LIMIT" in result.sql.upper()


@pytest.mark.asyncio
async def test_oversized_limit_clamped():
    result = await validate_sql("SELECT customer_id FROM analytics.customers LIMIT 999999")
    assert result.ok
    assert "999999" not in result.sql
