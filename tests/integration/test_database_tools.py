"""Phase 3: the unified query tool (validate -> execute -> normalize).
Requires a seeded, migrated DB — both analytics and olist schemas.
"""

from __future__ import annotations

import pytest

from app.tools.database_tools import run_query


@pytest.mark.asyncio
async def test_run_query_returns_normalized_result():
    result = await run_query("SELECT region_id, name FROM analytics.regions LIMIT 3")
    assert result.ok
    assert result.row_count == len(result.rows)
    assert result.columns == ["region_id", "name"]
    assert result.is_empty is False
    assert result.tables_referenced == ["analytics.regions"]


@pytest.mark.asyncio
async def test_run_query_handles_empty_result_set():
    result = await run_query("SELECT * FROM analytics.regions WHERE 1 = 0")
    assert result.ok
    assert result.is_empty
    assert result.row_count == 0
    assert result.rows == []


@pytest.mark.asyncio
async def test_run_query_rejects_invalid_sql_without_raising():
    result = await run_query("SELECT * FROM nowhere.customers")
    assert not result.ok
    assert result.rejection_reason is not None
    assert result.rows == []


@pytest.mark.asyncio
async def test_run_query_order_by_alias_and_group_by():
    result = await run_query(
        "SELECT segment, COUNT(*) AS n FROM analytics.customers GROUP BY segment ORDER BY n DESC"
    )
    assert result.ok
    assert set(result.columns) == {"segment", "n"}


@pytest.mark.asyncio
async def test_run_query_cte_is_accepted():
    result = await run_query(
        """
        WITH by_segment AS (
            SELECT segment, COUNT(*) AS n FROM analytics.customers GROUP BY segment
        )
        SELECT segment, n FROM by_segment ORDER BY n DESC
        """
    )
    assert result.ok


@pytest.mark.asyncio
async def test_run_query_against_olist_schema():
    result = await run_query(
        "SELECT customer_state, COUNT(*) AS n FROM olist.customers GROUP BY customer_state ORDER BY n DESC LIMIT 3"
    )
    assert result.ok
    assert result.row_count == 3
