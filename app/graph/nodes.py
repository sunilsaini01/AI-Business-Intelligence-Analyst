"""Phase 0 walking skeleton (Sec 12): exactly two nodes, no LLM, no critic loop yet.

Proves the plumbing — Postgres -> LangGraph -> FastAPI -> client — before any
agent logic exists. `fetch_node` and `respond_node` will be replaced by the
full Supervisor -> SQL -> Analysis -> [ML] -> Viz -> Critic -> Report chain
described in Sec 1 Fig. 1/2; this file is the seam where that expansion happens.
"""

from __future__ import annotations

import time

from app.db.database import analytics_readonly_connection
from app.graph.state import AgentState, trace_event

# Hard-coded on purpose for Phase 0 — the SQL Agent (Sec 5) will replace this
# with LLM-generated, sqlglot-validated, allow-listed queries (Sec 4).
_PHASE0_QUERY = """
    SELECT r.name AS region, COUNT(*) AS customer_count
    FROM analytics.customers c
    JOIN analytics.regions r ON c.region_id = r.region_id
    GROUP BY r.name
    ORDER BY customer_count DESC
"""


async def fetch_node(state: AgentState) -> AgentState:
    state["trace"].append(trace_event("fetch", "enter"))
    started = time.perf_counter()

    async with analytics_readonly_connection() as conn:
        records = await conn.fetch(_PHASE0_QUERY)

    rows = [dict(r) for r in records]
    exec_ms = (time.perf_counter() - started) * 1000

    state["sql_queries"].append(
        {
            "text": _PHASE0_QUERY.strip(),
            "validated_ok": True,
            "rejection_reason": None,
            "rows": rows,
            "row_count": len(rows),
            "exec_ms": exec_ms,
        }
    )
    state["trace"].append(trace_event("fetch", "exit", duration_ms=exec_ms))
    return state


async def respond_node(state: AgentState) -> AgentState:
    state["trace"].append(trace_event("respond", "enter"))
    started = time.perf_counter()

    rows = state["sql_queries"][-1]["rows"] if state["sql_queries"] else []
    state["analysis_results"]["customers_by_region"] = rows

    state["report"] = {
        "executive_summary": (
            f"Customer counts by region ({len(rows)} regions found)."
            if rows
            else "No data returned for this query."
        ),
        "key_findings": [f"{r['region']}: {r['customer_count']} customers" for r in rows],
        "evidence": [{"source": "sql_queries[0]", "row_count": len(rows)}],
        "recommendations": [],
        "confidence": "Medium",
        "limitations": (
            "Phase 0 walking skeleton: hard-coded query, no Supervisor/Critic/"
            "Report agents yet — this is plumbing verification, not real analysis."
        ),
    }

    state["trace"].append(
        trace_event("respond", "exit", duration_ms=(time.perf_counter() - started) * 1000)
    )
    return state
