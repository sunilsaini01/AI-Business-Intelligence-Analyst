"""Phase 0 walking skeleton, graph-level (not through the API — see
tests/api/test_analysis.py for the full HTTP round trip). Requires a seeded DB.

Superseded as the *default* graph by Phase 4/5 (see test_supervisor_sql_workflow.py)
but still exercises the original hard-coded-query plumbing proof.
"""

from __future__ import annotations

import pytest

from app.graph.state import new_state
from app.graph.workflow import get_phase0_graph


@pytest.mark.asyncio
async def test_phase0_graph_reaches_report():
    graph = get_phase0_graph()
    result = await graph.ainvoke(new_state("customers by region"))

    assert result["report"] is not None
    assert result["sql_queries"][0]["validated_ok"] is True
    assert len(result["trace"]) == 4  # fetch enter/exit, respond enter/exit
