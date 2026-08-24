"""Analysis Agent orchestration tests (Phase 6) — no LLM, no DB. Builds
AgentState by hand with pre-fabricated SQL evidence (as if the SQL Agent
had already run) and checks that analysis_agent_node dispatches to the
right deterministic analyses and writes a sensible state["analysis_results"].
"""

from __future__ import annotations

import pytest

from app.agents.analysis_agent import analysis_agent_node
from app.graph.state import new_state


def _query_record(rows: list[dict]) -> dict:
    return {
        "text": "SELECT ...",
        "validated_ok": True,
        "rejection_reason": None,
        "rows": rows,
        "row_count": len(rows),
        "exec_ms": 1.0,
    }


@pytest.mark.asyncio
async def test_dimension_value_query_produces_contribution_and_top_n():
    state = new_state("How many customers per region?")
    state["sql_queries"] = [
        _query_record(
            [
                {"region_name": "Central", "customer_count": 94},
                {"region_name": "North", "customer_count": 89},
            ]
        )
    ]
    result = await analysis_agent_node(state)

    analysis = result["analysis_results"]
    assert analysis["insufficient_evidence"] is False
    assert len(analysis["contributions"]) == 1
    assert analysis["contributions"][0]["dimension_col"] == "region_name"
    assert len(analysis["top_n"]) == 1


@pytest.mark.asyncio
async def test_period_value_query_produces_period_comparison():
    state = new_state("Why did revenue decrease?")
    state["intent"] = "diagnostic"
    state["sql_queries"] = [
        _query_record([{"month": "2026-06", "revenue": 100.0}, {"month": "2026-07", "revenue": 70.0}])
    ]
    result = await analysis_agent_node(state)

    analysis = result["analysis_results"]
    assert len(analysis["period_comparisons"]) == 1
    assert analysis["period_comparisons"][0]["direction"] == "decrease"
    # Diagnostic intent but no contribution breakdown available -> still
    # flagged insufficient for a *diagnosis*, even though a comparison exists.
    assert analysis["diagnostic"]["insufficient_evidence"] is True


@pytest.mark.asyncio
async def test_period_and_dimension_query_produces_full_diagnostic():
    state = new_state("Why did revenue decrease in July?")
    state["intent"] = "diagnostic"
    state["sql_queries"] = [
        _query_record([{"month": "2026-06", "revenue": 100.0}, {"month": "2026-07", "revenue": 70.0}]),
        _query_record(
            [
                {"segment": "Enterprise", "month": "2026-06", "revenue": 60.0},
                {"segment": "SMB", "month": "2026-06", "revenue": 40.0},
                {"segment": "Enterprise", "month": "2026-07", "revenue": 10.0},
                {"segment": "SMB", "month": "2026-07", "revenue": 60.0},
            ]
        ),
    ]
    result = await analysis_agent_node(state)

    analysis = result["analysis_results"]
    assert analysis["insufficient_evidence"] is False
    assert len(analysis["contributions"]) == 1  # one dimension column: segment
    diagnostic = analysis["diagnostic"]
    assert diagnostic["insufficient_evidence"] is False
    assert any("Enterprise" in i for i in diagnostic["interpretations"])


@pytest.mark.asyncio
async def test_no_analyzable_evidence_is_insufficient_evidence():
    state = new_state("Some question")
    state["sql_queries"] = [_query_record([{"free_text_note": "hello there"}])]
    result = await analysis_agent_node(state)
    assert result["analysis_results"]["insufficient_evidence"] is True


@pytest.mark.asyncio
async def test_rejected_queries_are_ignored():
    state = new_state("Some question")
    state["sql_queries"] = [
        {
            "text": "bad sql",
            "validated_ok": False,
            "rejection_reason": "nope",
            "rows": [],
            "row_count": 0,
            "exec_ms": 0.0,
        }
    ]
    result = await analysis_agent_node(state)
    assert result["analysis_results"]["insufficient_evidence"] is True


@pytest.mark.asyncio
async def test_empty_sql_queries_is_insufficient_evidence():
    state = new_state("Some question")
    result = await analysis_agent_node(state)
    assert result["analysis_results"]["insufficient_evidence"] is True


@pytest.mark.asyncio
async def test_writes_trace_events():
    state = new_state("Some question")
    result = await analysis_agent_node(state)
    node_names = [t["node"] for t in result["trace"]]
    assert node_names == ["analysis_agent", "analysis_agent"]
