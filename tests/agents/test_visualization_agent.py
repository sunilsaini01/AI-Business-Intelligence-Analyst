"""Visualization Agent orchestration tests (Phase 7) — no LLM, no DB. Builds
AgentState by hand with pre-fabricated analysis_results/sql_queries (as if
analysis_agent had already run) and checks visualization_agent_node's
priority/anti-spam logic and state["charts"] output.
"""

from __future__ import annotations

import pytest

from app.agents.visualization_agent import visualization_agent_node
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
async def test_no_analysis_results_and_no_evidence_produces_no_charts():
    state = new_state("some question")
    result = await visualization_agent_node(state)
    assert result["charts"] == []


@pytest.mark.asyncio
async def test_falls_back_to_table_when_analysis_results_empty_but_evidence_exists():
    state = new_state("some question")
    state["sql_queries"] = [_query_record([{"free_text": "unstructured note"}])]
    result = await visualization_agent_node(state)
    assert len(result["charts"]) == 1
    assert result["charts"][0]["chart_type"] == "table"


@pytest.mark.asyncio
async def test_trend_only_produces_one_line_chart():
    state = new_state("show me the monthly trend")
    state["analysis_results"] = {
        "period_comparisons": [],
        "trends": [
            {
                "ok": True,
                "period_col": "month",
                "value_col": "revenue",
                "points": [{"period": f"2026-0{i}", "value": float(i), "pct_change_from_prior": None} for i in range(1, 4)],
                "min_value": 1.0,
                "max_value": 3.0,
                "mean_value": 2.0,
                "direction": "increasing",
                "insufficient_evidence": False,
                "reason": None,
            }
        ],
        "contributions": [],
        "top_n": [],
        "distributions": [],
        "diagnostic": None,
        "insufficient_evidence": False,
        "reason": None,
    }
    result = await visualization_agent_node(state)
    assert len(result["charts"]) == 1
    assert result["charts"][0]["chart_type"] == "line"


@pytest.mark.asyncio
async def test_diagnostic_with_two_dominant_dimensions_caps_at_three_charts_not_spam():
    """Mirrors the July benchmark shape: 1 period comparison + 2 dominant
    contribution breakdowns (segment, region) -> 3 charts, never one chart
    per analysis_results entry (which would also include a redundant trend).
    """
    state = new_state("Why did revenue decrease in July?")
    state["intent"] = "diagnostic"

    def _contribution(dim: str, dominant_pct: float) -> dict:
        return {
            "ok": True,
            "dimension_col": dim,
            "value_col": "revenue",
            "total_current": 70.0,
            "total_prior": 100.0,
            "total_change": -30.0,
            "baseline_period": "2026-06",
            "current_period": "2026-07",
            "contributors": [
                {
                    "group": "A", "current_value": 10.0, "prior_value": 60.0, "change": -50.0,
                    "pct_change": -83.3, "pct_of_total_current": 14.3, "pct_of_total_change": dominant_pct, "rank": 1,
                },
                {
                    "group": "B", "current_value": 60.0, "prior_value": 40.0, "change": 20.0,
                    "pct_change": 50.0, "pct_of_total_current": 85.7, "pct_of_total_change": None, "rank": 2,
                },
            ],
            "insufficient_evidence": False,
            "reason": None,
        }

    state["analysis_results"] = {
        "period_comparisons": [
            {
                "ok": True, "period_col": "month", "value_col": "revenue",
                "baseline_period": "2026-06", "current_period": "2026-07",
                "baseline_value": 100.0, "current_value": 70.0, "absolute_change": -30.0,
                "percentage_change": -30.0, "direction": "decrease", "note": None,
                "insufficient_evidence": False, "reason": None,
            }
        ],
        "trends": [
            {  # present but must NOT get its own chart in the diagnostic path
                "ok": True, "period_col": "month", "value_col": "revenue",
                "points": [{"period": "2026-06", "value": 100.0, "pct_change_from_prior": None},
                           {"period": "2026-07", "value": 70.0, "pct_change_from_prior": -30.0}],
                "min_value": 70.0, "max_value": 100.0, "mean_value": 85.0, "direction": "decreasing",
                "insufficient_evidence": False, "reason": None,
            }
        ],
        "contributions": [_contribution("segment", 100.0), _contribution("region", 65.0)],
        "top_n": [],
        "distributions": [],
        "diagnostic": {
            "ok": True,
            "facts": ["revenue went from 100.00 in 2026-06 to 70.00 in 2026-07 (decrease (-30.0%))."],
            "interpretations": ["By segment, 'A' appears to be the dominant contributor..."],
            "limitations": [],
            "insufficient_evidence": False,
            "reason": None,
        },
        "insufficient_evidence": False,
        "reason": None,
    }

    result = await visualization_agent_node(state)
    charts = result["charts"]
    assert len(charts) == 3  # comparison + segment + region, NOT +1 for the trend too
    assert charts[0]["chart_type"] == "bar"
    assert charts[1]["chart_type"] == "horizontal_bar"
    assert charts[2]["chart_type"] == "horizontal_bar"
    chart_types_seen = {c["source_analysis"] for c in charts}
    assert "trend" not in chart_types_seen  # the redundant trend was skipped


@pytest.mark.asyncio
async def test_diagnostic_with_no_dominant_contributor_still_shows_overall_comparison():
    state = new_state("Why did revenue change?")
    state["intent"] = "diagnostic"
    state["analysis_results"] = {
        "period_comparisons": [
            {
                "ok": True, "period_col": "month", "value_col": "revenue",
                "baseline_period": "2026-06", "current_period": "2026-07",
                "baseline_value": 100.0, "current_value": 90.0, "absolute_change": -10.0,
                "percentage_change": -10.0, "direction": "decrease", "note": None,
                "insufficient_evidence": False, "reason": None,
            }
        ],
        "trends": [],
        "contributions": [],
        "top_n": [],
        "distributions": [],
        "diagnostic": {
            "ok": True,
            "facts": ["revenue went from 100.00 in 2026-06 to 90.00 in 2026-07 (decrease (-10.0%))."],
            "interpretations": [],
            "limitations": ["No segment/region/category breakdown with matching prior-period data was available..."],
            "insufficient_evidence": True,
            "reason": "A change was confirmed, but no contribution breakdown could be computed.",
        },
        "insufficient_evidence": False,
        "reason": None,
    }
    result = await visualization_agent_node(state)
    # diagnose_decline found no dominant contributor (insufficient_evidence
    # True on the diagnostic composite), but the underlying period_comparison
    # is still valid — the overall June-vs-July bar chart must not be lost
    # just because no single segment/region explained the change.
    assert len(result["charts"]) == 1
    assert result["charts"][0]["chart_type"] == "bar"


@pytest.mark.asyncio
async def test_writes_trace_events():
    state = new_state("some question")
    result = await visualization_agent_node(state)
    node_names = [t["node"] for t in result["trace"]]
    assert node_names == ["visualization_agent", "visualization_agent"]


@pytest.mark.asyncio
async def test_never_produces_more_than_max_charts():
    state = new_state("some question")
    state["analysis_results"] = {
        "period_comparisons": [],
        "trends": [
            {
                "ok": True, "period_col": "month", "value_col": f"metric_{i}",
                "points": [{"period": f"2026-0{j}", "value": float(j), "pct_change_from_prior": None} for j in range(1, 4)],
                "min_value": 1.0, "max_value": 3.0, "mean_value": 2.0, "direction": "increasing",
                "insufficient_evidence": False, "reason": None,
            }
            for i in range(10)  # far more trends than any sane cap
        ],
        "contributions": [],
        "top_n": [],
        "distributions": [],
        "diagnostic": None,
        "insufficient_evidence": False,
        "reason": None,
    }
    result = await visualization_agent_node(state)
    assert len(result["charts"]) <= 3
