"""Visualization Agent (Sec 1, Sec 5; user Phase 7 spec).

Turns Phase 6's deterministic `state["analysis_results"]` into 0-N chart
specs (`state["charts"]`) — see app/tools/chart_selector.py for the actual
selection rules and why none of this needs an LLM. Never queries Postgres
(reads only already-fetched `state["sql_queries"]`, and only as a fallback
when analysis_results has nothing usable) and never calls an LLM.

Priority / anti-spam (Sec 7 spec: "do not create four charts automatically,
avoid visualization spam"): a diagnostic question can have a period
comparison AND several contribution breakdowns available at once — this
picks the overall before/after comparison plus only the breakdowns that
diagnose_decline itself flagged as dominant (same >=20% threshold it uses),
not every dimension that happened to get computed. Everything is capped at
`_MAX_CHARTS` regardless.
"""

from __future__ import annotations

import time

from app.graph.state import AgentState, ChartRecord, trace_event
from app.tools import chart_selector as cs
from app.tools.column_classifier import find_scatter_candidate

_MAX_CHARTS = 3

# Must match app/tools/analysis_tools.py::diagnose_decline's default
# contribution_threshold_pct — this recomputes "was this contribution called
# out as dominant" using the same rule, rather than parsing diagnose_decline's
# free-text interpretation strings back apart.
_DOMINANT_CONTRIBUTION_THRESHOLD_PCT = 20.0


def _is_dominant(contribution: dict) -> bool:
    if contribution.get("total_change") is None or not contribution.get("contributors"):
        return False
    top_pct = contribution["contributors"][0].get("pct_of_total_change")
    return top_pct is not None and abs(top_pct) >= _DOMINANT_CONTRIBUTION_THRESHOLD_PCT


def _to_chart_record(spec: cs.VisualizationSpec) -> ChartRecord:
    return ChartRecord(
        chart_type=spec.chart_type or "table",
        title=spec.title,
        subtitle=spec.subtitle,
        x_axis=spec.x_axis,
        y_axis=spec.y_axis,
        group_by=spec.group_by,
        sort=spec.sort,
        data=spec.data,
        units=spec.units,
        source_analysis=spec.source_analysis,
        reason=spec.reason,
        limitations=spec.limitations,
        path="",
        spec_summary={
            "chart_type": spec.chart_type,
            "title": spec.title,
            "subtitle": spec.subtitle,
            "x_axis": spec.x_axis,
            "y_axis": spec.y_axis,
            "group_by": spec.group_by,
            "sort": spec.sort,
            "data": spec.data,
            "units": spec.units,
            "source_analysis": spec.source_analysis,
            "reason": spec.reason,
            "limitations": spec.limitations,
        },
    )


def _select_specs(state: AgentState) -> list[cs.VisualizationSpec]:
    analysis = state.get("analysis_results") or {}
    contributions = analysis.get("contributions") or []
    period_comparisons = analysis.get("period_comparisons") or []
    trends = analysis.get("trends") or []
    top_n_list = analysis.get("top_n") or []
    distributions = analysis.get("distributions") or []

    specs: list[cs.VisualizationSpec] = []

    # Diagnostic path is keyed off the intent + an actual period comparison
    # being available — NOT off diagnose_decline's own insufficient_evidence
    # flag, which can be true for reasons unrelated to whether the overall
    # comparison itself is worth showing (e.g. "a change was confirmed but no
    # contribution breakdown could be computed" still has a good comparison
    # chart, it just won't get a contribution breakdown chart alongside it).
    if state.get("intent") == "diagnostic" and period_comparisons:
        specs.append(cs.select_period_comparison_chart(period_comparisons[0]))
        for contribution in contributions:
            if _is_dominant(contribution):
                specs.append(cs.select_contribution_chart(contribution, dominant_only=True))
    elif trends:
        for trend in trends:
            specs.append(cs.select_trend_chart(trend))
    elif contributions:
        specs.append(cs.select_contribution_chart(contributions[0]))
    elif top_n_list:
        for entry in top_n_list:
            specs.append(cs.select_top_n_chart(entry))
    elif distributions:
        for dist in distributions:
            specs.append(cs.select_distribution_chart(dist))
    else:
        # Nothing in analysis_results at all — fall back to the first usable
        # raw SQL result (Sec 7: "prefer analysis_results, sql_queries where
        # necessary"). Only the first, to avoid spamming a table/scatter per
        # query when nothing was analyzable.
        for q in state["sql_queries"]:
            if not q["validated_ok"] or not q["rows"]:
                continue
            scatter_cols = find_scatter_candidate(q["rows"])
            if scatter_cols:
                specs.append(cs.select_scatter_chart(q["rows"], *scatter_cols))
            else:
                specs.append(cs.select_table_fallback(q["rows"]))
            break

    return specs


async def visualization_agent_node(state: AgentState) -> AgentState:
    state["trace"].append(trace_event("visualization_agent", "enter"))
    started = time.perf_counter()

    specs = [s for s in _select_specs(state) if s.visualization_needed][:_MAX_CHARTS]
    state["charts"] = [_to_chart_record(s) for s in specs]

    state["trace"].append(
        trace_event("visualization_agent", "exit", duration_ms=(time.perf_counter() - started) * 1000)
    )
    return state
