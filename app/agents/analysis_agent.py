"""Analysis Agent (Sec 1, Sec 5; user Phase 6 spec).

Turns raw SQL evidence (state["sql_queries"]) into deterministic analytical
evidence (state["analysis_results"]) using only pandas/NumPy — see
app/tools/analysis_tools.py for the actual math and
app/tools/column_classifier.py for how it figures out which column is a
period/dimension/metric. This agent never queries Postgres itself (only
app/tools/database_tools.py::run_query does that, via the SQL Agent) and
never calls an LLM.

Sec 5 RULE: this module must never import app.core.llm. CI greps for that
(`rg "llm_client" app/agents/analysis_agent.py` must return nothing) — do
not add an LLM call to this file.

Column-role detection is necessarily heuristic — the SQL Agent's queries are
LLM-generated per question, so column names vary run to run. When a query's
shape doesn't match any supported pattern, it's simply left unanalyzed (the
raw rows are still available to the Supervisor via state["sql_queries"])
rather than forcing a bad guess.
"""

from __future__ import annotations

import dataclasses
import time

from app.graph.state import AgentState, trace_event
from app.tools import analysis_tools as at
from app.tools.column_classifier import classify_columns


async def analysis_agent_node(state: AgentState) -> AgentState:
    state["trace"].append(trace_event("analysis_agent", "enter"))
    started = time.perf_counter()

    validated_queries = [q for q in state["sql_queries"] if q["validated_ok"] and q["rows"]]

    period_comparisons: list[at.PeriodComparison] = []
    trends: list[at.TrendAnalysis] = []
    contributions: list[at.ContributionAnalysis] = []
    top_n_results: list[dict] = []
    distributions: list[at.DistributionStats] = []

    for q in validated_queries:
        rows = q["rows"]
        period_col, dimension_cols, value_col = classify_columns(rows)
        if value_col is None:
            continue  # nothing numeric to analyze in this query's result shape

        if period_col and not dimension_cols:
            distinct_periods = {r.get(period_col) for r in rows}
            comparison = at.compare_periods(rows, period_col, value_col)
            if comparison.ok:
                period_comparisons.append(comparison)
            if len(distinct_periods) >= 3:
                trend = at.analyze_trend(rows, period_col, value_col)
                if trend.ok:
                    trends.append(trend)

        elif period_col and dimension_cols:
            # Dimension(s) broken down by period -> contribution-to-change,
            # one analysis per dimension column, each independently deriving
            # its own baseline/current period from what's actually present
            # in this query (never cross-referenced against another query's
            # period labels, which could use a different format).
            periods_present = sorted({r.get(period_col) for r in rows if r.get(period_col) is not None})
            if len(periods_present) >= 2:
                baseline_period, current_period = periods_present[-2], periods_present[-1]
                prior_rows = [r for r in rows if r.get(period_col) == baseline_period]
                current_rows = [r for r in rows if r.get(period_col) == current_period]
                for dim in dimension_cols:
                    contribution = at.analyze_contribution(
                        current_rows,
                        dim,
                        value_col,
                        prior_rows=prior_rows,
                        baseline_period=str(baseline_period),
                        current_period=str(current_period),
                    )
                    if contribution.ok:
                        contributions.append(contribution)

        elif dimension_cols and not period_col:
            for dim in dimension_cols:
                contribution = at.analyze_contribution(rows, dim, value_col)
                if contribution.ok:
                    contributions.append(contribution)
                top = at.top_n(rows, dim, value_col, n=5)
                if top:
                    top_n_results.append({"dimension": dim, "value_col": value_col, "rows": top})

        else:
            dist = at.distribution_stats(rows, value_col)
            if dist.ok:
                distributions.append(dist)

    diagnostic: at.DiagnosticResult | None = None
    if state["intent"] == "diagnostic" and period_comparisons:
        diagnostic = at.diagnose_decline(period_comparisons[0], contributions)

    has_any_analysis = bool(period_comparisons or trends or contributions or top_n_results or distributions)
    insufficient_evidence = not has_any_analysis
    reason = None
    if insufficient_evidence:
        reason = (
            "No SQL evidence matched a supported analysis pattern "
            "(period comparison, contribution, trend, or distribution)."
        )
    elif state["intent"] == "diagnostic" and diagnostic is None:
        insufficient_evidence = True
        reason = "Diagnostic question, but no period-over-period comparison could be established from the evidence."

    state["analysis_results"] = {
        "period_comparisons": [dataclasses.asdict(c) for c in period_comparisons],
        "trends": [dataclasses.asdict(t) for t in trends],
        "contributions": [dataclasses.asdict(c) for c in contributions],
        "top_n": top_n_results,
        "distributions": [dataclasses.asdict(d) for d in distributions],
        "diagnostic": dataclasses.asdict(diagnostic) if diagnostic is not None else None,
        "insufficient_evidence": insufficient_evidence,
        "reason": reason,
    }

    state["trace"].append(
        trace_event("analysis_agent", "exit", duration_ms=(time.perf_counter() - started) * 1000)
    )
    return state
