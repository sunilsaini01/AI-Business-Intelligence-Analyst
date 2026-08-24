"""Deterministic chart-type selection (Sec 5, Phase 7). Pure functions only —
no LLM, no I/O, no DB access. Each function turns one piece of Phase 6's
already-computed `analysis_results` (or, only when nothing there fits, raw
SQL evidence — see app/agents/visualization_agent.py) into a
`VisualizationSpec`. Every plotted value is copied from that input, never
invented.

Why no LLM for chart-type selection (project rule: prefer deterministic
logic, isolate any genuinely-needed LLM use and explain why): each entry in
`analysis_results` already carries an explicit type tag from the Analysis
Agent — "this is a period_comparison", "this is a contribution breakdown
with N groups", "this is a trend with N points". Which chart fits which tag
is a fixed, small lookup (Sec 7 spec's own examples: time+metric -> line,
category+metric -> bar, top-N -> horizontal bar, ...), not a judgment call
that needs semantic interpretation of the question text. There's nothing
here an LLM would decide differently from a lookup table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ChartType = Literal["bar", "horizontal_bar", "line", "area", "pie", "scatter", "kpi", "table"]

# Hard caps so a chart never ships with more data than a reader (or a
# frontend) can reasonably render — matches the "don't blindly visualize
# thousands of rows" rule. Aggregation/ranking already happened upstream in
# SQL/pandas; this is just the final display-size cap.
_MAX_CATEGORIES = 8
_MAX_SCATTER_POINTS = 200
_MAX_TABLE_ROWS = 20

# Above this many groups, a pie/donut is unreadable — fall back to a ranked
# horizontal bar instead ("too many categories -> horizontal bar or table").
_MAX_PIE_CATEGORIES = 4

# A trend longer than this reads better filled (area) than as a sparse line
# of discrete points — a standard dataviz convention, not an arbitrary cutoff.
_AREA_CHART_MIN_POINTS = 7


@dataclass
class VisualizationSpec:
    visualization_needed: bool
    chart_type: ChartType | None = None
    title: str = ""
    subtitle: str | None = None
    x_axis: str | None = None
    y_axis: str | None = None
    group_by: str | None = None
    sort: Literal["asc", "desc", "none"] | None = None
    data: list[dict[str, Any]] = field(default_factory=list)
    units: str | None = None
    source_analysis: str = ""
    reason: str = ""
    limitations: list[str] = field(default_factory=list)


def _no_viz(reason: str, *, source_analysis: str = "") -> VisualizationSpec:
    return VisualizationSpec(visualization_needed=False, reason=reason, source_analysis=source_analysis)


def select_period_comparison_chart(pc: dict[str, Any]) -> VisualizationSpec:
    """Two periods, one metric -> a 2-bar comparison (Sec 7 spec's own
    example: "June vs July revenue -> comparison bar chart")."""
    if not pc.get("ok"):
        return _no_viz(pc.get("reason") or "Period comparison not available.", source_analysis="period_comparison")

    data = [
        {"label": str(pc["baseline_period"]), "value": pc["baseline_value"]},
        {"label": str(pc["current_period"]), "value": pc["current_value"]},
    ]
    return VisualizationSpec(
        visualization_needed=True,
        chart_type="bar",
        title=f"{pc['value_col']}: {pc['baseline_period']} vs {pc['current_period']}",
        subtitle=f"{pc['direction']}" + (f" ({pc['percentage_change']:+.1f}%)" if pc.get("percentage_change") is not None else ""),
        x_axis=pc["period_col"],
        y_axis=pc["value_col"],
        sort="none",
        data=data,
        source_analysis="period_comparison",
        reason="Exactly two periods being compared — a 2-bar chart shows the before/after directly.",
        limitations=[pc["note"]] if pc.get("note") else [],
    )


def select_trend_chart(trend: dict[str, Any]) -> VisualizationSpec:
    """Time + numeric metric, 3+ points -> line (or area for a longer series)."""
    if not trend.get("ok"):
        return _no_viz(trend.get("reason") or "Trend not available.", source_analysis="trend")

    points = trend["points"]
    chart_type: ChartType = "area" if len(points) > _AREA_CHART_MIN_POINTS else "line"
    data = [{"label": str(p["period"]), "value": p["value"]} for p in points]
    return VisualizationSpec(
        visualization_needed=True,
        chart_type=chart_type,
        title=f"{trend['value_col']} over time",
        subtitle=f"Trend: {trend['direction']}",
        x_axis=trend["period_col"],
        y_axis=trend["value_col"],
        sort="none",
        data=data,
        source_analysis="trend",
        reason=(
            f"Time series with {len(points)} points -> "
            + ("area (longer series reads better filled)." if chart_type == "area" else "line (few points, precise values matter).")
        ),
    )


def select_contribution_chart(contrib: dict[str, Any], *, dominant_only: bool = False) -> VisualizationSpec:
    """Category + metric breakdown. Rules (Sec 7 spec):
    - has a prior period (i.e. this is a "who changed" breakdown) -> horizontal
      bar, sorted by |change|, capped at _MAX_CATEGORIES (matches "segment/
      region contribution to decline -> horizontal bar chart").
    - single period, small cardinality (<= _MAX_PIE_CATEGORIES), all
      non-negative -> pie ("small part-to-whole ... only when appropriate").
    - single period, larger cardinality -> bar (<= _MAX_CATEGORIES groups) or
      horizontal bar (more — "too many categories -> horizontal bar").
    - exactly one group -> KPI card instead of a one-bar chart.
    """
    if not contrib.get("ok") or not contrib.get("contributors"):
        return _no_viz(contrib.get("reason") or "Contribution breakdown not available.", source_analysis="contribution")

    contributors = contrib["contributors"]
    has_prior = contrib.get("total_change") is not None

    if len(contributors) == 1:
        c = contributors[0]
        return VisualizationSpec(
            visualization_needed=True,
            chart_type="kpi",
            title=f"{contrib['dimension_col']}: {c['group']}",
            y_axis=contrib["value_col"],
            data=[{"label": c["group"], "value": c["current_value"]}],
            source_analysis="contribution",
            reason="Exactly one group — a single value reads better as a KPI than a one-bar chart.",
        )

    if has_prior:
        ranked = sorted(contributors, key=lambda c: abs(c["change"] or 0.0), reverse=True)
        shown = ranked[:_MAX_CATEGORIES]
        limitations = []
        if len(ranked) > _MAX_CATEGORIES:
            limitations.append(f"{len(ranked)} groups present; showing the top {_MAX_CATEGORIES} by size of change.")
        title = f"{contrib['dimension_col']} contribution to the change in {contrib['value_col']}"
        if dominant_only:
            title = f"{contrib['dimension_col']} — dominant contributor to the change"
        return VisualizationSpec(
            visualization_needed=True,
            chart_type="horizontal_bar",
            title=title,
            x_axis=contrib["value_col"],
            y_axis=contrib["dimension_col"],
            sort="desc",
            data=[{"label": c["group"], "value": c["change"]} for c in shown],
            units="change",
            source_analysis="contribution",
            reason="Period-over-period breakdown by group -> horizontal bar ranked by size of change.",
            limitations=limitations,
        )

    # Single-period ranking/share-of-total.
    ranked = sorted(contributors, key=lambda c: c["current_value"], reverse=True)
    all_non_negative = all((c["current_value"] or 0) >= 0 for c in ranked)

    if len(ranked) <= _MAX_PIE_CATEGORIES and all_non_negative:
        return VisualizationSpec(
            visualization_needed=True,
            chart_type="pie",
            title=f"{contrib['dimension_col']} share of {contrib['value_col']}",
            group_by=contrib["dimension_col"],
            data=[{"label": c["group"], "value": c["current_value"]} for c in ranked],
            source_analysis="contribution",
            reason=f"{len(ranked)} groups, single period, non-negative values -> a small part-to-whole pie fits.",
        )

    chart_type: ChartType = "bar" if len(ranked) <= _MAX_CATEGORIES else "horizontal_bar"
    shown = ranked[:_MAX_CATEGORIES]
    limitations = []
    if len(ranked) > _MAX_CATEGORIES:
        limitations.append(f"{len(ranked)} groups present; showing the top {_MAX_CATEGORIES} by {contrib['value_col']}.")
    return VisualizationSpec(
        visualization_needed=True,
        chart_type=chart_type,
        title=f"{contrib['dimension_col']} by {contrib['value_col']}",
        x_axis=contrib["dimension_col"] if chart_type == "bar" else contrib["value_col"],
        y_axis=contrib["value_col"] if chart_type == "bar" else contrib["dimension_col"],
        sort="desc",
        data=[{"label": c["group"], "value": c["current_value"]} for c in shown],
        source_analysis="contribution",
        reason=(
            f"{len(ranked)} groups, single period -> "
            + ("bar chart." if chart_type == "bar" else "too many categories for a bar/pie -> horizontal bar.")
        ),
        limitations=limitations,
    )


def select_top_n_chart(top_n_entry: dict[str, Any]) -> VisualizationSpec:
    """Sec 7 spec: "Top-N categories -> horizontal bar chart", always — top_n
    is inherently a ranking, not a share-of-whole, so pie never applies here."""
    rows = top_n_entry.get("rows") or []
    if not rows:
        return _no_viz("No top-N rows available.", source_analysis="top_n")

    dim, value_col = top_n_entry["dimension"], top_n_entry["value_col"]
    shown = rows[:_MAX_CATEGORIES]
    return VisualizationSpec(
        visualization_needed=True,
        chart_type="horizontal_bar",
        title=f"Top {len(shown)} {dim} by {value_col}",
        x_axis=value_col,
        y_axis=dim,
        sort="desc",
        data=[{"label": str(r[dim]), "value": r[value_col]} for r in shown],
        source_analysis="top_n",
        reason="Top-N ranking -> horizontal bar chart (Sec 7 rule), already sorted by the Analysis Agent.",
    )


def select_distribution_chart(dist: dict[str, Any]) -> VisualizationSpec:
    """A single-column summary. One value (count==1) -> KPI card. Otherwise
    the 5-number summary reads better as a small table than as any chart."""
    if not dist.get("ok"):
        return _no_viz(dist.get("reason") or "Distribution not available.", source_analysis="distribution")

    if dist["count"] == 1:
        return VisualizationSpec(
            visualization_needed=True,
            chart_type="kpi",
            title=dist["column"],
            y_axis=dist["column"],
            data=[{"label": dist["column"], "value": dist["mean"]}],
            source_analysis="distribution",
            reason="Single value -> KPI card.",
        )

    return VisualizationSpec(
        visualization_needed=True,
        chart_type="table",
        title=f"{dist['column']} — summary statistics",
        data=[
            {"label": "count", "value": dist["count"]},
            {"label": "mean", "value": dist["mean"]},
            {"label": "median", "value": dist["median"]},
            {"label": "min", "value": dist["min"]},
            {"label": "max", "value": dist["max"]},
            {"label": "std", "value": dist["std"]},
            {"label": "q25", "value": dist["q25"]},
            {"label": "q75", "value": dist["q75"]},
        ],
        source_analysis="distribution",
        reason="Summary statistics (count/mean/median/min/max/std/quantiles) -> table, not a chart.",
    )


def select_scatter_chart(rows: list[dict[str, Any]], x_col: str, y_col: str) -> VisualizationSpec:
    """Two independent numeric columns, no period/dimension role -> scatter,
    capped at _MAX_SCATTER_POINTS points (raw-row data, so this is the one
    chart type that can see a lot of rows — cap it explicitly)."""
    points = [
        {"x": r[x_col], "y": r[y_col]}
        for r in rows
        if r.get(x_col) is not None and r.get(y_col) is not None
    ]
    if not points:
        return _no_viz("No rows with both numeric values present.", source_analysis="raw_evidence")

    limitations = []
    if len(points) > _MAX_SCATTER_POINTS:
        limitations.append(f"{len(points)} points available; showing the first {_MAX_SCATTER_POINTS}.")
        points = points[:_MAX_SCATTER_POINTS]

    return VisualizationSpec(
        visualization_needed=True,
        chart_type="scatter",
        title=f"{y_col} vs {x_col}",
        x_axis=x_col,
        y_axis=y_col,
        data=points,
        source_analysis="raw_evidence",
        reason="Two independent numeric columns, no time/category role -> scatter plot.",
        limitations=limitations,
    )


def select_table_fallback(rows: list[dict[str, Any]]) -> VisualizationSpec:
    """No supported chart pattern matched, but there's structured evidence —
    show it as a table rather than nothing ("no meaningful visual
    relationship -> table")."""
    if not rows:
        return _no_viz("No evidence available to display.", source_analysis="raw_evidence")

    shown = rows[:_MAX_TABLE_ROWS]
    limitations = []
    if len(rows) > _MAX_TABLE_ROWS:
        limitations.append(f"{len(rows)} rows available; showing the first {_MAX_TABLE_ROWS}.")

    return VisualizationSpec(
        visualization_needed=True,
        chart_type="table",
        title="Query results",
        data=shown,
        source_analysis="raw_evidence",
        reason="No period/dimension/trend pattern matched this evidence -> table fallback.",
        limitations=limitations,
    )
