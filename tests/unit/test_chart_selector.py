"""Deterministic unit tests for app/tools/chart_selector.py (Phase 7). Small
fixed fixtures only — no DB, no LLM, no network. Covers: time-series->line,
category+metric->bar, top-N->horizontal bar, KPI, scatter, too-many-
categories, missing/empty/null data, invalid types, unsupported analysis,
no hallucinated values, sorting, axis correctness.
"""

from __future__ import annotations

from app.tools.analysis_tools import (
    analyze_contribution,
    analyze_trend,
    compare_periods,
    distribution_stats,
)
from app.tools.chart_selector import (
    select_contribution_chart,
    select_distribution_chart,
    select_period_comparison_chart,
    select_scatter_chart,
    select_table_fallback,
    select_top_n_chart,
    select_trend_chart,
)
from app.tools.column_classifier import find_scatter_candidate
from dataclasses import asdict


# --- period comparison -> bar chart, axis/sort correctness -----------------


def test_period_comparison_selects_bar_chart_with_two_points():
    pc = asdict(compare_periods(
        [{"month": "2026-06", "revenue": 161445.80}, {"month": "2026-07", "revenue": 150633.02}],
        "month", "revenue",
    ))
    spec = select_period_comparison_chart(pc)
    assert spec.visualization_needed
    assert spec.chart_type == "bar"
    assert spec.x_axis == "month"
    assert spec.y_axis == "revenue"
    assert len(spec.data) == 2
    # No hallucinated values — every plotted number traces back to the input.
    assert spec.data[0]["value"] == 161445.80
    assert spec.data[1]["value"] == 150633.02


def test_period_comparison_not_ok_yields_no_visualization():
    pc = asdict(compare_periods([{"month": "2026-07", "revenue": 1.0}], "month", "revenue"))  # only 1 period
    spec = select_period_comparison_chart(pc)
    assert spec.visualization_needed is False
    assert spec.reason


# --- trend -> line (short) / area (long) ------------------------------------


def test_short_trend_selects_line_chart():
    rows = [{"month": f"2026-0{i}", "revenue": float(i * 10)} for i in range(1, 4)]  # 3 points
    trend = asdict(analyze_trend(rows, "month", "revenue"))
    spec = select_trend_chart(trend)
    assert spec.visualization_needed
    assert spec.chart_type == "line"
    assert spec.x_axis == "month"
    assert spec.y_axis == "revenue"
    assert [d["value"] for d in spec.data] == [10.0, 20.0, 30.0]


def test_long_trend_selects_area_chart():
    rows = [{"month": f"2026-{i:02d}", "revenue": float(i * 10)} for i in range(1, 10)]  # 9 points
    trend = asdict(analyze_trend(rows, "month", "revenue"))
    spec = select_trend_chart(trend)
    assert spec.chart_type == "area"


def test_trend_not_ok_yields_no_visualization():
    rows = [{"month": "2026-06", "revenue": 1.0}, {"month": "2026-07", "revenue": 2.0}]  # only 2 -> insufficient
    trend = asdict(analyze_trend(rows, "month", "revenue"))
    spec = select_trend_chart(trend)
    assert spec.visualization_needed is False


# --- contribution -> bar / horizontal_bar / pie / kpi, sorting -------------


def test_category_metric_single_period_small_n_selects_pie():
    rows = [{"method": "credit_card", "amount": 70.0}, {"method": "ach", "amount": 30.0}]
    contrib = asdict(analyze_contribution(rows, "method", "amount"))
    spec = select_contribution_chart(contrib)
    assert spec.chart_type == "pie"
    assert spec.group_by == "method"


def test_category_metric_moderate_n_selects_bar_sorted_desc():
    # 5 categories: above the pie threshold (<=4) but at/below the bar cap
    # (<=8) -> bar, not pie and not horizontal_bar.
    rows = [{"category": c, "revenue": v} for c, v in [("A", 10.0), ("B", 50.0), ("C", 30.0), ("D", 5.0), ("E", 20.0)]]
    contrib = asdict(analyze_contribution(rows, "category", "revenue"))
    spec = select_contribution_chart(contrib)
    assert spec.chart_type == "bar"
    assert spec.sort == "desc"
    assert [d["value"] for d in spec.data] == [50.0, 30.0, 20.0, 10.0, 5.0]  # sorted, not input order


def test_too_many_categories_selects_horizontal_bar_and_caps_at_eight():
    rows = [{"category": f"cat-{i}", "revenue": float(i)} for i in range(15)]
    contrib = asdict(analyze_contribution(rows, "category", "revenue"))
    spec = select_contribution_chart(contrib)
    assert spec.chart_type == "horizontal_bar"
    assert len(spec.data) == 8  # capped, not all 15
    assert spec.limitations  # notes that rows were truncated


def test_contribution_with_prior_period_selects_horizontal_bar_ranked_by_change():
    current = [{"segment": "Enterprise", "revenue": 10.0}, {"segment": "SMB", "revenue": 60.0}]
    prior = [{"segment": "Enterprise", "revenue": 60.0}, {"segment": "SMB", "revenue": 40.0}]
    contrib = asdict(analyze_contribution(current, "segment", "revenue", prior_rows=prior))
    spec = select_contribution_chart(contrib)
    assert spec.chart_type == "horizontal_bar"
    assert spec.units == "change"
    assert spec.data[0]["label"] == "Enterprise"  # largest |change| first


def test_single_group_contribution_selects_kpi():
    rows = [{"category": "Software", "revenue": 737525.145}]
    contrib = asdict(analyze_contribution(rows, "category", "revenue"))
    spec = select_contribution_chart(contrib)
    assert spec.chart_type == "kpi"
    assert spec.data == [{"label": "Software", "value": 737525.145}]  # exact value, not rounded/invented


def test_contribution_empty_yields_no_visualization():
    contrib = asdict(analyze_contribution([], "category", "revenue"))
    spec = select_contribution_chart(contrib)
    assert spec.visualization_needed is False


# --- top_n -> horizontal bar, always (never pie, per Sec 7 rule) -----------


def test_top_n_selects_horizontal_bar():
    top_n_entry = {
        "dimension": "category",
        "value_col": "revenue",
        "rows": [{"category": "Software", "revenue": 800.0}, {"category": "Hardware", "revenue": 300.0}],
    }
    spec = select_top_n_chart(top_n_entry)
    assert spec.chart_type == "horizontal_bar"
    assert spec.x_axis == "revenue"
    assert spec.y_axis == "category"
    assert spec.data[0]["value"] == 800.0


def test_top_n_empty_rows_yields_no_visualization():
    spec = select_top_n_chart({"dimension": "category", "value_col": "revenue", "rows": []})
    assert spec.visualization_needed is False


# --- distribution -> KPI (n=1) or table (n>1) -------------------------------


def test_distribution_single_value_selects_kpi():
    dist = asdict(distribution_stats([{"n": 42}], "n"))
    spec = select_distribution_chart(dist)
    assert spec.chart_type == "kpi"
    assert spec.data == [{"label": "n", "value": 42.0}]


def test_distribution_multi_value_selects_table():
    dist = asdict(distribution_stats([{"score": v} for v in [1, 2, 3, 4, 5]], "score"))
    spec = select_distribution_chart(dist)
    assert spec.chart_type == "table"
    labels = [d["label"] for d in spec.data]
    assert "mean" in labels and "median" in labels


def test_distribution_not_ok_yields_no_visualization():
    spec = select_distribution_chart(asdict(distribution_stats([], "score")))
    assert spec.visualization_needed is False


# --- scatter: two numeric columns, no period/dimension ----------------------


def test_two_numeric_columns_detected_and_selected_as_scatter():
    rows = [{"weight_g": 100.0 + i, "price": 10.0 + i} for i in range(5)]
    candidate = find_scatter_candidate(rows)
    assert candidate == ("weight_g", "price")
    spec = select_scatter_chart(rows, *candidate)
    assert spec.chart_type == "scatter"
    assert spec.x_axis == "weight_g"
    assert spec.y_axis == "price"
    assert len(spec.data) == 5
    assert spec.data[0] == {"x": 100.0, "y": 10.0}  # exact input values


def test_scatter_candidate_not_detected_when_dimension_present():
    rows = [{"category": "A", "weight_g": 100.0, "price": 10.0}]
    assert find_scatter_candidate(rows) is None


def test_scatter_caps_at_max_points():
    rows = [{"x": float(i), "y": float(i)} for i in range(500)]
    spec = select_scatter_chart(rows, "x", "y")
    assert len(spec.data) == 200
    assert spec.limitations


def test_scatter_skips_rows_with_null_values():
    rows = [{"x": 1.0, "y": 2.0}, {"x": None, "y": 3.0}, {"x": 4.0, "y": None}]
    spec = select_scatter_chart(rows, "x", "y")
    assert len(spec.data) == 1  # only the fully-populated row
    assert spec.data[0] == {"x": 1.0, "y": 2.0}


# --- table fallback: unsupported analysis type, missing/empty data --------


def test_table_fallback_for_unsupported_shape():
    rows = [{"free_text": "hello there", "another_free_text": "unrelated"}]
    spec = select_table_fallback(rows)
    assert spec.chart_type == "table"
    assert spec.data == rows


def test_table_fallback_empty_rows_yields_no_visualization():
    spec = select_table_fallback([])
    assert spec.visualization_needed is False
    assert spec.reason


def test_table_fallback_caps_rows():
    rows = [{"a": i} for i in range(50)]
    spec = select_table_fallback(rows)
    assert len(spec.data) == 20
    assert spec.limitations
