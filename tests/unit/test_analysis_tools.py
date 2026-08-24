"""Deterministic unit tests for app/tools/analysis_tools.py (Phase 6). Small
fixed fixtures only — no DB, no LLM, no network. Covers: percentage change,
absolute change, increase/decrease detection, contribution percentage,
ranking, top-N, trend, missing evidence, empty data, zero denominator,
duplicate rows, null values, insufficient periods, diagnostic analysis.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.tools.analysis_tools import (
    analyze_contribution,
    analyze_trend,
    compare_periods,
    compute_revenue,
    diagnose_decline,
    distribution_stats,
    top_n,
)


def test_compute_revenue_applies_discount():
    df = pd.DataFrame({"quantity": [2, 1], "unit_price": [10.0, 100.0], "discount": [0.0, 0.1]})
    result = compute_revenue(df)
    assert result.tolist() == [20.0, 90.0]


# --- compare_periods: absolute/percentage change, direction, edge cases ----


def test_absolute_and_percentage_change():
    rows = [{"month": "2026-06", "revenue": 161445.80}, {"month": "2026-07", "revenue": 150633.02}]
    result = compare_periods(rows, "month", "revenue")
    assert result.ok
    assert result.absolute_change == pytest.approx(150633.02 - 161445.80)
    assert result.percentage_change == pytest.approx(((150633.02 - 161445.80) / 161445.80) * 100)


def test_decrease_direction_detected():
    rows = [{"month": "2026-06", "revenue": 100.0}, {"month": "2026-07", "revenue": 70.0}]
    result = compare_periods(rows, "month", "revenue")
    assert result.direction == "decrease"


def test_increase_direction_detected():
    rows = [{"month": "2026-06", "revenue": 70.0}, {"month": "2026-07", "revenue": 100.0}]
    result = compare_periods(rows, "month", "revenue")
    assert result.direction == "increase"


def test_no_change_direction_detected():
    rows = [{"month": "2026-06", "revenue": 100.0}, {"month": "2026-07", "revenue": 100.0}]
    result = compare_periods(rows, "month", "revenue")
    assert result.direction == "no_change"
    assert result.absolute_change == 0.0
    assert result.percentage_change == 0.0


def test_zero_baseline_percentage_change_is_none_not_fabricated():
    rows = [{"month": "2026-06", "revenue": 0.0}, {"month": "2026-07", "revenue": 50.0}]
    result = compare_periods(rows, "month", "revenue")
    assert result.ok
    assert result.absolute_change == 50.0
    assert result.percentage_change is None  # undefined, not 0 or inf


def test_duplicate_rows_for_same_period_are_summed():
    rows = [
        {"month": "2026-06", "revenue": 40.0},
        {"month": "2026-06", "revenue": 60.0},  # same period, second row
        {"month": "2026-07", "revenue": 50.0},
    ]
    result = compare_periods(rows, "month", "revenue")
    assert result.ok
    assert result.baseline_value == 100.0  # 40 + 60


def test_null_values_are_dropped_not_treated_as_zero():
    rows = [
        {"month": "2026-06", "revenue": None},
        {"month": "2026-06", "revenue": 40.0},
        {"month": "2026-07", "revenue": 50.0},
    ]
    result = compare_periods(rows, "month", "revenue")
    assert result.ok
    assert result.baseline_value == 40.0  # null row excluded, not coerced to 0


def test_insufficient_periods_single_period():
    rows = [{"month": "2026-07", "revenue": 100.0}]
    result = compare_periods(rows, "month", "revenue")
    assert not result.ok
    assert result.insufficient_evidence
    assert "2" in result.reason


def test_empty_rows_is_insufficient_evidence():
    result = compare_periods([], "month", "revenue")
    assert not result.ok
    assert result.insufficient_evidence


def test_missing_column_is_insufficient_evidence():
    rows = [{"month": "2026-06", "revenue": 1.0}, {"month": "2026-07", "revenue": 2.0}]
    result = compare_periods(rows, "month", "amount")  # wrong column name
    assert not result.ok
    assert result.insufficient_evidence


def test_more_than_two_periods_compares_latest_two_and_notes_it():
    rows = [
        {"month": "2026-05", "revenue": 10.0},
        {"month": "2026-06", "revenue": 20.0},
        {"month": "2026-07", "revenue": 15.0},
    ]
    result = compare_periods(rows, "month", "revenue")
    assert result.ok
    assert result.baseline_period == "2026-06"
    assert result.current_period == "2026-07"
    assert result.note is not None


# --- analyze_trend -----------------------------------------------------


def test_trend_requires_at_least_three_periods():
    rows = [{"month": "2026-06", "revenue": 10.0}, {"month": "2026-07", "revenue": 20.0}]
    result = analyze_trend(rows, "month", "revenue")
    assert not result.ok
    assert result.insufficient_evidence


def test_trend_detects_increasing():
    rows = [{"month": f"2026-0{i}", "revenue": float(i * 10)} for i in range(1, 6)]
    result = analyze_trend(rows, "month", "revenue")
    assert result.ok
    assert result.direction == "increasing"
    assert result.min_value == 10.0
    assert result.max_value == 50.0


def test_trend_detects_decreasing():
    rows = [{"month": f"2026-0{i}", "revenue": float(60 - i * 10)} for i in range(1, 6)]
    result = analyze_trend(rows, "month", "revenue")
    assert result.ok
    assert result.direction == "decreasing"


def test_trend_detects_flat():
    rows = [{"month": f"2026-0{i}", "revenue": 100.0} for i in range(1, 6)]
    result = analyze_trend(rows, "month", "revenue")
    assert result.ok
    assert result.direction == "flat"


# --- analyze_contribution: contribution %, ranking, edge cases ---------


def test_contribution_percentage_and_ranking_single_period():
    rows = [
        {"region": "North", "revenue": 30.0},
        {"region": "South", "revenue": 70.0},
    ]
    result = analyze_contribution(rows, "region", "revenue")
    assert result.ok
    assert result.total_current == 100.0
    by_group = {c.group: c for c in result.contributors}
    assert by_group["South"].pct_of_total_current == 70.0
    assert by_group["North"].pct_of_total_current == 30.0
    assert by_group["South"].rank == 1  # highest current value ranks first (no prior period given)
    assert by_group["North"].rank == 2


def test_contribution_to_change_with_prior_period():
    current = [{"segment": "Enterprise", "revenue": 10.0}, {"segment": "SMB", "revenue": 60.0}]
    prior = [{"segment": "Enterprise", "revenue": 60.0}, {"segment": "SMB", "revenue": 40.0}]
    result = analyze_contribution(current, "segment", "revenue", prior_rows=prior)
    assert result.ok
    assert result.total_current == 70.0
    assert result.total_prior == 100.0
    assert result.total_change == -30.0
    by_group = {c.group: c for c in result.contributors}
    assert by_group["Enterprise"].change == -50.0
    assert by_group["SMB"].change == 20.0
    # Enterprise's decline (-50) is larger in magnitude than the net total
    # change (-30, since SMB partly offset it) -> Enterprise ranks first.
    assert result.contributors[0].group == "Enterprise"
    # pct_of_total_change is measured against the sum of declines only (SMB
    # increased, so it doesn't count toward "share of the decline") — with
    # Enterprise the only group that declined, it's 100% of that decline.
    # (Not 50/-30=166.7%: dividing by the *net* total_change would be
    # nonsensical here since SMB's offsetting increase shrinks that
    # denominator — see analyze_contribution's docstring.)
    assert by_group["Enterprise"].pct_of_total_change == pytest.approx(100.0)
    # SMB moved opposite the overall (declining) trend -> not a "% of decline".
    assert by_group["SMB"].pct_of_total_change is None


def test_contribution_group_absent_in_one_period_is_zero_not_fabricated():
    current = [{"segment": "Enterprise", "revenue": 0.0}]
    prior = [{"segment": "Enterprise", "revenue": 20.0}, {"segment": "SMB", "revenue": 5.0}]
    result = analyze_contribution(current, "segment", "revenue", prior_rows=prior)
    assert result.ok
    by_group = {c.group: c for c in result.contributors}
    # SMB has no row in `current` at all -> treated as 0 (aggregation semantics).
    assert by_group["SMB"].current_value == 0.0
    assert by_group["SMB"].change == -5.0


def test_contribution_zero_total_current_does_not_crash():
    rows = [{"region": "North", "revenue": 0.0}, {"region": "South", "revenue": 0.0}]
    result = analyze_contribution(rows, "region", "revenue")
    assert result.ok
    assert all(c.pct_of_total_current == 0.0 for c in result.contributors)


def test_contribution_zero_total_change_leaves_pct_of_change_none():
    current = [{"region": "North", "revenue": 50.0}, {"region": "South", "revenue": 50.0}]
    prior = [{"region": "North", "revenue": 60.0}, {"region": "South", "revenue": 40.0}]
    # total_current == total_prior == 100 -> total_change == 0
    result = analyze_contribution(current, "region", "revenue", prior_rows=prior)
    assert result.total_change == 0.0
    assert all(c.pct_of_total_change is None for c in result.contributors)


def test_contribution_empty_rows_is_insufficient_evidence():
    result = analyze_contribution([], "region", "revenue")
    assert not result.ok
    assert result.insufficient_evidence


def test_contribution_duplicate_rows_grouped_before_ranking():
    rows = [
        {"region": "North", "revenue": 10.0},
        {"region": "North", "revenue": 10.0},  # duplicate group, should sum
        {"region": "South", "revenue": 15.0},
    ]
    result = analyze_contribution(rows, "region", "revenue")
    by_group = {c.group: c for c in result.contributors}
    assert by_group["North"].current_value == 20.0
    assert result.contributors[0].group == "North"  # 20 > 15


# --- top_n ---------------------------------------------------------------


def test_top_n_deterministic_sort():
    rows = [
        {"category": "Software", "revenue": 800.0},
        {"category": "Hardware", "revenue": 300.0},
        {"category": "Services", "revenue": 500.0},
    ]
    result = top_n(rows, "category", "revenue", n=2)
    assert [r["category"] for r in result] == ["Software", "Services"]


def test_top_n_empty_rows_returns_empty_list():
    assert top_n([], "category", "revenue") == []


def test_top_n_missing_column_returns_empty_list():
    rows = [{"category": "Software", "revenue": 800.0}]
    assert top_n(rows, "category", "amount") == []


# --- distribution_stats ---------------------------------------------------


def test_distribution_stats_basic():
    rows = [{"score": v} for v in [1, 2, 3, 4, 5]]
    result = distribution_stats(rows, "score")
    assert result.ok
    assert result.count == 5
    assert result.mean == 3.0
    assert result.median == 3.0
    assert result.min == 1.0
    assert result.max == 5.0
    assert result.std is not None


def test_distribution_stats_single_value_std_is_none():
    result = distribution_stats([{"score": 5}], "score")
    assert result.ok
    assert result.count == 1
    assert result.std is None  # sample std undefined for n=1, not 0


def test_distribution_stats_empty_is_insufficient_evidence():
    result = distribution_stats([], "score")
    assert not result.ok
    assert result.insufficient_evidence


def test_distribution_stats_unexpected_data_type_coerced_or_dropped():
    rows = [{"score": "not_a_number"}, {"score": 5}, {"score": 7}]
    result = distribution_stats(rows, "score")
    assert result.ok
    assert result.count == 2  # the non-numeric row is dropped, not crashed on


# --- diagnose_decline: fact vs interpretation vs limitation --------------


def test_diagnostic_analysis_identifies_dominant_contributor():
    comparison = compare_periods(
        [{"month": "2026-06", "revenue": 100.0}, {"month": "2026-07", "revenue": 70.0}], "month", "revenue"
    )
    contribution = analyze_contribution(
        [{"segment": "Enterprise", "revenue": 10.0}, {"segment": "SMB", "revenue": 60.0}],
        "segment",
        "revenue",
        prior_rows=[{"segment": "Enterprise", "revenue": 60.0}, {"segment": "SMB", "revenue": 40.0}],
    )
    result = diagnose_decline(comparison, [contribution])
    assert result.ok
    assert result.facts  # a fact statement about the decline exists
    assert any("Enterprise" in i for i in result.interpretations)
    assert not result.insufficient_evidence


def test_diagnostic_analysis_no_dominant_contributor_stays_a_limitation():
    comparison = compare_periods(
        [{"month": "2026-06", "revenue": 100.0}, {"month": "2026-07", "revenue": 90.0}], "month", "revenue"
    )
    # Every segment declined by roughly the same modest share -> no single dominant driver.
    contribution = analyze_contribution(
        [{"segment": "Enterprise", "revenue": 30.0}, {"segment": "SMB", "revenue": 30.0}, {"segment": "MM", "revenue": 30.0}],
        "segment",
        "revenue",
        prior_rows=[{"segment": "Enterprise", "revenue": 33.0}, {"segment": "SMB", "revenue": 33.0}, {"segment": "MM", "revenue": 34.0}],
    )
    result = diagnose_decline(comparison, [contribution], contribution_threshold_pct=50.0)
    assert result.ok
    assert result.facts
    assert result.interpretations == []
    assert result.limitations  # explicitly notes no dominant contributor, doesn't overreach


def test_diagnostic_analysis_no_change_reports_fact_only():
    comparison = compare_periods(
        [{"month": "2026-06", "revenue": 100.0}, {"month": "2026-07", "revenue": 100.0}], "month", "revenue"
    )
    result = diagnose_decline(comparison, [])
    assert result.ok
    assert result.facts
    assert result.interpretations == []
    assert result.limitations == []


def test_diagnostic_analysis_insufficient_comparison_is_insufficient_evidence():
    comparison = compare_periods([{"month": "2026-07", "revenue": 100.0}], "month", "revenue")  # only 1 period
    result = diagnose_decline(comparison, [])
    assert not result.ok
    assert result.insufficient_evidence


def test_diagnostic_analysis_change_confirmed_but_no_contribution_data():
    comparison = compare_periods(
        [{"month": "2026-06", "revenue": 100.0}, {"month": "2026-07", "revenue": 70.0}], "month", "revenue"
    )
    result = diagnose_decline(comparison, [])  # no contribution breakdowns available
    assert result.ok
    assert result.facts
    assert result.insufficient_evidence
    assert result.limitations


