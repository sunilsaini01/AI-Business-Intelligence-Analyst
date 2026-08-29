"""Deterministic tests for app/tools/ml_tools.py (Phase 15, Objective 4).
No DB, no LLM — every function here takes plain Python/pandas structures
and is tested against hand-built data, so "deterministic metrics" and
"correct train/test separation" are provable without needing the real
seeded database (see tests/agents/test_ml_agent.py for the DB-backed,
end-to-end version of the same guarantees).
"""

from __future__ import annotations

import numpy as np
import pytest

from app.tools.ml_tools import (
    MIN_CHURN_CLASS_COUNT,
    MIN_CHURN_CUSTOMERS,
    MIN_FORECAST_POINTS,
    baseline_forecast,
    build_churn_feature_table,
    evaluate_and_forecast,
    feature_columns_for,
    fit_churn_classifier,
)


def test_baseline_forecast_extends_a_perfect_linear_trend():
    values = np.array([10.0, 20.0, 30.0, 40.0])
    forecast = baseline_forecast(values, periods_ahead=2)
    np.testing.assert_allclose(forecast, [50.0, 60.0], atol=1e-6)


# --- evaluate_and_forecast: time-aware split, insufficient data -----------


def test_forecast_reports_insufficient_data_below_the_minimum_history():
    result = evaluate_and_forecast([100.0] * (MIN_FORECAST_POINTS - 1))
    assert result.ok is False
    assert "at least" in result.reason.lower()
    assert result.metrics == {}


def test_forecast_on_a_perfect_linear_series_has_near_zero_error():
    """A perfectly linear series is the deterministic-metrics proof: the
    held-out tail is 100% predictable from the trend, so MAE/RMSE must be
    ~0 — not a hand-waved "reasonable" number."""
    values = [100.0 + 10.0 * i for i in range(10)]  # 100, 110, ..., 190
    result = evaluate_and_forecast(values, periods_ahead=1, test_size=2)
    assert result.ok is True
    assert result.metrics["mae"] < 1e-6
    assert result.metrics["rmse"] < 1e-6
    assert result.forecast_next[0] == pytest.approx(200.0, abs=1e-6)


def test_forecast_never_uses_the_held_out_tail_to_fit_the_evaluated_model():
    """The actual train/test-separation guarantee: a wildly different final
    point (a one-off spike the model was never allowed to see while being
    scored) must NOT make the evaluation metrics look artificially good —
    if the split were random/leaky, the model could "cheat" by fitting
    through the spike; a genuinely time-aware split can't."""
    values = [100.0] * 8 + [100000.0]  # flat history, then one huge spike as the held-out point
    result = evaluate_and_forecast(values, periods_ahead=1, test_size=1)
    assert result.ok is True
    # The model was fit on the flat 100.0 history only — it has no way to
    # have "seen" the spike, so its prediction for that point stays near
    # 100, and the resulting error is large, not suspiciously small.
    assert result.sample_predictions[0]["predicted"] == pytest.approx(100.0, abs=1.0)
    assert result.metrics["mae"] > 1000


def test_forecast_train_and_test_sizes_are_reported_and_sum_to_the_input():
    values = [float(i) for i in range(12)]
    result = evaluate_and_forecast(values, periods_ahead=1, test_size=3)
    assert result.train_size + result.test_size == len(values)
    assert result.test_size == 3


def test_forecast_is_fully_deterministic_across_repeated_calls():
    values = [140000.0, 152000.0, 149500.0, 161000.0, 150600.0, 158200.0, 163000.0]
    first = evaluate_and_forecast(values, periods_ahead=1, test_size=2)
    second = evaluate_and_forecast(values, periods_ahead=1, test_size=2)
    assert first.metrics == second.metrics
    assert first.forecast_next == second.forecast_next


# --- churn: feature engineering + classifier -------------------------------


def _customer_row(cid: int, segment: str, order_count: int, revenue: float, last_order_date, signup_date="2025-01-01"):
    return {
        "customer_id": cid, "segment": segment, "signup_date": signup_date,
        "order_count": order_count, "total_revenue": revenue, "last_order_date": last_order_date,
    }


def _synthetic_customers(n: int, *, reference: str = "2026-06-01"):
    """Deterministic, hand-built customer population — half clearly active
    (recent orders, many logins), half clearly churned (no recent order,
    no activity) — so the classifier has a genuinely learnable signal."""
    customers = []
    activity = []
    for i in range(n):
        active = i % 2 == 0
        cid = i + 1
        customers.append(
            _customer_row(
                cid, "SMB", order_count=5 if active else 0, revenue=500.0 if active else 0.0,
                last_order_date="2026-05-20" if active else None,
            )
        )
        if active:
            activity.append({"customer_id": cid, "activity_type": "login", "cnt": 10})
    return customers, activity


def test_build_churn_feature_table_labels_never_ordered_customers_as_churned():
    customers, activity = _synthetic_customers(4)
    table = build_churn_feature_table(customers, activity, churn_window_days=90)
    never_ordered = table[table["order_count"] == 0]
    assert (never_ordered["churned"] == 1).all()


def test_build_churn_feature_table_labels_recent_orders_as_active():
    customers, activity = _synthetic_customers(4)
    table = build_churn_feature_table(customers, activity, churn_window_days=90)
    recent = table[table["order_count"] > 0]
    assert (recent["churned"] == 0).all()


def test_feature_columns_for_includes_activity_and_segment_columns():
    customers, activity = _synthetic_customers(4)
    table = build_churn_feature_table(customers, activity)
    columns = feature_columns_for(table)
    assert "login_count" in columns
    assert any(c.startswith("segment_") for c in columns)
    assert "churned" not in columns  # the label itself is never a feature


def test_feature_columns_for_never_leaks_the_columns_the_label_is_derived_from():
    """Phase 16 data-leakage audit: `churned` is computed FROM
    days_since_last_order/last_order_date (build_churn_feature_table) —
    none of the three may ever appear in the columns actually fed to
    fit_churn_classifier, or the model would trivially "predict" its own
    label instead of learning a genuine behavioral signal. This is the
    regression test protecting that separation: it fails immediately if a
    future change to _BASE_FEATURE_COLUMNS (or feature_columns_for's
    segment-column derivation) ever reintroduces one of these."""
    customers, activity = _synthetic_customers(80)
    table = build_churn_feature_table(customers, activity, churn_window_days=90)
    columns = set(feature_columns_for(table))
    leaking_columns = {"churned", "days_since_last_order", "last_order_date", "signup_date"}
    assert not (columns & leaking_columns)


def test_fitted_churn_model_coefficients_never_include_a_leaking_column():
    """Same guarantee, proven at the ACTUAL fit — not just the column list
    fit_churn_classifier intended to use, but what the real, returned
    feature_importance mapping (keyed by the columns genuinely passed to
    LogisticRegression.fit) contains."""
    customers, activity = _synthetic_customers(80)
    table = build_churn_feature_table(customers, activity, churn_window_days=90)
    result = fit_churn_classifier(table)
    assert result.ok is True
    leaking_columns = {"churned", "days_since_last_order", "last_order_date", "signup_date"}
    assert not (set(result.feature_importance.keys()) & leaking_columns)


def test_churn_classifier_reports_insufficient_data_below_minimum_customers():
    customers, activity = _synthetic_customers(MIN_CHURN_CUSTOMERS - 2)
    table = build_churn_feature_table(customers, activity, churn_window_days=90)
    result = fit_churn_classifier(table)
    assert result.ok is False
    assert "customers" in result.reason.lower()


def test_churn_classifier_reports_insufficient_data_when_one_class_is_too_rare():
    """Enough TOTAL customers, but almost all in one class — a classifier
    "trained" on 2 examples of the minority class isn't reliable, and
    fit_churn_classifier must say so rather than silently returning a
    misleadingly confident-looking result."""
    customers = [_customer_row(i, "SMB", 5, 500.0, "2026-05-20") for i in range(1, 41)]
    customers += [_customer_row(41, "SMB", 0, 0.0, None), _customer_row(42, "SMB", 0, 0.0, None)]
    table = build_churn_feature_table(customers, [], churn_window_days=90)
    result = fit_churn_classifier(table)
    assert result.ok is False
    assert str(MIN_CHURN_CLASS_COUNT) in result.reason


def test_churn_classifier_produces_real_metrics_and_train_test_separation():
    customers, activity = _synthetic_customers(80)
    table = build_churn_feature_table(customers, activity, churn_window_days=90)
    result = fit_churn_classifier(table)
    assert result.ok is True
    assert result.train_size + result.test_size == len(table)
    assert result.test_size == pytest.approx(len(table) * 0.25, abs=1)
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert key in result.metrics
        assert 0.0 <= result.metrics[key] <= 1.0
    # A clean, separable synthetic signal (active customers have orders AND
    # logins, churned customers have neither) should be learned well.
    assert result.metrics["accuracy"] > 0.9


def test_churn_classifier_is_deterministic_across_repeated_calls():
    customers, activity = _synthetic_customers(80)
    table = build_churn_feature_table(customers, activity, churn_window_days=90)
    first = fit_churn_classifier(table)
    second = fit_churn_classifier(table)
    assert first.metrics == second.metrics
    assert first.feature_importance == second.feature_importance


def test_churn_classifier_sample_predictions_reference_real_customer_ids():
    customers, activity = _synthetic_customers(80)
    table = build_churn_feature_table(customers, activity, churn_window_days=90)
    result = fit_churn_classifier(table)
    real_ids = set(table["customer_id"])
    for pred in result.sample_predictions:
        assert pred["customer_id"] in real_ids
        assert pred["predicted_churn"] in (0, 1)
        assert 0.0 <= pred["churn_probability"] <= 1.0


def test_build_churn_feature_table_handles_an_empty_customer_list():
    table = build_churn_feature_table([], [])
    assert table.empty


def test_build_churn_feature_table_handles_no_activity_rows_at_all():
    customers, _ = _synthetic_customers(10)
    table = build_churn_feature_table(customers, [], churn_window_days=90)
    assert "login_count" in table.columns
    assert (table["login_count"] == 0).all()
