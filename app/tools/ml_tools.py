"""Deterministic sklearn helpers for the ML Agent (Sec 2, Sec 5). No LLM
calls anywhere in this module (Sec 5 "0 LLM calls" rule) — every number
here comes from an actual model fit/evaluation against real query results,
never invented or estimated by an agent.

Two supported tasks, deliberately simple/explainable rather than reaching
for anything heavier (Phase 15, Objective 4 — "prefer simple, explainable
models"):

- `evaluate_and_forecast`: linear-trend baseline (`baseline_forecast`,
  unchanged since Phase 8) with a TIME-AWARE train/test split — the held-
  out points are always the most recent ones, never a random sample,
  because shuffling a time series lets the model "see the future" during
  evaluation and reports a misleadingly good score.
- `fit_churn_classifier`: logistic regression over deterministic, per-
  customer features (order history + activity counts) built by
  `build_churn_feature_table` — coefficients double as feature importance,
  which is the point of choosing this over a black-box model here. A
  RANDOM (stratified) train/test split is appropriate for this one — each
  customer is an independent row, not a time-ordered sequence.

Both entry points return a dataclass with `ok: bool`; `ok=False` means
"not enough data to fit/evaluate reliably", never an exception — the ML
Agent (app/agents/ml_agent.py) turns that into a graceful "ML not
appropriate" result rather than a crash, per Sec 9.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Forecasting -------------------------------------------------------------

MIN_FORECAST_POINTS = 6  # fewer than this and a held-out test tail is meaningless


def baseline_forecast(monthly_values: np.ndarray, periods_ahead: int = 1) -> np.ndarray:
    """Linear-trend baseline via numpy.polyfit — the first thing to try, not XGBoost."""
    x = np.arange(len(monthly_values))
    coeffs = np.polyfit(x, monthly_values, deg=1)
    future_x = np.arange(len(monthly_values), len(monthly_values) + periods_ahead)
    return np.polyval(coeffs, future_x)


@dataclass
class ForecastResult:
    ok: bool
    reason: str | None = None
    train_size: int = 0
    test_size: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    sample_predictions: list[dict[str, Any]] = field(default_factory=list)
    forecast_next: list[float] = field(default_factory=list)


def evaluate_and_forecast(
    monthly_values: list[float], *, periods_ahead: int = 1, test_size: int = 2
) -> ForecastResult:
    """Time-aware evaluation: the LAST `test_size` points are held out
    (never shuffled) and never used to fit the model being scored against
    them — the baseline is fit only on everything before the held-out
    tail, then scored against the real values there. `forecast_next` is a
    separate fit against the FULL history (train+test): once a model's
    accuracy has been honestly evaluated on held-out data, using all
    available history for the actual forward-looking forecast is standard
    practice, not a second "test" of the same claim.
    """
    n = len(monthly_values)
    if n < MIN_FORECAST_POINTS:
        return ForecastResult(
            ok=False, reason=f"Need at least {MIN_FORECAST_POINTS} historical monthly periods, got {n}."
        )

    effective_test_size = test_size if 1 <= test_size < n else max(1, min(2, n - 1))
    values = np.asarray(monthly_values, dtype=float)
    train, test = values[: n - effective_test_size], values[n - effective_test_size :]

    predicted_test = baseline_forecast(train, periods_ahead=effective_test_size)
    mae = float(mean_absolute_error(test, predicted_test))
    rmse = float(np.sqrt(np.mean((test - predicted_test) ** 2)))

    metrics = {"mae": round(mae, 4), "rmse": round(rmse, 4)}
    if np.all(test != 0):
        mape = float(np.mean(np.abs((test - predicted_test) / test))) * 100
        metrics["mape_pct"] = round(mape, 2)

    sample_predictions = [
        {"period_index": n - effective_test_size + i, "actual": float(a), "predicted": float(p)}
        for i, (a, p) in enumerate(zip(test, predicted_test))
    ]

    forecast_next = [float(v) for v in baseline_forecast(values, periods_ahead=periods_ahead)]

    return ForecastResult(
        ok=True,
        train_size=len(train),
        test_size=len(test),
        metrics=metrics,
        sample_predictions=sample_predictions,
        forecast_next=forecast_next,
    )


# --- Churn / risk classification ---------------------------------------------

MIN_CHURN_CUSTOMERS = 30
MIN_CHURN_CLASS_COUNT = 5  # need at least this many examples of EACH class
DEFAULT_CHURN_WINDOW_DAYS = 180

_ACTIVITY_FEATURE_COLUMNS = ("login_count", "support_ticket_count", "cart_abandon_count")
_BASE_FEATURE_COLUMNS = ("order_count", "total_revenue", "tenure_days")


def build_churn_feature_table(
    customer_rows: list[dict[str, Any]],
    activity_rows: list[dict[str, Any]],
    *,
    churn_window_days: int = DEFAULT_CHURN_WINDOW_DAYS,
) -> pd.DataFrame:
    """Deterministic feature engineering — every value traces back to a
    real row from the safe query layer (app/tools/database_tools.py::
    run_query), nothing invented, no LLM involved. One row per customer:

    - order_count, total_revenue: from the customer's own order history
      (0 for a customer who never ordered).
    - tenure_days: days between signup and the dataset's own most recent
      order date (used as "now" — the data is synthetic/future-dated, so
      the real wall-clock date would be meaningless here).
    - login_count / support_ticket_count / cart_abandon_count: from
      analytics.customer_activity, 0 where a customer has none.
    - one-hot columns for `segment`.
    - `churned` (the label): 1 if the customer's last order is more than
      `churn_window_days` before the dataset's most recent order date, or
      they never ordered at all; 0 otherwise. Derived purely from the
      data — never asked of an LLM, never a claim this table itself makes
      about causality.
    """
    customers = pd.DataFrame(customer_rows)
    if customers.empty:
        return customers

    for money_col in ("total_revenue",):
        customers[money_col] = customers[money_col].astype(float)
    customers["order_count"] = customers["order_count"].astype(float)
    customers["signup_date"] = pd.to_datetime(customers["signup_date"])
    customers["last_order_date"] = pd.to_datetime(customers["last_order_date"])

    activity = pd.DataFrame(activity_rows)
    if not activity.empty:
        pivot = activity.pivot_table(
            index="customer_id", columns="activity_type", values="cnt", fill_value=0, aggfunc="sum"
        )
        pivot.columns = [f"{c}_count" for c in pivot.columns]
    else:
        pivot = pd.DataFrame(index=pd.Index([], name="customer_id"))

    df = customers.set_index("customer_id").join(pivot, how="left")
    for col in _ACTIVITY_FEATURE_COLUMNS:
        df[col] = df[col].fillna(0.0) if col in df.columns else 0.0

    reference_date = customers["last_order_date"].max()
    tenure = (reference_date - df["signup_date"]).dt.days
    df["tenure_days"] = tenure.clip(lower=0).fillna(0.0)

    days_since_last_order = (reference_date - df["last_order_date"]).dt.days
    df["churned"] = ((days_since_last_order > churn_window_days) | df["last_order_date"].isna()).astype(int)

    segment_dummies = pd.get_dummies(df["segment"], prefix="segment", dtype=float)
    df = pd.concat([df, segment_dummies], axis=1)

    return df.reset_index()


def feature_columns_for(feature_table: pd.DataFrame) -> list[str]:
    """The exact columns fit_churn_classifier will use — base behavioral/
    activity features plus whichever segment one-hot columns this table
    actually produced (data-driven, never hardcoded against a specific
    segment list that could drift out of sync with the seed data)."""
    segment_cols = [c for c in feature_table.columns if c.startswith("segment_")]
    return [*_BASE_FEATURE_COLUMNS, *_ACTIVITY_FEATURE_COLUMNS, *segment_cols]


@dataclass
class ChurnResult:
    ok: bool
    reason: str | None = None
    train_size: int = 0
    test_size: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    sample_predictions: list[dict[str, Any]] = field(default_factory=list)
    feature_importance: dict[str, float] = field(default_factory=dict)
    features: list[str] = field(default_factory=list)


def fit_churn_classifier(feature_table: pd.DataFrame, *, sample_size: int = 10) -> ChurnResult:
    """Logistic regression, stratified random train/test split (each
    customer is an independent row — no time-ordering to respect, unlike
    the forecast). Standardizes features before fitting (logistic
    regression's coefficients are only comparable as "importance" on a
    common scale) — never done on churn_window_days or the label itself.
    """
    n = len(feature_table)
    if n < MIN_CHURN_CUSTOMERS:
        return ChurnResult(ok=False, reason=f"Need at least {MIN_CHURN_CUSTOMERS} customers with data, got {n}.")

    y = feature_table["churned"]
    class_counts = y.value_counts()
    if len(class_counts) < 2 or class_counts.min() < MIN_CHURN_CLASS_COUNT:
        return ChurnResult(
            ok=False,
            reason=(
                "Not enough examples of both churned and active customers to train/evaluate a "
                f"classifier reliably (need >= {MIN_CHURN_CLASS_COUNT} of each class)."
            ),
        )

    columns = feature_columns_for(feature_table)
    X = feature_table[columns].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
    }
    if len(set(y_test)) > 1:
        metrics["roc_auc"] = round(float(roc_auc_score(y_test, probabilities)), 4)

    feature_importance = {col: round(float(coef), 4) for col, coef in zip(columns, model.coef_[0])}

    customer_ids = feature_table.loc[X_test.index, "customer_id"]
    sample_predictions = [
        {"customer_id": int(cid), "predicted_churn": int(pred), "churn_probability": round(float(prob), 4)}
        for cid, pred, prob in list(zip(customer_ids, predictions, probabilities))[:sample_size]
    ]

    return ChurnResult(
        ok=True,
        train_size=len(X_train),
        test_size=len(X_test),
        metrics=metrics,
        sample_predictions=sample_predictions,
        feature_importance=feature_importance,
        features=columns,
    )
