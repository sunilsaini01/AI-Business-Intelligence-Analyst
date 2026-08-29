"""ML Agent (Sec 1, Sec 2, Sec 5). Phase 15, Objective 4 — the final
missing analytical component from the original scaffolding.

Runs on every request (same "always in the linear chain, decides
internally whether it has anything to do" pattern as analysis_agent/
visualization_agent — see app/graph/workflow.py), but only ever does real
work when `state["intent"] == "predictive"` (set by the Supervisor's plan,
app/agents/schemas.py::SupervisorPlan). Two supported tasks, chosen by
keyword match against the question, never guessed by an LLM:

- forecasting: linear-trend baseline over monthly revenue, time-aware
  train/test split (app/tools/ml_tools.py::evaluate_and_forecast).
- churn_risk: logistic regression over deterministic per-customer
  features (app/tools/ml_tools.py::build_churn_feature_table /
  fit_churn_classifier).

Reads: state["question"], state["intent"] (set by the Supervisor's plan,
before sql_agent/analysis_agent/this node ever run — see
app/graph/workflow.py's node order). Writes: state["ml_results"].

Data access is exclusively through app/tools/database_tools.py::run_query
— the SAME safety pipeline (AST validation, schema allow-list, LIMIT
clamp, readonly role) every other query in this system goes through. The
SQL here is fixed/reviewed, not LLM-generated, but it is NOT exempt from
validation — run_query re-validates every query regardless of origin.

Sec 5 RULE: this module must never import app.core.llm (same CI grep as
analysis_agent.py) — zero LLM calls for fit/predict/evaluate. Every
number in `ml_results` comes from an actual model run against real query
results; nothing here is narrated, and nothing here decides what the
final report says — app/agents/supervisor.py's synthesis step and
app/agents/report_agent.py's deterministic formatting own that (Sec
"never duplicate report generation logic inside the ML Agent").

On "not appropriate" or "insufficient data": `state["ml_results"]` is
still a structured dict (not silently `None`) with `ok=False` and a plain-
English `reason` — Report Agent surfaces that honestly rather than
inventing a prediction or silently pretending nothing was asked. `None`
is reserved for the different case where a predictive task was never
attempted at all (the question wasn't predictive to begin with).
"""

from __future__ import annotations

import time
from typing import Any

from app.graph.state import AgentState, trace_event
from app.tools.database_tools import run_query
from app.tools.ml_tools import (
    DEFAULT_CHURN_WINDOW_DAYS,
    build_churn_feature_table,
    evaluate_and_forecast,
    fit_churn_classifier,
)

_FORECAST_KEYWORDS = (
    "forecast", "predict", "projection", "next month", "next quarter",
    "next year", "trend", "expected revenue", "future revenue",
)
_CHURN_KEYWORDS = (
    "churn", "risk", "retention", "attrition", "likely to leave",
    "likely to cancel", "at risk", "cancel",
)

# Fixed, reviewed queries — not LLM-generated — but still run through the
# exact same validate_sql() + readonly-role pipeline as every other query
# in this system (app/tools/database_tools.py::run_query never skips
# validation based on where a query came from).
_FORECAST_SQL = """
SELECT date_trunc('month', o.order_date)::date AS month,
       SUM(oi.quantity * oi.unit_price * (1 - oi.discount)) AS revenue
FROM analytics.orders o
JOIN analytics.order_items oi ON oi.order_id = o.order_id
GROUP BY 1
ORDER BY 1
"""

_CHURN_CUSTOMER_SQL = """
SELECT c.customer_id, c.segment, c.signup_date,
       COUNT(DISTINCT o.order_id) AS order_count,
       COALESCE(SUM(oi.quantity * oi.unit_price * (1 - oi.discount)), 0) AS total_revenue,
       MAX(o.order_date) AS last_order_date
FROM analytics.customers c
LEFT JOIN analytics.orders o ON o.customer_id = c.customer_id
LEFT JOIN analytics.order_items oi ON oi.order_id = o.order_id
GROUP BY c.customer_id, c.segment, c.signup_date
"""

_CHURN_ACTIVITY_SQL = """
SELECT customer_id, activity_type, COUNT(*) AS cnt
FROM analytics.customer_activity
GROUP BY customer_id, activity_type
"""


def _select_task(question: str) -> str | None:
    q = question.lower()
    if any(k in q for k in _CHURN_KEYWORDS):
        return "churn_risk"
    if any(k in q for k in _FORECAST_KEYWORDS):
        return "forecasting"
    return None


def _not_appropriate(reason: str) -> dict[str, Any]:
    return {"ok": False, "status": "not_appropriate", "task": None, "reason": reason}


def _insufficient_data(task: str, reason: str) -> dict[str, Any]:
    return {"ok": False, "status": "insufficient_data", "task": task, "reason": reason}


def _ml_error(task: str, exc: Exception) -> dict[str, Any]:
    """Phase 16: an unexpected failure INSIDE the model computation itself
    (as opposed to a DB-layer rejection, which run_query already turns into
    a safe result) — e.g. a future feature-engineering change that raises
    on some data shape. Same `ok=False` shape as _insufficient_data (Report
    Agent's _format_ml_summary already renders any ok=False result via its
    `reason` uniformly), just a distinct `status` for honest diagnostics.
    Never the raw exception message — only its type name, same
    non-leaking convention as app/tools/database_tools.py::run_query and
    app/core/errors.py. This is what keeps a genuine ML computation bug
    from crashing the ENTIRE analysis (Sec 9 graceful degradation) instead
    of just the predictive portion of it."""
    return {
        "ok": False,
        "status": "error",
        "task": task,
        "reason": f"ML computation failed unexpectedly ({type(exc).__name__}).",
    }


async def _run_forecast() -> dict[str, Any]:
    result = await run_query(_FORECAST_SQL)
    if not result.ok or not result.rows:
        return _insufficient_data(
            "forecasting",
            "Could not retrieve a monthly revenue history to forecast from"
            + (f" ({result.rejection_reason})." if result.rejection_reason else "."),
        )

    values = [float(row["revenue"]) for row in result.rows if row.get("revenue") is not None]
    try:
        forecast = evaluate_and_forecast(values, periods_ahead=1, test_size=2)
    except Exception as exc:  # noqa: BLE001 — degrade gracefully, never crash the whole analysis (Sec 9)
        return _ml_error("forecasting", exc)
    if not forecast.ok:
        return _insufficient_data("forecasting", forecast.reason or "Insufficient history to forecast.")

    mape = forecast.metrics.get("mape_pct")
    return {
        "ok": True,
        "status": "ok",
        "task": "forecasting",
        "target": "monthly_revenue",
        "features": ["month_index"],
        "model_name": "linear_trend_baseline",
        "train_size": forecast.train_size,
        "test_size": forecast.test_size,
        "metrics": forecast.metrics,
        "sample_predictions": forecast.sample_predictions,
        "forecast_next": forecast.forecast_next,
        "feature_importance": None,
        "limitations": [
            "A simple linear-trend baseline, not a seasonal or causal model — it will not capture "
            "seasonality, promotions, or one-off events.",
            "Evaluated on a short held-out tail (time-aware split, never shuffled); treat the error "
            "metrics as directional, not a precise guarantee.",
        ],
        "confidence": "Medium" if mape is not None and mape < 20 else "Low",
    }


async def _run_churn() -> dict[str, Any]:
    customer_result = await run_query(_CHURN_CUSTOMER_SQL)
    if not customer_result.ok or not customer_result.rows:
        return _insufficient_data(
            "churn_risk",
            "Could not retrieve customer order history to assess churn risk"
            + (f" ({customer_result.rejection_reason})." if customer_result.rejection_reason else "."),
        )
    activity_result = await run_query(_CHURN_ACTIVITY_SQL)

    try:
        feature_table = build_churn_feature_table(
            customer_result.rows, activity_result.rows if activity_result.ok else []
        )
        churn = fit_churn_classifier(feature_table)
    except Exception as exc:  # noqa: BLE001 — degrade gracefully, never crash the whole analysis (Sec 9)
        return _ml_error("churn_risk", exc)
    if not churn.ok:
        return _insufficient_data("churn_risk", churn.reason or "Insufficient data to train a churn model.")

    roc_auc = churn.metrics.get("roc_auc", 0.0)
    return {
        "ok": True,
        "status": "ok",
        "task": "churn_risk",
        "target": "churned",
        "features": churn.features,
        "model_name": "logistic_regression",
        "train_size": churn.train_size,
        "test_size": churn.test_size,
        "metrics": churn.metrics,
        "sample_predictions": churn.sample_predictions,
        "feature_importance": churn.feature_importance,
        "limitations": [
            f"Churn is DEFINED here as no order within {DEFAULT_CHURN_WINDOW_DAYS} days of the "
            "dataset's most recent order — a modeling choice, not a fact reported by customers.",
            "Feature importance reflects statistical association in this dataset, not proven cause — "
            "no causal claim is made or implied.",
        ],
        "confidence": "Medium" if roc_auc >= 0.7 else "Low",
    }


async def ml_agent_node(state: AgentState) -> AgentState:
    state["trace"].append(trace_event("ml_agent", "enter"))
    started = time.perf_counter()

    if state["intent"] != "predictive":
        state["ml_results"] = None
    else:
        task = _select_task(state["question"])
        if task == "forecasting":
            state["ml_results"] = await _run_forecast()
        elif task == "churn_risk":
            state["ml_results"] = await _run_churn()
        else:
            state["ml_results"] = _not_appropriate(
                "Question was classified as predictive, but didn't match a supported ML task "
                "(revenue forecasting or customer churn/risk prediction)."
            )

    state["trace"].append(
        trace_event("ml_agent", "exit", duration_ms=(time.perf_counter() - started) * 1000)
    )
    return state
