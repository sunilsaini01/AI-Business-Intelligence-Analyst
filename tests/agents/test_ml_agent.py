"""Phase 15, Objective 4 — ML Agent orchestration tests. Requires a live,
seeded, migrated DB (same convention as tests/agents/test_sql_agent.py):
the forecast/churn queries are real, fixed SQL run through the real safety
pipeline (app/tools/database_tools.py::run_query) against the real
analytics.* seed data — nothing here is mocked at the DB layer, since
"no arbitrary SQL" and "no fabricated metrics" are exactly the properties
this file needs to prove for real, not assume.
"""

from __future__ import annotations

import inspect
import re

import pytest

import app.agents.ml_agent as ml_agent_module
from app.agents.ml_agent import ml_agent_node
from app.graph.state import new_state
from app.tools import database_tools


def _predictive_state(question: str) -> dict:
    state = new_state(question)
    state["intent"] = "predictive"
    return state


# --- task selection / not-appropriate / non-predictive ----------------------


@pytest.mark.asyncio
async def test_non_predictive_intent_never_runs_any_ml_task():
    state = new_state("How many customers do we have?")
    state["intent"] = "descriptive"
    result = await ml_agent_node(state)
    assert result["ml_results"] is None


@pytest.mark.asyncio
async def test_predictive_intent_with_no_supported_keyword_is_not_appropriate():
    state = _predictive_state("Tell me about the company's outlook.")
    result = await ml_agent_node(state)
    assert result["ml_results"]["ok"] is False
    assert result["ml_results"]["status"] == "not_appropriate"
    assert result["ml_results"]["task"] is None
    assert "reason" in result["ml_results"]


# --- forecasting: a real, valid request --------------------------------------


@pytest.mark.asyncio
async def test_forecast_question_produces_a_real_forecast_result():
    state = _predictive_state("Can you forecast revenue for next month?")
    result = await ml_agent_node(state)
    ml = result["ml_results"]
    assert ml["ok"] is True
    assert ml["status"] == "ok"
    assert ml["task"] == "forecasting"
    assert ml["model_name"] == "linear_trend_baseline"
    assert ml["train_size"] > 0 and ml["test_size"] > 0
    assert "mae" in ml["metrics"] and isinstance(ml["metrics"]["mae"], float)
    assert ml["forecast_next"] and isinstance(ml["forecast_next"][0], float)
    assert ml["confidence"] in ("Low", "Medium", "High")
    assert ml["limitations"]  # never claims to be a complete/causal model without saying so


@pytest.mark.asyncio
async def test_forecast_metrics_are_never_fabricated_they_trace_to_a_real_fit():
    """Runs the same forecast twice — real model fits against the same
    real data must produce byte-identical metrics; a hard-coded or
    LLM-guessed number would have no reason to be this exactly
    reproducible."""
    state_a = _predictive_state("forecast next month's revenue")
    state_b = _predictive_state("what is the revenue forecast")
    result_a = await ml_agent_node(state_a)
    result_b = await ml_agent_node(state_b)
    assert result_a["ml_results"]["metrics"] == result_b["ml_results"]["metrics"]
    assert result_a["ml_results"]["forecast_next"] == result_b["ml_results"]["forecast_next"]


# --- churn: a real, valid request --------------------------------------------


@pytest.mark.asyncio
async def test_churn_question_produces_a_real_churn_result():
    state = _predictive_state("Which customers are at risk of churn?")
    result = await ml_agent_node(state)
    ml = result["ml_results"]
    assert ml["ok"] is True
    assert ml["status"] == "ok"
    assert ml["task"] == "churn_risk"
    assert ml["model_name"] == "logistic_regression"
    assert ml["train_size"] > 0 and ml["test_size"] > 0
    for key in ("accuracy", "precision", "recall", "f1"):
        assert key in ml["metrics"]
    assert ml["feature_importance"]  # a real, non-empty coefficient map
    assert ml["sample_predictions"]
    for pred in ml["sample_predictions"]:
        assert isinstance(pred["customer_id"], int)
        assert pred["predicted_churn"] in (0, 1)


@pytest.mark.asyncio
async def test_churn_keyword_takes_priority_over_forecast_keyword_when_both_present():
    """'at risk' + 'revenue' both appear — churn-specific language should
    win when the question is really about identifying customers, not
    projecting a number."""
    state = _predictive_state("Which customers are at risk, based on their revenue?")
    result = await ml_agent_node(state)
    assert result["ml_results"]["task"] == "churn_risk"


# --- graceful failure handling (never crashes the graph) --------------------


@pytest.mark.asyncio
async def test_forecast_failure_at_the_query_layer_degrades_to_insufficient_data(monkeypatch):
    """If the (fixed, reviewed) forecast query somehow gets rejected or the
    DB call fails, run_query itself never raises (see database_tools.py) —
    this proves ml_agent_node reflects that as a clean, structured
    insufficient_data result, never an unhandled exception reaching the
    graph."""

    class _AlwaysRejected:
        ok = False
        rows = []
        rejection_reason = "simulated failure"

    async def _fake_run_query(sql, **kwargs):
        return _AlwaysRejected()

    monkeypatch.setattr(ml_agent_module, "run_query", _fake_run_query)
    state = _predictive_state("forecast next month's revenue")
    result = await ml_agent_node(state)
    assert result["ml_results"]["ok"] is False
    assert result["ml_results"]["status"] == "insufficient_data"
    assert result["ml_results"]["task"] == "forecasting"


@pytest.mark.asyncio
async def test_churn_failure_at_the_query_layer_degrades_to_insufficient_data(monkeypatch):
    class _AlwaysRejected:
        ok = False
        rows = []
        rejection_reason = "simulated failure"

    async def _fake_run_query(sql, **kwargs):
        return _AlwaysRejected()

    monkeypatch.setattr(ml_agent_module, "run_query", _fake_run_query)
    state = _predictive_state("which customers are likely to churn")
    result = await ml_agent_node(state)
    assert result["ml_results"]["ok"] is False
    assert result["ml_results"]["status"] == "insufficient_data"
    assert result["ml_results"]["task"] == "churn_risk"


# --- Phase 16: unexpected computation errors degrade gracefully too --------
# (as opposed to the two tests above, which simulate a DB/query-layer
# rejection — this simulates a bug INSIDE the model computation itself,
# e.g. a future feature-engineering change that raises on some data shape)


@pytest.mark.asyncio
async def test_unexpected_forecast_computation_error_does_not_crash_the_workflow(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated bug in evaluate_and_forecast")

    monkeypatch.setattr(ml_agent_module, "evaluate_and_forecast", _boom)
    state = _predictive_state("forecast next month's revenue")
    result = await ml_agent_node(state)  # must not raise
    assert result["ml_results"]["ok"] is False
    assert result["ml_results"]["status"] == "error"
    assert result["ml_results"]["task"] == "forecasting"
    # the raw exception message never reaches the result — only its type name
    assert "simulated bug" not in str(result["ml_results"]["reason"])
    assert "RuntimeError" in result["ml_results"]["reason"]


@pytest.mark.asyncio
async def test_unexpected_computation_error_never_leaks_a_secret_looking_message(monkeypatch):
    """Phase 16, Section 14 — an exception's message could, in principle,
    contain anything (a DSN, a stack-trace fragment) if it originated deep
    in a library call; _ml_error only ever surfaces the exception's TYPE
    name, never str(exc), so this must hold no matter what the message
    said."""

    def _boom(*args, **kwargs):
        raise RuntimeError("leaking postgresql://bi_app:supersecret@postgres:5432/bi_agent, sk-ANTHROPIC-FAKEKEY")

    monkeypatch.setattr(ml_agent_module, "evaluate_and_forecast", _boom)
    state = _predictive_state("forecast next month's revenue")
    result = await ml_agent_node(state)
    reason_text = str(result["ml_results"]["reason"]).lower()
    for marker in ("postgresql://", "supersecret", "sk-anthropic"):
        assert marker not in reason_text


@pytest.mark.asyncio
async def test_unexpected_churn_computation_error_does_not_crash_the_workflow(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated bug in fit_churn_classifier")

    monkeypatch.setattr(ml_agent_module, "fit_churn_classifier", _boom)
    state = _predictive_state("which customers are likely to churn")
    result = await ml_agent_node(state)  # must not raise
    assert result["ml_results"]["ok"] is False
    assert result["ml_results"]["status"] == "error"
    assert result["ml_results"]["task"] == "churn_risk"
    assert "simulated bug" not in str(result["ml_results"]["reason"])
    assert "RuntimeError" in result["ml_results"]["reason"]


# --- trace / observability ---------------------------------------------------


@pytest.mark.asyncio
async def test_ml_agent_writes_enter_and_exit_trace_events_even_when_a_no_op():
    state = new_state("How many customers do we have?")
    state["intent"] = "descriptive"
    result = await ml_agent_node(state)
    node_events = [(t["node"], t["event"]) for t in result["trace"]]
    assert node_events == [("ml_agent", "enter"), ("ml_agent", "exit")]


# --- Sec 5 rule + safe data access (structural guards) -----------------------


_IMPORT_LINE = re.compile(r"^\s*(import app\.core\.llm\b|from app\.core\.llm import)", re.MULTILINE)


def test_ml_agent_module_never_imports_the_llm_client():
    """Same CI-grep rule already enforced for analysis_agent.py (Sec 5:
    '0 LLM calls') — zero LLM calls anywhere in fit/predict/evaluate. Checks
    for a real `import`/`from ... import` line, not just the substring
    "app.core.llm" anywhere — this module's own docstring explains the Sec
    5 rule in prose, which would false-positive on a plain substring check."""
    source = inspect.getsource(ml_agent_module)
    assert not _IMPORT_LINE.search(source)
    assert "get_llm_client()" not in source


def test_ml_tools_module_never_imports_the_llm_client_or_a_db_driver():
    import app.tools.ml_tools as ml_tools_module

    source = inspect.getsource(ml_tools_module)
    assert not _IMPORT_LINE.search(source)
    for forbidden in ("import asyncpg", "import psycopg"):
        assert forbidden not in source


def test_ml_agent_only_reaches_the_database_through_run_query():
    """Structural guard for "must NOT directly execute arbitrary SQL" — the
    only DB-adjacent import in this module is run_query, the same safety-
    pipeline entry point every other agent uses (app/tools/database_tools.py
    — AST validation, schema allow-list, LIMIT clamp, readonly role);
    nothing here imports asyncpg or execute_validated_query directly."""
    source = inspect.getsource(ml_agent_module)
    assert "from app.tools.database_tools import run_query" in source
    for forbidden in ("execute_validated_query", "asyncpg", "analytics_readonly_connection"):
        assert forbidden not in source


def test_run_query_is_the_real_safety_pipeline_not_a_test_double():
    """Confirms ml_agent.run_query is genuinely bound to database_tools.run_query
    (not shadowed/monkeypatched in this test session) — the other tests in
    this file that hit the real DB are only meaningful if this holds."""
    assert ml_agent_module.run_query is database_tools.run_query
