"""Tests for app/evaluation/evaluator.py.

`evaluate_case_from_state` is pure (no LLM, no DB) and tested against
hand-built AgentState fixtures — no live anything required.

`run_case_live` tests use ScriptedLLMClient (never a real LLM call) but DO
run the real graph against the real seeded DB, same as
tests/integration/test_workflow.py — only the quota-handling and generic-
error-handling tests substitute a fake client that raises directly, so the
whole suite still needs no network/API key.
"""

from __future__ import annotations

import httpx
import pytest
import groq

from app.agents.schemas import SQLGeneration, SupervisorPlan, SupervisorSynthesis
from app.evaluation.benchmark import load_benchmark
from app.evaluation.evaluator import evaluate_case_from_state, run_case_live
from app.graph.state import new_state
from tests.fakes import ScriptedLLMClient


def _rate_limit_error() -> groq.RateLimitError:
    resp = httpx.Response(429, request=httpx.Request("POST", "http://test"))
    return groq.RateLimitError("quota exhausted", response=resp, body=None)


class _AlwaysRaisesLLMClient:
    """Not a ScriptedLLMClient: every call raises immediately, standing in
    for "the provider rejected the very first request" (quota already
    exhausted before this case even started)."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def complete(self, **kwargs):
        raise self._exc

    async def complete_structured(self, **kwargs):
        raise self._exc


BI_004 = next(c for c in load_benchmark("evaluation/datasets/benchmark.json") if c["id"] == "bi-004")
BI_001 = next(c for c in load_benchmark("evaluation/datasets/benchmark.json") if c["id"] == "bi-001")


def _grounded_diagnostic_state() -> dict:
    state = new_state(BI_004["question"])
    state["intent"] = "diagnostic"
    state["sql_queries"] = [{"text": "SELECT 1", "validated_ok": True, "rejection_reason": None, "rows": [], "row_count": 0, "exec_ms": 1.0}]
    state["analysis_results"] = {
        "period_comparisons": [
            {
                "baseline_period": "2026-06", "current_period": "2026-07",
                "baseline_value": 161445.80, "current_value": 150633.02,
                "absolute_change": -10812.78, "percentage_change": -6.7,
            }
        ],
        "contributions": [
            {
                "total_change": -10812.78,
                "contributors": [
                    {"group": "Enterprise", "current_value": 6171.99, "prior_value": 16782.83,
                     "change": -10610.84, "pct_of_total_change": 74.4},
                ],
            }
        ],
    }
    state["charts"] = [{"chart_type": "bar", "title": "Revenue", "subtitle": None, "x_axis": None, "y_axis": None,
                         "group_by": None, "sort": None, "data": [], "units": None, "source_analysis": "period_comparison",
                         "reason": "", "limitations": [], "path": "", "spec_summary": {}}]
    state["critic_feedback"] = {
        "status": "PASS", "score": 1.0, "findings": [], "verified_claims": [], "unsupported_claims": [],
        "recommendations": [], "target_agent": None,
    }
    state["report"] = {
        "executive_summary": "Revenue fell from 161445.80 to 150633.02 (-6.7%), driven mainly by Enterprise.",
        "key_findings": ["June revenue: 161445.80", "July revenue: 150633.02", "Enterprise change: -10610.84"],
        "evidence": [], "recommendations": [], "confidence": "Medium", "limitations": "",
    }
    state["trace"] = [
        {"node": "supervisor", "event": "enter", "timestamp": "", "duration_ms": None},
        {"node": "supervisor", "event": "exit", "timestamp": "", "duration_ms": 50.0},
    ]
    return state


def test_evaluate_case_from_state_passes_for_a_grounded_diagnostic_report():
    result = evaluate_case_from_state(BI_004, _grounded_diagnostic_state())
    assert result.status == "PASSED"
    assert result.sql_correct is True
    assert result.analysis_correct is True
    assert result.answer_correct is True
    assert result.grounded is True
    assert result.hallucination_detected is False
    assert result.critic_correct is True
    assert result.first_failing_level is None
    assert result.latency_ms == 50.0
    # Phase 10: this fixture predates the Report Generator (its report dict
    # has none of the 5 new presentation fields) — report_completeness is
    # correctly False, and crucially does NOT affect PASSED/first_failing_level
    # above (additive-only metric, see app/evaluation/metrics.py::
    # evaluate_report_completeness's docstring).
    assert result.report_completeness_correct is False


def test_evaluate_case_from_state_reports_error_when_no_report_was_produced():
    state = new_state(BI_004["question"])
    result = evaluate_case_from_state(BI_004, state)
    assert result.status == "ERROR"
    assert result.errors


def test_evaluate_case_from_state_fails_and_localizes_a_fabricated_number():
    state = _grounded_diagnostic_state()
    state["report"]["executive_summary"] += " Adjusted figure: 999999.99."
    result = evaluate_case_from_state(BI_004, state)
    assert result.status == "FAILED"
    assert result.grounded is False
    assert result.hallucination_detected is True
    assert result.first_failing_level == "end_to_end"


def test_evaluate_case_from_state_out_of_scope_case_passes_on_correct_short_circuit():
    case = dict(BI_004)
    case["expected_behavior"] = "out_of_scope"
    state = new_state("hello")
    state["intent"] = "out_of_scope"
    state["report"] = {
        "executive_summary": "I can't answer that.", "key_findings": [], "evidence": [],
        "recommendations": [], "confidence": "Low", "limitations": "",
    }
    result = evaluate_case_from_state(case, state)
    assert result.status == "PASSED"


def test_evaluate_case_from_state_out_of_scope_case_fails_if_sql_ran_anyway():
    case = dict(BI_004)
    case["expected_behavior"] = "out_of_scope"
    state = new_state("hello")
    state["intent"] = "descriptive"
    state["sql_queries"] = [{"text": "SELECT 1", "validated_ok": True, "rejection_reason": None, "rows": [], "row_count": 0, "exec_ms": 1.0}]
    state["report"] = {
        "executive_summary": "x", "key_findings": [], "evidence": [], "recommendations": [], "confidence": "Low", "limitations": "",
    }
    result = evaluate_case_from_state(case, state)
    assert result.status == "FAILED"


@pytest.mark.asyncio
async def test_run_case_live_reports_skipped_quota_not_a_failure():
    result = await run_case_live(BI_001, llm=_AlwaysRaisesLLMClient(_rate_limit_error()))
    assert result.status == "SKIPPED_QUOTA"
    assert result.errors == []  # not an application error — explicitly not conflated with one
    assert "quota" in result.notes[0].lower() or "rate" in result.notes[0].lower()


@pytest.mark.asyncio
async def test_run_case_live_reports_error_status_on_a_genuine_exception():
    result = await run_case_live(BI_001, llm=_AlwaysRaisesLLMClient(ValueError("boom")))
    assert result.status == "ERROR"
    assert "boom" in result.errors[0]


@pytest.mark.asyncio
async def test_run_case_live_runs_the_real_graph_end_to_end():
    """Same shape as tests/integration/test_workflow.py's round trip — real
    DB, real deterministic tools, only the LLM is scripted — proving
    run_case_live drives the SAME production graph, not a second pipeline.
    """
    fake_llm = ScriptedLLMClient(
        {
            SupervisorPlan: [
                SupervisorPlan(
                    out_of_scope=False, intent="descriptive", target_schema="analytics",
                    steps=["Count total customers"], reasoning="Simple count query.",
                )
            ],
            SQLGeneration: [SQLGeneration(sql="SELECT COUNT(*) AS n FROM analytics.customers", purpose="Total customer count")],
            SupervisorSynthesis: [
                SupervisorSynthesis(
                    insufficient_evidence=False, executive_summary="There are customers in the database.",
                    key_findings=["Customer count evidence retrieved."], confidence="High", limitations="",
                )
            ],
        }
    )
    case = {"id": "adhoc", "question": "How many customers do we have?", "expected_tables": ["analytics.customers"]}
    result = await run_case_live(case, llm=fake_llm)
    assert result.status in ("PASSED", "FAILED")  # reached scoring, not an error/skip
    assert result.sql_correct is True
    assert result.latency_ms is not None
    # ScriptedLLMClient (a test fake, not a real LLMClient/GroqLLMClient) has
    # no total_usage attribute — run_case_live must tolerate that (getattr
    # default), not crash, since token accounting is best-effort.
    assert result.token_usage is None
    # Phase 10: run_case_live drives the real graph end to end, which now
    # includes report_agent — the finalized report should carry its
    # presentation-layer sections for real, not just in a hand-built fixture.
    assert result.report_completeness_correct is True
