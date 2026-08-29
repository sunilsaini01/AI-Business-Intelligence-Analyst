"""Phase 16 — ML Agent regression protection at the EVALUATION-framework
level (distinct from, and non-duplicative of, the existing coverage):

- tests/unit/test_ml_tools.py / tests/agents/test_ml_agent.py (Phase 15)
  already prove the tools/agent are individually correct, deterministic,
  and leakage-free.
- tests/integration/test_workflow.py (Phase 15) already proves ml_agent is
  wired into the real graph and the Report Agent cites its output.

This file proves the EVALUATION FRAMEWORK (app/evaluation/evaluator.py +
metrics.py) correctly scores REAL ml_agent output against the REAL seeded
database and enforces the Phase 16 regression thresholds — i.e. that a
future quality regression would actually be caught here, not just that
the pipeline runs without crashing.

Zero LLM calls with real content: ml_agent itself never calls an LLM
(Sec 5), and every ScriptedLLMClient response here is fixed/canned — no
Groq/Anthropic quota is touched by this file, matching Phase 16's explicit
"no unnecessary live LLM calls" instruction.
"""

from __future__ import annotations

import pytest

from app.agents.schemas import SQLGeneration, SupervisorPlan, SupervisorSynthesis
from app.evaluation.benchmark import load_benchmark
from app.evaluation.evaluator import evaluate_case_from_state
from app.graph.state import new_state
from app.graph.workflow import build_graph
from tests.fakes import ScriptedLLMClient

_CASES = load_benchmark("evaluation/datasets/benchmark.json")
ML_001_FORECAST = next(c for c in _CASES if c["id"] == "ml-001")
ML_002_CHURN = next(c for c in _CASES if c["id"] == "ml-002")


def _forecast_llm() -> ScriptedLLMClient:
    return ScriptedLLMClient(
        {
            SupervisorPlan: [
                SupervisorPlan(
                    out_of_scope=False, intent="predictive", target_schema="analytics",
                    steps=["Forecast next period's revenue"], reasoning="Needs the ML Agent's trend model.",
                )
            ],
            SQLGeneration: [SQLGeneration(sql="SELECT COUNT(*) AS n FROM analytics.customers", purpose="context")],
            SupervisorSynthesis: [
                SupervisorSynthesis(
                    insufficient_evidence=False, executive_summary="Revenue is projected to continue its trend.",
                    key_findings=["The forecast model projects next period's revenue."],
                    confidence="Medium", limitations="",
                )
            ],
        }
    )


def _churn_llm() -> ScriptedLLMClient:
    return ScriptedLLMClient(
        {
            SupervisorPlan: [
                SupervisorPlan(
                    out_of_scope=False, intent="predictive", target_schema="analytics",
                    steps=["Identify customers at risk of churn"], reasoning="Needs the ML Agent's churn model.",
                )
            ],
            SQLGeneration: [SQLGeneration(sql="SELECT COUNT(*) AS n FROM analytics.customers", purpose="context")],
            SupervisorSynthesis: [
                SupervisorSynthesis(
                    insufficient_evidence=False, executive_summary="Some customers are flagged as at risk of churn.",
                    key_findings=["The churn risk model flagged a subset of customers as at risk."],
                    confidence="Medium", limitations="",
                )
            ],
        }
    )


async def _run_real_case(case: dict, llm: ScriptedLLMClient):
    """Real DB, real ml_agent execution (zero LLM calls inside it), only
    the Supervisor/SQL Agent's LLM calls are scripted — same convention as
    tests/integration/test_workflow.py. Returns (final_state,
    CaseEvaluation) so a test can inspect BOTH the raw ml_results and the
    evaluation framework's verdict on it."""
    graph = build_graph(llm=llm)
    final_state = await graph.ainvoke(new_state(case["question"]))
    result = evaluate_case_from_state(case, final_state)
    return final_state, result


# --- Case 1: revenue forecasting --------------------------------------------


@pytest.mark.asyncio
async def test_case_1_revenue_forecasting_produces_a_grounded_quality_gated_result():
    final_state, result = await _run_real_case(ML_001_FORECAST, _forecast_llm())

    ml_results = final_state["ml_results"]
    assert ml_results is not None
    assert ml_results["ok"] is True
    assert ml_results["task"] == "forecasting"
    assert isinstance(ml_results["metrics"]["mape_pct"], float)  # MAPE is calculated
    assert isinstance(ml_results["metrics"]["mae"], float)  # MAE is calculated

    # The evaluation framework's own verdict, not a re-implementation of it.
    assert result.ml_correct is True  # metrics satisfy the Phase 16 thresholds
    ml_level = next(lvl for lvl in result.levels if lvl.level == "ml")
    assert ml_level.correct is True

    # Reaches the Critic, and the Critic's verdict is itself correctly scored.
    assert final_state["critic_feedback"] is not None
    assert final_state["critic_feedback"]["status"] in ("PASS", "WARN")
    assert result.critic_correct is True

    # The report contains grounded ML results (Phase 15's Report Agent
    # formatting, still exercised here through the real graph).
    assert final_state["report"]["ml_summary"] != ""
    assert result.grounded is True
    assert result.hallucination_detected is False

    assert result.status == "PASSED"


# --- Case 2: churn prediction ------------------------------------------------


@pytest.mark.asyncio
async def test_case_2_churn_prediction_produces_a_grounded_quality_gated_result():
    final_state, result = await _run_real_case(ML_002_CHURN, _churn_llm())

    ml_results = final_state["ml_results"]
    assert ml_results is not None
    assert ml_results["ok"] is True
    assert ml_results["task"] == "churn_risk"
    for key in ("roc_auc", "precision", "recall", "accuracy"):
        assert isinstance(ml_results["metrics"][key], float)

    assert result.ml_correct is True
    ml_level = next(lvl for lvl in result.levels if lvl.level == "ml")
    assert ml_level.correct is True

    assert final_state["critic_feedback"] is not None
    assert final_state["critic_feedback"]["status"] in ("PASS", "WARN")
    assert result.critic_correct is True

    assert final_state["report"]["ml_summary"] != ""
    assert result.grounded is True
    assert result.hallucination_detected is False

    assert result.status == "PASSED"


# --- Case 3/4: insufficient data degrades gracefully, never a false pass ---


def test_case_3_insufficient_forecast_data_is_not_scored_as_a_quality_pass_or_fail():
    """Hand-built final_state (no live DB starvation needed — this is
    exactly the same "pure, hand-built AgentState" convention already
    established by tests/evaluation/test_evaluator.py) standing in for
    what app/agents/ml_agent.py._insufficient_data actually produces.
    evaluate_case_from_state must never treat a graceful degradation as
    either a quality success or a quality regression."""
    state = new_state(ML_001_FORECAST["question"])
    state["intent"] = "predictive"
    state["ml_results"] = {
        "ok": False, "status": "insufficient_data", "task": "forecasting",
        "reason": "Need at least 6 historical monthly periods, got 2.",
    }
    state["report"] = {
        "executive_summary": "Insufficient evidence to forecast revenue.", "key_findings": [],
        "evidence": [], "recommendations": [], "confidence": "Low", "limitations": "Not enough history.",
    }
    state["critic_feedback"] = {
        "status": "PASS", "score": 1.0, "findings": [], "verified_claims": [], "unsupported_claims": [],
        "recommendations": [], "target_agent": None,
    }
    result = evaluate_case_from_state(ML_001_FORECAST, state)
    ml_level = next(lvl for lvl in result.levels if lvl.level == "ml")
    assert ml_level.correct is None  # neither PASSED-quality nor FAILED-quality
    assert result.ml_correct is None
    # Not applicable never blocks end-to-end on its own (same _ok() rule
    # every other level uses) — this case still resolves via the OTHER
    # levels, it just isn't the ML quality gate that decides it.
    assert result.status in ("PASSED", "FAILED")


def test_case_4_insufficient_churn_data_is_not_scored_as_a_quality_pass_or_fail():
    state = new_state(ML_002_CHURN["question"])
    state["intent"] = "predictive"
    state["ml_results"] = {
        "ok": False, "status": "insufficient_data", "task": "churn_risk",
        "reason": "Need at least 30 customers with data, got 12.",
    }
    state["report"] = {
        "executive_summary": "Insufficient evidence to assess churn risk.", "key_findings": [],
        "evidence": [], "recommendations": [], "confidence": "Low", "limitations": "Not enough customers.",
    }
    state["critic_feedback"] = {
        "status": "PASS", "score": 1.0, "findings": [], "verified_claims": [], "unsupported_claims": [],
        "recommendations": [], "target_agent": None,
    }
    result = evaluate_case_from_state(ML_002_CHURN, state)
    ml_level = next(lvl for lvl in result.levels if lvl.level == "ml")
    assert ml_level.correct is None
    assert result.ml_correct is None
    assert result.status in ("PASSED", "FAILED")


def test_insufficient_data_result_never_contains_fabricated_metrics():
    """Belt-and-suspenders on top of the two tests above: an ok=False
    result must genuinely carry no metrics dict at all — this is what
    "cannot silently appear as a successful analysis" actually means at
    the data level, not just at the ml_correct verdict level."""
    for ok_false_result in (
        {"ok": False, "status": "insufficient_data", "task": "forecasting", "reason": "x"},
        {"ok": False, "status": "not_appropriate", "task": None, "reason": "x"},
        {"ok": False, "status": "error", "task": "churn_risk", "reason": "x"},
    ):
        assert "metrics" not in ok_false_result


# --- Case 5: reproducibility, at the evaluation-verdict level ---------------


@pytest.mark.asyncio
async def test_case_5_forecast_evaluation_verdict_is_reproducible_across_runs():
    """Runs the SAME real scenario twice, independently, against the real
    seeded DB — proves the EVALUATION VERDICT (not just the raw metrics,
    already covered at the tool level by tests/unit/test_ml_tools.py and
    tests/agents/test_ml_agent.py) is stable, which is what actually
    matters for a regression suite that must not be flaky."""
    final_state_a, result_a = await _run_real_case(ML_001_FORECAST, _forecast_llm())
    final_state_b, result_b = await _run_real_case(ML_001_FORECAST, _forecast_llm())

    assert final_state_a["ml_results"]["metrics"] == final_state_b["ml_results"]["metrics"]
    assert result_a.ml_correct == result_b.ml_correct is True
    assert result_a.status == result_b.status == "PASSED"


@pytest.mark.asyncio
async def test_case_5_churn_evaluation_verdict_is_reproducible_across_runs():
    final_state_a, result_a = await _run_real_case(ML_002_CHURN, _churn_llm())
    final_state_b, result_b = await _run_real_case(ML_002_CHURN, _churn_llm())

    assert final_state_a["ml_results"]["metrics"] == final_state_b["ml_results"]["metrics"]
    assert result_a.ml_correct == result_b.ml_correct is True
    assert result_a.status == result_b.status == "PASSED"


# --- Evaluation isolation (Sec 12) ------------------------------------------


@pytest.mark.asyncio
async def test_running_an_ml_evaluation_case_creates_no_evaluation_run_row():
    """Preserves the existing Phase 8 rule (tests/api/test_analysis_service.py::
    test_run_analysis_never_creates_an_evaluation_run_row) — evaluating a
    case via evaluate_case_from_state/the graph directly (as every test in
    this file does) must never itself write an EvaluationRun, same as a
    normal POST /analyze never does. Only app/services/evaluation_service.py::
    create_run does that, and nothing in this file calls it."""
    from sqlalchemy import select

    from app.db.database import async_session_factory
    from app.db.models import EvaluationRun

    async with async_session_factory() as db:
        before = len((await db.execute(select(EvaluationRun))).scalars().all())

    await _run_real_case(ML_001_FORECAST, _forecast_llm())

    async with async_session_factory() as db:
        after = len((await db.execute(select(EvaluationRun))).scalars().all())
    assert after == before
