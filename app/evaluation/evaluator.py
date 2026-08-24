"""Runs the benchmark set (Sec 6) against the SAME production graph used by
the API (app.graph.workflow.get_graph) — no second execution pipeline.
`evaluate_case_from_state` is pure/deterministic (no LLM, no I/O) so it's
independently unit-testable against a hand-built AgentState; `run_case_live`
and `run_benchmark` are the only functions here that touch the graph/LLM/
filesystem.

Live LLM calls are optional and isolated (Groq quota handling, per user
instruction): a RateLimitError from either provider is caught per-case and
recorded as `status="SKIPPED_QUOTA"` — never hidden, never hard-coded around,
and never allowed to abort the rest of the run.
"""

from __future__ import annotations

import datetime
import json
import time
from pathlib import Path
from typing import Any

import anthropic
import groq

from app.core.config import get_settings
from app.core.llm import LLMClientProtocol, get_llm_client
from app.evaluation.benchmark import BenchmarkCase, load_benchmark
from app.evaluation.failure_analysis import summarize_failures
from app.evaluation.metrics import (
    evaluate_analysis_correctness,
    evaluate_answer_correctness,
    evaluate_critic_effectiveness,
    evaluate_critic_verdict,
    evaluate_grounding,
    evaluate_report_completeness,
    evaluate_sql_correctness,
    evaluate_visualization_correctness,
    hallucination_rate,
    jaccard_similarity,
    latency_from_trace,
    overall_task_success,
)
from app.evaluation.models import CaseEvaluation, EvaluationRunSummary, LevelResult
from app.graph.state import AgentState, new_state
from app.graph.workflow import build_graph, get_graph

_RATE_LIMIT_ERRORS = (anthropic.RateLimitError, groq.RateLimitError)


def evaluate_case_from_state(case: BenchmarkCase, final_state: AgentState) -> CaseEvaluation:
    """Pure: no LLM call, no DB, no network. Everything it needs is already
    in `final_state`, which is exactly what makes it testable with a
    hand-built state (tests/evaluation/test_evaluator.py) instead of a live
    graph run.
    """
    ground_truth = dict(case.get("ground_truth") or {})
    tolerance = dict(case.get("tolerance") or {"abs": 0.02, "rel": 0.01})
    expected_tables = case.get("expected_tables", [])
    expected_chart_types = case.get("expected_chart_types")
    expected_behavior = case.get("expected_behavior", "answerable")

    report = final_state.get("report")
    analysis_results = final_state.get("analysis_results") or {}
    sql_queries = final_state.get("sql_queries") or []
    charts = final_state.get("charts") or []
    critic_feedback = final_state.get("critic_feedback")
    trace = final_state.get("trace") or []

    errors: list[str] = []
    notes: list[str] = []
    levels: list[LevelResult] = []

    if expected_behavior == "out_of_scope":
        # A fixed decline message with no evidence gathered is the CORRECT
        # outcome here (app/graph/workflow.py's out-of-scope short-circuit),
        # not a failure to synthesize a real analysis.
        correct = report is not None and final_state.get("intent") == "out_of_scope" and not sql_queries
        levels.append(LevelResult(level="end_to_end", correct=correct, details="out_of_scope short-circuit"))
        return CaseEvaluation(
            case_id=case["id"],
            question=case["question"],
            status="PASSED" if correct else "FAILED",
            expected={"expected_behavior": "out_of_scope"},
            actual={"intent": final_state.get("intent"), "n_sql_queries": len(sql_queries)},
            levels=levels,
            first_failing_level=None if correct else "end_to_end",
            latency_ms=latency_from_trace(trace).get("total"),
            notes=notes,
        )

    if report is None:
        errors.append("Pipeline did not produce a report.")
        return CaseEvaluation(
            case_id=case["id"],
            question=case["question"],
            status="ERROR",
            expected=ground_truth,
            actual={},
            errors=errors,
            latency_ms=latency_from_trace(trace).get("total"),
        )

    expected_tools = set(case.get("expected_tools", []))
    if expected_tools:
        tool_jaccard = jaccard_similarity(expected_tools, set(final_state.get("required_tools", [])))
        notes.append(f"tool_selection_jaccard={tool_jaccard:.2f}")

    sql_eval = evaluate_sql_correctness(sql_queries, expected_tables)
    levels.append(LevelResult(level="sql", correct=sql_eval["correct"], details=str(sql_eval)))

    analysis_eval: dict[str, Any] = {"correct": None}
    answer_eval: dict[str, Any] = {"correct": None}
    if ground_truth:
        analysis_eval = evaluate_analysis_correctness(analysis_results, ground_truth, tolerance)
        answer_eval = evaluate_answer_correctness(report, ground_truth, tolerance)
    levels.append(LevelResult(level="analysis", correct=analysis_eval["correct"], details=str(analysis_eval)))

    viz_eval = evaluate_visualization_correctness(charts, expected_chart_types, ground_truth)
    levels.append(LevelResult(level="visualization", correct=viz_eval["correct"], details=str(viz_eval)))

    grounding_eval = evaluate_grounding(report, analysis_results, sql_queries)
    if grounding_eval["hallucination_detected"]:
        notes.append("Numerical grounding check found unsupported number(s) in the report text.")

    critic_eval = evaluate_critic_verdict(critic_feedback, expected_valid=True)
    levels.append(LevelResult(level="critic", correct=critic_eval["correct"], details=str(critic_eval)))

    critic_effectiveness = evaluate_critic_effectiveness(report, analysis_results, sql_queries, charts)
    completeness_eval = evaluate_report_completeness(report, expected_chart_types=expected_chart_types)

    def _ok(v: bool | None) -> bool:
        return v is not False  # None ("not applicable") never blocks end-to-end success

    end_to_end_ok = bool(
        _ok(sql_eval["correct"])
        and _ok(analysis_eval["correct"])
        and _ok(answer_eval["correct"])
        and _ok(viz_eval["correct"])
        and grounding_eval["grounded"]
        and _ok(critic_eval["correct"])
    )
    levels.append(LevelResult(level="end_to_end", correct=end_to_end_ok))
    first_failing = next((lvl.level for lvl in levels if lvl.correct is False), None)

    return CaseEvaluation(
        case_id=case["id"],
        question=case["question"],
        status="PASSED" if end_to_end_ok else "FAILED",
        expected=ground_truth,
        actual={
            "executive_summary": report["executive_summary"],
            "analysis_keys": sorted(analysis_results.keys()),
            "chart_types": [c["chart_type"] for c in charts],
            "critic_status": critic_feedback["status"] if critic_feedback else None,
        },
        sql_correct=sql_eval["correct"],
        answer_correct=answer_eval["correct"],
        analysis_correct=analysis_eval["correct"],
        visualization_correct=viz_eval["correct"],
        critic_correct=critic_eval["correct"],
        critic_effectiveness_correct=critic_effectiveness["correct"],
        report_completeness_correct=completeness_eval["correct"],
        grounded=grounding_eval["grounded"],
        hallucination_detected=grounding_eval["hallucination_detected"],
        levels=levels,
        first_failing_level=first_failing,
        latency_ms=latency_from_trace(trace).get("total"),
        errors=errors,
        notes=notes,
    )


async def run_case_live(case: BenchmarkCase, llm: LLMClientProtocol | None = None) -> CaseEvaluation:
    """Invokes the real production graph (app.graph.workflow) — same code
    path the API uses, not a parallel evaluation-only pipeline. `llm=None`
    uses the real configured provider; tests pass a ScriptedLLMClient.
    """
    graph = get_graph() if llm is None else build_graph(llm=llm)
    settings = get_settings()
    active_llm = llm if llm is not None else get_llm_client()
    # `total_usage` on LLMClient/GroqLLMClient is a cumulative-since-process-
    # start counter (it's a cached singleton across the whole app — see
    # app/core/llm.py::get_llm_client), not a per-call figure. Snapshot it
    # before this case's graph run so the case's token_usage is a DELTA, not
    # the running total polluted by every other call this process has ever
    # made (other cases in this same benchmark run included).
    usage_before = dict(getattr(active_llm, "total_usage", {}) or {})
    start = time.perf_counter()
    try:
        final_state = await graph.ainvoke(new_state(case["question"], max_retries=settings.critic_max_retries))
    except _RATE_LIMIT_ERRORS as exc:
        return CaseEvaluation(
            case_id=case["id"],
            question=case["question"],
            status="SKIPPED_QUOTA",
            expected=dict(case.get("ground_truth") or {}),
            actual={},
            notes=[f"Live evaluation skipped: LLM provider rate-limited/quota exhausted ({exc})."],
        )
    except Exception as exc:  # noqa: BLE001 — a genuine pipeline error must land as a case result, not crash the whole run
        return CaseEvaluation(
            case_id=case["id"],
            question=case["question"],
            status="ERROR",
            expected=dict(case.get("ground_truth") or {}),
            actual={},
            errors=[f"{type(exc).__name__}: {exc}"],
        )
    latency_ms = (time.perf_counter() - start) * 1000

    result = evaluate_case_from_state(case, final_state)
    result.latency_ms = latency_ms  # wall-clock (includes the live LLM round trips); supersedes the trace-only figure
    usage_after = getattr(active_llm, "total_usage", None)
    if usage_after is not None:
        result.token_usage = {
            key: usage_after.get(key, 0) - usage_before.get(key, 0) for key in usage_after
        }
    return result


def _aggregate_scores(case_results: list[CaseEvaluation]) -> dict[str, float]:
    scored = [c for c in case_results if c.status in ("PASSED", "FAILED")]

    def _rate(flags: list[bool | None]) -> float | None:
        present = [f for f in flags if f is not None]
        return sum(1 for f in present if f) / len(present) if present else None

    # Deterministic quality scores (higher is better) — feeds Sec 6's
    # overall_task_success formula. hallucination_rate is excluded on
    # purpose: it's a defect-rate metric (lower is better), not a quality
    # score, so it's tracked separately rather than averaged in inverted.
    quality_scores = {
        "sql_correctness": _rate([c.sql_correct for c in scored]),
        "answer_correctness": _rate([c.answer_correct for c in scored]),
        "analysis_correctness": _rate([c.analysis_correct for c in scored]),
        "visualization_correctness": _rate([c.visualization_correct for c in scored]),
        "critic_correctness": _rate([c.critic_correct for c in scored]),
        "critic_effectiveness": _rate([c.critic_effectiveness_correct for c in scored]),
        "report_completeness": _rate([c.report_completeness_correct for c in scored]),
        "groundedness": _rate([c.grounded for c in scored]),
        "end_to_end_success_rate": (
            sum(1 for c in scored if c.status == "PASSED") / len(scored) if scored else 0.0
        ),
    }
    scores: dict[str, float | None] = dict(quality_scores)
    scores["hallucination_rate"] = hallucination_rate([c.hallucination_detected for c in scored])
    deterministic = [v for v in quality_scores.values() if v is not None]
    scores["overall_task_success"] = overall_task_success(deterministic, judge_scores=[])
    return {k: v for k, v in scores.items() if v is not None}


async def run_benchmark(
    dataset_path: str = "evaluation/datasets/benchmark.json",
    label: str = "manual-run",
    llm: LLMClientProtocol | None = None,
    reports_dir: str = "evaluation/reports",
) -> EvaluationRunSummary:
    cases = load_benchmark(dataset_path)
    settings = get_settings()
    model_name = (
        settings.groq_model_strong if settings.llm_provider == "groq" else settings.llm_model_strong
    )

    case_results: list[CaseEvaluation] = []
    for case in cases:
        case_results.append(await run_case_live(case, llm=llm))

    passed = sum(1 for c in case_results if c.status == "PASSED")
    failed = sum(1 for c in case_results if c.status == "FAILED")
    errored = sum(1 for c in case_results if c.status == "ERROR")
    skipped = sum(1 for c in case_results if c.status == "SKIPPED_QUOTA")
    latencies = [c.latency_ms for c in case_results if c.latency_ms is not None]

    summary = EvaluationRunSummary(
        label=label,
        model_name=model_name,
        total_cases=len(case_results),
        passed=passed,
        failed=failed,
        errored=errored,
        skipped=skipped,
        end_to_end_success_rate=(passed / (passed + failed)) if (passed + failed) else 0.0,
        hallucination_rate=hallucination_rate(
            [c.hallucination_detected for c in case_results if c.status in ("PASSED", "FAILED")]
        ),
        mean_latency_ms=(sum(latencies) / len(latencies)) if latencies else None,
        aggregate_scores=_aggregate_scores(case_results),
        case_results=case_results,
    )

    _write_report(summary, reports_dir)
    return summary


def _write_report(summary: EvaluationRunSummary, reports_dir: str) -> None:
    out_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"{timestamp}_{summary.label}.json"
    payload = {
        "label": summary.label,
        "model_name": summary.model_name,
        "generated_at": timestamp,
        "total_cases": summary.total_cases,
        "passed": summary.passed,
        "failed": summary.failed,
        "errored": summary.errored,
        "skipped": summary.skipped,
        "end_to_end_success_rate": summary.end_to_end_success_rate,
        "hallucination_rate": summary.hallucination_rate,
        "mean_latency_ms": summary.mean_latency_ms,
        "aggregate_scores": summary.aggregate_scores,
        "failure_summary": summarize_failures(summary.case_results),
        "cases": [
            {
                "case_id": c.case_id,
                "question": c.question,
                "status": c.status,
                "sql_correct": c.sql_correct,
                "answer_correct": c.answer_correct,
                "analysis_correct": c.analysis_correct,
                "visualization_correct": c.visualization_correct,
                "critic_correct": c.critic_correct,
                "critic_effectiveness_correct": c.critic_effectiveness_correct,
                "report_completeness_correct": c.report_completeness_correct,
                "grounded": c.grounded,
                "hallucination_detected": c.hallucination_detected,
                "first_failing_level": c.first_failing_level,
                "latency_ms": c.latency_ms,
                "token_usage": c.token_usage,
                "errors": c.errors,
                "notes": c.notes,
                "expected": c.expected,
                "actual": c.actual,
            }
            for c in summary.case_results
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
