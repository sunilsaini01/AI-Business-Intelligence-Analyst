"""Sec 6 metric definitions, extended for Phase 8 to cover the layers that
now actually exist (Analysis/Visualization/Critic agents didn't exist when
Sec 6 was first drafted). Deterministic metrics dominate by design — every
function here is plain Python; the one place an LLM genuinely helps
(qualitative relevance/recommendation judging) stays isolated in
app/evaluation/judges.py and is optional, matching Sec 6's own 0.70/0.30
weighting philosophy.

Reuses existing code rather than re-implementing it: numeric tolerance
(app/tools/critic_checks.py::values_are_close), evidence grounding
(critic_checks.check_numerical_grounding), and Critic effectiveness is
measured by literally running the real app/agents/critic.py against real
and deliberately-mutated versions of a real pipeline result — not a
re-implementation of what the Critic checks.
"""

from __future__ import annotations

import copy
from typing import Any

from app.tools.critic_checks import (
    check_numerical_grounding,
    run_all_deterministic_checks,
    summarize_findings,
    values_are_close,
)


# ---------------------------------------------------------------------------
# Trajectory / tool selection (pre-existing, Sec 6)
# ---------------------------------------------------------------------------


def jaccard_similarity(expected: set[str], actual: set[str]) -> float:
    """Tool selection accuracy (Sec 6)."""
    if not expected and not actual:
        return 1.0
    union = expected | actual
    if not union:
        return 1.0
    return len(expected & actual) / len(union)


def overall_task_success(deterministic_scores: list[float], judge_scores: list[float]) -> float:
    """0.70 * mean(deterministic) + 0.30 * mean(judge) — Sec 6."""
    det_mean = sum(deterministic_scores) / len(deterministic_scores) if deterministic_scores else 0.0
    judge_mean = sum(judge_scores) / len(judge_scores) if judge_scores else 0.0
    return 0.70 * det_mean + 0.30 * judge_mean


# ---------------------------------------------------------------------------
# Level 1 — SQL correctness
# ---------------------------------------------------------------------------


def evaluate_sql_correctness(
    sql_queries: list[dict[str, Any]], expected_tables: list[str]
) -> dict[str, Any]:
    """All queries must have validated_ok=True (proves read-only + allow-listed
    + within limits — app/tools/database_tools.py already enforces this, this
    metric just checks the run actually produced clean queries) and the
    tables referenced should overlap the case's expected tables.
    """
    if not sql_queries:
        return {"correct": False, "reason": "No SQL queries were executed.", "table_jaccard": 0.0}

    all_ok = all(q.get("validated_ok") for q in sql_queries)
    tables_used: set[str] = set()
    for q in sql_queries:
        tables_used.update(q.get("tables_referenced", []) or [])
    # tables_referenced isn't stored on SQLQueryRecord today — fall back to
    # nothing (Jaccard still works, just less informative) rather than error.
    table_jaccard = jaccard_similarity(set(expected_tables), tables_used) if tables_used else None

    return {
        "correct": all_ok,
        "all_validated_ok": all_ok,
        "n_queries": len(sql_queries),
        "n_rejected": sum(1 for q in sql_queries if not q.get("validated_ok")),
        "table_jaccard": table_jaccard,
    }


# ---------------------------------------------------------------------------
# Level 2/3 — Analysis & Answer correctness (type-dispatched on ground_truth)
# ---------------------------------------------------------------------------


def _get(d: dict[str, Any] | None, *keys: str, default: Any = None) -> Any:
    d = d or {}
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def evaluate_answer_correctness(
    report: dict[str, Any], ground_truth: dict[str, Any], tolerance: dict[str, float]
) -> dict[str, Any]:
    """Checks the FREE-TEXT report against ground truth — separate from
    evaluate_analysis_correctness, which checks the structured
    analysis_results directly (text can be right even if worded oddly, or
    wrong even if analysis_results was right, if synthesis mis-stated it —
    this is what actually distinguishes the two failure modes).
    """
    text = (report.get("executive_summary", "") + " " + " ".join(report.get("key_findings", []))).lower()
    abs_tol, rel_tol = tolerance.get("abs", 0.02), tolerance.get("rel", 0.01)
    gt_type = ground_truth.get("type")

    if gt_type == "category_values":
        matched = 0
        for label, value in ground_truth["values"].items():
            if label.lower() in text and _number_near_in_text(text, value, abs_tol, rel_tol):
                matched += 1
        total = len(ground_truth["values"])
        return {"correct": matched >= max(1, total // 2), "matched": matched, "total": total}

    if gt_type == "top_category":
        aliases = ground_truth.get("top_group_aliases", [ground_truth["top_group"]])
        name_ok = any(a.lower() in text for a in aliases)
        value_ok = _number_near_in_text(text, ground_truth["top_value"], abs_tol, rel_tol)
        return {"correct": name_ok and value_ok, "name_mentioned": name_ok, "value_mentioned": value_ok}

    if gt_type == "period_comparison_with_contribution":
        baseline_ok = _number_near_in_text(text, ground_truth["baseline_value"], abs_tol, rel_tol)
        current_ok = _number_near_in_text(text, ground_truth["current_value"], abs_tol, rel_tol)
        return {"correct": baseline_ok and current_ok, "baseline_mentioned": baseline_ok, "current_mentioned": current_ok}

    if gt_type == "trend_bounds":
        # Free text rarely quotes every month — this type is checked properly
        # in evaluate_analysis_correctness against the structured trend instead.
        return {"correct": True, "note": "trend_bounds checked at the analysis level, not the text level"}

    return {"correct": None, "reason": f"Unknown ground_truth type: {gt_type}"}


def _number_near_in_text(text: str, value: float, abs_tol: float, rel_tol: float) -> bool:
    from app.tools.critic_checks import _extract_numbers  # local import: private helper, intentionally internal

    for extracted, _is_pct in _extract_numbers(text):
        if values_are_close(extracted, value, abs_tol=abs_tol, rel_tol=rel_tol):
            return True
        if values_are_close(extracted, abs(value), abs_tol=abs_tol, rel_tol=rel_tol):
            return True
    return False


def evaluate_analysis_correctness(
    analysis_results: dict[str, Any], ground_truth: dict[str, Any], tolerance: dict[str, float]
) -> dict[str, Any]:
    """Checks state["analysis_results"] (Phase 6's structured output)
    directly against ground truth — the most precise check available, since
    it compares numbers to numbers, not numbers embedded in prose.
    """
    abs_tol, rel_tol = tolerance.get("abs", 0.02), tolerance.get("rel", 0.01)
    gt_type = ground_truth.get("type")

    if gt_type == "category_values":
        contributions = analysis_results.get("contributions", [])
        if not contributions:
            return {"correct": False, "reason": "No contribution analysis produced."}
        by_group = {c["group"]: c["current_value"] for c in contributions[0].get("contributors", [])}
        matched = sum(
            1
            for label, expected_v in ground_truth["values"].items()
            if label in by_group and values_are_close(by_group[label], expected_v, abs_tol=abs_tol, rel_tol=rel_tol)
        )
        total = len(ground_truth["values"])
        return {"correct": matched == total, "matched": matched, "total": total}

    if gt_type == "top_category":
        candidates = analysis_results.get("contributions", []) + [
            {"contributors": [{"group": r.get(e["dimension"]), "current_value": r.get(e["value_col"])} for r in e["rows"]]}
            for e in analysis_results.get("top_n", [])
            for _ in [None]
        ]
        aliases = {a.lower() for a in ground_truth.get("top_group_aliases", [ground_truth["top_group"]])}
        for c in candidates:
            contributors = c.get("contributors", [])
            if not contributors:
                continue
            top = contributors[0]
            if str(top.get("group", "")).lower() in aliases and values_are_close(
                float(top.get("current_value", float("nan"))), ground_truth["top_value"], abs_tol=abs_tol, rel_tol=rel_tol
            ):
                return {"correct": True, "found_group": top.get("group"), "found_value": top.get("current_value")}
        return {"correct": False, "reason": "Top group/value not found in contributions or top_n."}

    if gt_type == "period_comparison_with_contribution":
        pcs = analysis_results.get("period_comparisons", [])
        pc_ok = any(
            values_are_close(pc.get("baseline_value", float("nan")), ground_truth["baseline_value"], abs_tol=abs_tol, rel_tol=rel_tol)
            and values_are_close(pc.get("current_value", float("nan")), ground_truth["current_value"], abs_tol=abs_tol, rel_tol=rel_tol)
            for pc in pcs
        )
        dom = ground_truth.get("dominant_contributor")
        contrib_ok = True
        if dom:
            contrib_ok = False
            for contrib in analysis_results.get("contributions", []):
                for c in contrib.get("contributors", []):
                    if c.get("group") == dom["group"] and values_are_close(
                        c.get("change", float("nan")), dom["change"], abs_tol=abs_tol, rel_tol=rel_tol
                    ):
                        contrib_ok = True
        return {"correct": pc_ok and contrib_ok, "period_comparison_ok": pc_ok, "dominant_contributor_ok": contrib_ok}

    if gt_type == "trend_bounds":
        trends = analysis_results.get("trends", [])
        if not trends:
            return {"correct": False, "reason": "No trend analysis produced."}
        trend = trends[0]
        min_ok = values_are_close(trend.get("min_value", float("nan")), ground_truth["min_value"], abs_tol=abs_tol, rel_tol=rel_tol)
        max_ok = values_are_close(trend.get("max_value", float("nan")), ground_truth["max_value"], abs_tol=abs_tol, rel_tol=rel_tol)
        enough_months = len(trend.get("points", [])) >= ground_truth.get("num_months_at_least", 1)
        return {"correct": min_ok and max_ok and enough_months, "min_ok": min_ok, "max_ok": max_ok, "enough_months": enough_months}

    return {"correct": None, "reason": f"Unknown ground_truth type: {gt_type}"}


# ---------------------------------------------------------------------------
# Level 4 — Evidence grounding (reuses the Critic's own check directly)
# ---------------------------------------------------------------------------


def evaluate_grounding(
    report: dict[str, Any], analysis_results: dict[str, Any], sql_queries: list[dict[str, Any]]
) -> dict[str, Any]:
    findings = check_numerical_grounding(report, analysis_results, sql_queries)
    n_errors = sum(1 for f in findings if f["severity"] == "ERROR")
    return {
        "grounded": n_errors == 0,
        "hallucination_detected": n_errors > 0,
        "n_numerical_errors": n_errors,
        "findings": findings,
    }


# ---------------------------------------------------------------------------
# Level 3 — Visualization correctness
# ---------------------------------------------------------------------------


def evaluate_visualization_correctness(
    charts: list[dict[str, Any]], expected_chart_types: list[str] | None, ground_truth: dict[str, Any]
) -> dict[str, Any]:
    if not expected_chart_types:
        return {"correct": None, "reason": "No visualization expected for this case."}
    if not charts:
        return {"correct": False, "reason": "No chart was produced."}

    type_ok = any(c.get("chart_type") in expected_chart_types for c in charts)

    # Spot-check chart values against ground truth where the shape allows it
    # — reuses the same numbers the analysis-level check already verified,
    # this just confirms the CHART (not just the analysis) carries them.
    gt_type = ground_truth.get("type")
    value_ok = True
    if gt_type == "top_category":
        aliases = {a.lower() for a in ground_truth.get("top_group_aliases", [ground_truth["top_group"]])}
        value_ok = any(
            str(point.get("label", "")).lower() in aliases
            and values_are_close(float(point.get("value", float("nan"))), ground_truth["top_value"], abs_tol=0.05, rel_tol=0.01)
            for chart in charts
            for point in chart.get("data", [])
        ) or not any(c.get("chart_type") == "kpi" for c in charts)  # only enforce for KPI-shaped charts, where this is unambiguous

    return {"correct": type_ok and value_ok, "chart_type_ok": type_ok, "chart_value_ok": value_ok, "chart_types_seen": [c.get("chart_type") for c in charts]}


# ---------------------------------------------------------------------------
# Level 4 — Critic effectiveness (mutation testing against the REAL Critic)
# ---------------------------------------------------------------------------


def mutate_report_with_fabricated_number(report: dict[str, Any]) -> dict[str, Any]:
    """Injects an obviously-fabricated number into a real, good report — the
    Critic should reject this. Used to measure the Critic's true-positive
    detection rate against a real pipeline result, not a synthetic fixture.
    """
    mutated = copy.deepcopy(report)
    mutated["executive_summary"] = mutated["executive_summary"].rstrip(".") + " (adjusted figure: 999999.99)."
    mutated["key_findings"] = list(mutated["key_findings"]) + ["Adjusted total: 999999.99"]
    return mutated


def mutate_report_with_unsupported_causal_claim(report: dict[str, Any]) -> dict[str, Any]:
    """Injects a causal claim naming no real evidence — the Critic should
    reject this via check_causal_claims."""
    mutated = copy.deepcopy(report)
    mutated["executive_summary"] = (
        mutated["executive_summary"].rstrip(".") + " because customers lost trust in the brand."
    )
    return mutated


_STATUS_RANK = {"PASS": 0, "WARN": 1, "FAIL": 2}


def evaluate_critic_effectiveness(
    report: dict[str, Any],
    analysis_results: dict[str, Any],
    sql_queries: list[dict[str, Any]],
    charts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Mutation testing against the REAL deterministic checks (not a
    re-implementation of what they check): takes a real, already-produced
    good report and asks whether injecting a fabricated number / an
    unsupported causal claim makes run_all_deterministic_checks notice.
    Both checks are deterministic (no LLM), so this runs on every case in
    every benchmark run, live-LLM quota or not.
    """
    baseline_findings = run_all_deterministic_checks(report, analysis_results, sql_queries, charts)
    baseline_status, _ = summarize_findings(baseline_findings)

    fabricated = mutate_report_with_fabricated_number(report)
    fabricated_findings = run_all_deterministic_checks(fabricated, analysis_results, sql_queries, charts)
    fabricated_status, _ = summarize_findings(fabricated_findings)

    causal = mutate_report_with_unsupported_causal_claim(report)
    causal_findings = run_all_deterministic_checks(causal, analysis_results, sql_queries, charts)
    causal_status, _ = summarize_findings(causal_findings)

    fabricated_caught = _STATUS_RANK[fabricated_status] > _STATUS_RANK[baseline_status]
    causal_caught = _STATUS_RANK[causal_status] > _STATUS_RANK[baseline_status]

    return {
        "correct": bool(fabricated_caught and causal_caught),
        "baseline_status": baseline_status,
        "fabricated_number_status": fabricated_status,
        "fabricated_number_caught": fabricated_caught,
        "unsupported_causal_status": causal_status,
        "unsupported_causal_caught": causal_caught,
    }


def evaluate_report_completeness(
    report: dict[str, Any] | None, *, expected_chart_types: list[str] | None
) -> dict[str, Any]:
    """Phase 10 (Report Generator) quality metric: does the FINAL report
    actually carry the presentation-layer sections it's supposed to add
    (verified_claims, analysis_explanation, visualizations,
    technical_details), not whether their content happens to be right —
    that's already covered by evaluate_answer_correctness/
    evaluate_analysis_correctness/evaluate_grounding on the same report.

    Deliberately NOT folded into a case's end-to-end pass/fail (see
    app/evaluation/evaluator.py::evaluate_case_from_state) — it's an
    additive quality signal, not a new gate, so it can't change which of
    the 5 existing benchmark cases pass or fail.

    Uses `.get()` throughout: a hand-built report dict from before Phase 10
    (or a test fixture that doesn't exercise report_agent_node) simply
    scores incomplete rather than raising — that's accurate, not a bug.
    """
    if report is None:
        return {"correct": False, "reason": "No report produced."}

    checks = {
        "has_executive_summary": bool(report.get("executive_summary")),
        "has_confidence": report.get("confidence") in ("Low", "Medium", "High"),
        "has_analysis_explanation_or_insufficient_evidence": bool(report.get("analysis_explanation"))
        or "insufficient" in report.get("executive_summary", "").lower(),
        "has_visualizations_when_charts_expected": (
            bool(report.get("visualizations")) if expected_chart_types else True
        ),
        "has_technical_details": bool(report.get("technical_details")),
    }
    return {"correct": all(checks.values()), **checks}


# ---------------------------------------------------------------------------
# ML Agent quality (Phase 16) — a measurable regression contract on top of
# app/agents/ml_agent.py's real, deterministic output. This gates the ML
# Agent's OWN measured QUALITY, not whether it decided to run at all
# (data-sufficiency stays app/tools/ml_tools.py's job, unchanged here) — a
# future change to feature engineering/splitting/target creation that
# quietly makes the model worse must fail these checks, not silently pass
# because "it still produced *a* number". `None` ("not applicable", same
# convention as report_completeness_correct etc. above) covers both "not a
# predictive question" and "ML legitimately couldn't run" (insufficient
# data / not appropriate / a computation error, all `ok=False` — see
# app/agents/ml_agent.py) — a graceful degradation is not itself a quality
# regression and must never be scored as one.
# ---------------------------------------------------------------------------

# Forecast: Phase 15 observed ~11% MAPE on the current ~20-month seeded
# revenue series with a linear-trend baseline (Sec 2's own "baseline
# first" judgment call — it is not trying to be a seasonal/causal model).
# 25% leaves headroom for ordinary month-to-month noise while still
# catching a genuine regression (e.g. a broken time-alignment or an
# accidentally-shuffled split would push error far higher, not by a few
# points). MAE has no dataset-independent dollar ceiling that would
# survive the seed data ever being regenerated at a different revenue
# scale, so it's compared as a FRACTION of the held-out period's own mean
# actual value instead of a hard-coded number.
MAX_FORECAST_MAPE_PCT = 25.0
MAX_FORECAST_MAE_FRACTION_OF_MEAN = 0.20

# Churn: Phase 15 observed ROC-AUC ~0.82, accuracy ~75%, precision ~76%,
# recall ~73% (stratified split, 180-day window, ~500 seeded customers).
# ROC-AUC is treated as the PRIMARY signal per this phase's explicit
# instruction (accuracy alone is misleading under any class imbalance);
# 0.65 is meaningfully above the 0.5 "coin flip" floor while leaving room
# for legitimate split-to-split variance. The accuracy/precision/recall
# floors are deliberately well below the observed values — they exist to
# catch a genuinely broken model (e.g. a feature/label swap collapsing
# one class), not to lock in today's exact numbers.
MIN_CHURN_ROC_AUC = 0.65
MIN_CHURN_ACCURACY = 0.60
MIN_CHURN_PRECISION = 0.55
MIN_CHURN_RECALL = 0.55


def evaluate_forecast_quality(ml_results: dict[str, Any] | None) -> dict[str, Any]:
    """Regression gate for a `task == "forecasting"` ml_results. Never
    trusts a metric it can't find — a `ok=True` result missing `mape_pct`/
    `mae` is treated as `correct=False` (malformed/fabricated output), not
    silently skipped, since that shape should never occur from a real
    app/tools/ml_tools.py::evaluate_and_forecast call.
    """
    if not ml_results:
        return {"correct": None, "reason": "No ML result to evaluate (not a predictive question)."}
    if ml_results.get("task") != "forecasting":
        return {"correct": None, "reason": "Not a forecasting result."}
    if not ml_results.get("ok"):
        return {
            "correct": None,
            "reason": f"Forecast did not run (status={ml_results.get('status')}): {ml_results.get('reason')}",
        }

    metrics = ml_results.get("metrics") or {}
    mape = metrics.get("mape_pct")
    mae = metrics.get("mae")
    if mape is None or mae is None:
        return {
            "correct": False,
            "reason": "ok=True forecast result is missing mape_pct/mae — malformed or fabricated output.",
        }

    sample_predictions = ml_results.get("sample_predictions") or []
    actual_values = [p["actual"] for p in sample_predictions if isinstance(p.get("actual"), (int, float))]
    mean_actual = (sum(actual_values) / len(actual_values)) if actual_values else None
    mae_fraction = (mae / mean_actual) if mean_actual else None

    mape_ok = mape <= MAX_FORECAST_MAPE_PCT
    mae_ok = mae_fraction is None or mae_fraction <= MAX_FORECAST_MAE_FRACTION_OF_MEAN

    return {
        "correct": bool(mape_ok and mae_ok),
        "mape_pct": mape,
        "mae": mae,
        "mae_fraction_of_mean": mae_fraction,
        "mape_within_threshold": mape_ok,
        "mae_within_threshold": mae_ok,
        "threshold_mape_pct": MAX_FORECAST_MAPE_PCT,
        "threshold_mae_fraction": MAX_FORECAST_MAE_FRACTION_OF_MEAN,
    }


def evaluate_churn_quality(ml_results: dict[str, Any] | None) -> dict[str, Any]:
    """Regression gate for a `task == "churn_risk"` ml_results. All four
    metrics must clear their floor — ROC-AUC is the metric that would
    actually reveal degraded quality on an imbalanced label (the reason
    accuracy alone isn't trusted), but a genuinely broken model can also
    show up as a precision/recall collapse with ROC-AUC still looking
    passable, so none of the four is skipped.
    """
    if not ml_results:
        return {"correct": None, "reason": "No ML result to evaluate (not a predictive question)."}
    if ml_results.get("task") != "churn_risk":
        return {"correct": None, "reason": "Not a churn_risk result."}
    if not ml_results.get("ok"):
        return {
            "correct": None,
            "reason": f"Churn model did not run (status={ml_results.get('status')}): {ml_results.get('reason')}",
        }

    metrics = ml_results.get("metrics") or {}
    roc_auc, accuracy = metrics.get("roc_auc"), metrics.get("accuracy")
    precision, recall = metrics.get("precision"), metrics.get("recall")
    if None in (roc_auc, accuracy, precision, recall):
        return {
            "correct": False,
            "reason": "ok=True churn result is missing one or more required metrics — malformed or fabricated output.",
        }

    checks = {
        "roc_auc_within_threshold": roc_auc >= MIN_CHURN_ROC_AUC,
        "accuracy_within_threshold": accuracy >= MIN_CHURN_ACCURACY,
        "precision_within_threshold": precision >= MIN_CHURN_PRECISION,
        "recall_within_threshold": recall >= MIN_CHURN_RECALL,
    }
    return {
        "correct": all(checks.values()),
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "threshold_roc_auc": MIN_CHURN_ROC_AUC,
        "threshold_accuracy": MIN_CHURN_ACCURACY,
        "threshold_precision": MIN_CHURN_PRECISION,
        "threshold_recall": MIN_CHURN_RECALL,
        **checks,
    }


def evaluate_ml_quality(ml_results: dict[str, Any] | None) -> dict[str, Any]:
    """Dispatches on ml_results["task"] — the single entry point
    app/evaluation/evaluator.py::evaluate_case_from_state calls, mirroring
    how the SQL/analysis/visualization levels each have one clear owner
    function."""
    if not ml_results:
        return {"correct": None, "reason": "No ML result to evaluate (not a predictive question)."}
    task = ml_results.get("task")
    if task == "forecasting":
        return evaluate_forecast_quality(ml_results)
    if task == "churn_risk":
        return evaluate_churn_quality(ml_results)
    # task is None (not_appropriate) or something unrecognized — never a
    # quality failure, there's simply nothing to grade.
    return {"correct": None, "reason": f"No applicable ML quality check for task={task!r}."}


def evaluate_critic_verdict(critic_feedback: dict[str, Any] | None, expected_valid: bool) -> dict[str, Any]:
    """Did the Critic's actual verdict match what a report of this quality
    should get: PASS/WARN for a genuinely good report, FAIL for a
    deliberately-broken one."""
    if critic_feedback is None:
        return {"correct": False, "reason": "No critic_feedback present."}
    status = critic_feedback.get("status")
    if expected_valid:
        correct = status in ("PASS", "WARN")
    else:
        correct = status == "FAIL"
    return {"correct": correct, "status": status, "expected_valid": expected_valid}


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


def hallucination_rate(case_hallucination_flags: list[bool]) -> float:
    if not case_hallucination_flags:
        return 0.0
    return sum(1 for f in case_hallucination_flags if f) / len(case_hallucination_flags)


def end_to_end_success_rate(case_statuses: list[str]) -> float:
    if not case_statuses:
        return 0.0
    return sum(1 for s in case_statuses if s == "PASSED") / len(case_statuses)


def latency_from_trace(trace: list[dict[str, Any]]) -> dict[str, float]:
    """Sums each node's own enter->exit duration (already recorded on every
    trace event — Sec 9 observability) — no new instrumentation needed."""
    per_node: dict[str, float] = {}
    total = 0.0
    for event in trace:
        if event.get("event") == "exit" and event.get("duration_ms") is not None:
            node = event["node"]
            per_node[node] = per_node.get(node, 0.0) + event["duration_ms"]
            total += event["duration_ms"]
    per_node["total"] = total
    return per_node
