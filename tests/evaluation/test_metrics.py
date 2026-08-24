"""Deterministic unit tests for app/evaluation/metrics.py — no DB, no LLM,
no network. Each metric function is tested against small, hand-built
fixtures shaped like the real state pieces it consumes (SQLQueryRecord,
BusinessReport, analysis_results, ChartRecord).
"""

from __future__ import annotations

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
    end_to_end_success_rate,
    jaccard_similarity,
    latency_from_trace,
    mutate_report_with_fabricated_number,
    mutate_report_with_unsupported_causal_claim,
    overall_task_success,
    values_are_close,
)

# --- numeric tolerance (reused from critic_checks, exercised again here since
# metrics.py's answer/analysis correctness dispatch depends on it directly) ---


def test_values_are_close_absorbs_rounding():
    assert values_are_close(161445.80, 161445.79, abs_tol=0.02, rel_tol=0.01)


def test_values_are_close_rejects_a_different_number():
    assert not values_are_close(161445.80, 150633.02, abs_tol=0.02, rel_tol=0.01)


# --- SQL correctness ---------------------------------------------------------


def test_sql_correctness_all_validated_ok():
    queries = [{"validated_ok": True, "rows": []}, {"validated_ok": True, "rows": []}]
    result = evaluate_sql_correctness(queries, expected_tables=["analytics.customers"])
    assert result["correct"] is True
    assert result["n_rejected"] == 0


def test_sql_correctness_flags_rejected_query():
    queries = [{"validated_ok": True, "rows": []}, {"validated_ok": False, "rejection_reason": "unsafe"}]
    result = evaluate_sql_correctness(queries, expected_tables=[])
    assert result["correct"] is False
    assert result["n_rejected"] == 1


def test_sql_correctness_no_queries_is_incorrect():
    result = evaluate_sql_correctness([], expected_tables=["analytics.customers"])
    assert result["correct"] is False


# --- answer / analysis correctness, type-dispatched on ground_truth ---------


def _report(summary: str, findings: list[str] | None = None) -> dict:
    return {
        "executive_summary": summary,
        "key_findings": findings or [],
        "evidence": [],
        "recommendations": [],
        "confidence": "Medium",
        "limitations": "",
    }


def test_answer_correctness_top_category():
    gt = {"type": "top_category", "top_group": "Software", "top_value": 737525.145}
    report = _report("Software generated the highest revenue at $737,525.15.")
    result = evaluate_answer_correctness(report, gt, {"abs": 5.0, "rel": 0.005})
    assert result["correct"] is True


def test_answer_correctness_top_category_rejects_wrong_value():
    gt = {"type": "top_category", "top_group": "Software", "top_value": 737525.145}
    report = _report("Software generated the highest revenue at $999,999.00.")
    result = evaluate_answer_correctness(report, gt, {"abs": 5.0, "rel": 0.005})
    assert result["correct"] is False


def test_answer_correctness_period_comparison():
    gt = {
        "type": "period_comparison_with_contribution",
        "baseline_value": 161445.80,
        "current_value": 150633.02,
    }
    report = _report("Revenue fell from 161445.80 in June to 150633.02 in July.")
    result = evaluate_answer_correctness(report, gt, {"abs": 5.0, "rel": 0.01})
    assert result["correct"] is True


def test_analysis_correctness_period_comparison_and_dominant_contributor():
    gt = {
        "type": "period_comparison_with_contribution",
        "baseline_value": 161445.80,
        "current_value": 150633.02,
        "dominant_contributor": {"group": "Enterprise", "change": -10610.84},
    }
    analysis_results = {
        "period_comparisons": [
            {"baseline_value": 161445.80, "current_value": 150633.02}
        ],
        "contributions": [
            {"contributors": [{"group": "Enterprise", "current_value": 6171.99, "prior_value": 16782.83, "change": -10610.84}]}
        ],
    }
    result = evaluate_analysis_correctness(analysis_results, gt, {"abs": 5.0, "rel": 0.01})
    assert result["correct"] is True
    assert result["period_comparison_ok"] is True
    assert result["dominant_contributor_ok"] is True


def test_analysis_correctness_trend_bounds():
    gt = {"type": "trend_bounds", "min_value": 110404.49, "max_value": 216951.19, "num_months_at_least": 12}
    analysis_results = {
        "trends": [
            {"min_value": 110404.49, "max_value": 216951.19, "points": [{"period": f"2025-{m:02d}"} for m in range(1, 13)]}
        ]
    }
    result = evaluate_analysis_correctness(analysis_results, gt, {"abs": 5.0, "rel": 0.01})
    assert result["correct"] is True


def test_analysis_correctness_missing_data_is_incorrect_not_error():
    gt = {"type": "trend_bounds", "min_value": 1.0, "max_value": 2.0, "num_months_at_least": 12}
    result = evaluate_analysis_correctness({}, gt, {"abs": 0.02, "rel": 0.01})
    assert result["correct"] is False


# --- evidence grounding (wraps critic_checks.check_numerical_grounding) -----


def test_grounding_flags_a_fabricated_number():
    report = _report("Revenue was $999999.99 in July.", ["July revenue: 999999.99"])
    result = evaluate_grounding(report, analysis_results={}, sql_queries=[])
    assert result["grounded"] is False
    assert result["hallucination_detected"] is True


def test_grounding_passes_when_numbers_match_evidence():
    analysis_results = {"period_comparisons": [{"baseline_value": 100.0, "current_value": 90.0, "percentage_change": -10.0}]}
    report = _report("Revenue fell from 100.0 to 90.0, a 10.0% decline.")
    result = evaluate_grounding(report, analysis_results, sql_queries=[])
    assert result["grounded"] is True
    assert result["hallucination_detected"] is False


# --- visualization correctness ----------------------------------------------


def test_visualization_correctness_not_applicable_when_none_expected():
    result = evaluate_visualization_correctness([], expected_chart_types=None, ground_truth={})
    assert result["correct"] is None


def test_visualization_correctness_fails_when_no_chart_produced():
    result = evaluate_visualization_correctness([], expected_chart_types=["bar"], ground_truth={})
    assert result["correct"] is False


def test_visualization_correctness_matches_expected_type():
    charts = [{"chart_type": "bar", "data": [{"label": "Central", "value": 94}]}]
    result = evaluate_visualization_correctness(charts, expected_chart_types=["bar", "horizontal_bar"], ground_truth={})
    assert result["correct"] is True


# --- critic effectiveness (mutation testing against the real deterministic checks) --


def test_critic_effectiveness_catches_both_mutations_on_a_good_report():
    report = _report(
        "Revenue fell from 161445.80 to 150633.02, a -6.7% decline.",
        ["June revenue: 161445.80", "July revenue: 150633.02"],
    )
    analysis_results = {
        "period_comparisons": [
            {"baseline_value": 161445.80, "current_value": 150633.02, "absolute_change": -10812.78, "percentage_change": -6.7}
        ]
    }
    result = evaluate_critic_effectiveness(report, analysis_results, sql_queries=[], charts=[])
    assert result["correct"] is True
    assert result["fabricated_number_caught"] is True
    assert result["unsupported_causal_caught"] is True


def test_mutation_helpers_actually_change_the_report():
    report = _report("Revenue was stable.", ["No major changes."])
    fabricated = mutate_report_with_fabricated_number(report)
    causal = mutate_report_with_unsupported_causal_claim(report)
    assert "999999.99" in fabricated["executive_summary"]
    assert "because" in causal["executive_summary"].lower()
    # original untouched (deep-copied, not mutated in place)
    assert "999999.99" not in report["executive_summary"]


# --- report completeness (Phase 10 quality metric) ---------------------------


def _complete_report(**overrides) -> dict:
    base = {
        "executive_summary": "Revenue fell.",
        "key_findings": ["June: 100", "July: 90"],
        "confidence": "Medium",
        "verified_claims": ["June: 100"],
        "analysis_explanation": "Revenue moved from 100.0 to 90.0.",
        "visualizations": [{"chart_type": "bar", "title": "Revenue", "subtitle": None}],
        "technical_details": {"critic_status": "PASS", "critic_score": 1.0},
        "narrative": None,
    }
    base.update(overrides)
    return base


def test_report_completeness_passes_for_a_fully_populated_report():
    result = evaluate_report_completeness(_complete_report(), expected_chart_types=["bar"])
    assert result["correct"] is True


def test_report_completeness_none_report_is_incorrect():
    result = evaluate_report_completeness(None, expected_chart_types=None)
    assert result["correct"] is False


def test_report_completeness_missing_visualizations_when_charts_expected_is_incorrect():
    result = evaluate_report_completeness(_complete_report(visualizations=[]), expected_chart_types=["bar"])
    assert result["correct"] is False
    assert result["has_visualizations_when_charts_expected"] is False


def test_report_completeness_no_charts_expected_does_not_require_visualizations():
    result = evaluate_report_completeness(_complete_report(visualizations=[]), expected_chart_types=None)
    assert result["correct"] is True


def test_report_completeness_insufficient_evidence_summary_satisfies_analysis_explanation_check():
    result = evaluate_report_completeness(
        _complete_report(executive_summary="Insufficient evidence to determine the cause.", analysis_explanation=""),
        expected_chart_types=None,
    )
    assert result["has_analysis_explanation_or_insufficient_evidence"] is True


def test_report_completeness_tolerates_a_pre_phase10_report_missing_new_fields():
    old_shaped_report = {"executive_summary": "x", "key_findings": [], "confidence": "Low"}
    result = evaluate_report_completeness(old_shaped_report, expected_chart_types=["bar"])
    assert result["correct"] is False  # honest: this report genuinely lacks the new sections
    assert result["has_confidence"] is True


# --- critic verdict correctness ----------------------------------------------


def test_critic_verdict_correct_when_pass_matches_expected_valid():
    assert evaluate_critic_verdict({"status": "PASS"}, expected_valid=True)["correct"] is True


def test_critic_verdict_incorrect_when_fail_but_expected_valid():
    assert evaluate_critic_verdict({"status": "FAIL"}, expected_valid=True)["correct"] is False


def test_critic_verdict_missing_feedback_is_incorrect():
    assert evaluate_critic_verdict(None, expected_valid=True)["correct"] is False


# --- aggregates ----------------------------------------------------------------


def test_hallucination_rate():
    assert hallucination_rate([True, False, False, False]) == 0.25
    assert hallucination_rate([]) == 0.0


def test_end_to_end_success_rate():
    assert end_to_end_success_rate(["PASSED", "PASSED", "FAILED"]) == 2 / 3
    assert end_to_end_success_rate([]) == 0.0


def test_latency_from_trace_sums_exit_durations_per_node_and_total():
    trace = [
        {"node": "supervisor", "event": "enter", "duration_ms": None},
        {"node": "supervisor", "event": "exit", "duration_ms": 120.0},
        {"node": "sql_agent", "event": "enter", "duration_ms": None},
        {"node": "sql_agent", "event": "exit", "duration_ms": 80.0},
    ]
    result = latency_from_trace(trace)
    assert result["supervisor"] == 120.0
    assert result["sql_agent"] == 80.0
    assert result["total"] == 200.0


def test_jaccard_and_overall_task_success_unchanged_from_seed():
    assert jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0
    deterministic = [1.0, 0.8]
    judge = [0.6]
    expected = 0.70 * (sum(deterministic) / 2) + 0.30 * 0.6
    assert overall_task_success(deterministic, judge) == expected
