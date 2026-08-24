"""Deterministic unit tests for app/tools/critic_checks.py (Phase 9). Small
fixed fixtures only — no DB, no LLM, no network. Covers: correct evidence ->
PASS, wrong number/percentage -> FAIL, unsupported/supported causal claims,
wrong period, wrong category, chart/data mismatch, missing evidence,
opposite-direction contribution handling, hallucinated values, empty
results, the real July benchmark, and Olist-shaped results.
"""

from __future__ import annotations

from app.tools.critic_checks import (
    check_causal_claims,
    check_chart_consistency,
    check_contribution_arithmetic,
    check_evidence_sufficiency,
    check_numerical_grounding,
    check_period_consistency,
    check_visualization_presence,
    run_all_deterministic_checks,
    summarize_findings,
)

# --- fixtures ---------------------------------------------------------------


def _report(summary: str, findings: list[str], *, confidence: str = "High", limitations: str = "") -> dict:
    return {
        "executive_summary": summary,
        "key_findings": findings,
        "evidence": [],
        "recommendations": [],
        "confidence": confidence,
        "limitations": limitations,
    }


def _period_comparison(baseline_period="2026-06", current_period="2026-07", baseline=161445.80, current=150633.02) -> dict:
    absolute_change = current - baseline
    return {
        "ok": True,
        "period_col": "month",
        "value_col": "revenue",
        "baseline_period": baseline_period,
        "current_period": current_period,
        "baseline_value": baseline,
        "current_value": current,
        "absolute_change": absolute_change,
        "percentage_change": (absolute_change / baseline) * 100,
        "direction": "decrease",
        "note": None,
        "insufficient_evidence": False,
        "reason": None,
    }


def _contribution(dominant_group="Enterprise", dominant_pct=74.4) -> dict:
    return {
        "ok": True,
        "dimension_col": "segment",
        "value_col": "revenue",
        "total_current": 6172.0,
        "total_prior": 16782.83,
        "total_change": -10610.83,
        "baseline_period": "2026-06",
        "current_period": "2026-07",
        "contributors": [
            {
                "group": dominant_group, "current_value": 6172.0, "prior_value": 16782.83,
                "change": -10610.83, "pct_change": -63.2, "pct_of_total_current": 100.0,
                "pct_of_total_change": dominant_pct, "rank": 1,
            },
        ],
    }


def _july_benchmark_analysis_results() -> dict:
    return {
        "period_comparisons": [_period_comparison()],
        "trends": [],
        "contributions": [_contribution()],
        "top_n": [],
        "distributions": [],
        "diagnostic": {
            "ok": True,
            "facts": ["revenue went from 161,445.80 in 2026-06 to 150,633.02 in 2026-07 (decrease (-6.7%))."],
            "interpretations": [
                "By segment, 'Enterprise' appears to be the dominant contributor, accounting for "
                "approximately 74.4% of the total change (-10,610.83)."
            ],
            "limitations": [],
            "insufficient_evidence": False,
            "reason": None,
        },
        "insufficient_evidence": False,
        "reason": None,
    }


# --- 1/13: correct numerical evidence -> PASS; hallucinated value -> FAIL --


def test_correct_numbers_produce_no_findings():
    report = _report(
        "Revenue decreased from 161445.80 to 150633.02, a 6.7% decline.",
        ["June revenue was 161445.80", "July revenue was 150633.02"],
    )
    findings = check_numerical_grounding(report, _july_benchmark_analysis_results(), [])
    assert findings == []


def test_wrong_number_is_flagged_error():
    report = _report("July revenue was $170,000.", ["July revenue = 170000"])
    findings = check_numerical_grounding(report, _july_benchmark_analysis_results(), [])
    assert any(f["severity"] == "ERROR" and f["category"] == "numerical" for f in findings)


def test_wrong_percentage_is_flagged_error():
    report = _report("Revenue declined by 45%.", [])
    findings = check_numerical_grounding(report, _july_benchmark_analysis_results(), [])
    assert any(f["severity"] == "ERROR" for f in findings)


def test_hallucinated_value_with_no_evidence_at_all_is_flagged():
    report = _report("Revenue was 999999.99.", [])
    findings = check_numerical_grounding(report, {}, [])
    assert any(f["severity"] == "ERROR" for f in findings)


# --- 4/5: causal claims ------------------------------------------------------


def test_unsupported_causal_claim_no_dominant_contributor_fails():
    report = _report("Revenue decreased because customers stopped buying.", [])
    analysis = {"period_comparisons": [_period_comparison()], "contributions": []}
    findings = check_causal_claims(report, analysis)
    assert any(f["severity"] == "ERROR" and f["category"] == "causal_claim" for f in findings)


def test_supported_causal_claim_naming_dominant_contributor_passes():
    report = _report("Revenue decreased because Enterprise revenue declined sharply.", [])
    analysis = {"contributions": [_contribution(dominant_group="Enterprise", dominant_pct=74.4)]}
    findings = check_causal_claims(report, analysis)
    assert findings == []


def test_causal_claim_naming_wrong_entity_is_warned():
    report = _report("Revenue decreased because the West region underperformed.", [])
    analysis = {"contributions": [_contribution(dominant_group="Enterprise", dominant_pct=74.4)]}
    findings = check_causal_claims(report, analysis)
    assert any(f["severity"] == "WARNING" for f in findings)


def test_non_causal_text_produces_no_causal_findings():
    report = _report("Revenue was 150633.02 in July.", [])
    findings = check_causal_claims(report, {"contributions": []})
    assert findings == []


# --- 6/7: wrong period / wrong category in a chart --------------------------


def test_chart_with_wrong_period_label_fails():
    analysis = {"period_comparisons": [_period_comparison(baseline_period="2026-06", current_period="2026-07")]}
    chart = {
        "chart_type": "bar", "title": "revenue: May vs June", "subtitle": None,
        "x_axis": "month", "y_axis": "revenue", "group_by": None, "sort": "none",
        "data": [{"label": "2026-05", "value": 161445.80}, {"label": "2026-06", "value": 150633.02}],
        "units": None, "source_analysis": "period_comparison", "reason": "", "limitations": [],
        "path": "", "spec_summary": {},
    }
    findings = check_chart_consistency([chart], analysis)
    assert any(f["category"] == "chart_consistency" and "2026-05" in f["message"] for f in findings)


def test_chart_with_correct_period_labels_passes():
    analysis = {"period_comparisons": [_period_comparison()]}
    chart = {
        "chart_type": "bar", "title": "revenue: June vs July", "subtitle": None,
        "x_axis": "month", "y_axis": "revenue", "group_by": None, "sort": "none",
        "data": [{"label": "2026-06", "value": 161445.80}, {"label": "2026-07", "value": 150633.02}],
        "units": None, "source_analysis": "period_comparison", "reason": "", "limitations": [],
        "path": "", "spec_summary": {},
    }
    assert check_chart_consistency([chart], analysis) == []


def test_chart_with_wrong_category_fails():
    analysis = {"contributions": [_contribution(dominant_group="Enterprise")]}
    chart = {
        "chart_type": "horizontal_bar", "title": "segment contribution", "subtitle": None,
        "x_axis": "revenue", "y_axis": "segment", "group_by": None, "sort": "desc",
        "data": [{"label": "SMB", "value": -10610.83}],  # SMB never appeared in the contribution
        "units": "change", "source_analysis": "contribution", "reason": "", "limitations": [],
        "path": "", "spec_summary": {},
    }
    findings = check_chart_consistency([chart], analysis)
    assert any("SMB" in f["message"] for f in findings)


# --- 8/9: chart/data value mismatch vs correct chart ------------------------


def test_chart_value_mismatch_fails():
    analysis = {"period_comparisons": [_period_comparison()]}
    chart = {
        "chart_type": "bar", "title": "t", "subtitle": None, "x_axis": None, "y_axis": None,
        "group_by": None, "sort": "none",
        "data": [{"label": "2026-06", "value": 161445.80}, {"label": "2026-07", "value": 177767.61}],
        "units": None, "source_analysis": "period_comparison", "reason": "", "limitations": [],
        "path": "", "spec_summary": {},
    }
    findings = check_chart_consistency([chart], analysis)
    assert any(f["severity"] == "ERROR" and "177767" in f["message"] for f in findings)


def test_correct_chart_produces_no_findings():
    analysis = _july_benchmark_analysis_results()
    chart = {
        "chart_type": "bar", "title": "t", "subtitle": None, "x_axis": None, "y_axis": None,
        "group_by": None, "sort": "none",
        "data": [{"label": "2026-06", "value": 161445.80}, {"label": "2026-07", "value": 150633.02}],
        "units": None, "source_analysis": "period_comparison", "reason": "", "limitations": [],
        "path": "", "spec_summary": {},
    }
    assert check_chart_consistency([chart], analysis) == []


# --- 10: missing/insufficient evidence --------------------------------------


def test_insufficient_evidence_with_high_confidence_fails():
    report = _report("Revenue changed.", [], confidence="High", limitations="")
    analysis = {"insufficient_evidence": True, "reason": "no comparison possible"}
    findings = check_evidence_sufficiency(report, analysis)
    assert any(f["severity"] == "ERROR" for f in findings)


def test_insufficient_evidence_with_low_confidence_and_limitations_is_clean():
    report = _report("Insufficient evidence.", [], confidence="Low", limitations="Not enough data.")
    analysis = {"insufficient_evidence": True, "reason": "no comparison possible"}
    findings = check_evidence_sufficiency(report, analysis)
    assert findings == []


def test_sufficient_evidence_skips_this_check_entirely():
    report = _report("All good.", [], confidence="High", limitations="")
    findings = check_evidence_sufficiency(report, {"insufficient_evidence": False})
    assert findings == []


# --- 11: opposite-direction contribution handled correctly -----------------


def test_opposite_direction_contributor_with_leaked_pct_is_flagged():
    analysis = {
        "contributions": [
            {
                "ok": True, "dimension_col": "segment", "value_col": "revenue",
                "total_current": 70.0, "total_prior": 100.0, "total_change": -30.0,
                "contributors": [
                    {"group": "A", "current_value": 10.0, "prior_value": 60.0, "change": -50.0, "pct_of_total_change": 100.0},
                    # SMB moved the OPPOSITE way (increase) but has a leaked pct_of_total_change -> bug
                    {"group": "B", "current_value": 60.0, "prior_value": 40.0, "change": 20.0, "pct_of_total_change": 50.0},
                ],
            }
        ]
    }
    findings = check_contribution_arithmetic(analysis)
    assert any("opposite" in f["message"] for f in findings)


def test_opposite_direction_contributor_correctly_none_produces_no_finding():
    analysis = {
        "contributions": [
            {
                "ok": True, "dimension_col": "segment", "value_col": "revenue",
                "total_current": 70.0, "total_prior": 100.0, "total_change": -30.0,
                "contributors": [
                    {"group": "A", "current_value": 10.0, "prior_value": 60.0, "change": -50.0, "pct_of_total_change": 100.0},
                    {"group": "B", "current_value": 60.0, "prior_value": 40.0, "change": 20.0, "pct_of_total_change": None},
                ],
            }
        ]
    }
    assert check_contribution_arithmetic(analysis) == []


def test_wrong_contribution_arithmetic_is_flagged():
    analysis = {
        "contributions": [
            {
                "dimension_col": "segment", "total_change": -10.0,
                "contributors": [{"group": "A", "current_value": 10.0, "prior_value": 60.0, "change": -999.0, "pct_of_total_change": None}],
            }
        ]
    }
    findings = check_contribution_arithmetic(analysis)
    assert any(f["category"] == "contribution_arithmetic" for f in findings)


# --- 14: empty analysis results ---------------------------------------------


def test_empty_analysis_results_does_not_crash_any_check():
    report = _report("Nothing to report.", [])
    assert check_numerical_grounding(report, {}, []) == []
    assert check_period_consistency(report, {}) == []
    assert check_contribution_arithmetic({}) == []
    assert check_evidence_sufficiency(report, {}) == []
    assert check_causal_claims(report, {}) == []
    assert check_chart_consistency([], {}) == []
    assert check_visualization_presence({}, []) == []


# --- 15: missing visualization ----------------------------------------------


def test_missing_visualization_when_analysis_had_content_is_info():
    analysis = {"period_comparisons": [_period_comparison()], "insufficient_evidence": False}
    findings = check_visualization_presence(analysis, [])
    assert any(f["severity"] == "INFO" and f["category"] == "missing_visualization" for f in findings)


def test_missing_visualization_not_flagged_when_evidence_was_insufficient():
    findings = check_visualization_presence({"insufficient_evidence": True}, [])
    assert findings == []


# --- 16/17: Olist + July benchmark end-to-end deterministic pass -----------


def test_olist_shaped_result_passes_clean():
    analysis = {
        "period_comparisons": [], "trends": [], "top_n": [], "distributions": [],
        "contributions": [
            {
                "ok": True, "dimension_col": "product_category_name", "value_col": "avg_score",
                "total_current": 4.64, "total_prior": None, "total_change": None,
                "contributors": [
                    {"group": "cds_dvds_musicais", "current_value": 4.642857142857143, "prior_value": None,
                     "change": None, "pct_of_total_current": 100.0, "pct_of_total_change": None, "rank": 1},
                ],
            }
        ],
        "diagnostic": None, "insufficient_evidence": False, "reason": None,
    }
    report = _report(
        "The product category with the highest average review score is cds_dvds_musicais, at 4.642857142857143.",
        ["cds_dvds_musicais has an average score of 4.642857142857143"],
        confidence="High",
    )
    findings = run_all_deterministic_checks(report, analysis, [], [])
    status, _ = summarize_findings(findings)
    assert status == "PASS"


def test_july_benchmark_correct_report_passes():
    report = _report(
        "Revenue decreased from 161445.80 in June to 150633.02 in July, a 6.7% decline, "
        "because Enterprise revenue declined sharply (74.4% of the total change).",
        ["June revenue: 161445.80", "July revenue: 150633.02", "Enterprise change: -10610.83"],
        confidence="Medium",
    )
    findings = run_all_deterministic_checks(report, _july_benchmark_analysis_results(), [], [])
    status, score = summarize_findings(findings)
    assert status == "PASS"
    # Not a perfect 1.0: charts=[] here (this test isn't about visualization)
    # correctly earns one INFO note from check_visualization_presence — a
    # real, usable analysis with no chart generated for it. INFO findings
    # don't affect PASS/WARN/FAIL status, just a small score deduction.
    assert score >= 0.9
    assert any(f["severity"] == "INFO" and f["category"] == "missing_visualization" for f in findings)


def test_july_benchmark_wrong_report_fails():
    report = _report(
        "Revenue decreased from 161445.80 in June to 170000 in July, a 45% decline, "
        "because customers stopped buying.",
        [],
        confidence="High",
    )
    findings = run_all_deterministic_checks(report, _july_benchmark_analysis_results(), [], [])
    status, _ = summarize_findings(findings)
    assert status == "FAIL"


# --- summarize_findings: PASS/WARN/FAIL routing -----------------------------


def test_summarize_no_findings_is_pass():
    assert summarize_findings([]) == ("PASS", 1.0)


def test_summarize_warning_only_is_warn():
    status, score = summarize_findings([{"severity": "WARNING", "category": "x", "message": "m"}])
    assert status == "WARN"
    assert 0.0 < score < 1.0


def test_summarize_any_error_is_fail():
    status, _ = summarize_findings(
        [{"severity": "WARNING", "category": "x", "message": "m"}, {"severity": "ERROR", "category": "y", "message": "m2"}]
    )
    assert status == "FAIL"
