"""Deterministic Critic checks (Sec 5, Phase 9). Pure functions only — no
LLM, no DB access, no I/O. Each function inspects the already-produced
`report`/`analysis_results`/`charts` for one class of problem and returns a
list of `CriticFinding` dicts (app/graph/state.py). The one genuinely
semantic check (does the wording overstate what the evidence supports) lives
in app/agents/critic.py behind the LLM abstraction instead — everything here
is checkable with arithmetic and string matching, so none of it needs an LLM.

Numbers-in-text matching (`_extract_numbers`) is the load-bearing piece: it
pulls every numeric token out of the report's free text and verifies each
one against a pool of numbers actually present in analysis_results/evidence,
within a tolerance that absorbs rounding but not a genuinely different
number (Sec 6's "no fabricated numbers" rule, made mechanical).
"""

from __future__ import annotations

import re
from typing import Any

from app.graph.state import BusinessReport, ChartRecord, CriticFinding

_NUMBER_PATTERN = re.compile(r"[-+]?\$?\d[\d,]*\.?\d*%?")
_CAUSAL_PATTERN = re.compile(
    r"\b(because|due to|caused by|drove|driving|driven by|led to|leads? to|resulted? in|the reason)\b",
    re.IGNORECASE,
)
_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}
_MONTH_NAME_PATTERN = re.compile(r"\b(" + "|".join(_MONTH_NAMES) + r")\b", re.IGNORECASE)

_REL_TOL = 0.01  # 1% relative tolerance — absorbs rounding, not a different number
_ABS_TOL = 0.02  # plus a small absolute floor for near-zero values


_YEAR_RANGE = range(1900, 2100)


def _extract_numbers(text: str) -> list[tuple[float, bool]]:
    """Returns [(value, is_percentage), ...] parsed out of free text.

    A bare 4-digit integer in a plausible year range (e.g. "July 2026") is
    excluded — it's a label/date component, not a claimed data value, and
    treating it as one would flag every report that mentions a year as an
    unsupported number.
    """
    out = []
    for match in _NUMBER_PATTERN.finditer(text):
        token = match.group()
        is_pct = token.endswith("%")
        cleaned = token.replace("$", "").replace(",", "").rstrip("%")
        try:
            value = float(cleaned)
        except ValueError:
            continue
        looks_like_bare_year = (
            not is_pct and "$" not in token and "." not in cleaned
            and int(value) in _YEAR_RANGE
        )
        if looks_like_bare_year:
            continue
        out.append((value, is_pct))
    return out


def values_are_close(a: float, b: float, *, abs_tol: float = _ABS_TOL, rel_tol: float = _REL_TOL) -> bool:
    """Shared tolerance rule (Phase 8's app/evaluation/metrics.py reuses this
    directly rather than duplicating it) — tolerant of rounding, not of a
    genuinely different number. `rel_tol` scales with the larger magnitude
    so a big number's tolerance doesn't shrink to `abs_tol` just because the
    OTHER side of the comparison happens to be tiny.
    """
    tol = max(abs_tol, max(abs(a), abs(b)) * rel_tol)
    return abs(a - b) <= tol


def _matches_any(value: float, pool: set[float]) -> bool:
    return any(values_are_close(value, known) for known in pool)


def _add_pct(pool: set[float], value: float) -> None:
    """Percentages get quoted in report text both signed ("-6.7%") and
    unsigned ("a 6.7% decline" — the direction word already carries the
    sign, so restating it would read as a double negative) — both are
    legitimate phrasings of the same real number, so both go in the pool.
    """
    pool.add(value)
    pool.add(abs(value))


def _collect_known_values(
    analysis_results: dict[str, Any],
    sql_queries: list[dict[str, Any]],
    ml_results: dict[str, Any] | None = None,
) -> tuple[set[float], set[float]]:
    """Returns (values_pool, percentages_pool) — every number the report is
    allowed to cite, gathered from analysis_results and the raw evidence
    rows underneath it. Deliberately broad (includes raw SQL row values too)
    since a broader "known good" pool only makes this check less trigger-happy
    on things that are actually fine — the failure mode we care about
    (a genuinely fabricated number) won't be in the pool no matter how broad.

    `ml_results` (Phase 15, Objective 4) is optional and additive: when the
    ML Agent produced a real result, its metrics/predictions/feature-
    importance values are genuine, deterministically-computed numbers
    (never LLM-invented — app/agents/ml_agent.py never calls an LLM) and
    are just as legitimate for the report to cite as anything from
    analysis_results. Only numeric VALUES go in the pool — `reason`/
    `limitations` text is never scanned, and a failed/insufficient-data
    result (`ml_results["ok"] is False`) contributes nothing.
    """
    values: set[float] = set()
    percentages: set[float] = set()

    for pc in analysis_results.get("period_comparisons", []):
        for key in ("baseline_value", "current_value", "absolute_change"):
            if pc.get(key) is not None:
                values.add(float(pc[key]))
        if pc.get("percentage_change") is not None:
            _add_pct(percentages, float(pc["percentage_change"]))

    for trend in analysis_results.get("trends", []):
        for point in trend.get("points", []):
            if point.get("value") is not None:
                values.add(float(point["value"]))
            if point.get("pct_change_from_prior") is not None:
                _add_pct(percentages, float(point["pct_change_from_prior"]))
        for key in ("min_value", "max_value", "mean_value"):
            if trend.get(key) is not None:
                values.add(float(trend[key]))

    for contrib in analysis_results.get("contributions", []):
        for key in ("total_current", "total_prior", "total_change"):
            if contrib.get(key) is not None:
                values.add(float(contrib[key]))
        for c in contrib.get("contributors", []):
            for key in ("current_value", "prior_value", "change"):
                if c.get(key) is not None:
                    values.add(float(c[key]))
            for key in ("pct_change", "pct_of_total_current", "pct_of_total_change"):
                if c.get(key) is not None:
                    _add_pct(percentages, float(c[key]))

    for entry in analysis_results.get("top_n", []):
        for row in entry.get("rows", []):
            for v in row.values():
                if isinstance(v, (int, float)):
                    values.add(float(v))

    for dist in analysis_results.get("distributions", []):
        for key in ("count", "mean", "median", "min", "max", "std", "q25", "q75"):
            if dist.get(key) is not None:
                values.add(float(dist[key]))

    for q in sql_queries:
        if not q.get("validated_ok"):
            continue
        for row in q.get("rows", []):
            for v in row.values():
                if isinstance(v, (int, float)):
                    values.add(float(v))

    if ml_results and ml_results.get("ok"):
        ml_numbers: list[Any] = list(ml_results.get("metrics", {}).values())
        ml_numbers += list(ml_results.get("forecast_next") or [])
        ml_numbers += [v for pred in (ml_results.get("sample_predictions") or []) for v in pred.values()]
        ml_numbers += list((ml_results.get("feature_importance") or {}).values())
        for v in ml_numbers:
            if isinstance(v, (int, float)):
                values.add(float(v))
                # A fractional metric (e.g. accuracy=0.752) is often quoted
                # in report text as a percentage ("75.2% accuracy") — both
                # phrasings are the same real number.
                if 0 <= v <= 1:
                    _add_pct(percentages, float(v) * 100)

    return values, percentages


def check_numerical_grounding(
    report: BusinessReport,
    analysis_results: dict[str, Any],
    sql_queries: list[dict[str, Any]],
    ml_results: dict[str, Any] | None = None,
) -> list[CriticFinding]:
    """Every number in the report's free text must trace back to a number
    that actually appears in analysis_results, the raw evidence, or a real
    ML Agent result (Phase 15) — Sec 6's diagnostic Example 4 ("July
    revenue = $170,000" when the real figure is $150,633.02) is exactly
    what this catches.
    """
    values_pool, pct_pool = _collect_known_values(analysis_results, sql_queries, ml_results)
    findings: list[CriticFinding] = []

    texts = [("executive_summary", report["executive_summary"])] + [
        ("key_findings", f) for f in report["key_findings"]
    ]
    for source, text in texts:
        for value, is_pct in _extract_numbers(text):
            pool = pct_pool if is_pct else values_pool
            # A percentage that also happens to equal a plain value (rare but
            # possible, e.g. "100%") is allowed to match either pool.
            if not _matches_any(value, pool) and not (is_pct and _matches_any(value, values_pool)):
                findings.append(
                    {
                        "severity": "ERROR",
                        "category": "numerical",
                        "message": f"{source} states {value}{'%' if is_pct else ''}, which does not match any value in the evidence.",
                    }
                )
    return findings


def _extract_period_mentions(text: str) -> set[int]:
    """Month numbers mentioned by name in free text (e.g. "June" -> 6)."""
    return {_MONTH_NAMES[m.lower()] for m in _MONTH_NAME_PATTERN.findall(text)}


def _period_to_month(period_label: Any) -> int | None:
    s = str(period_label)
    match = re.match(r"^\d{4}-(\d{2})", s)
    if match:
        return int(match.group(1))
    try:
        n = int(s)
        if 1 <= n <= 12:
            return n
    except ValueError:
        pass
    return None


def check_period_consistency(report: BusinessReport, analysis_results: dict[str, Any]) -> list[CriticFinding]:
    """Months named in the report text should be among the months the
    analysis actually covers — catches a report describing a different
    period than the one the evidence is about.
    """
    known_months: set[int] = set()
    for pc in analysis_results.get("period_comparisons", []):
        for key in ("baseline_period", "current_period"):
            m = _period_to_month(pc.get(key))
            if m:
                known_months.add(m)
    for contrib in analysis_results.get("contributions", []):
        for key in ("baseline_period", "current_period"):
            m = _period_to_month(contrib.get(key))
            if m:
                known_months.add(m)
    for trend in analysis_results.get("trends", []):
        for point in trend.get("points", []):
            m = _period_to_month(point.get("period"))
            if m:
                known_months.add(m)

    if not known_months:
        return []  # nothing period-shaped in the analysis — this check doesn't apply

    findings: list[CriticFinding] = []
    text = report["executive_summary"] + " " + " ".join(report["key_findings"])
    mentioned = _extract_period_mentions(text)
    unknown = mentioned - known_months
    if unknown:
        month_names = {v: k for k, v in _MONTH_NAMES.items()}
        findings.append(
            {
                "severity": "WARNING",
                "category": "period_consistency",
                "message": (
                    f"Report mentions {', '.join(month_names[m].title() for m in sorted(unknown))}, "
                    f"not present in the analyzed periods."
                ),
            }
        )
    return findings


def check_contribution_arithmetic(analysis_results: dict[str, Any]) -> list[CriticFinding]:
    """Re-verifies the Analysis Agent's own arithmetic: change == current -
    prior, and (Phase 6 bug #4/#5 regression guard) a contributor moving
    opposite the overall trend must have pct_of_total_change == None, never a
    number — re-checked here independently of analysis_tools.py's own logic
    so a future regression there gets caught, not just trusted.
    """
    findings: list[CriticFinding] = []
    for contrib in analysis_results.get("contributions", []):
        total_change = contrib.get("total_change")
        for c in contrib.get("contributors", []):
            current_v, prior_v, change = c.get("current_value"), c.get("prior_value"), c.get("change")
            if prior_v is not None and change is not None and current_v is not None:
                expected = current_v - prior_v
                if abs(expected - change) > max(_ABS_TOL, abs(expected) * _REL_TOL):
                    findings.append(
                        {
                            "severity": "ERROR",
                            "category": "contribution_arithmetic",
                            "message": (
                                f"{contrib.get('dimension_col')}='{c.get('group')}': change {change} "
                                f"does not equal current ({current_v}) - prior ({prior_v})."
                            ),
                        }
                    )
            if total_change is not None and total_change != 0 and change is not None:
                moves_with_total = (total_change < 0 and change < 0) or (total_change > 0 and change > 0)
                if not moves_with_total and c.get("pct_of_total_change") is not None:
                    findings.append(
                        {
                            "severity": "ERROR",
                            "category": "contribution_arithmetic",
                            "message": (
                                f"{contrib.get('dimension_col')}='{c.get('group')}' moved opposite the "
                                f"overall trend but has a non-null pct_of_total_change — should be None."
                            ),
                        }
                    )
    return findings


def check_evidence_sufficiency(report: BusinessReport, analysis_results: dict[str, Any]) -> list[CriticFinding]:
    """When the Analysis Agent flagged insufficient evidence, the report
    must not claim High confidence, and should disclose a limitation."""
    diagnostic = analysis_results.get("diagnostic") or {}
    insufficient = bool(analysis_results.get("insufficient_evidence") or diagnostic.get("insufficient_evidence"))
    if not insufficient:
        return []

    findings: list[CriticFinding] = []
    if report["confidence"] == "High":
        findings.append(
            {
                "severity": "ERROR",
                "category": "missing_evidence",
                "message": "Evidence was flagged insufficient, but the report claims High confidence.",
            }
        )
    if not report["limitations"].strip():
        findings.append(
            {
                "severity": "WARNING",
                "category": "missing_evidence",
                "message": "Evidence was flagged insufficient, but no limitation was disclosed in the report.",
            }
        )
    return findings


def _dominant_contribution_groups(analysis_results: dict[str, Any], threshold_pct: float = 20.0) -> set[str]:
    groups: set[str] = set()
    for contrib in analysis_results.get("contributions", []):
        if contrib.get("total_change") is None or not contrib.get("contributors"):
            continue
        top = contrib["contributors"][0]
        pct = top.get("pct_of_total_change")
        if pct is not None and abs(pct) >= threshold_pct:
            groups.add(str(top["group"]))
    return groups


def check_causal_claims(report: BusinessReport, analysis_results: dict[str, Any]) -> list[CriticFinding]:
    """A causal claim ("revenue decreased *because* ...") needs a dominant
    contributor in the evidence to back it, and should name one. Sec 6
    Example 1 (causal claim, no dominant contributor -> FAIL) and Example 5
    (causal claim naming the actual dominant contributor -> PASS) are exactly
    what this distinguishes. Numeric overstatement in a causal claim (Example
    2, "90%" when the real figure is 74.4%) is caught by
    check_numerical_grounding instead, not duplicated here.
    """
    dominant = _dominant_contribution_groups(analysis_results)
    findings: list[CriticFinding] = []

    texts = [("executive_summary", report["executive_summary"])] + [
        ("key_findings", f) for f in report["key_findings"]
    ]
    for source, text in texts:
        if not _CAUSAL_PATTERN.search(text):
            continue
        if not dominant:
            findings.append(
                {
                    "severity": "ERROR",
                    "category": "causal_claim",
                    "message": f"{source} makes a causal claim, but no dominant contributor was identified in the evidence.",
                }
            )
        elif not any(g.lower() in text.lower() for g in dominant):
            findings.append(
                {
                    "severity": "WARNING",
                    "category": "causal_claim",
                    "message": (
                        f"{source} makes a causal claim not clearly tied to an identified dominant "
                        f"contributor ({', '.join(sorted(dominant))})."
                    ),
                }
            )
    return findings


def _known_numbers_and_labels_for_chart(
    analysis_results: dict[str, Any], source_analysis: str
) -> tuple[set[float], set[str]]:
    """The specific value pool AND label pool (period names / category names)
    a chart's own `source_analysis` tag implies — narrower than
    check_numerical_grounding's pool on purpose, so a chart genuinely can't
    borrow a number, period, or category from an unrelated analysis entry.
    """
    values: set[float] = set()
    labels: set[str] = set()
    if source_analysis == "period_comparison":
        for pc in analysis_results.get("period_comparisons", []):
            for key in ("baseline_value", "current_value"):
                if pc.get(key) is not None:
                    values.add(float(pc[key]))
            for key in ("baseline_period", "current_period"):
                if pc.get(key) is not None:
                    labels.add(str(pc[key]))
    elif source_analysis == "trend":
        for trend in analysis_results.get("trends", []):
            for point in trend.get("points", []):
                if point.get("value") is not None:
                    values.add(float(point["value"]))
                if point.get("period") is not None:
                    labels.add(str(point["period"]))
    elif source_analysis == "contribution":
        for contrib in analysis_results.get("contributions", []):
            for c in contrib.get("contributors", []):
                for key in ("current_value", "change"):
                    if c.get(key) is not None:
                        values.add(float(c[key]))
                if c.get("group") is not None:
                    labels.add(str(c["group"]))
    elif source_analysis == "top_n":
        for entry in analysis_results.get("top_n", []):
            dim = entry.get("dimension")
            for row in entry.get("rows", []):
                for k, v in row.items():
                    if isinstance(v, (int, float)):
                        values.add(float(v))
                    elif k == dim:
                        labels.add(str(v))
    elif source_analysis == "distribution":
        for dist in analysis_results.get("distributions", []):
            for key in ("count", "mean", "median", "min", "max", "std", "q25", "q75"):
                if dist.get(key) is not None:
                    values.add(float(dist[key]))
    return values, labels


def check_chart_consistency(charts: list[ChartRecord], analysis_results: dict[str, Any]) -> list[CriticFinding]:
    """Every value AND label a chart plots must trace back to the specific
    analysis_results entry it claims as its source (`source_analysis`) — Sec
    6's visualization-validation example (chart shows July=$177,767 when the
    analysis says July=$150,633.02 -> FAIL) and the period-mismatch example
    (Analysis: June->July, Chart: May->June -> FAIL) are both this check —
    the first is a value mismatch, the second a label mismatch.
    """
    findings: list[CriticFinding] = []
    for chart in charts:
        source = chart.get("source_analysis", "")
        if source in ("raw_evidence", ""):
            continue  # table/scatter fallback reads straight from SQL rows, not analysis_results
        values_pool, labels_pool = _known_numbers_and_labels_for_chart(analysis_results, source)

        for point in chart.get("data", []):
            value = point.get("value", point.get("y"))
            if value is not None and isinstance(value, (int, float)) and values_pool:
                if not _matches_any(float(value), values_pool):
                    findings.append(
                        {
                            "severity": "ERROR",
                            "category": "chart_consistency",
                            "message": (
                                f"Chart '{chart.get('title')}' plots {value}, not found in its source "
                                f"analysis ({source})."
                            ),
                        }
                    )
            label = point.get("label")
            if label is not None and labels_pool and str(label) not in labels_pool:
                findings.append(
                    {
                        "severity": "ERROR",
                        "category": "chart_consistency",
                        "message": (
                            f"Chart '{chart.get('title')}' shows '{label}', not one of the periods/"
                            f"categories in its source analysis ({source}: {sorted(labels_pool)})."
                        ),
                    }
                )
    return findings


def check_visualization_presence(analysis_results: dict[str, Any], charts: list[ChartRecord]) -> list[CriticFinding]:
    """If the analysis produced real, sufficient results but no chart came
    out of it, that's worth flagging (low severity — the answer is still
    usable without a chart, it's just a missed opportunity)."""
    if charts:
        return []
    if analysis_results.get("insufficient_evidence"):
        return []  # nothing to visualize — expected, not a gap
    has_content = any(
        analysis_results.get(key)
        for key in ("period_comparisons", "trends", "contributions", "top_n", "distributions")
    )
    if not has_content:
        return []
    return [
        {
            "severity": "INFO",
            "category": "missing_visualization",
            "message": "Analysis produced usable results but no chart was generated.",
        }
    ]


def run_all_deterministic_checks(
    report: BusinessReport,
    analysis_results: dict[str, Any],
    sql_queries: list[dict[str, Any]],
    charts: list[ChartRecord],
    ml_results: dict[str, Any] | None = None,
) -> list[CriticFinding]:
    findings: list[CriticFinding] = []
    findings += check_numerical_grounding(report, analysis_results, sql_queries, ml_results)
    findings += check_period_consistency(report, analysis_results)
    findings += check_contribution_arithmetic(analysis_results)
    findings += check_evidence_sufficiency(report, analysis_results)
    findings += check_causal_claims(report, analysis_results)
    findings += check_chart_consistency(charts, analysis_results)
    findings += check_visualization_presence(analysis_results, charts)
    return findings


def summarize_findings(findings: list[CriticFinding]) -> tuple[str, float]:
    """(status, score). ERROR present -> FAIL. Else WARNING present -> WARN.
    Else PASS. Score is a simple deterministic penalty, not a statistical
    model — it's a rough usability signal, not itself something the report
    should quote as a fact.
    """
    n_errors = sum(1 for f in findings if f["severity"] == "ERROR")
    n_warnings = sum(1 for f in findings if f["severity"] == "WARNING")
    n_info = sum(1 for f in findings if f["severity"] == "INFO")

    if n_errors:
        status = "FAIL"
    elif n_warnings:
        status = "WARN"
    else:
        status = "PASS"

    score = max(0.0, min(1.0, 1.0 - 0.34 * n_errors - 0.1 * n_warnings - 0.02 * n_info))
    return status, score
