"""Report Generator (Sec 1, Sec 5; user Phase 10 spec).

Runs AFTER the Critic (Phase 9) has validated — or exhausted retries on —
the Supervisor's synthesized report; never before, never in place of it
(app/graph/workflow.py's `_route_after_critic` sends every terminal state
with `report is not None` here, whether that's a clean PASS/WARN or a
force-degraded FAIL-exhausted-retries report). This is a presentation/
finalization layer, not another analysis agent:

    Critic            = "Is this answer trustworthy?"
    Report Generator  = "How do we present this validated answer clearly?"

It never re-derives facts: every number/entity/period in the enrichments it
adds comes verbatim from state["critic_feedback"] (Phase 9),
state["analysis_results"] (Phase 6), and state["charts"] (Phase 7) — this
module contains zero SQL, zero pandas arithmetic, zero chart-selection
logic, and zero independent fact discovery. It also never touches the 6
fields the Supervisor/Critic already own (`executive_summary`,
`key_findings`, `evidence`, `recommendations`, `confidence`,
`limitations`) — seem app/graph/state.py's `BusinessReport` docstring for
the full field-ownership split. On a degraded FAIL-exhausted report this
matters most: this node presents what the Critic validated, it does not
spin it into something more confident-sounding.

The one optional LLM call (`ReportNarrative`) is a pure wording pass over
already-approved content, re-validated against the same numeric-grounding
check the Critic uses before being accepted — if it fails, invents
anything, or the call errors out (including a Groq/Anthropic quota
exhaustion, Sec "Known previous issues" Issue 1/2), the narrative is simply
omitted (`None`), never a hard failure and never blocking the rest of the
report. It is skipped entirely on a FAIL-exhausted report, REGARDLESS of
configuration — a wording pass over content that just failed verification
is exactly the wrong moment to make it sound more polished.

Gated behind `Settings.report_narrative_enabled`
(`REPORT_NARRATIVE_ENABLED`, off by default — Phase 10 cleanup) since it's
an extra LLM call, and therefore extra quota consumption, on every single
successful (PASS/WARN) analysis; every deterministic field above is
generated regardless of this flag. See `report_agent_node`'s
`narrative_enabled` parameter for how tests override it directly.
"""

from __future__ import annotations

import time
from typing import Any

from app.agents.schemas import ReportNarrative
from app.core.config import get_settings
from app.core.errors import ErrorCategory, classify_exception
from app.core.llm import LLMClientProtocol, ModelTier, get_llm_client
from app.graph.state import AgentState, BusinessReport, trace_event
from app.tools.critic_checks import check_numerical_grounding

_NARRATIVE_SYSTEM_PROMPT = """You are preparing a business-intelligence report for a non-technical
stakeholder. You are given an executive summary and key findings that have ALREADY been verified —
your only job is to reorganize and clarify the wording, not to add anything.

STRICT RULES:
- Do not introduce any number, percentage, entity name, period, or claim that is not already
  present verbatim in the text below.
- Do not soften or remove the stated confidence/limitations.
- 2-4 sentences, plain business language, no jargon.

Executive summary:
{executive_summary}

Key findings:
{key_findings}
"""


def _build_verified_claims(critic_feedback: dict[str, Any] | None) -> list[str]:
    """Never invented — verbatim from the Critic's own verdict (Phase 9,
    app/agents/critic.py), which already sets this to the report's key
    findings on PASS/WARN and to [] on an unresolved FAIL."""
    if critic_feedback is None:
        return []
    return list(critic_feedback.get("verified_claims", []))


def _format_top_n_lines(analysis_results: dict[str, Any]) -> list[str]:
    """Renders state["analysis_results"]["top_n"] (Phase 6, produced by
    app/tools/analysis_tools.py::top_n via the Analysis Agent) — the ranked
    rows are already-computed dicts keyed by the actual dimension/value
    column names (e.g. {"category": "Software", "revenue": 737525.14}), so
    this only formats them, never re-ranks or recomputes anything.

    Only reached as a fallback when `contributions` produced no usable line
    (see `_format_analysis_explanation`) — when a contribution breakdown IS
    available, it already says who the top contributor is and by how much
    of the TOTAL/CHANGE, which top_n's plain per-item ranking doesn't add to.
    Malformed/empty entries (missing dimension/value_col/rows, a row missing
    either key, a non-numeric value) are skipped individually rather than
    raising or discarding the whole entry.
    """
    lines: list[str] = []
    for entry in analysis_results.get("top_n", []):
        dim = entry.get("dimension")
        value_col = entry.get("value_col")
        rows = entry.get("rows") or []
        if not dim or not value_col or not rows:
            continue

        ranked: list[str] = []
        for row in rows:
            label = row.get(dim)
            value = row.get(value_col)
            if label is None or value is None:
                continue
            try:
                ranked.append(f"{label} ({float(value):,.2f})")
            except (TypeError, ValueError):
                continue  # a non-numeric value_col entry -> skip this row, not the whole ranking

        if ranked:
            lines.append(f"Top {dim} by {value_col}: " + "; ".join(ranked) + ".")
    return lines


def _format_analysis_explanation(analysis_results: dict[str, Any] | None) -> str:
    """Deterministic prose rendering of state["analysis_results"] (Phase 6)
    — reuses the Analysis Agent's own already-computed values, never
    recomputes anything. Diagnostic facts/interpretations are already full,
    stakeholder-readable sentences (app/tools/analysis_tools.py::
    diagnose_decline) and are used verbatim when present.
    """
    if not analysis_results:
        return ""

    diagnostic = analysis_results.get("diagnostic")
    if diagnostic:
        lines = list(diagnostic.get("facts", [])) + list(diagnostic.get("interpretations", []))
        for limitation in diagnostic.get("limitations", []):
            lines.append(f"Note: {limitation}")
        if lines:
            return " ".join(lines)

    lines: list[str] = []
    for pc in analysis_results.get("period_comparisons", []):
        if not pc.get("ok"):
            continue
        pct = pc.get("percentage_change")
        pct_text = f", a {pct:+.1f}% change" if pct is not None else ""
        lines.append(
            f"{pc['value_col']} moved from {pc['baseline_value']:,.2f} ({pc['baseline_period']}) to "
            f"{pc['current_value']:,.2f} ({pc['current_period']}){pct_text}."
        )

    contribution_lines: list[str] = []
    for contrib in analysis_results.get("contributions", []):
        if not contrib.get("ok") or not contrib.get("contributors"):
            continue
        top = contrib["contributors"][0]
        contribution_lines.append(
            f"By {contrib['dimension_col']}, '{top['group']}' was the largest contributor at "
            f"{top['current_value']:,.2f} ({top['pct_of_total_current']:.1f}% of the total)."
        )
    if contribution_lines:
        lines.extend(contribution_lines)
    else:
        # Bug fix (Phase 10 cleanup): a dimension breakdown with no period to
        # compare against computes BOTH `contributions` and `top_n`
        # independently (app/agents/analysis_agent.py's
        # "dimension_cols and not period_col" branch) — if `contributions`
        # comes back empty/failed (analyze_contribution's own edge cases)
        # while `top_n` still succeeded, that ranked data was previously
        # silently dropped here, so a genuinely available answer produced an
        # empty analysis_explanation. Falls back to top_n ONLY when
        # contributions gave us nothing, so a report where contributions did
        # succeed renders byte-identical to before this fix.
        lines.extend(_format_top_n_lines(analysis_results))

    for trend in analysis_results.get("trends", []):
        if not trend.get("ok"):
            continue
        lines.append(
            f"{trend['value_col']} showed a {trend['direction']} trend, ranging from "
            f"{trend['min_value']:,.2f} to {trend['max_value']:,.2f} (average {trend['mean_value']:,.2f})."
        )
    for dist in analysis_results.get("distributions", []):
        if not dist.get("ok"):
            continue
        lines.append(
            f"{dist['column']}: {dist['count']} values, averaging {dist['mean']:,.2f} "
            f"(range {dist['min']:,.2f}-{dist['max']:,.2f})."
        )

    if not lines and analysis_results.get("insufficient_evidence"):
        return analysis_results.get("reason") or "Insufficient evidence to produce a detailed analysis."

    return " ".join(lines)


def _format_ml_summary(ml_results: dict[str, Any] | None) -> str:
    """Deterministic, no-LLM formatting of state["ml_results"] (Phase 15,
    Objective 4) — same "verbatim from already-computed state" contract as
    _format_analysis_explanation: every number here comes straight from
    app/agents/ml_agent.py's real model run, this function only renders
    it. `None` (the question wasn't predictive) renders nothing; a
    structured `ok=False` result (not appropriate / insufficient data)
    renders an honest one-line explanation rather than silently omitting
    the fact that a prediction was attempted — see ml_agent.py's own
    docstring on why that distinction matters.
    """
    if not ml_results:
        return ""
    if not ml_results.get("ok"):
        reason = ml_results.get("reason") or "not available for this question."
        return f"Predictive analysis: {reason}"

    if ml_results.get("task") == "forecasting":
        forecast_next = ml_results.get("forecast_next") or []
        next_value = f"{forecast_next[0]:,.2f}" if forecast_next else "n/a"
        mae = ml_results.get("metrics", {}).get("mae")
        mae_text = f", MAE {mae:,.2f} on held-out history" if mae is not None else ""
        return (
            f"Forecast ({ml_results.get('model_name', 'model')}): next-period "
            f"{ml_results.get('target', 'value')} projected at {next_value}{mae_text}."
        )

    if ml_results.get("task") == "churn_risk":
        metrics = ml_results.get("metrics", {})
        accuracy = metrics.get("accuracy")
        roc_auc = metrics.get("roc_auc")
        parts = []
        if accuracy is not None:
            parts.append(f"{accuracy * 100:.1f}% accuracy")
        if roc_auc is not None:
            parts.append(f"{roc_auc:.2f} ROC-AUC")
        score_text = " (" + ", ".join(parts) + ")" if parts else ""
        return (
            f"Churn risk model ({ml_results.get('model_name', 'model')}) evaluated on "
            f"{ml_results.get('test_size', 0)} held-out customers{score_text}."
        )

    return ""


def _build_visualizations(charts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """References the Visualization Agent's own already-selected charts
    (Phase 7) — never re-derives chart type or re-reads chart data; the full
    spec is served separately via GET /analysis/{id}/charts."""
    return [{"chart_type": c["chart_type"], "title": c["title"], "subtitle": c.get("subtitle")} for c in charts]


def _build_technical_details(
    critic_feedback: dict[str, Any] | None,
    state: AgentState,
    *,
    narrative_error_category: str | None = None,
) -> dict[str, Any]:
    return {
        "critic_status": critic_feedback["status"] if critic_feedback else None,
        "critic_score": critic_feedback["score"] if critic_feedback else None,
        "retry_count": state["retry_count"],
        "sql_queries_run": len(state["sql_queries"]),
        "charts_generated": len(state["charts"]),
        # Phase 13, Objective A — set only when the optional narrative call
        # was attempted AND failed/was discarded; None when it was never
        # attempted (disabled, or FAIL-exhausted) or succeeded outright. See
        # app/core/errors.py::ErrorCategory. Never changes whether
        # `narrative` ends up None — same outcome as before, just recorded.
        "narrative_error_category": narrative_error_category,
    }


async def _try_narrative(
    report: BusinessReport,
    analysis_results: dict[str, Any],
    sql_queries: list[dict[str, Any]],
    llm: LLMClientProtocol,
    ml_results: dict[str, Any] | None = None,
) -> tuple[str | None, ErrorCategory | None]:
    """One optional, isolated LLM call — pure wording, re-validated against
    the same numeric-grounding check the Critic uses before being trusted.
    Any failure (LLM error, quota, invented content) degrades to `None`,
    never a hard failure — `executive_summary` is always the report's real,
    primary, already-validated text; this is a bonus, not a dependency.

    Returns (narrative, error_category) — error_category is set only when
    the LLM call itself raised (Phase 13, Objective A classification); the
    "invented content, discarded" path returns (None, None) since that's
    not a provider/infra failure, the grounding check itself worked
    correctly and did its job.
    """
    try:
        result = await llm.complete_structured(
            tier=ModelTier.STRONG,
            system=_NARRATIVE_SYSTEM_PROMPT.format(
                executive_summary=report["executive_summary"],
                key_findings="\n".join(f"- {f}" for f in report["key_findings"]),
            ),
            messages=[{"role": "user", "content": "Write the stakeholder narrative now."}],
            response_model=ReportNarrative,
        )
    except Exception as exc:  # noqa: BLE001 — quota/infra/schema failure -> omit, never a hard failure (Issue 1/2)
        return None, classify_exception(exc)

    probe: BusinessReport = {**report, "executive_summary": result.narrative, "key_findings": []}
    findings = check_numerical_grounding(probe, analysis_results, sql_queries, ml_results)
    if any(f["severity"] == "ERROR" for f in findings):
        return None, None  # the rewrite introduced something not in the evidence -> discard, don't ship it
    return result.narrative, None


async def report_agent_node(
    state: AgentState,
    llm: LLMClientProtocol | None = None,
    narrative_enabled: bool | None = None,
) -> AgentState:
    """Finalizes the report the Critic just reviewed. A no-op (besides the
    trace event) if `state["report"]` is somehow None — defensive only, the
    graph's routing (`_route_after_critic`) never reaches this node in that
    case.

    `narrative_enabled=None` (production default, and every real call in
    app/graph/workflow.py) reads `Settings.report_narrative_enabled`
    (`REPORT_NARRATIVE_ENABLED`, off by default — see app/core/config.py).
    Explicit `True`/`False` — same dependency-injection pattern as `llm` —
    lets tests exercise either path directly without touching env vars or
    the settings cache.
    """
    state["trace"].append(trace_event("report_agent", "enter"))
    started = time.perf_counter()

    report = state["report"]
    if report is not None:
        critic_feedback = state.get("critic_feedback")
        report["verified_claims"] = _build_verified_claims(critic_feedback)
        report["analysis_explanation"] = _format_analysis_explanation(state.get("analysis_results"))
        report["ml_summary"] = _format_ml_summary(state.get("ml_results"))
        report["visualizations"] = _build_visualizations(state.get("charts", []))

        if narrative_enabled is None:
            narrative_enabled = get_settings().report_narrative_enabled

        status = critic_feedback["status"] if critic_feedback else None
        narrative_error_category: str | None = None
        if narrative_enabled and status in ("PASS", "WARN"):
            llm = llm or get_llm_client()
            report["narrative"], narrative_error_category = await _try_narrative(
                report,
                state.get("analysis_results") or {},
                state.get("sql_queries") or [],
                llm,
                state.get("ml_results"),
            )
        else:
            report["narrative"] = None

        report["technical_details"] = _build_technical_details(
            critic_feedback, state, narrative_error_category=narrative_error_category
        )
        state["report"] = report

    state["trace"].append(
        trace_event("report_agent", "exit", duration_ms=(time.perf_counter() - started) * 1000)
    )
    return state
