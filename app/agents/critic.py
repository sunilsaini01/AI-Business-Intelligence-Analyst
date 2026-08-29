"""Critic (Sec 1 Fig. 2, Sec 5; user Phase 9 spec).

Reviews the report the Supervisor just synthesized against the deterministic
evidence (state["analysis_results"], state["sql_queries"], state["charts"]).
Most of the review is plain Python (app/tools/critic_checks.py) — numeric
grounding, period consistency, contribution arithmetic, chart-vs-analysis
consistency, evidence-sufficiency vs claimed confidence, and a causal-claim
heuristic all only need arithmetic and string matching, not language
understanding. The ONE genuinely semantic question — "does this wording
overstate what the evidence supports" — is isolated behind the LLM
abstraction (CriticSemanticCheck, ModelTier.STRONG) and kept small: it never
sees raw SQL rows, only the executive summary/key findings and the Analysis
Agent's own facts/interpretations.

Routing (bounded retry loop, matches the original blueprint's Fig. 2 exactly
— do not simplify away the retry_count == max_retries branch, that's how an
unbounded loop gets avoided): FAIL with retry_count < max_retries -> clear
`report` to None and increment retry_count, which routes back to the
Supervisor (app/graph/workflow.py's `report is None` check re-triggers
`_synthesize()`, now with `critic_feedback` in state so it can actually
address what failed). FAIL with retry_count == max_retries -> leave `report`
set but force confidence down to "Low" and append the unresolved findings to
`limitations` — never loop a third time, never drop the disclosure. PASS/WARN
-> leave `report` as-is, done.

An LLM/infra error calling the semantic check is NOT a content FAIL — it's
caught and recorded as an INFO finding (deterministic checks alone still
decide PASS/WARN/FAIL), consistent with Sec 9's "infra failure isn't a
content failure" rule. Phase 13 (Objective A) added exception
CLASSIFICATION on top of this — see app/core/errors.py — without changing
that outcome at all: `critic_feedback.semantic_check_error_category`
records rate_limit/timeout/provider_error/validation_error/
application_error for observability; it is still always an INFO finding,
never re-raised, never a content FAIL.
"""

from __future__ import annotations

import time

from app.agents.schemas import CriticSemanticCheck
from app.core.errors import classify_exception
from app.core.llm import LLMClientProtocol, ModelTier, get_llm_client
from app.graph.state import AgentState, CriticFinding, trace_event
from app.tools.critic_checks import run_all_deterministic_checks, summarize_findings

_SEMANTIC_SYSTEM_PROMPT = """You are reviewing a business-intelligence report for overstatement.
Given the report's executive summary and key findings, and the deterministic facts/interpretations
that were computed by code (not by you), decide whether the report's wording is fully supported —
no invented claims, and no causal or certainty language stronger than the facts/interpretations
justify. Interpretations are already hedged (they say "appears to be" / "likely") — the report
restating an interpretation as a plain fact, or adding a cause the interpretations don't mention,
counts as unsupported.

Facts (ground truth, computed by code):
{facts}

Interpretations (the analysis system's own hedged reasoning, computed by code):
{interpretations}

Report executive summary:
{executive_summary}

Report key findings:
{key_findings}
"""


async def _semantic_check(
    report: dict, analysis_results: dict, llm: LLMClientProtocol
) -> list[CriticFinding]:
    diagnostic = analysis_results.get("diagnostic") or {}
    facts = diagnostic.get("facts") or []
    interpretations = diagnostic.get("interpretations") or []
    if not facts and not interpretations:
        return []  # nothing for a semantic check to compare against — deterministic checks cover this case

    result = await llm.complete_structured(
        tier=ModelTier.STRONG,
        system=_SEMANTIC_SYSTEM_PROMPT.format(
            facts="\n".join(f"- {f}" for f in facts) or "(none)",
            interpretations="\n".join(f"- {i}" for i in interpretations) or "(none)",
            executive_summary=report["executive_summary"],
            key_findings="\n".join(f"- {f}" for f in report["key_findings"]),
        ),
        messages=[{"role": "user", "content": "Review the report now."}],
        response_model=CriticSemanticCheck,
    )

    if result.supported:
        return []
    if result.unsupported_claims:
        return [
            {"severity": "ERROR", "category": "semantic", "message": f"Unsupported claim: {claim}"}
            for claim in result.unsupported_claims
        ]
    return [{"severity": "WARNING", "category": "semantic", "message": result.reasoning}]


def _force_degrade(report: dict, findings: list[CriticFinding]) -> None:
    """Mutates `report` in place — retries are exhausted, the run still has
    to answer, but confidence must reflect that verification didn't fully
    pass (Sec 9's degrade-and-disclose rule)."""
    report["confidence"] = "Low"
    error_messages = [f["message"] for f in findings if f["severity"] == "ERROR"]
    note = "Automated review found unresolved issues: " + "; ".join(error_messages) if error_messages else (
        "Automated review flagged unresolved concerns."
    )
    report["limitations"] = f"{report['limitations']} {note}".strip() if report["limitations"] else note


async def critic_node(state: AgentState, llm: LLMClientProtocol | None = None) -> AgentState:
    state["trace"].append(trace_event("critic", "enter"))
    started = time.perf_counter()

    report = state["report"]
    semantic_check_error_category: str | None = None
    if report is None:
        findings: list[CriticFinding] = [
            {"severity": "ERROR", "category": "missing_evidence", "message": "No report was produced to review."}
        ]
    else:
        findings = run_all_deterministic_checks(
            report, state["analysis_results"], state["sql_queries"], state["charts"], state.get("ml_results")
        )
        llm = llm or get_llm_client()
        try:
            findings += await _semantic_check(report, state["analysis_results"], llm)
        except Exception as exc:  # noqa: BLE001 — infra/LLM error is not a content FAIL
            # Classification only (Phase 13, Objective A) — the OUTCOME is
            # unchanged from before: still one INFO finding, still resolved
            # entirely by the deterministic checks, never a content FAIL and
            # never re-raised. This just records WHICH kind of failure it
            # was (rate_limit/timeout/provider_error/application_error) so
            # the orchestration layer/observability doesn't have to
            # string-match the message to tell them apart.
            semantic_check_error_category = classify_exception(exc)
            findings.append(
                {
                    "severity": "INFO",
                    "category": "semantic",
                    "message": (
                        f"Semantic check unavailable ({semantic_check_error_category}) — "
                        "deterministic checks only."
                    ),
                }
            )

    status, score = summarize_findings(findings)

    if status == "FAIL" and state["retry_count"] < state["max_retries"]:
        state["retry_count"] += 1
        state["report"] = None  # routes back to the Supervisor for revision
        target_agent = "supervisor"
    elif status == "FAIL":
        target_agent = None
        if report is not None:
            _force_degrade(report, findings)
    else:
        target_agent = None

    state["critic_feedback"] = {
        "status": status,
        "score": score,
        "findings": findings,
        "verified_claims": [] if status == "FAIL" else list(report["key_findings"]) if report else [],
        "unsupported_claims": [f["message"] for f in findings if f["severity"] == "ERROR"],
        "recommendations": [f["message"] for f in findings if f["severity"] in ("ERROR", "WARNING")],
        "target_agent": target_agent,
        "semantic_check_error_category": semantic_check_error_category,
    }

    state["trace"].append(
        trace_event("critic", "exit", duration_ms=(time.perf_counter() - started) * 1000)
    )
    return state
