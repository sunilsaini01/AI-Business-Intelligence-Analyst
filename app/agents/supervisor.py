"""Supervisor (Sec 1, Sec 5; user Phase 4/6/9 spec). Runs twice per request
— or more, if the Critic (Phase 9) rejects a synthesis and there are retries
left:

1. First entry (no plan yet): classify intent, pick which dataset answers the
   question (`analytics` vs `olist`), and produce a short plan — structured
   output via SupervisorPlan, not free text (Phase 4 requirement).
2. Second (and any later revision) entry (plan set, report is None): synthesize
   a final answer. As of Phase 6, this step gets BOTH the raw SQL evidence
   AND the Analysis Agent's deterministic facts/interpretations — the LLM is
   told to treat `facts` as ground truth, cite `interpretations` as the
   analysis system's own already-computed reasoning (not to independently
   invent a different one), and fall back to the raw evidence only when no
   analysis was possible. Structured via SupervisorSynthesis; told explicitly
   to say "insufficient evidence" rather than invent a cause. As of Phase 9,
   if `state["critic_feedback"]` shows a FAIL from a previous attempt (the
   Critic cleared `report` back to None to trigger this re-entry — see
   app/agents/critic.py and app/graph/workflow.py's routing), the specific
   issues it found are added to the prompt so the revision actually addresses
   them instead of repeating the same mistake.

`llm` is dependency-injected (defaults to app.core.llm.get_llm_client()) so
tests can pass a fake and never touch ANTHROPIC_API_KEY — see
tests/agents/test_supervisor.py.

Sec 5 note: this *is* the interim Report Agent too — Phase 10 will pull
synthesis out into its own agent. The Critic (Phase 9) now provides the
formal groundedness checking this docstring used to defer to Phase 10 for;
the system prompt's own strict-grounding rules are the first line of
defense, the Critic is the second.
"""

from __future__ import annotations

import time

from app.agents.prompt_utils import compact_rows_json
from app.agents.schemas import SupervisorPlan, SupervisorSynthesis
from app.core.llm import LLMClientProtocol, ModelTier, get_llm_client
from app.graph.state import AgentState, trace_event
from app.tools.schema_tools import format_schema_for_prompt, get_analytics_schema

_PLANNING_SYSTEM_PROMPT = """You are the Supervisor of a business-intelligence analysis system.
Given a user's question and the available database schemas, decide:
- whether the question is answerable from the data at all (out_of_scope=true if not — e.g.
  small talk, requests for opinions, or anything neither dataset could contain)
- which dataset answers it best (analytics vs olist — see schema descriptions)
- the intent category
- a short plan of what evidence to gather

Each step becomes ONE SQL query (with joins/aggregation/CTEs as needed — a single query can
already do a lot). Do NOT split one query's worth of work into multiple steps like "join A and B",
then "join with C", then "aggregate" — that wastes 3 queries fetching thousands of unneeded raw
rows where 1 aggregate query would do. Write the fewest steps that get the full answer: usually
1 step for a simple total/breakdown, up to 3-4 only for a genuinely multi-part comparison (e.g. a
diagnostic question needing revenue-by-month AND a regional breakdown are two different
aggregates, so two steps — but don't add a third step just to re-fetch what the first two already
covered).

Do not write SQL yourself — a separate SQL Agent does that from your plan.
Available schemas:
{schema_text}
"""

_SYNTHESIS_SYSTEM_PROMPT = """You are the Supervisor of a business-intelligence analysis system,
now synthesizing a final answer from evidence already gathered by SQL queries and analyzed by a
deterministic Analysis Agent (pandas/NumPy, not an LLM — every number in it was computed by code).

STRICT RULES:
- Every number in your answer must appear verbatim in the Deterministic Analysis or the raw
  Evidence below. Never compute, estimate, or infer a number that isn't already there — the
  Analysis Agent already did the arithmetic; you narrate it, you don't redo it.
- Deterministic Analysis "facts" are ground truth — state them as fact.
- Deterministic Analysis "interpretations" are the analysis system's own reasoning about which
  contributor likely explains a change — you may cite them, but phrase them as what the evidence
  suggests, not as a certainty, and do not substitute a different explanation of your own.
- If Deterministic Analysis reports insufficient_evidence=true for a diagnostic question, your
  answer must also acknowledge that — do not override it with your own reading of the raw rows.
- If there is no Deterministic Analysis at all, fall back to the raw Evidence directly, same
  strict-grounding rule.
- key_findings must each cite a specific number from the Deterministic Analysis or the Evidence.
{critic_feedback_section}
Original question: {question}
Plan that was executed: {plan}

Deterministic Analysis (facts/interpretations/limitations, computed by pandas — not the LLM):
{analysis_text}

Raw evidence gathered (query purpose, and the actual rows returned):
{evidence_text}
"""

_CRITIC_FEEDBACK_SECTION = """
A previous attempt at this answer was reviewed and rejected. Fix these specific issues — do
not repeat them:
{issues}
"""


def _direct_out_of_scope_report(reason: str) -> dict:
    return {
        "executive_summary": "I can't answer that from this dataset.",
        "key_findings": [],
        "evidence": [],
        "recommendations": [],
        "confidence": "Low",
        "limitations": reason,
        # out_of_scope short-circuits straight to END (app/graph/workflow.py) —
        # the Report Generator (Phase 10) never runs on this path, so these
        # stay at their empty defaults rather than being left unset.
        "verified_claims": [],
        "analysis_explanation": "",
        "visualizations": [],
        "technical_details": {},
        "narrative": None,
    }


async def supervisor_node(state: AgentState, llm: LLMClientProtocol | None = None) -> AgentState:
    llm = llm or get_llm_client()
    state["trace"].append(trace_event("supervisor", "enter"))
    started = time.perf_counter()

    if not state["plan"] and state["report"] is None:
        state = await _plan(state, llm)
    elif state["report"] is None:
        state = await _synthesize(state, llm)
    # else: already resolved (shouldn't normally be re-entered)

    state["trace"].append(
        trace_event("supervisor", "exit", duration_ms=(time.perf_counter() - started) * 1000)
    )
    return state


async def _plan(state: AgentState, llm: LLMClientProtocol) -> AgentState:
    schema = await get_analytics_schema()
    schema_text = format_schema_for_prompt(schema)

    plan = await llm.complete_structured(
        tier=ModelTier.STRONG,
        system=_PLANNING_SYSTEM_PROMPT.format(schema_text=schema_text),
        messages=[{"role": "user", "content": state["question"]}],
        response_model=SupervisorPlan,
    )

    if plan.out_of_scope:
        state["intent"] = "out_of_scope"
        state["report"] = _direct_out_of_scope_report(plan.reasoning)
        return state

    state["intent"] = plan.intent
    state["target_schema"] = plan.target_schema
    state["plan"] = plan.steps
    state["required_tools"] = ["sql_agent"]
    return state


def _format_analysis_for_prompt(analysis: dict | None) -> str:
    """Renders state["analysis_results"] (Phase 6, app/agents/analysis_agent.py)
    into a compact text block for the synthesis prompt. Facts/interpretations/
    limitations come first (the diagnostic composite, when present) since
    that's the direct answer to a "why" question; the other analysis types
    follow as supporting detail.
    """
    if not analysis:
        return "(no deterministic analysis was performed)"

    lines: list[str] = []

    diagnostic = analysis.get("diagnostic")
    if diagnostic:
        for f in diagnostic.get("facts", []):
            lines.append(f"FACT: {f}")
        for i in diagnostic.get("interpretations", []):
            lines.append(f"INTERPRETATION: {i}")
        for l in diagnostic.get("limitations", []):
            lines.append(f"LIMITATION: {l}")
        if diagnostic.get("insufficient_evidence"):
            lines.append(f"INSUFFICIENT_EVIDENCE: {diagnostic.get('reason', '')}")

    for pc in analysis.get("period_comparisons", []):
        pct = pc.get("percentage_change")
        pct_text = f", {pct:+.1f}%" if pct is not None else ""
        lines.append(
            f"Period comparison ({pc['value_col']}): {pc['baseline_period']}={pc['baseline_value']:,.2f} -> "
            f"{pc['current_period']}={pc['current_value']:,.2f} ({pc['direction']}{pct_text})"
        )

    for contrib in analysis.get("contributions", []):
        top = contrib.get("contributors", [])[:5]
        top_text = "; ".join(
            f"{c['group']}={c['current_value']:,.2f}"
            + (f" (change {c['change']:+,.2f}, {c['pct_of_total_change']:.1f}% of total change)"
               if c.get("pct_of_total_change") is not None else "")
            for c in top
        )
        lines.append(f"Contribution by {contrib['dimension_col']}: {top_text}")

    for trend in analysis.get("trends", []):
        lines.append(
            f"Trend ({trend['value_col']} over {trend['period_col']}): direction={trend['direction']}, "
            f"min={trend['min_value']:,.2f}, max={trend['max_value']:,.2f}, mean={trend['mean_value']:,.2f}"
        )

    for dist in analysis.get("distributions", []):
        lines.append(
            f"Distribution ({dist['column']}): count={dist['count']}, mean={dist['mean']:,.2f}, "
            f"median={dist['median']:,.2f}, min={dist['min']:,.2f}, max={dist['max']:,.2f}"
        )

    if analysis.get("insufficient_evidence") and not diagnostic:
        lines.append(f"INSUFFICIENT_EVIDENCE: {analysis.get('reason', '')}")

    return "\n".join(lines) if lines else "(no deterministic analysis was performed)"


async def _synthesize(state: AgentState, llm: LLMClientProtocol) -> AgentState:
    evidence_lines = []
    for q in state["sql_queries"]:
        if q["validated_ok"]:
            evidence_lines.append(
                f"- Query: {q['text']}\n  Rows ({q['row_count']}): {compact_rows_json(q['rows'], max_rows=40)}"
            )
        else:
            evidence_lines.append(f"- Query rejected: {q['rejection_reason']}")
    evidence_text = "\n".join(evidence_lines) if evidence_lines else "(no evidence gathered)"
    analysis_text = _format_analysis_for_prompt(state.get("analysis_results"))

    critic_feedback = state.get("critic_feedback")
    critic_feedback_section = ""
    if critic_feedback and critic_feedback["status"] == "FAIL":
        issues = "\n".join(f"- {f['message']}" for f in critic_feedback["findings"] if f["severity"] in ("ERROR", "WARNING"))
        critic_feedback_section = _CRITIC_FEEDBACK_SECTION.format(issues=issues or "(see recommendations)")

    synthesis = await llm.complete_structured(
        tier=ModelTier.STRONG,
        system=_SYNTHESIS_SYSTEM_PROMPT.format(
            question=state["question"],
            plan="; ".join(state["plan"]),
            analysis_text=analysis_text,
            evidence_text=evidence_text,
            critic_feedback_section=critic_feedback_section,
        ),
        messages=[{"role": "user", "content": "Synthesize the final answer now."}],
        response_model=SupervisorSynthesis,
    )

    state["report"] = {
        "executive_summary": synthesis.executive_summary,
        "key_findings": synthesis.key_findings,
        "evidence": [
            {"query": q["text"], "row_count": q["row_count"]} for q in state["sql_queries"] if q["validated_ok"]
        ],
        "recommendations": [],
        "confidence": "Low" if synthesis.insufficient_evidence else synthesis.confidence,
        "limitations": synthesis.limitations,
        # Populated by the Report Generator (Phase 10, app/agents/report_agent.py)
        # once the Critic has reviewed this synthesis — empty until then.
        "verified_claims": [],
        "analysis_explanation": "",
        "visualizations": [],
        "technical_details": {},
        "narrative": None,
    }
    return state
