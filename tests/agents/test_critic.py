"""Critic Agent tests (Phase 9). Deterministic-check coverage lives in
tests/unit/test_critic_checks.py — these test the orchestrator: LLM semantic
check wiring (via ScriptedLLMClient, no network), LLM failure handling, and
PASS/WARN/FAIL retry routing. No live LLM calls in this file — see
tests/agents/test_critic_live_llm.py for the one optional live test.
"""

from __future__ import annotations

import groq
import httpx
import pytest

from app.agents.critic import critic_node
from app.agents.schemas import CriticSemanticCheck
from app.graph.state import new_state
from tests.fakes import ScriptedLLMClient


def _diagnostic_state(
    *, executive_summary: str, key_findings: list[str], confidence: str = "Medium", with_dominant_contribution: bool = True
) -> dict:
    """`with_dominant_contribution=False` builds the Sec 6 Example 1 shape:
    a confirmed change with NO contribution breakdown at all — nothing in
    the evidence could back *any* causal claim, as opposed to a claim that
    just names the wrong entity (a different, WARNING-level case)."""
    state = new_state("Why did revenue decrease in July?")
    state["intent"] = "diagnostic"
    contributions = []
    interpretations = []
    if with_dominant_contribution:
        contributions = [
            {
                "ok": True, "dimension_col": "segment", "value_col": "revenue",
                "total_current": 6172.0, "total_prior": 16782.83, "total_change": -10610.83,
                "baseline_period": "2026-06", "current_period": "2026-07",
                "contributors": [
                    {"group": "Enterprise", "current_value": 6172.0, "prior_value": 16782.83,
                     "change": -10610.83, "pct_change": -63.2, "pct_of_total_current": 100.0,
                     "pct_of_total_change": 74.4, "rank": 1},
                ],
            }
        ]
        interpretations = [
            "By segment, 'Enterprise' appears to be the dominant contributor, accounting for "
            "approximately 74.4% of the total change (-10,610.83)."
        ]
    state["analysis_results"] = {
        "period_comparisons": [
            {
                "ok": True, "period_col": "month", "value_col": "revenue",
                "baseline_period": "2026-06", "current_period": "2026-07",
                "baseline_value": 161445.80, "current_value": 150633.02,
                "absolute_change": -10812.78, "percentage_change": -6.7,
                "direction": "decrease", "note": None, "insufficient_evidence": False, "reason": None,
            }
        ],
        "trends": [], "top_n": [], "distributions": [],
        "contributions": contributions,
        "diagnostic": {
            "ok": True,
            "facts": ["revenue went from 161,445.80 in 2026-06 to 150,633.02 in 2026-07 (decrease (-6.7%))."],
            "interpretations": interpretations,
            "limitations": [], "insufficient_evidence": False, "reason": None,
        },
        "insufficient_evidence": False, "reason": None,
    }
    state["charts"] = []
    state["report"] = {
        "executive_summary": executive_summary,
        "key_findings": key_findings,
        "evidence": [],
        "recommendations": [],
        "confidence": confidence,
        "limitations": "",
    }
    return state


@pytest.mark.asyncio
async def test_grounded_report_with_supported_semantic_check_passes():
    state = _diagnostic_state(
        executive_summary=(
            "Revenue decreased from 161445.80 to 150633.02 (-6.7%), driven mainly by Enterprise "
            "(74.4% of the total change)."
        ),
        key_findings=["June revenue: 161445.80", "July revenue: 150633.02", "Enterprise change: -10610.83"],
    )
    fake_llm = ScriptedLLMClient(
        {CriticSemanticCheck: [CriticSemanticCheck(supported=True, unsupported_claims=[], reasoning="Fully grounded.")]}
    )

    result = await critic_node(state, llm=fake_llm)

    assert result["critic_feedback"]["status"] == "PASS"
    assert result["report"] is not None  # not cleared for retry


@pytest.mark.asyncio
async def test_flags_unsupported_causal_claim_and_routes_for_retry():
    state = _diagnostic_state(
        executive_summary="Revenue decreased because customers churned en masse.",
        key_findings=[],
        with_dominant_contribution=False,  # Sec 6 Example 1: no evidence backs ANY cause
    )
    # The deterministic causal-claim check alone already produces an ERROR
    # (the claim names no dominant-contributor entity from the evidence) —
    # the semantic check still runs too (facts/interpretations are present in
    # this fixture), scripted here to agree ("supported") so the FAIL below
    # is unambiguously coming from the deterministic check, not the LLM one.
    fake_llm = ScriptedLLMClient({CriticSemanticCheck: [CriticSemanticCheck(supported=True, unsupported_claims=[], reasoning="")]})

    result = await critic_node(state, llm=fake_llm)

    assert result["critic_feedback"]["status"] == "FAIL"
    assert any(f["category"] == "causal_claim" for f in result["critic_feedback"]["findings"])
    assert result["report"] is None  # cleared -> routes back to supervisor for revision
    assert result["retry_count"] == 1


@pytest.mark.asyncio
async def test_semantic_check_unsupported_claim_fails():
    state = _diagnostic_state(
        executive_summary="Revenue decreased from 161445.80 to 150633.02, driven by Enterprise.",
        key_findings=["June: 161445.80", "July: 150633.02"],
    )
    fake_llm = ScriptedLLMClient(
        {
            CriticSemanticCheck: [
                CriticSemanticCheck(
                    supported=False,
                    unsupported_claims=["Revenue decreased ... driven by Enterprise (stated as fact, not hedged)"],
                    reasoning="The interpretation is hedged; the summary states it as certain.",
                )
            ]
        }
    )

    result = await critic_node(state, llm=fake_llm)

    assert result["critic_feedback"]["status"] == "FAIL"
    assert any(f["category"] == "semantic" for f in result["critic_feedback"]["findings"])


@pytest.mark.asyncio
async def test_llm_failure_does_not_crash_and_is_not_a_content_fail():
    class _RaisingLLM:
        async def complete(self, **kwargs):
            raise RuntimeError("should not be called")

        async def complete_structured(self, **kwargs):
            raise RuntimeError("simulated LLM/infra failure")

    state = _diagnostic_state(
        executive_summary="Revenue decreased from 161445.80 to 150633.02 (-6.7%).",
        key_findings=["June revenue: 161445.80", "July revenue: 150633.02"],
    )

    result = await critic_node(state, llm=_RaisingLLM())

    findings = result["critic_feedback"]["findings"]
    assert any(f["severity"] == "INFO" and f["category"] == "semantic" for f in findings)
    # Deterministic checks alone are clean here -> the LLM failure must not
    # force a FAIL by itself (infra error != content problem).
    assert result["critic_feedback"]["status"] == "PASS"
    # Phase 13, Objective A: an unrecognized exception classifies as
    # "application_error" — still just an INFO finding, still PASS.
    assert result["critic_feedback"]["semantic_check_error_category"] == "application_error"


@pytest.mark.asyncio
async def test_rate_limit_during_semantic_check_is_classified_but_still_not_a_content_fail():
    """Phase 13, Objective A: a genuine provider RateLimitError degrades
    exactly the same way any other LLM failure does (INFO finding, PASS
    stands if deterministic checks are clean) — classification changes
    WHAT gets recorded, never the outcome."""

    class _RateLimitedLLM:
        async def complete(self, **kwargs):
            raise RuntimeError("should not be called")

        async def complete_structured(self, **kwargs):
            resp = httpx.Response(429, request=httpx.Request("POST", "http://test"))
            raise groq.RateLimitError("rate limited", response=resp, body=None)

    state = _diagnostic_state(
        executive_summary="Revenue decreased from 161445.80 to 150633.02 (-6.7%).",
        key_findings=["June revenue: 161445.80", "July revenue: 150633.02"],
    )

    result = await critic_node(state, llm=_RateLimitedLLM())

    assert result["critic_feedback"]["status"] == "PASS"
    assert result["critic_feedback"]["semantic_check_error_category"] == "rate_limit"
    findings = result["critic_feedback"]["findings"]
    assert any(
        f["severity"] == "INFO" and f["category"] == "semantic" and "rate_limit" in f["message"]
        for f in findings
    )


@pytest.mark.asyncio
async def test_semantic_check_error_category_is_none_when_the_check_succeeds():
    state = _diagnostic_state(
        executive_summary="Revenue was 150633.02 in July.",
        key_findings=[],
    )
    fake_llm = ScriptedLLMClient(
        {CriticSemanticCheck: [CriticSemanticCheck(supported=True, unsupported_claims=[], reasoning="")]}
    )
    result = await critic_node(state, llm=fake_llm)
    assert result["critic_feedback"]["semantic_check_error_category"] is None


@pytest.mark.asyncio
async def test_semantic_check_error_category_is_none_when_report_is_missing():
    state = new_state("some question")
    state["report"] = None
    result = await critic_node(state, llm=ScriptedLLMClient({}))
    assert result["critic_feedback"]["semantic_check_error_category"] is None


@pytest.mark.asyncio
async def test_fail_at_max_retries_forces_low_confidence_and_exits():
    state = _diagnostic_state(
        executive_summary="Revenue decreased because customers churned en masse.",
        key_findings=[],
        confidence="High",
        with_dominant_contribution=False,
    )
    state["retry_count"] = state["max_retries"]  # already exhausted
    fake_llm = ScriptedLLMClient({})

    result = await critic_node(state, llm=fake_llm)

    assert result["critic_feedback"]["status"] == "FAIL"
    assert result["report"] is not None  # NOT cleared — forced exit, not another retry
    assert result["report"]["confidence"] == "Low"
    assert "unresolved issues" in result["report"]["limitations"]
    assert result["retry_count"] == state["max_retries"]  # not incremented further


@pytest.mark.asyncio
async def test_missing_report_is_a_hard_fail():
    state = new_state("some question")
    state["report"] = None
    result = await critic_node(state, llm=ScriptedLLMClient({}))
    assert result["critic_feedback"]["status"] == "FAIL"


@pytest.mark.asyncio
async def test_writes_trace_events():
    state = _diagnostic_state(executive_summary="Revenue was 150633.02 in July.", key_findings=[])
    fake_llm = ScriptedLLMClient(
        {CriticSemanticCheck: [CriticSemanticCheck(supported=True, unsupported_claims=[], reasoning="")]}
    )
    result = await critic_node(state, llm=fake_llm)
    node_names = [t["node"] for t in result["trace"]]
    assert node_names == ["critic", "critic"]
