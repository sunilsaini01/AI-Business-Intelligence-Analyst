"""Deterministic tests for app/agents/report_agent.py (Phase 10). No live
LLM calls — the one optional call (`ReportNarrative`) is exercised via
`ScriptedLLMClient`/a fake that raises, never the real network. See
tests/agents/test_report_agent_live_llm.py for the one live-LLM exception.
"""

from __future__ import annotations

import json

import groq
import httpx
import pytest

from app.agents.report_agent import report_agent_node
from app.agents.schemas import ReportNarrative
from app.core.config import get_settings
from app.graph.state import new_state
from tests.fakes import ScriptedLLMClient

# --- fixtures ----------------------------------------------------------------


def _report(**overrides) -> dict:
    base = {
        "executive_summary": "Revenue fell from 161445.80 to 150633.02 (-6.7%), driven mainly by Enterprise.",
        "key_findings": ["June revenue: 161445.80", "July revenue: 150633.02", "Enterprise change: -10610.84"],
        "evidence": [{"query": "SELECT ...", "row_count": 2}],
        "recommendations": [],
        "confidence": "Medium",
        "limitations": "",
        "verified_claims": [],
        "analysis_explanation": "",
        "visualizations": [],
        "technical_details": {},
        "narrative": None,
    }
    base.update(overrides)
    return base


def _diagnostic_analysis_results() -> dict:
    return {
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
        "contributions": [
            {
                "ok": True, "dimension_col": "segment", "value_col": "revenue",
                "total_current": 6172.0, "total_prior": 16782.83, "total_change": -10610.83,
                "baseline_period": "2026-06", "current_period": "2026-07",
                "contributors": [
                    {"group": "Enterprise", "current_value": 6172.0, "prior_value": 16782.83,
                     "change": -10610.84, "pct_change": -63.2, "pct_of_total_current": 100.0,
                     "pct_of_total_change": 74.4, "rank": 1},
                ],
            }
        ],
        "diagnostic": {
            "ok": True,
            "facts": ["revenue went from 161,445.80 in 2026-06 to 150,633.02 in 2026-07 (decrease (-6.7%))."],
            "interpretations": [
                "By segment, 'Enterprise' appears to be the dominant contributor, accounting for "
                "approximately 74.4% of the total change (-10,610.84)."
            ],
            "limitations": [],
            "insufficient_evidence": False,
            "reason": None,
        },
        "insufficient_evidence": False,
        "reason": None,
    }


def _state_with(*, report=None, analysis_results=None, charts=None, critic_feedback=None, retry_count=0) -> dict:
    state = new_state("Why did revenue decrease in July?")
    state["intent"] = "diagnostic"
    state["report"] = report if report is not None else _report()
    state["analysis_results"] = analysis_results if analysis_results is not None else _diagnostic_analysis_results()
    state["charts"] = charts if charts is not None else [
        {"chart_type": "bar", "title": "Revenue: June vs July", "subtitle": None, "x_axis": None, "y_axis": None,
         "group_by": None, "sort": None, "data": [], "units": None, "source_analysis": "period_comparison",
         "reason": "", "limitations": [], "path": "", "spec_summary": {}},
    ]
    state["critic_feedback"] = critic_feedback if critic_feedback is not None else {
        "status": "PASS", "score": 1.0, "findings": [],
        "verified_claims": list(state["report"]["key_findings"]),
        "unsupported_claims": [], "recommendations": [], "target_agent": None,
    }
    state["retry_count"] = retry_count
    return state


# --- 1/9. correct report generation, Critic PASS -----------------------------


@pytest.mark.asyncio
async def test_report_agent_populates_all_five_new_fields_on_pass():
    state = _state_with()
    result = await report_agent_node(state, llm=ScriptedLLMClient({}))
    report = result["report"]

    assert report["verified_claims"] == state["critic_feedback"]["verified_claims"]
    assert report["analysis_explanation"]
    assert report["visualizations"] == [{"chart_type": "bar", "title": "Revenue: June vs July", "subtitle": None}]
    assert report["technical_details"]["critic_status"] == "PASS"
    assert report["technical_details"]["retry_count"] == 0
    assert report["narrative"] is None  # REPORT_NARRATIVE_ENABLED defaults to False — see narrative-config tests below


@pytest.mark.asyncio
async def test_report_agent_records_trace_events():
    state = _state_with()
    result = await report_agent_node(state, llm=ScriptedLLMClient({}))
    nodes = [t["node"] for t in result["trace"]]
    assert nodes == ["report_agent", "report_agent"]
    assert result["trace"][0]["event"] == "enter"
    assert result["trace"][1]["event"] == "exit"


# --- 2/3/8. never touches the 6 Supervisor/Critic-owned fields ---------------


@pytest.mark.asyncio
async def test_report_agent_never_touches_executive_summary_or_key_findings():
    state = _state_with()
    original_summary = state["report"]["executive_summary"]
    original_findings = list(state["report"]["key_findings"])
    result = await report_agent_node(state, llm=ScriptedLLMClient({}))
    assert result["report"]["executive_summary"] == original_summary
    assert result["report"]["key_findings"] == original_findings


@pytest.mark.asyncio
async def test_report_agent_never_touches_limitations_confidence_or_evidence():
    state = _state_with(report=_report(limitations="Some caveat.", confidence="Low"))
    result = await report_agent_node(state, llm=ScriptedLLMClient({}))
    assert result["report"]["limitations"] == "Some caveat."
    assert result["report"]["confidence"] == "Low"
    assert result["report"]["evidence"] == [{"query": "SELECT ...", "row_count": 2}]


# --- 9/10/11. Critic PASS / WARN / FAIL behavior ------------------------------


@pytest.mark.asyncio
async def test_report_agent_on_warn_preserves_findings_and_still_enriches():
    warn_feedback = {
        "status": "WARN", "score": 0.8,
        "findings": [{"severity": "WARNING", "category": "period_consistency", "message": "minor mismatch"}],
        "verified_claims": ["June revenue: 161445.80"],
        "unsupported_claims": [], "recommendations": ["minor mismatch"], "target_agent": None,
    }
    state = _state_with(critic_feedback=warn_feedback)
    result = await report_agent_node(state, llm=ScriptedLLMClient({}))
    assert result["report"]["verified_claims"] == ["June revenue: 161445.80"]
    assert result["report"]["technical_details"]["critic_status"] == "WARN"
    # WARN still gets the deterministic enrichments — nothing is suppressed.
    assert result["report"]["analysis_explanation"]


@pytest.mark.asyncio
async def test_report_agent_on_fail_exhausted_skips_llm_and_preserves_degraded_state():
    fail_feedback = {
        "status": "FAIL", "score": 0.2,
        "findings": [{"severity": "ERROR", "category": "numerical", "message": "unsupported number"}],
        "verified_claims": [],  # critic.py sets this to [] on FAIL
        "unsupported_claims": ["unsupported number"], "recommendations": ["unsupported number"],
        "target_agent": None,
    }
    degraded_report = _report(confidence="Low", limitations="Automated review found unresolved issues: unsupported number")
    state = _state_with(report=degraded_report, critic_feedback=fail_feedback, retry_count=2)

    # An LLM client that would raise if ever called — proves the narrative
    # step is genuinely skipped on FAIL, not just usually empty.
    class _ExplodingLLM:
        async def complete(self, **kwargs):
            raise AssertionError("must not be called on a FAIL-exhausted report")

        async def complete_structured(self, **kwargs):
            raise AssertionError("must not be called on a FAIL-exhausted report")

    # narrative_enabled=True on purpose: proves FAIL-exhausted skips the LLM
    # call REGARDLESS of configuration, not just because it happens to be
    # off by default (Phase 10 cleanup requirement #2/#3).
    result = await report_agent_node(state, llm=_ExplodingLLM(), narrative_enabled=True)

    assert result["report"]["verified_claims"] == []
    assert result["report"]["narrative"] is None
    assert result["report"]["confidence"] == "Low"
    assert "unresolved issues" in result["report"]["limitations"]


# --- 5/6/7. numeric / period / category preservation in analysis_explanation --


@pytest.mark.asyncio
async def test_analysis_explanation_preserves_exact_numbers_periods_and_category():
    state = _state_with()
    result = await report_agent_node(state, llm=ScriptedLLMClient({}))
    explanation = result["report"]["analysis_explanation"]
    # Verified benchmark values (evaluation/datasets/benchmark.json bi-004) —
    # must appear byte-for-byte as produced by app/tools/analysis_tools.py,
    # never recomputed here.
    assert "161,445.80" in explanation
    assert "150,633.02" in explanation
    assert "-6.7%" in explanation
    assert "Enterprise" in explanation
    assert "74.4%" in explanation
    assert "-10,610.84" in explanation


# --- 13. empty / insufficient evidence ----------------------------------------


@pytest.mark.asyncio
async def test_analysis_explanation_when_insufficient_evidence():
    state = _state_with(
        analysis_results={
            "period_comparisons": [], "trends": [], "contributions": [], "top_n": [], "distributions": [],
            "diagnostic": None, "insufficient_evidence": True, "reason": "No SQL evidence matched a supported pattern.",
        }
    )
    result = await report_agent_node(state, llm=ScriptedLLMClient({}))
    assert result["report"]["analysis_explanation"] == "No SQL evidence matched a supported pattern."


@pytest.mark.asyncio
async def test_analysis_explanation_empty_dict_is_empty_string():
    state = _state_with(analysis_results={})
    result = await report_agent_node(state, llm=ScriptedLLMClient({}))
    assert result["report"]["analysis_explanation"] == ""


# --- 14. non-diagnostic (Olist-shaped) contribution analysis -----------------


@pytest.mark.asyncio
async def test_analysis_explanation_for_olist_shaped_contribution_analysis():
    state = _state_with(
        analysis_results={
            "period_comparisons": [], "trends": [], "top_n": [], "distributions": [],
            "diagnostic": None, "insufficient_evidence": False, "reason": None,
            "contributions": [
                {
                    "ok": True, "dimension_col": "category", "value_col": "review_score",
                    "total_current": 450.2, "total_prior": None, "total_change": None,
                    "baseline_period": None, "current_period": None,
                    "contributors": [
                        {"group": "cds_dvds_musicais", "current_value": 4.6429, "prior_value": None,
                         "change": None, "pct_change": None, "pct_of_total_current": 12.3,
                         "pct_of_total_change": None, "rank": 1},
                    ],
                }
            ],
        },
        charts=[],
    )
    result = await report_agent_node(state, llm=ScriptedLLMClient({}))
    explanation = result["report"]["analysis_explanation"]
    assert "cds_dvds_musicais" in explanation
    assert "4.64" in explanation  # 4.6429 rendered to 2dp by the ,.2f format
    assert result["report"]["visualizations"] == []


# --- top_n bug fix (Phase 10 cleanup #1) --------------------------------------


def _top_n_analysis_results(*, contributions: list[dict] | None = None) -> dict:
    return {
        "period_comparisons": [], "trends": [], "distributions": [],
        "diagnostic": None, "insufficient_evidence": False, "reason": None,
        "contributions": contributions if contributions is not None else [],
        "top_n": [
            {
                "dimension": "category", "value_col": "revenue",
                "rows": [
                    {"category": "Software", "revenue": 737525.145},
                    {"category": "Hardware", "revenue": 500000.0},
                ],
            }
        ],
    }


@pytest.mark.asyncio
async def test_analysis_explanation_falls_back_to_top_n_when_contributions_empty():
    """The exact bug scenario: top_n has real ranked data, contributions is
    an empty list (e.g. analyze_contribution's own ok=False edge case never
    got appended by the Analysis Agent) — analysis_explanation must not come
    back empty."""
    state = _state_with(analysis_results=_top_n_analysis_results(contributions=[]), charts=[])
    result = await report_agent_node(state, llm=ScriptedLLMClient({}))
    explanation = result["report"]["analysis_explanation"]
    assert explanation != ""
    assert "Software" in explanation
    assert "737,525.15" in explanation or "737,525.14" in explanation  # float rounding, either is a correct .2f
    assert "category" in explanation
    assert "revenue" in explanation


@pytest.mark.asyncio
async def test_analysis_explanation_falls_back_to_top_n_when_contributions_all_failed():
    """Same scenario, but `contributions` has an entry present that failed
    (ok=False) rather than an empty list — must still fall back to top_n."""
    failed_contribution = {
        "ok": False, "dimension_col": "category", "value_col": "revenue",
        "total_current": None, "total_prior": None, "total_change": None,
        "baseline_period": None, "current_period": None, "contributors": [],
    }
    state = _state_with(
        analysis_results=_top_n_analysis_results(contributions=[failed_contribution]), charts=[]
    )
    result = await report_agent_node(state, llm=ScriptedLLMClient({}))
    explanation = result["report"]["analysis_explanation"]
    assert explanation != ""
    assert "Software" in explanation
    assert "Hardware" in explanation


@pytest.mark.asyncio
async def test_analysis_explanation_prefers_contributions_over_top_n_when_both_available():
    """Existing behavior for successful contributions is unchanged (Phase 10
    cleanup requirement) — when contributions DID succeed, top_n is not also
    rendered (would be redundant), so this must match
    test_analysis_explanation_for_olist_shaped_contribution_analysis's output
    exactly, byte for byte, even with top_n data also present."""
    contributions = [
        {
            "ok": True, "dimension_col": "category", "value_col": "review_score",
            "total_current": 450.2, "total_prior": None, "total_change": None,
            "baseline_period": None, "current_period": None,
            "contributors": [
                {"group": "cds_dvds_musicais", "current_value": 4.6429, "prior_value": None,
                 "change": None, "pct_change": None, "pct_of_total_current": 12.3,
                 "pct_of_total_change": None, "rank": 1},
            ],
        }
    ]
    without_top_n = _state_with(
        analysis_results={
            "period_comparisons": [], "trends": [], "top_n": [], "distributions": [],
            "diagnostic": None, "insufficient_evidence": False, "reason": None,
            "contributions": contributions,
        },
        charts=[],
    )
    with_top_n = _state_with(
        analysis_results={
            "period_comparisons": [], "trends": [], "distributions": [],
            "diagnostic": None, "insufficient_evidence": False, "reason": None,
            "contributions": contributions,
            "top_n": [{"dimension": "category", "value_col": "review_score", "rows": [{"category": "cds_dvds_musicais", "review_score": 4.6429}]}],
        },
        charts=[],
    )
    result_without = await report_agent_node(without_top_n, llm=ScriptedLLMClient({}))
    result_with = await report_agent_node(with_top_n, llm=ScriptedLLMClient({}))
    assert result_without["report"]["analysis_explanation"] == result_with["report"]["analysis_explanation"]
    assert "Software" not in result_with["report"]["analysis_explanation"]  # top_n's own extra rows never leak in either


@pytest.mark.asyncio
async def test_analysis_explanation_top_n_handles_malformed_entries_gracefully():
    """Missing dimension/value_col/rows keys, an empty rows list, and a row
    missing either key must all be skipped individually, never raise."""
    state = _state_with(
        analysis_results={
            "period_comparisons": [], "trends": [], "distributions": [],
            "diagnostic": None, "insufficient_evidence": False, "reason": None,
            "contributions": [],
            "top_n": [
                {},  # missing everything
                {"dimension": "category", "value_col": "revenue", "rows": []},  # empty rows
                {"dimension": "category", "value_col": "revenue", "rows": [{"category": "Software"}]},  # missing value_col in row
                {"dimension": "category", "rows": [{"category": "Software", "revenue": 1.0}]},  # missing value_col key on entry
            ],
        },
        charts=[],
    )
    result = await report_agent_node(state, llm=ScriptedLLMClient({}))
    assert result["report"]["analysis_explanation"] == ""  # nothing usable anywhere -> empty, not a crash


@pytest.mark.asyncio
async def test_analysis_explanation_empty_top_n_and_empty_contributions_falls_back_to_insufficient_evidence():
    state = _state_with(
        analysis_results={
            "period_comparisons": [], "trends": [], "distributions": [], "contributions": [], "top_n": [],
            "diagnostic": None, "insufficient_evidence": True, "reason": "No SQL evidence matched a supported pattern.",
        },
        charts=[],
    )
    result = await report_agent_node(state, llm=ScriptedLLMClient({}))
    assert result["report"]["analysis_explanation"] == "No SQL evidence matched a supported pattern."


# --- narrative configuration (Phase 10 cleanup, REPORT_NARRATIVE_ENABLED) ----


class _CountingLLM:
    """Records whether it was ever called — the sharpest way to prove "zero
    narrative LLM call", stronger than just asserting narrative is None
    (which could also happen because a real call returned nothing usable).
    """

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, **kwargs):
        self.calls += 1
        raise AssertionError("complete() should never be called by report_agent_node")

    async def complete_structured(self, **kwargs):
        self.calls += 1
        raise AssertionError("must not be called when narrative_enabled=False")


def test_report_narrative_enabled_defaults_to_false():
    """Locks in the safe default (Phase 10 cleanup #2) — regresses loudly if
    someone flips the Settings default without updating .env.example too."""
    assert get_settings().report_narrative_enabled is False


@pytest.mark.asyncio
async def test_narrative_disabled_by_default_makes_zero_llm_call():
    llm = _CountingLLM()
    state = _state_with()  # critic_feedback status="PASS" by default
    result = await report_agent_node(state, llm=llm)  # narrative_enabled not passed -> reads settings (False)
    assert result["report"]["narrative"] is None
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_narrative_disabled_explicitly_makes_zero_llm_call_even_on_warn():
    warn_feedback = {
        "status": "WARN", "score": 0.8, "findings": [],
        "verified_claims": ["June revenue: 161445.80"],
        "unsupported_claims": [], "recommendations": [], "target_agent": None,
    }
    llm = _CountingLLM()
    state = _state_with(critic_feedback=warn_feedback)
    result = await report_agent_node(state, llm=llm, narrative_enabled=False)
    assert result["report"]["narrative"] is None
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_narrative_enabled_calls_the_llm_on_pass():
    llm = ScriptedLLMClient({ReportNarrative: [ReportNarrative(narrative="Revenue declined 6.7% in July, led by Enterprise.")]})
    state = _state_with()
    result = await report_agent_node(state, llm=llm, narrative_enabled=True)
    assert result["report"]["narrative"] == "Revenue declined 6.7% in July, led by Enterprise."
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_narrative_enabled_calls_the_llm_on_warn():
    warn_feedback = {
        "status": "WARN", "score": 0.8, "findings": [],
        "verified_claims": ["June revenue: 161445.80"],
        "unsupported_claims": [], "recommendations": [], "target_agent": None,
    }
    llm = ScriptedLLMClient({ReportNarrative: [ReportNarrative(narrative="Revenue declined 6.7% in July, led by Enterprise.")]})
    state = _state_with(critic_feedback=warn_feedback)
    result = await report_agent_node(state, llm=llm, narrative_enabled=True)
    assert result["report"]["narrative"] == "Revenue declined 6.7% in July, led by Enterprise."


@pytest.mark.asyncio
async def test_narrative_discarded_when_it_invents_a_number():
    """Grounding rejection behavior (Phase 10 cleanup #2, requirement 4) —
    unchanged by the new config flag, just now requires narrative_enabled=True
    to actually reach the check."""
    llm = ScriptedLLMClient({ReportNarrative: [ReportNarrative(narrative="Revenue declined by 999999.99 in July.")]})
    state = _state_with()
    result = await report_agent_node(state, llm=llm, narrative_enabled=True)
    assert result["report"]["narrative"] is None
    # the rest of the report is untouched by the discarded narrative
    assert "999999.99" not in result["report"]["executive_summary"]


@pytest.mark.asyncio
async def test_narrative_omitted_when_llm_call_raises():
    class _RaisingLLM:
        async def complete(self, **kwargs):
            raise RuntimeError("boom")

        async def complete_structured(self, **kwargs):
            raise RuntimeError("boom")

    state = _state_with()
    result = await report_agent_node(state, llm=_RaisingLLM(), narrative_enabled=True)
    assert result["report"]["narrative"] is None


@pytest.mark.asyncio
async def test_narrative_rate_limit_is_classified_in_technical_details():
    """Phase 13, Objective A: a genuine provider RateLimitError during the
    narrative call is classified and recorded — narrative still degrades
    to None, exactly as before; only the recorded category is new."""

    class _RateLimitedLLM:
        async def complete(self, **kwargs):
            raise RuntimeError("should not be called")

        async def complete_structured(self, **kwargs):
            resp = httpx.Response(429, request=httpx.Request("POST", "http://test"))
            raise groq.RateLimitError("rate limited", response=resp, body=None)

    state = _state_with()
    result = await report_agent_node(state, llm=_RateLimitedLLM(), narrative_enabled=True)
    assert result["report"]["narrative"] is None
    assert result["report"]["technical_details"]["narrative_error_category"] == "rate_limit"


@pytest.mark.asyncio
async def test_narrative_error_category_is_none_when_narrative_succeeds():
    llm = ScriptedLLMClient({ReportNarrative: [ReportNarrative(narrative="Revenue declined 6.7% in July, led by Enterprise.")]})
    state = _state_with()
    result = await report_agent_node(state, llm=llm, narrative_enabled=True)
    assert result["report"]["narrative"] is not None
    assert result["report"]["technical_details"]["narrative_error_category"] is None


@pytest.mark.asyncio
async def test_narrative_error_category_is_none_when_discarded_for_grounding_not_a_provider_failure():
    """Inventing an ungrounded number is a content problem the grounding
    check itself correctly caught — not a provider/infra failure, so it
    must NOT be classified as one."""
    llm = ScriptedLLMClient({ReportNarrative: [ReportNarrative(narrative="Revenue declined by 999999.99 in July.")]})
    state = _state_with()
    result = await report_agent_node(state, llm=llm, narrative_enabled=True)
    assert result["report"]["narrative"] is None
    assert result["report"]["technical_details"]["narrative_error_category"] is None


@pytest.mark.asyncio
async def test_narrative_error_category_is_none_when_narrative_disabled():
    state = _state_with()
    result = await report_agent_node(state, llm=ScriptedLLMClient({}))  # narrative_enabled defaults to False
    assert result["report"]["technical_details"]["narrative_error_category"] is None


@pytest.mark.asyncio
async def test_narrative_omitted_when_scripted_client_has_nothing_queued():
    """No ReportNarrative response scripted -> ScriptedLLMClient raises
    AssertionError -> report_agent_node must swallow it, not crash (matches
    the real behavior every pre-Phase-10 integration test exercises when the
    flag is on)."""
    state = _state_with()
    result = await report_agent_node(state, llm=ScriptedLLMClient({}), narrative_enabled=True)
    assert result["report"]["narrative"] is None


# --- 15. July diagnostic benchmark, exact verified values --------------------


@pytest.mark.asyncio
async def test_preserves_july_diagnostic_benchmark_values_exactly():
    """evaluation/datasets/benchmark.json bi-004's verified ground truth:
    June $161,445.80 -> July $150,633.02 (-6.7%), Enterprise -$10,610.84
    (74.4% of the decline). Report Generator must reproduce these exactly —
    never recompute, never round differently, never substitute a different
    entity/period (Sec "Known previous issues" 5/6/7)."""
    state = _state_with()
    result = await report_agent_node(state, llm=ScriptedLLMClient({}))
    td = result["report"]["technical_details"]
    assert td["critic_status"] == "PASS"

    pc = state["analysis_results"]["period_comparisons"][0]
    assert pc["baseline_value"] == 161445.80
    assert pc["current_value"] == 150633.02
    assert pc["percentage_change"] == -6.7
    contributor = state["analysis_results"]["contributions"][0]["contributors"][0]
    assert contributor["group"] == "Enterprise"
    assert contributor["change"] == -10610.84
    assert contributor["pct_of_total_change"] == 74.4
    # And the Report Generator's own output actually carries them, verbatim.
    explanation = result["report"]["analysis_explanation"]
    assert "161,445.80" in explanation and "150,633.02" in explanation and "74.4%" in explanation


# --- 16. report serialization --------------------------------------------------


@pytest.mark.asyncio
async def test_final_report_is_json_serializable():
    state = _state_with()
    result = await report_agent_node(state, llm=ScriptedLLMClient({}))
    serialized = json.dumps(result["report"])
    assert json.loads(serialized)["confidence"] == "Medium"


# --- defensive: report is None (shouldn't happen given graph routing) --------


@pytest.mark.asyncio
async def test_report_agent_is_a_safe_noop_when_report_is_none():
    state = new_state("test")
    assert state["report"] is None
    result = await report_agent_node(state, llm=ScriptedLLMClient({}))
    assert result["report"] is None
    assert [t["node"] for t in result["trace"]] == ["report_agent", "report_agent"]
