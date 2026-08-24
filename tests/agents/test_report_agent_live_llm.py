"""The one Report Generator test that calls a real LLM (Phase 10's optional
`ReportNarrative` polish step). Skipped automatically without a configured
key for the active provider, same as tests/agents/test_critic_live_llm.py
and tests/api/test_analyze_live_llm.py. Also skips gracefully at RUNTIME on
a rate-limit/quota error rather than failing the suite — Groq daily quota
exhaustion is a known, external, non-code condition (see
docs/architecture.md), not something a live test should report as broken.

Run explicitly once quota is available:
    docker compose exec api pytest tests/agents/test_report_agent_live_llm.py -v
"""

from __future__ import annotations

import anthropic
import groq
import pytest

from app.agents.report_agent import report_agent_node
from app.core.config import get_settings
from app.graph.state import new_state

_settings = get_settings()
_active_key = _settings.groq_api_key if _settings.llm_provider == "groq" else _settings.anthropic_api_key

pytestmark = pytest.mark.skipif(
    not _active_key,
    reason=f"No API key configured for LLM_PROVIDER={_settings.llm_provider} — live-LLM test skipped, not failed.",
)


def _grounded_pass_state() -> dict:
    state = new_state("Why did revenue decrease in July?")
    state["intent"] = "diagnostic"
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
            "limitations": [], "insufficient_evidence": False, "reason": None,
        },
        "insufficient_evidence": False, "reason": None,
    }
    state["charts"] = []
    state["critic_feedback"] = {
        "status": "PASS", "score": 1.0, "findings": [],
        "verified_claims": ["June revenue: 161445.80", "July revenue: 150633.02"],
        "unsupported_claims": [], "recommendations": [], "target_agent": None,
    }
    state["report"] = {
        "executive_summary": (
            "Revenue decreased from 161445.80 to 150633.02 (-6.7%), driven mainly by Enterprise "
            "(74.4% of the total change)."
        ),
        "key_findings": ["June revenue: 161445.80", "July revenue: 150633.02", "Enterprise change: -10610.84"],
        "evidence": [], "recommendations": [], "confidence": "Medium", "limitations": "",
        "verified_claims": [], "analysis_explanation": "", "visualizations": [],
        "technical_details": {}, "narrative": None,
    }
    return state


@pytest.mark.asyncio
async def test_narrative_is_grounded_or_gracefully_omitted_via_real_llm():
    state = _grounded_pass_state()
    try:
        # narrative_enabled=True is required here — REPORT_NARRATIVE_ENABLED
        # defaults to False (Phase 10 cleanup), and this test's whole point
        # is to exercise the real LLM call, not silently skip it.
        result = await report_agent_node(state, narrative_enabled=True)  # llm=None -> real get_llm_client()
    except (anthropic.RateLimitError, groq.RateLimitError) as exc:
        pytest.skip(f"LLM provider rate-limited/quota exhausted, not a code failure: {exc}")

    report = result["report"]
    # Regardless of whether the real model produced a usable narrative
    # (it might invent something and get discarded — that's the safety net
    # working, not a bug), the report must remain valid either way.
    assert report["narrative"] is None or isinstance(report["narrative"], str)
    assert report["executive_summary"]  # never overwritten by the narrative attempt
    assert report["analysis_explanation"]  # deterministic, always present regardless of the LLM call's outcome
