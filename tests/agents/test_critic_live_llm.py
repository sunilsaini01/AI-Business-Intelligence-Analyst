"""Live-provider tests for the Critic's ONE semantic LLM call
(app/agents/critic.py::_semantic_check). Skipped automatically without a
configured key, same as the other live tests.

Phase 13, Objective C rewrite. The original version of this file asserted
one absolute verdict ("this exact fixture must always get PASS/WARN") on a
fixture whose executive_summary ("driven mainly by Enterprise") was less
hedged than its own interpretation ("Enterprise *appears to be* the
dominant contributor") — the system prompt's own rule ("restating a hedged
interpretation as a plain fact... counts as unsupported") makes a strict
FAIL on that exact wording defensible, not obviously wrong. A live model
occasionally returning that stricter, defensible read isn't a code bug;
repeatedly treating it as one would have meant either weakening the
assertion (masking a REAL regression next time) or leaving a flaky test
that fails on legitimate model judgment. Two changes fix this without
weakening what's actually being verified:

1. The well-grounded fixture's wording now mirrors its interpretation's
   hedge level exactly, removing the genuine ambiguity — this tests
   whether the Critic correctly PASSES an appropriately-hedged, fully
   grounded report, not whether the model always agrees with strict
   over-claiming.
2. A second, genuinely adversarial fixture (a fabricated root cause —
   "customer churn due to a competitor's pricing change" — never present
   anywhere in the evidence) must reliably FAIL. This is the harder,
   deterministic-checks-CAN'T-catch-this case (app/tools/critic_checks.py::
   check_causal_claims only checks whether the named ENTITY matches a
   dominant contributor, which "Enterprise" does here — the fabricated
   CAUSE is only catchable by the semantic check), so this is a genuine
   contract test of the one thing this LLM call exists for, not incidental
   wording.

Both tests classify their result via `critic_feedback.semantic_check_error_
category` (Phase 13, Objective A) rather than a try/except around
`critic_node` — `critic_node` NEVER lets an LLM failure propagate out of it
by design (Sec 9's "infra failure isn't a content failure" rule), so a
try/except-based skip wrapper here would be unreachable dead code (the
exact gap flagged in the Phase 11/12 audits). Checking the classification
field instead means quota/timeout/provider-error conditions are correctly
skipped, while a real semantic misclassification still fails the test —
never converted into a skip just to stay green.

Run explicitly once quota is available:
    docker compose exec api pytest tests/agents/test_critic_live_llm.py -v
"""

from __future__ import annotations

import pytest

from app.agents.critic import critic_node
from app.core.config import get_settings
from app.graph.state import new_state

_settings = get_settings()
_active_key = _settings.groq_api_key if _settings.llm_provider == "groq" else _settings.anthropic_api_key

pytestmark = pytest.mark.skipif(
    not _active_key,
    reason=f"No API key configured for LLM_PROVIDER={_settings.llm_provider} — live-LLM test skipped, not failed.",
)

# rate_limit/timeout/provider_error (app/core/errors.py::ErrorCategory) all
# mean "the provider was unavailable", never "the model gave a wrong
# verdict" — validation_error/application_error are NOT skip-worthy here,
# since those would indicate a real bug in how this test drives critic_node.
_INFRASTRUCTURE_CATEGORIES = {"rate_limit", "timeout", "provider_error"}


def _base_diagnostic_state() -> dict:
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
                     "change": -10610.83, "pct_change": -63.2, "pct_of_total_current": 100.0,
                     "pct_of_total_change": 74.4, "rank": 1},
                ],
            }
        ],
        "diagnostic": {
            "ok": True,
            "facts": ["revenue went from 161,445.80 in 2026-06 to 150,633.02 in 2026-07 (decrease (-6.7%))."],
            "interpretations": [
                "By segment, 'Enterprise' appears to be the dominant contributor, accounting for "
                "approximately 74.4% of the total change (-10,610.83)."
            ],
            "limitations": [], "insufficient_evidence": False, "reason": None,
        },
        "insufficient_evidence": False, "reason": None,
    }
    state["charts"] = []
    return state


def _grounded_hedged_state() -> dict:
    """Wording deliberately mirrors the interpretation's own hedge level —
    "appears to be the dominant contributor", not a stronger unhedged
    causal claim — so a PASS/WARN verdict isn't dependent on the model
    resolving a genuinely ambiguous hedging judgment call."""
    state = _base_diagnostic_state()
    state["report"] = {
        "executive_summary": (
            "Revenue decreased from 161445.80 to 150633.02 (-6.7%). By segment, Enterprise appears to be "
            "the dominant contributor, accounting for approximately 74.4% of the total change."
        ),
        "key_findings": ["June revenue: 161445.80", "July revenue: 150633.02", "Enterprise change: -10610.83"],
        "evidence": [], "recommendations": [], "confidence": "Medium", "limitations": "",
    }
    return state


def _fabricated_root_cause_state() -> dict:
    """Same evidence, but the executive_summary now asserts a specific root
    cause (customer churn, a competitor's pricing) that appears NOWHERE in
    facts/interpretations — the entity name ("Enterprise") still matches
    the dominant contributor, so the deterministic check_causal_claims
    check (entity-name matching only) can't catch this; only the semantic
    check can."""
    state = _base_diagnostic_state()
    state["report"] = {
        "executive_summary": (
            "Revenue decreased from 161445.80 to 150633.02 (-6.7%) because a large number of Enterprise "
            "customers churned in response to a competitor's aggressive pricing change."
        ),
        "key_findings": ["June revenue: 161445.80", "July revenue: 150633.02"],
        "evidence": [], "recommendations": [], "confidence": "Medium", "limitations": "",
    }
    return state


@pytest.mark.asyncio
async def test_semantic_check_supports_a_well_grounded_appropriately_hedged_report_via_real_llm():
    state = _grounded_hedged_state()
    result = await critic_node(state)  # llm=None -> real get_llm_client(); never raises for an LLM failure

    category = result["critic_feedback"]["semantic_check_error_category"]
    if category in _INFRASTRUCTURE_CATEGORIES:
        pytest.skip(f"LLM provider unavailable ({category}), not a code failure.")

    assert result["critic_feedback"]["status"] in ("PASS", "WARN"), (
        f"real model verdict: {result['critic_feedback']['status']}, "
        f"findings: {result['critic_feedback']['findings']}"
    )


@pytest.mark.asyncio
async def test_semantic_check_catches_a_fabricated_root_cause_via_real_llm():
    state = _fabricated_root_cause_state()
    result = await critic_node(state)

    category = result["critic_feedback"]["semantic_check_error_category"]
    if category in _INFRASTRUCTURE_CATEGORIES:
        pytest.skip(f"LLM provider unavailable ({category}), not a code failure.")

    # Asserting the OVERALL Critic verdict (not just the semantic check's
    # own opinion in isolation) is what makes this a genuine end-to-end
    # contract test — a real miss here is a real gap in the one check this
    # LLM call exists for, not a wording nitpick, and must remain a FAIL.
    assert result["critic_feedback"]["status"] == "FAIL", (
        "expected the Critic to catch a fabricated root cause (churn/competitor pricing — "
        f"present nowhere in the evidence); got {result['critic_feedback']['status']}, "
        f"findings: {result['critic_feedback']['findings']}"
    )
