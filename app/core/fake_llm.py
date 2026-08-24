"""Deterministic, offline `LLMClientProtocol` implementation (Phase 13,
Objective B). Used ONLY when `LLM_PROVIDER=fake` is explicitly set — never
the default (`Settings.llm_provider` stays `"anthropic"`, unchanged; see
app/core/config.py). Exists so the browser E2E test (tests/e2e/) can drive
the REAL running backend through a REAL HTTP/browser flow without
depending on live Groq/Anthropic quota, per the explicit instruction not to
make browser E2E testing dependent on real provider quota.

This is NOT a language model — it recognizes exactly one canned question
(keyword-matched) and returns fixed, schema-valid, deliberately
NUMBER-FREE synthesis text (see `_FAKE_SYNTHESIS` below) so nothing it
returns can ever be flagged as an unsupported numeric claim by the Critic's
real, unweakened grounding checks (app/tools/critic_checks.py) — the E2E
run exercises the REAL Supervisor -> SQL Agent -> Analysis Agent ->
Visualization Agent -> Critic -> Report Generator pipeline against the
REAL seeded database, only the two LLM-authored text fields (plan
reasoning, synthesis wording) are canned, everything numeric in the final
report still comes from real, computed evidence.

Any question this fake doesn't recognize gets a safe `out_of_scope`
response rather than a guess — this must never silently produce a
plausible-looking but ungrounded answer.
"""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

_KNOWN_QUESTION_KEYWORDS = ("customer", "region")

_FAKE_SQL = (
    "SELECT r.name AS region, COUNT(*) AS customer_count "
    "FROM analytics.customers c JOIN analytics.regions r ON r.region_id = c.region_id "
    "GROUP BY r.name ORDER BY r.name"
)

# Deliberately cites no specific numbers — see module docstring. Real
# figures reach the user via the Analysis Agent's deterministic
# `analysis_explanation` (Phase 10) and the actual chart data, both
# computed from real query results, not from this fake client.
_FAKE_EXECUTIVE_SUMMARY = "Customer counts differ across regions, as shown in the accompanying breakdown."
_FAKE_KEY_FINDINGS = [
    "Regional customer distribution was retrieved successfully.",
    "See the visualization for the full breakdown by region.",
]


class FakeLLMClient:
    """Matches `LLMClientProtocol` structurally (duck-typed, like every
    other implementation in app/core/llm.py) — no shared base class."""

    def __init__(self) -> None:
        self.total_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

    async def complete(self, *, tier: Any, system: str, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        return "This is a deterministic response from the fake LLM provider (LLM_PROVIDER=fake)."

    async def complete_structured(
        self,
        *,
        tier: Any,
        system: str,
        messages: list[dict[str, Any]],
        response_model: type[T],
        **kwargs: Any,
    ) -> T:
        # Only the actual message content (the real user question, per
        # app/agents/supervisor.py::_plan's `messages=[{"role": "user",
        # "content": state["question"]}]`) is used for keyword matching —
        # NOT `system`, which embeds the full database schema description
        # and therefore already contains the words "customers"/"regions"
        # for EVERY question regardless of what was actually asked. Matching
        # against `system` too was a real bug caught by the E2E browser
        # test itself: "What's the weather today?" was answered as if it
        # were the customers-by-region question, because the schema text in
        # the system prompt matched the keywords even though the user's
        # actual question didn't.
        question_text = " ".join(m.get("content", "") for m in messages)
        name = response_model.__name__

        if name == "SupervisorPlan":
            return self._plan(response_model, question_text)
        if name == "SQLGeneration":
            return response_model(sql=_FAKE_SQL, purpose="Count customers by region")
        if name == "SupervisorSynthesis":
            return response_model(
                insufficient_evidence=False,
                executive_summary=_FAKE_EXECUTIVE_SUMMARY,
                key_findings=_FAKE_KEY_FINDINGS,
                confidence="Medium",
                limitations="",
            )
        if name == "CriticSemanticCheck":
            return response_model(supported=True, unsupported_claims=[], reasoning="Fake provider: no claims to review.")
        if name == "ReportNarrative":
            return response_model(narrative=_FAKE_EXECUTIVE_SUMMARY)

        raise ValueError(f"FakeLLMClient has no canned response for {name} — extend it or don't use LLM_PROVIDER=fake here.")

    def _plan(self, response_model: type[T], question_text: str) -> T:
        recognized = any(keyword in question_text.lower() for keyword in _KNOWN_QUESTION_KEYWORDS)
        if not recognized:
            return response_model(
                out_of_scope=True,
                intent="out_of_scope",
                target_schema="analytics",
                steps=[],
                reasoning="Fake provider: question not in the small set of canned scenarios it recognizes.",
            )
        return response_model(
            out_of_scope=False,
            intent="descriptive",
            target_schema="analytics",
            steps=["Count customers by region"],
            reasoning="Fake provider deterministic plan for the customers-by-region scenario.",
        )
