"""The 0.30 "LLM-judge" half of Sec 6's weighting (relevance, recommendation
quality) — the one part of evaluation that genuinely needs an LLM rather
than arithmetic. Deliberately isolated here and NOT called by
app/evaluation/evaluator.py::run_benchmark by default: deterministic
evaluation (the other 0.70, and everything Phase 8 added on top of it —
SQL/analysis/visualization/critic correctness, grounding, hallucination
detection) must be runnable with zero LLM calls and zero quota risk. A
caller that explicitly wants the judge scores calls these directly and
handles the same RateLimitError-means-skip pattern as every other live-LLM
path in this codebase (see app/agents/critic.py, tests/agents/test_critic_live_llm.py).

Provider-agnostic: takes `llm: LLMClientProtocol`, the same abstraction every
production agent uses — never imports `anthropic`/`groq` directly, so
swapping LLM_PROVIDER doesn't touch this file.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.core.llm import LLMClientProtocol, ModelTier


class RelevanceJudgment(BaseModel):
    """Does the report actually answer the question that was asked."""

    relevance: int = Field(ge=1, le=5, description="1 (off-topic/non-answer) to 5 (directly and fully answers the question).")
    reasoning: str = Field(description="One or two sentences justifying the score.")


class RecommendationQualityJudgment(BaseModel):
    """Are the report's recommendations (if any) specific and actionable,
    grounded in the findings, rather than generic advice."""

    quality: int = Field(ge=1, le=5, description="1 (generic/unhelpful or absent when warranted) to 5 (specific, actionable, grounded in the findings).")
    reasoning: str = Field(description="One or two sentences justifying the score.")


_RELEVANCE_SYSTEM = (
    "You are grading whether a business intelligence report answers the question it was asked. "
    "Score strictly on relevance to the QUESTION, not on writing quality or factual accuracy "
    "(those are checked separately). Respond only via the given tool."
)

_RECOMMENDATION_SYSTEM = (
    "You are grading the quality of a business intelligence report's recommendations: are they "
    "specific, actionable, and clearly grounded in the report's own findings — not generic advice "
    "that could apply to any report. If the report has no recommendations and the question didn't "
    "call for any, score 3 (neutral, not a defect). Respond only via the given tool."
)


async def judge_relevance(question: str, report: dict, llm: LLMClientProtocol) -> RelevanceJudgment:
    return await llm.complete_structured(
        tier=ModelTier.STRONG,
        system=_RELEVANCE_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Executive summary: {report.get('executive_summary', '')}\n"
                    f"Key findings: {report.get('key_findings', [])}"
                ),
            }
        ],
        response_model=RelevanceJudgment,
    )


async def judge_recommendation_quality(report: dict, llm: LLMClientProtocol) -> RecommendationQualityJudgment:
    return await llm.complete_structured(
        tier=ModelTier.STRONG,
        system=_RECOMMENDATION_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Executive summary: {report.get('executive_summary', '')}\n"
                    f"Recommendations: {report.get('recommendations', [])}"
                ),
            }
        ],
        response_model=RecommendationQualityJudgment,
    )
