"""Deterministic tests for app/core/fake_llm.py (Phase 13, Objective B) —
the offline stand-in used ONLY by the browser E2E test. No network call
anywhere in this file; that's the entire point of this module.
"""

from __future__ import annotations

import pytest

from app.agents.schemas import (
    CriticSemanticCheck,
    ReportNarrative,
    SQLGeneration,
    SupervisorPlan,
    SupervisorSynthesis,
)
from app.core.fake_llm import FakeLLMClient
from app.core.llm import ModelTier
from app.tools.database_tools import validate_sql


@pytest.mark.asyncio
async def test_recognized_question_produces_an_in_scope_plan():
    client = FakeLLMClient()
    plan = await client.complete_structured(
        tier=ModelTier.STRONG,
        system="plan this",
        messages=[{"role": "user", "content": "How many customers do we have per region?"}],
        response_model=SupervisorPlan,
    )
    assert plan.out_of_scope is False
    assert plan.target_schema == "analytics"
    assert plan.steps


@pytest.mark.asyncio
async def test_plan_matching_ignores_the_system_prompt_only_the_user_question_counts():
    """Regression test: the E2E browser test caught this for real —
    "What's the weather like today?" was being answered as a real data
    question because the SYSTEM prompt (schema description) mentions
    "customers"/"regions" for every single question. Only `messages` (the
    actual user question) may drive keyword matching."""
    client = FakeLLMClient()
    plan = await client.complete_structured(
        tier=ModelTier.STRONG,
        system=(
            "You are the Supervisor... Available schemas: analytics.customers, "
            "analytics.regions, olist.customers ..."
        ),
        messages=[{"role": "user", "content": "What's the weather like today?"}],
        response_model=SupervisorPlan,
    )
    assert plan.out_of_scope is True


@pytest.mark.asyncio
async def test_unrecognized_question_produces_an_out_of_scope_plan_not_a_guess():
    client = FakeLLMClient()
    plan = await client.complete_structured(
        tier=ModelTier.STRONG,
        system="plan this",
        messages=[{"role": "user", "content": "What's the weather like today?"}],
        response_model=SupervisorPlan,
    )
    assert plan.out_of_scope is True
    assert plan.steps == []


@pytest.mark.asyncio
async def test_sql_generation_is_schema_qualified_and_passes_the_real_validator():
    """The fake client's canned SQL must be genuinely valid against the
    real, unweakened SQL safety pipeline (Sec 4) — an E2E test that only
    "succeeds" because the safety layer was bypassed would be worthless."""
    client = FakeLLMClient()
    sql_gen = await client.complete_structured(
        tier=ModelTier.FAST,
        system="write sql",
        messages=[{"role": "user", "content": "count customers by region"}],
        response_model=SQLGeneration,
    )
    result = await validate_sql(sql_gen.sql)
    assert result.ok, result.rejection_reason


@pytest.mark.asyncio
async def test_synthesis_cites_no_numbers_so_it_can_never_be_flagged_as_ungrounded():
    client = FakeLLMClient()
    synthesis = await client.complete_structured(
        tier=ModelTier.STRONG,
        system="synthesize",
        messages=[{"role": "user", "content": "synthesize now"}],
        response_model=SupervisorSynthesis,
    )
    assert synthesis.insufficient_evidence is False
    combined_text = synthesis.executive_summary + " ".join(synthesis.key_findings)
    assert not any(char.isdigit() for char in combined_text)


@pytest.mark.asyncio
async def test_critic_semantic_check_and_narrative_have_canned_responses_too():
    client = FakeLLMClient()
    check = await client.complete_structured(
        tier=ModelTier.STRONG, system="s", messages=[{"role": "user", "content": "x"}],
        response_model=CriticSemanticCheck,
    )
    assert check.supported is True

    narrative = await client.complete_structured(
        tier=ModelTier.STRONG, system="s", messages=[{"role": "user", "content": "x"}],
        response_model=ReportNarrative,
    )
    assert narrative.narrative


@pytest.mark.asyncio
async def test_unknown_response_model_raises_rather_than_guessing():
    from pydantic import BaseModel

    class _SomeOtherModel(BaseModel):
        x: int

    client = FakeLLMClient()
    with pytest.raises(ValueError, match="no canned response"):
        await client.complete_structured(
            tier=ModelTier.STRONG, system="s", messages=[{"role": "user", "content": "x"}],
            response_model=_SomeOtherModel,
        )


@pytest.mark.asyncio
async def test_complete_returns_a_fixed_non_empty_string():
    client = FakeLLMClient()
    text = await client.complete(tier=ModelTier.FAST, system="s", messages=[{"role": "user", "content": "x"}])
    assert isinstance(text, str) and text
