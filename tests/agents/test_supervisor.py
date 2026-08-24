"""Phase 4: Supervisor unit tests with a ScriptedLLMClient. Still touches the
DB (schema introspection feeds the prompt regardless of outcome) — not a pure
unit test, consistent with the rest of this project's "unit" tests that
validate against the live allow-list.
"""

from __future__ import annotations

import pytest

from app.agents.schemas import SupervisorPlan, SupervisorSynthesis
from app.agents.supervisor import supervisor_node
from app.graph.state import new_state
from tests.fakes import ScriptedLLMClient


@pytest.mark.asyncio
async def test_plan_sets_intent_target_schema_and_required_tools():
    fake = ScriptedLLMClient(
        {
            SupervisorPlan: [
                SupervisorPlan(
                    out_of_scope=False,
                    intent="comparative",
                    target_schema="olist",
                    steps=["Compare SP vs RJ order counts"],
                    reasoning="Direct state comparison.",
                )
            ]
        }
    )
    state = new_state("Compare orders between Sao Paulo and Rio de Janeiro")
    result = await supervisor_node(state, llm=fake)

    assert result["intent"] == "comparative"
    assert result["target_schema"] == "olist"
    assert result["plan"] == ["Compare SP vs RJ order counts"]
    assert result["required_tools"] == ["sql_agent"]
    assert result["report"] is None  # not yet — SQL agent hasn't run


@pytest.mark.asyncio
async def test_out_of_scope_question_gets_direct_report_no_plan():
    fake = ScriptedLLMClient(
        {
            SupervisorPlan: [
                SupervisorPlan(
                    out_of_scope=True,
                    intent="out_of_scope",
                    target_schema="analytics",
                    steps=["n/a"],
                    reasoning="Not a data question.",
                )
            ]
        }
    )
    state = new_state("What's your favorite color?")
    result = await supervisor_node(state, llm=fake)

    assert result["report"] is not None
    assert result["report"]["confidence"] == "Low"
    assert result["plan"] == []


@pytest.mark.asyncio
async def test_synthesize_builds_report_from_gathered_evidence():
    fake = ScriptedLLMClient(
        {
            SupervisorSynthesis: [
                SupervisorSynthesis(
                    insufficient_evidence=False,
                    executive_summary="Revenue was flat.",
                    key_findings=["Revenue: 1000 in June, 1000 in July"],
                    confidence="Medium",
                    limitations="Single-query evidence only.",
                )
            ]
        }
    )
    state = new_state("Why did revenue change?")
    state["plan"] = ["Compare June and July revenue"]
    state["sql_queries"] = [
        {
            "text": "SELECT 1000 AS revenue",
            "validated_ok": True,
            "rejection_reason": None,
            "rows": [{"revenue": 1000}],
            "row_count": 1,
            "exec_ms": 1.0,
        }
    ]

    result = await supervisor_node(state, llm=fake)

    assert result["report"]["executive_summary"] == "Revenue was flat."
    assert result["report"]["confidence"] == "Medium"
    assert result["report"]["evidence"] == [{"query": "SELECT 1000 AS revenue", "row_count": 1}]


@pytest.mark.asyncio
async def test_synthesize_forces_low_confidence_when_evidence_insufficient():
    fake = ScriptedLLMClient(
        {
            SupervisorSynthesis: [
                SupervisorSynthesis(
                    insufficient_evidence=True,
                    executive_summary="Insufficient evidence to determine the cause.",
                    key_findings=[],
                    confidence="High",  # model's own confidence claim should be overridden
                    limitations="Query only returned aggregate totals, not a breakdown.",
                )
            ]
        }
    )
    state = new_state("Why did revenue decrease in July?")
    state["plan"] = ["Compare June and July revenue"]
    state["sql_queries"] = []

    result = await supervisor_node(state, llm=fake)

    assert result["report"]["confidence"] == "Low"
    assert "Insufficient evidence" in result["report"]["executive_summary"]
