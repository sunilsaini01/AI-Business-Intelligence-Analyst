"""Phase 5: SQL Agent unit tests with a ScriptedLLMClient — real SQL still
executes against the seeded DB through the actual safety pipeline (Sec 4);
only the "write the SQL" step is faked, exactly as it would be for a real
LLM, so the retry-on-rejection path is genuinely exercised.
"""

from __future__ import annotations

import pytest

from app.agents.schemas import SQLGeneration
from app.agents.sql_agent import sql_agent_node
from app.graph.state import new_state
from tests.fakes import ScriptedLLMClient


@pytest.mark.asyncio
async def test_generates_and_executes_a_valid_query():
    fake = ScriptedLLMClient(
        {
            SQLGeneration: [
                SQLGeneration(
                    sql="SELECT segment, COUNT(*) AS n FROM analytics.customers GROUP BY segment",
                    purpose="Customer count by segment",
                )
            ]
        }
    )
    state = new_state("How many customers per segment?")
    state["plan"] = ["Count customers by segment"]
    state["target_schema"] = "analytics"

    result = await sql_agent_node(state, llm=fake)

    assert len(result["sql_queries"]) == 1
    record = result["sql_queries"][0]
    assert record["validated_ok"] is True
    assert record["row_count"] > 0


@pytest.mark.asyncio
async def test_retries_once_after_rejected_sql_and_succeeds():
    fake = ScriptedLLMClient(
        {
            SQLGeneration: [
                SQLGeneration(sql="SELECT * FROM customers", purpose="unqualified, will be rejected"),
                SQLGeneration(sql="SELECT COUNT(*) AS n FROM analytics.customers", purpose="fixed"),
            ]
        }
    )
    state = new_state("How many customers total?")
    state["plan"] = ["Count all customers"]
    state["target_schema"] = "analytics"

    result = await sql_agent_node(state, llm=fake)

    assert len(fake.calls) == 2  # confirms the retry actually happened
    record = result["sql_queries"][0]
    assert record["validated_ok"] is True


@pytest.mark.asyncio
async def test_records_rejection_without_crashing_when_retry_also_fails():
    fake = ScriptedLLMClient(
        {
            SQLGeneration: [
                SQLGeneration(sql="SELECT * FROM customers", purpose="bad"),
                SQLGeneration(sql="DROP TABLE analytics.customers", purpose="still bad"),
            ]
        }
    )
    state = new_state("How many customers total?")
    state["plan"] = ["Count all customers"]
    state["target_schema"] = "analytics"

    result = await sql_agent_node(state, llm=fake)

    record = result["sql_queries"][0]
    assert record["validated_ok"] is False
    assert record["rejection_reason"] is not None
    assert record["rows"] == []


@pytest.mark.asyncio
async def test_caps_queries_at_max_steps():
    fake = ScriptedLLMClient(
        {
            SQLGeneration: [
                SQLGeneration(sql="SELECT COUNT(*) AS n FROM analytics.customers", purpose=f"step {i}")
                for i in range(4)
            ]
        }
    )
    state = new_state("Many-step question")
    state["plan"] = [f"step {i}" for i in range(10)]  # more than _MAX_STEPS
    state["target_schema"] = "analytics"

    result = await sql_agent_node(state, llm=fake)

    assert len(result["sql_queries"]) == 4
