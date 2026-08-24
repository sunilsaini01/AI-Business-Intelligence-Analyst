"""Phase 4/5/6/7/9 default graph, end to end at the graph level (not through
the API — see tests/api/test_analysis.py for that), with a ScriptedLLMClient
standing in for Claude so this runs with no network/API key and no
flakiness. Requires a seeded, migrated DB (real SQL still executes — only
the LLM calls are faked). analysis_agent, visualization_agent, and the
Critic's deterministic checks are all real (no LLM), only the Critic's one
semantic check is faked alongside the Supervisor/SQL Agent calls.
"""

from __future__ import annotations

import pytest

from app.agents.schemas import CriticSemanticCheck, SQLGeneration, SupervisorPlan, SupervisorSynthesis
from app.graph.state import new_state
from app.graph.workflow import build_graph
from tests.fakes import ScriptedLLMClient


@pytest.mark.asyncio
async def test_supervisor_sql_agent_round_trip_reaches_report():
    fake_llm = ScriptedLLMClient(
        {
            SupervisorPlan: [
                SupervisorPlan(
                    out_of_scope=False,
                    intent="descriptive",
                    target_schema="analytics",
                    steps=["Count total customers"],
                    reasoning="Simple count query answers this directly.",
                )
            ],
            SQLGeneration: [
                SQLGeneration(
                    sql="SELECT COUNT(*) AS n FROM analytics.customers",
                    purpose="Total customer count",
                )
            ],
            SupervisorSynthesis: [
                SupervisorSynthesis(
                    insufficient_evidence=False,
                    executive_summary="There are customers in the database.",
                    key_findings=["Customer count evidence retrieved."],
                    confidence="High",
                    limitations="",
                )
            ],
        }
    )

    graph = build_graph(llm=fake_llm)
    result = await graph.ainvoke(new_state("How many customers do we have?"))

    assert result["intent"] == "descriptive"
    assert result["target_schema"] == "analytics"
    assert len(result["sql_queries"]) == 1
    assert result["sql_queries"][0]["validated_ok"] is True
    assert result["sql_queries"][0]["rows"][0]["n"] >= 0
    assert result["report"] is not None
    assert result["report"]["confidence"] == "High"
    assert result["analysis_results"] is not None  # Phase 6: analysis_agent always writes something
    # Phase 7: a single-value result (COUNT(*) AS n, one row) -> distribution
    # with count==1 -> a KPI card, not left empty.
    assert len(result["charts"]) == 1
    assert result["charts"][0]["chart_type"] == "kpi"
    # Phase 9: a clean, un-embellished report with no diagnostic composite ->
    # no LLM semantic call needed at all (nothing to compare against), the
    # deterministic checks alone clear it -> PASS, no retry.
    assert result["critic_feedback"]["status"] == "PASS"
    assert result["retry_count"] == 0

    node_names = [t["node"] for t in result["trace"]]
    assert node_names == [
        "supervisor", "supervisor",
        "sql_agent", "sql_agent",
        "analysis_agent", "analysis_agent",
        "visualization_agent", "visualization_agent",
        "supervisor", "supervisor",
        "critic", "critic",
        "report_agent", "report_agent",
    ]
    # Phase 10: the Report Generator ran and added its presentation-only
    # fields without touching anything the Supervisor/Critic already set.
    assert result["report"]["executive_summary"] == "There are customers in the database."
    assert result["report"]["verified_claims"] == ["Customer count evidence retrieved."]
    assert result["report"]["visualizations"] == [{"chart_type": "kpi", "title": result["charts"][0]["title"], "subtitle": result["charts"][0].get("subtitle")}]
    assert result["report"]["technical_details"]["critic_status"] == "PASS"


@pytest.mark.asyncio
async def test_diagnostic_question_runs_analysis_agent_and_reaches_supervisor():
    """Phase 6 acceptance shape at the graph level: a diagnostic plan with
    two steps (overall period comparison + segment-by-month breakdown)
    should produce a real diagnostic composite that reaches the synthesis
    prompt. Uses real SQL against the real seeded benchmark data (the fixed
    July 2026 Enterprise/North dip — see scripts/generate_data.py) rather
    than UNION-of-literals: a UNION query's root node is sqlglot's `Union`,
    not `Select`, so the real validator correctly rejects it — that's not a
    bug, it's the validator doing its job, so this test uses real SQL
    instead of fighting it.
    """
    fake_llm = ScriptedLLMClient(
        {
            SupervisorPlan: [
                SupervisorPlan(
                    out_of_scope=False,
                    intent="diagnostic",
                    target_schema="analytics",
                    steps=["Get June and July revenue", "Break down revenue by segment for June and July"],
                    reasoning="Diagnostic question needs a comparison plus a contribution breakdown.",
                )
            ],
            SQLGeneration: [
                SQLGeneration(
                    sql=(
                        "SELECT (date_trunc('month', o.order_date)::date)::text AS month, "
                        "SUM(oi.quantity * oi.unit_price * (1 - oi.discount)) AS revenue "
                        "FROM analytics.orders o JOIN analytics.order_items oi ON oi.order_id = o.order_id "
                        "WHERE o.order_date >= '2026-06-01' AND o.order_date < '2026-08-01' "
                        "GROUP BY 1 ORDER BY 1"
                    ),
                    purpose="Overall June vs July 2026 revenue",
                ),
                SQLGeneration(
                    sql=(
                        "SELECT c.segment AS segment, (date_trunc('month', o.order_date)::date)::text AS month, "
                        "SUM(oi.quantity * oi.unit_price * (1 - oi.discount)) AS revenue "
                        "FROM analytics.orders o "
                        "JOIN analytics.order_items oi ON oi.order_id = o.order_id "
                        "JOIN analytics.customers c ON c.customer_id = o.customer_id "
                        "WHERE o.order_date >= '2026-06-01' AND o.order_date < '2026-08-01' "
                        "GROUP BY c.segment, 2 ORDER BY 1, 2"
                    ),
                    purpose="Revenue by segment for June and July 2026",
                ),
            ],
            SupervisorSynthesis: [
                SupervisorSynthesis(
                    insufficient_evidence=False,
                    executive_summary="Revenue declined from June to July 2026, driven by Enterprise.",
                    key_findings=["Total revenue declined June to July 2026", "Enterprise drove most of the decline"],
                    confidence="Medium",
                    limitations="",
                )
            ],
            CriticSemanticCheck: [
                CriticSemanticCheck(supported=True, unsupported_claims=[], reasoning="Grounded in the facts/interpretations.")
            ],
        }
    )

    graph = build_graph(llm=fake_llm)
    result = await graph.ainvoke(new_state("Why did revenue decrease in July?"))

    assert result["intent"] == "diagnostic"
    assert result["report"] is not None

    analysis = result["analysis_results"]
    assert analysis["insufficient_evidence"] is False
    assert len(analysis["period_comparisons"]) == 1
    assert analysis["period_comparisons"][0]["direction"] == "decrease"
    diagnostic = analysis["diagnostic"]
    assert diagnostic is not None
    assert diagnostic["facts"]
    # Real seeded benchmark data (see scripts/generate_data.py's fixed
    # BENCHMARK_DIP_* constants): Enterprise is the dominant contributor to
    # the June->July decline — should be called out as an interpretation.
    assert any("Enterprise" in i for i in diagnostic["interpretations"])

    # Phase 7: diagnostic path -> overall comparison + only the dominant
    # contribution breakdown(s), never one chart per analysis_results entry
    # ("avoid visualization spam" — Sec 7 spec).
    charts = result["charts"]
    assert 1 <= len(charts) <= 2
    assert charts[0]["chart_type"] == "bar"  # the June-vs-July overall comparison, first
    if len(charts) == 2:
        assert charts[1]["chart_type"] == "horizontal_bar"  # segment contribution to the decline
        assert charts[1]["source_analysis"] == "contribution"

    # The synthesis call the fake LLM actually received must contain the
    # deterministic analysis, not just the raw evidence — proves the wiring
    # from analysis_agent -> supervisor's synthesis prompt actually works.
    synthesis_call = next(c for c in fake_llm.calls if c["response_model"] is SupervisorSynthesis)
    assert "FACT" in synthesis_call["system"]
    assert "Enterprise" in synthesis_call["system"]

    # Phase 9: a well-grounded, correctly-attributed report (the causal claim
    # names the real dominant contributor, every number traces to real
    # evidence) clears the Critic on the first attempt — no retry needed.
    assert result["critic_feedback"]["status"] == "PASS"
    assert result["retry_count"] == 0
    assert result["trace"][-1]["node"] == "report_agent"
    assert result["trace"][-1]["event"] == "exit"
    # Phase 10: analysis_explanation is built deterministically from the
    # SAME diagnostic facts/interpretations just asserted above — verbatim,
    # not re-derived.
    assert result["report"]["analysis_explanation"]
    assert "Enterprise" in result["report"]["analysis_explanation"]
    assert len(result["report"]["visualizations"]) == len(charts)


@pytest.mark.asyncio
async def test_critic_fail_triggers_retry_and_second_attempt_passes():
    """Phase 9 acceptance shape: the Critic rejects a first synthesis that
    invents a number (real evidence is real customer rows, not "999999"),
    which clears `report` and routes back to the Supervisor for revision —
    the second, honest synthesis then clears the Critic. Proves the retry
    loop both fires AND terminates (not just that PASS works on the first
    try, which the tests above already cover). sql_agent/analysis_agent/
    visualization_agent do NOT re-run on retry (same evidence, only the
    Supervisor's wording needed fixing) — SQLGeneration is only queued once.
    """
    fake_llm = ScriptedLLMClient(
        {
            SupervisorPlan: [
                SupervisorPlan(
                    out_of_scope=False,
                    intent="descriptive",
                    target_schema="analytics",
                    steps=["Count total customers"],
                    reasoning="Simple count query answers this directly.",
                )
            ],
            SQLGeneration: [
                SQLGeneration(sql="SELECT COUNT(*) AS n FROM analytics.customers", purpose="Total customer count")
            ],
            SupervisorSynthesis: [
                SupervisorSynthesis(
                    insufficient_evidence=False,
                    executive_summary="There are 999999 customers in the database.",
                    key_findings=["Customer count: 999999"],
                    confidence="High",
                    limitations="",
                ),
                SupervisorSynthesis(
                    insufficient_evidence=False,
                    executive_summary="Customer count evidence was retrieved successfully.",
                    key_findings=["Customer count evidence retrieved."],
                    confidence="Medium",
                    limitations="",
                ),
            ],
        }
    )

    graph = build_graph(llm=fake_llm)
    result = await graph.ainvoke(new_state("How many customers do we have?"))

    assert result["retry_count"] == 1  # exactly one retry consumed, not more
    assert result["critic_feedback"]["status"] == "PASS"  # the SECOND attempt's verdict
    assert result["report"]["executive_summary"] == "Customer count evidence was retrieved successfully."

    # Two full Supervisor synthesis attempts, one Critic review each — proves
    # the loop actually looped, not just that PASS-on-first-try works.
    synthesis_calls = [c for c in fake_llm.calls if c["response_model"] is SupervisorSynthesis]
    assert len(synthesis_calls) == 2
    node_names = [t["node"] for t in result["trace"]]
    assert node_names.count("critic") == 4  # 2 critic_node calls x (enter+exit)


@pytest.mark.asyncio
async def test_out_of_scope_question_short_circuits_without_sql():
    fake_llm = ScriptedLLMClient(
        {
            SupervisorPlan: [
                SupervisorPlan(
                    out_of_scope=True,
                    intent="out_of_scope",
                    target_schema="analytics",
                    steps=["n/a"],
                    reasoning="This is a greeting, not a data question.",
                )
            ],
        }
    )

    graph = build_graph(llm=fake_llm)
    result = await graph.ainvoke(new_state("hello, how are you?"))

    assert result["intent"] == "out_of_scope"
    assert result["sql_queries"] == []
    assert result["report"] is not None
    assert result["report"]["confidence"] == "Low"
