"""Phase 11 tests for the orchestration layer (app/services/analysis_service.py)
and its FastAPI routes — session lifecycle, incremental trace persistence,
current_stage, concurrency, and cross-session isolation. Uses
`build_graph(llm=ScriptedLLMClient(...))` injected directly into
`run_analysis(..., graph=...)` (a DI seam added for Phase 11) so all of this
is deterministic: no live LLM call, no ASGI needed for the service-level
tests, and the `client` fixture is used only where the actual HTTP/route
contract (status codes, response schema) is what's under test.
"""

from __future__ import annotations

import asyncio
import uuid

import groq
import httpx
import pytest
from sqlalchemy import select

from app.agents.schemas import CriticSemanticCheck, ReportNarrative, SQLGeneration, SupervisorPlan, SupervisorSynthesis
from app.core.config import get_settings
from app.db.database import async_session_factory
from app.db.models import EvaluationRun, SessionStatus
from app.graph.workflow import build_graph
from app.services import analysis_service
from tests.fakes import ScriptedLLMClient


def _happy_path_llm() -> ScriptedLLMClient:
    return ScriptedLLMClient(
        {
            SupervisorPlan: [
                SupervisorPlan(
                    out_of_scope=False, intent="descriptive", target_schema="analytics",
                    steps=["Count total customers"], reasoning="Simple count query.",
                )
            ],
            SQLGeneration: [SQLGeneration(sql="SELECT COUNT(*) AS n FROM analytics.customers", purpose="Total customer count")],
            SupervisorSynthesis: [
                SupervisorSynthesis(
                    insufficient_evidence=False, executive_summary="There are customers.",
                    key_findings=["Customer count retrieved."], confidence="High", limitations="",
                )
            ],
        }
    )


def _out_of_scope_llm() -> ScriptedLLMClient:
    return ScriptedLLMClient(
        {
            SupervisorPlan: [
                SupervisorPlan(
                    out_of_scope=True, intent="out_of_scope", target_schema="analytics",
                    steps=[], reasoning="Not a data question.",
                )
            ],
        }
    )


class _RaisingLLM:
    async def complete(self, **kwargs):
        raise RuntimeError("simulated LLM outage")

    async def complete_structured(self, **kwargs):
        raise RuntimeError("simulated LLM outage")


class _FailsAfterPlanLLM:
    """Succeeds on the Supervisor's planning call, then raises on every
    subsequent call — used to prove a mid-pipeline failure still leaves the
    already-completed earlier steps persisted (Phase 11's incremental
    persistence, not just an all-or-nothing batch write)."""

    async def complete(self, **kwargs):
        raise RuntimeError("simulated outage")

    async def complete_structured(self, **kwargs):
        if kwargs.get("response_model") is SupervisorPlan:
            return SupervisorPlan(
                out_of_scope=False, intent="descriptive", target_schema="analytics",
                steps=["Count total customers"], reasoning="x",
            )
        raise RuntimeError("simulated SQL Agent outage")


# --- session lifecycle ---------------------------------------------------


@pytest.mark.asyncio
async def test_create_session_starts_pending():
    analysis_id = await analysis_service.create_session("How many customers?")
    session = await analysis_service.get_session(analysis_id)
    assert session is not None
    assert session.status == SessionStatus.PENDING
    assert session.error_message is None


@pytest.mark.asyncio
async def test_run_analysis_transitions_to_done_on_success():
    analysis_id = await analysis_service.create_session("How many customers?")
    await analysis_service.run_analysis(analysis_id, "How many customers?", graph=build_graph(llm=_happy_path_llm()))
    session = await analysis_service.get_session(analysis_id)
    assert session.status == SessionStatus.DONE
    assert session.error_message is None


@pytest.mark.asyncio
async def test_run_analysis_transitions_to_failed_not_left_pending_forever():
    analysis_id = await analysis_service.create_session("How many customers?")
    await analysis_service.run_analysis(analysis_id, "How many customers?", graph=build_graph(llm=_RaisingLLM()))
    session = await analysis_service.get_session(analysis_id)
    assert session.status == SessionStatus.FAILED
    assert session.status != SessionStatus.PENDING
    # generic message only — never the real exception text/stack trace (Sec 7)
    assert session.error_message == "Analysis failed. Please try again."


@pytest.mark.asyncio
async def test_run_analysis_missing_session_is_a_safe_noop():
    """A session_id that doesn't exist (e.g. deleted between create and the
    background task starting) must not crash the background task."""
    await analysis_service.run_analysis(uuid.uuid4(), "anything", graph=build_graph(llm=_happy_path_llm()))


# --- incremental trace persistence + current_stage (Phase 11) ------------


@pytest.mark.asyncio
async def test_run_analysis_persists_full_trace_through_report_agent():
    analysis_id = await analysis_service.create_session("How many customers?")
    await analysis_service.run_analysis(analysis_id, "How many customers?", graph=build_graph(llm=_happy_path_llm()))
    trace = await analysis_service.get_trace(analysis_id)
    assert [t.agent_name for t in trace] == [
        "supervisor", "supervisor",
        "sql_agent", "sql_agent",
        "analysis_agent", "analysis_agent",
        "ml_agent", "ml_agent",  # Phase 15: always in the chain, no-op here (intent != "predictive")
        "visualization_agent", "visualization_agent",
        "supervisor", "supervisor",
        "critic", "critic",
        "report_agent", "report_agent",
    ]


@pytest.mark.asyncio
async def test_current_stage_is_none_before_anything_ran_and_report_agent_after():
    analysis_id = await analysis_service.create_session("How many customers?")
    assert await analysis_service.get_current_stage(analysis_id) is None
    await analysis_service.run_analysis(analysis_id, "How many customers?", graph=build_graph(llm=_happy_path_llm()))
    assert await analysis_service.get_current_stage(analysis_id) == "report_agent"


@pytest.mark.asyncio
async def test_failed_run_still_persists_the_partial_trace_it_reached():
    """The Phase 11 improvement over the old batch-at-the-end write: a node
    that raises mid-pipeline no longer wipes out the earlier, genuinely
    completed nodes' trace."""
    analysis_id = await analysis_service.create_session("How many customers?")
    await analysis_service.run_analysis(analysis_id, "How many customers?", graph=build_graph(llm=_FailsAfterPlanLLM()))
    session = await analysis_service.get_session(analysis_id)
    assert session.status == SessionStatus.FAILED
    trace = await analysis_service.get_trace(analysis_id)
    assert [t.agent_name for t in trace] == ["supervisor", "supervisor"]


@pytest.mark.asyncio
async def test_out_of_scope_run_persists_only_supervisor_and_completes_done():
    analysis_id = await analysis_service.create_session("hello, how are you?")
    await analysis_service.run_analysis(analysis_id, "hello, how are you?", graph=build_graph(llm=_out_of_scope_llm()))
    session = await analysis_service.get_session(analysis_id)
    assert session.status == SessionStatus.DONE
    trace = await analysis_service.get_trace(analysis_id)
    assert [t.agent_name for t in trace] == ["supervisor", "supervisor"]  # critic/report_agent skipped by design


# --- execution_metadata / error classification (Phase 13, Objectives A & D) --


async def _get_execution_metadata(analysis_id) -> dict:
    session = await analysis_service.get_session(analysis_id)
    return session.execution_metadata


@pytest.mark.asyncio
async def test_execution_metadata_populated_on_success():
    analysis_id = await analysis_service.create_session("How many customers?")
    await analysis_service.run_analysis(analysis_id, "How many customers?", graph=build_graph(llm=_happy_path_llm()))
    metadata = await _get_execution_metadata(analysis_id)

    assert metadata["final_status"] == "DONE"
    assert metadata["current_stage"] == "report_agent"
    assert metadata["completed_nodes"] == [
        "supervisor", "sql_agent", "analysis_agent", "ml_agent", "visualization_agent", "critic", "report_agent",
    ]
    assert metadata["failed_node"] is None
    assert metadata["error_category"] is None
    assert metadata["retry_count"] == 0
    assert metadata["report_generated"] is True
    assert metadata["narrative_enabled"] is False  # REPORT_NARRATIVE_ENABLED default
    assert metadata["total_duration_ms"] >= 0
    assert metadata["start_time"] < metadata["end_time"]


@pytest.mark.asyncio
async def test_execution_metadata_classifies_a_generic_exception_as_application_error():
    analysis_id = await analysis_service.create_session("How many customers?")
    await analysis_service.run_analysis(analysis_id, "How many customers?", graph=build_graph(llm=_RaisingLLM()))
    metadata = await _get_execution_metadata(analysis_id)

    assert metadata["final_status"] == "FAILED"
    assert metadata["error_category"] == "application_error"
    assert metadata["report_generated"] is False
    assert metadata["completed_nodes"] == []  # RuntimeError on the very first (planning) call


@pytest.mark.asyncio
async def test_execution_metadata_classifies_a_rate_limit_error_distinctly():
    class _RateLimitedLLM:
        async def complete(self, **kwargs):
            raise RuntimeError("should not be called")

        async def complete_structured(self, **kwargs):
            resp = httpx.Response(429, request=httpx.Request("POST", "http://test"))
            raise groq.RateLimitError("rate limited", response=resp, body=None)

    analysis_id = await analysis_service.create_session("How many customers?")
    await analysis_service.run_analysis(analysis_id, "How many customers?", graph=build_graph(llm=_RateLimitedLLM()))

    session = await analysis_service.get_session(analysis_id)
    assert session.status == SessionStatus.FAILED
    assert "rate-limited or out of quota" in session.error_message

    metadata = await _get_execution_metadata(analysis_id)
    assert metadata["error_category"] == "rate_limit"
    assert metadata["final_status"] == "FAILED"


@pytest.mark.asyncio
async def test_execution_metadata_infers_the_failed_node_from_completed_nodes():
    """_FailsAfterPlanLLM succeeds on the Supervisor's plan, then fails on
    the SQL Agent's call — completed_nodes should show only "supervisor",
    and failed_node should be inferred as the very next canonical stage."""
    analysis_id = await analysis_service.create_session("How many customers?")
    await analysis_service.run_analysis(analysis_id, "How many customers?", graph=build_graph(llm=_FailsAfterPlanLLM()))
    metadata = await _get_execution_metadata(analysis_id)

    assert metadata["completed_nodes"] == ["supervisor"]
    assert metadata["failed_node"] == "sql_agent"
    assert metadata["error_category"] == "application_error"


# --- Phase 14, Issue 5: execution_metadata reflects ACTUAL behavior, ------
# --- not just key presence -------------------------------------------------


@pytest.mark.asyncio
async def test_execution_metadata_reflects_a_critic_fail_exhausted_degradation():
    """A FAIL that survives all retries still completes the run (DONE, not
    FAILED — app/agents/critic.py::_force_degrade downgrades confidence and
    discloses the issue, it doesn't abort). execution_metadata must show
    the real retry_count actually consumed (== critic_max_retries) and
    report_generated=True, since the degraded report genuinely is
    persisted — never left showing a success shape for what was actually a
    contested run.

    Forces the FAIL via a deterministic check (an executive_summary citing
    a number nowhere in the evidence -> check_numerical_grounding, app/tools/
    critic_checks.py) rather than the semantic LLM check — the semantic
    check only runs when analysis_results has diagnostic facts/
    interpretations (app/agents/critic.py::_semantic_check), which a plain
    COUNT query never produces, so relying on it here would silently no-op
    instead of failing."""
    settings = get_settings()
    forced_fail = ScriptedLLMClient(
        {
            SupervisorPlan: [
                SupervisorPlan(
                    out_of_scope=False, intent="descriptive", target_schema="analytics",
                    steps=["Count total customers"], reasoning="x",
                )
            ],
            SQLGeneration: [SQLGeneration(sql="SELECT COUNT(*) AS n FROM analytics.customers", purpose="count")],
            # One synthesis per attempt: 1 initial + one per retry
            # (critic_max_retries) = 1 + settings.critic_max_retries. Every
            # attempt repeats the same fabricated, ungrounded figure, so
            # every attempt FAILs the same deterministic check — this is
            # what proves retries genuinely exhaust rather than happening
            # to pass on a later attempt.
            SupervisorSynthesis: [
                SupervisorSynthesis(
                    insufficient_evidence=False,
                    executive_summary="There are 4210000 customers, up 87.3%.",
                    key_findings=["Customer count retrieved."], confidence="Medium", limitations="",
                )
                for _ in range(1 + settings.critic_max_retries)
            ],
        }
    )

    analysis_id = await analysis_service.create_session("How many customers?")
    await analysis_service.run_analysis(analysis_id, "How many customers?", graph=build_graph(llm=forced_fail))

    session = await analysis_service.get_session(analysis_id)
    assert session.status == SessionStatus.DONE  # degraded, not FAILED

    metadata = await _get_execution_metadata(analysis_id)
    assert metadata["final_status"] == "DONE"
    assert metadata["retry_count"] == settings.critic_max_retries
    assert metadata["report_generated"] is True
    assert "critic" in metadata["completed_nodes"]
    assert "report_agent" in metadata["completed_nodes"]

    report = await analysis_service.get_report(analysis_id)
    assert report.confidence == "Low"  # _force_degrade's actual effect, not assumed


@pytest.mark.asyncio
async def test_execution_metadata_when_narrative_is_enabled_end_to_end(monkeypatch):
    """The default-off path (narrative_enabled=False) is already covered by
    test_execution_metadata_populated_on_success. This exercises the
    opposite configuration through the real graph/settings wiring — not
    the narrative_enabled= parameter override tests already cover at the
    unit level (tests/agents/test_report_agent.py) — to prove
    execution_metadata's narrative_enabled flag reflects the SETTING that
    was actually in effect for this run, not a hard-coded default."""
    monkeypatch.setenv("REPORT_NARRATIVE_ENABLED", "true")
    get_settings.cache_clear()
    try:
        llm = ScriptedLLMClient(
            {
                SupervisorPlan: [
                    SupervisorPlan(
                        out_of_scope=False, intent="descriptive", target_schema="analytics",
                        steps=["Count total customers"], reasoning="x",
                    )
                ],
                SQLGeneration: [SQLGeneration(sql="SELECT COUNT(*) AS n FROM analytics.customers", purpose="count")],
                SupervisorSynthesis: [
                    SupervisorSynthesis(
                        insufficient_evidence=False, executive_summary="There are some customers.",
                        key_findings=["Customer count retrieved."], confidence="High", limitations="",
                    )
                ],
                CriticSemanticCheck: [CriticSemanticCheck(supported=True, unsupported_claims=[], reasoning="ok")],
                ReportNarrative: [ReportNarrative(narrative="There are some customers, per the data.")],
            }
        )
        analysis_id = await analysis_service.create_session("How many customers?")
        await analysis_service.run_analysis(analysis_id, "How many customers?", graph=build_graph(llm=llm))

        metadata = await _get_execution_metadata(analysis_id)
        assert metadata["final_status"] == "DONE"
        assert metadata["narrative_enabled"] is True
        assert metadata["report_generated"] is True

        report = await analysis_service.get_report(analysis_id)
        # Grounded verbatim in the source text -> passes report_agent's own
        # numeric-grounding re-check -> kept, not degraded to None.
        assert report.report_extras["narrative"] == "There are some customers, per the data."
    finally:
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_execution_metadata_infers_analysis_agent_as_the_failed_node(monkeypatch):
    """failed_node must name the ACTUAL failing node, not just "whichever
    node happens to be after the SQL Agent" — this forces the failure one
    stage later than the existing SQL-Agent-failure test
    (test_execution_metadata_infers_the_failed_node_from_completed_nodes)
    to prove the inference generalizes, by making the Analysis Agent's own
    (non-LLM) column-classification step raise."""
    import app.agents.analysis_agent as analysis_agent_module

    def _boom(rows):
        raise RuntimeError("simulated Analysis Agent bug")

    monkeypatch.setattr(analysis_agent_module, "classify_columns", _boom)

    llm = ScriptedLLMClient(
        {
            SupervisorPlan: [
                SupervisorPlan(
                    out_of_scope=False, intent="descriptive", target_schema="analytics",
                    steps=["Count total customers"], reasoning="x",
                )
            ],
            SQLGeneration: [SQLGeneration(sql="SELECT COUNT(*) AS n FROM analytics.customers", purpose="count")],
        }
    )
    analysis_id = await analysis_service.create_session("How many customers?")
    await analysis_service.run_analysis(analysis_id, "How many customers?", graph=build_graph(llm=llm))

    session = await analysis_service.get_session(analysis_id)
    assert session.status == SessionStatus.FAILED

    metadata = await _get_execution_metadata(analysis_id)
    assert metadata["completed_nodes"] == ["supervisor", "sql_agent"]
    assert metadata["failed_node"] == "analysis_agent"
    assert metadata["error_category"] == "application_error"


@pytest.mark.asyncio
async def test_execution_metadata_never_contains_secrets():
    analysis_id = await analysis_service.create_session("How many customers?")
    await analysis_service.run_analysis(analysis_id, "How many customers?", graph=build_graph(llm=_happy_path_llm()))
    metadata = await _get_execution_metadata(analysis_id)

    metadata_text = str(metadata).lower()
    for marker in ("postgresql://", "api_key", "password", "authorization", "bearer "):
        assert marker not in metadata_text


@pytest.mark.asyncio
async def test_execution_metadata_survives_a_missing_session_gracefully():
    """_persist_execution_metadata must not raise if the session row is
    somehow gone by the time it runs — mirrors run_analysis's own
    missing-session guard."""
    await analysis_service._persist_execution_metadata(uuid.uuid4(), {"final_status": "DONE"})


# --- report/charts/detail endpoints reflect what was actually persisted --


@pytest.mark.asyncio
async def test_report_endpoint_returns_phase10_fields_after_completion(client):
    analysis_id = await analysis_service.create_session("How many customers?")
    await analysis_service.run_analysis(analysis_id, "How many customers?", graph=build_graph(llm=_happy_path_llm()))

    status_resp = await client.get(f"/api/v1/analysis/{analysis_id}/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "DONE"
    assert status_resp.json()["current_stage"] == "report_agent"

    report_resp = await client.get(f"/api/v1/analysis/{analysis_id}/report")
    assert report_resp.status_code == 200
    body = report_resp.json()
    assert body["executive_summary"] == "There are customers."
    assert body["verified_claims"] == ["Customer count retrieved."]
    assert body["analysis_explanation"]
    assert body["visualizations"]
    assert body["technical_details"]["critic_status"] == "PASS"
    assert body["narrative"] is None  # REPORT_NARRATIVE_ENABLED defaults to False

    charts_resp = await client.get(f"/api/v1/analysis/{analysis_id}/charts")
    assert charts_resp.status_code == 200
    assert len(charts_resp.json()) >= 1  # persisted metadata, not regenerated here

    detail_resp = await client.get(f"/api/v1/analysis/{analysis_id}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["current_stage"] == "report_agent"
    assert detail["trace"][-1]["node"] == "report_agent"
    # no secrets in the trace — only node/event/timestamp/duration ever go in
    assert all(set(event.keys()) == {"node", "event", "timestamp", "duration_ms"} for event in detail["trace"])
    trace_text = str(detail["trace"]).lower()
    for secret_marker in ("api_key", "password", "secret", "postgresql://"):
        assert secret_marker not in trace_text


def _predictive_forecast_llm() -> ScriptedLLMClient:
    return ScriptedLLMClient(
        {
            SupervisorPlan: [
                SupervisorPlan(
                    out_of_scope=False, intent="predictive", target_schema="analytics",
                    steps=["Forecast next month's revenue"], reasoning="Needs the ML Agent's trend model.",
                )
            ],
            SQLGeneration: [SQLGeneration(sql="SELECT COUNT(*) AS n FROM analytics.customers", purpose="context")],
            SupervisorSynthesis: [
                SupervisorSynthesis(
                    insufficient_evidence=False, executive_summary="Revenue is projected to continue its trend.",
                    key_findings=["The forecast model projects next month's revenue."],
                    confidence="Medium", limitations="",
                )
            ],
        }
    )


@pytest.mark.asyncio
async def test_predictive_analysis_exposes_full_ml_results_via_the_api(client):
    """Phase 15, Objective 4 — end-to-end proof that the ML Agent's real,
    computed result (not a fabricated one — ml_agent never calls an LLM)
    reaches the API response, structured, not just as prose."""
    analysis_id = await analysis_service.create_session("How many customers?")
    await analysis_service.run_analysis(
        analysis_id, "Can you forecast revenue for next month?", graph=build_graph(llm=_predictive_forecast_llm())
    )

    report_resp = await client.get(f"/api/v1/analysis/{analysis_id}/report")
    assert report_resp.status_code == 200
    body = report_resp.json()

    assert body["ml_summary"] != ""
    ml_results = body["ml_results"]
    assert ml_results["ok"] is True
    assert ml_results["task"] == "forecasting"
    assert ml_results["model_name"] == "linear_trend_baseline"
    assert ml_results["train_size"] > 0
    assert ml_results["test_size"] > 0
    assert isinstance(ml_results["metrics"]["mae"], float)
    assert ml_results["forecast_next"]
    assert ml_results["confidence"] in ("Low", "Medium", "High")
    assert ml_results["limitations"]

    metadata = await _get_execution_metadata(analysis_id)
    assert metadata["final_status"] == "DONE"
    assert "ml_agent" in metadata["completed_nodes"]


@pytest.mark.asyncio
async def test_non_predictive_analysis_has_no_ml_results_via_the_api(client):
    analysis_id = await analysis_service.create_session("How many customers?")
    await analysis_service.run_analysis(analysis_id, "How many customers?", graph=build_graph(llm=_happy_path_llm()))

    report_resp = await client.get(f"/api/v1/analysis/{analysis_id}/report")
    body = report_resp.json()
    assert body["ml_summary"] == ""
    assert body["ml_results"] is None


def _predictive_churn_llm() -> ScriptedLLMClient:
    return ScriptedLLMClient(
        {
            SupervisorPlan: [
                SupervisorPlan(
                    out_of_scope=False, intent="predictive", target_schema="analytics",
                    steps=["Identify customers at risk of churn"], reasoning="Needs the ML Agent's churn model.",
                )
            ],
            SQLGeneration: [SQLGeneration(sql="SELECT COUNT(*) AS n FROM analytics.customers", purpose="context")],
            SupervisorSynthesis: [
                SupervisorSynthesis(
                    insufficient_evidence=False, executive_summary="Some customers are flagged as at risk.",
                    key_findings=["The churn model flagged a subset of customers as at risk."],
                    confidence="Medium", limitations="",
                )
            ],
        }
    )


@pytest.mark.asyncio
async def test_concurrent_forecast_and_churn_analyses_never_mix_ml_results():
    """Phase 16, Section 15 — simultaneous ML analyses (mixed forecast +
    churn, run via real asyncio.gather concurrency, no sleep-based
    synchronization, same convention as test_five_concurrent_analyses_
    remain_fully_isolated) must never cross-contaminate customer data,
    forecast/churn results, metrics, report_extras, or execution_metadata
    between sessions."""
    id_forecast_1 = await analysis_service.create_session("forecast 1")
    id_churn_1 = await analysis_service.create_session("churn 1")
    id_forecast_2 = await analysis_service.create_session("forecast 2")
    id_churn_2 = await analysis_service.create_session("churn 2")

    await asyncio.gather(
        analysis_service.run_analysis(
            id_forecast_1, "Forecast revenue for next month", graph=build_graph(llm=_predictive_forecast_llm())
        ),
        analysis_service.run_analysis(
            id_churn_1, "Identify customers at risk of churn", graph=build_graph(llm=_predictive_churn_llm())
        ),
        analysis_service.run_analysis(
            id_forecast_2, "Forecast revenue for next quarter", graph=build_graph(llm=_predictive_forecast_llm())
        ),
        analysis_service.run_analysis(
            id_churn_2, "Which customers are likely to cancel", graph=build_graph(llm=_predictive_churn_llm())
        ),
    )

    forecast_1 = await analysis_service.get_report(id_forecast_1)
    forecast_2 = await analysis_service.get_report(id_forecast_2)
    churn_1 = await analysis_service.get_report(id_churn_1)
    churn_2 = await analysis_service.get_report(id_churn_2)

    for report in (forecast_1, forecast_2, churn_1, churn_2):
        assert report is not None

    # Each session's ml_results genuinely reflects ITS OWN task — a forecast
    # session must never end up with a churn_risk result or vice versa.
    for report in (forecast_1, forecast_2):
        assert report.report_extras["ml_results"]["task"] == "forecasting"
        assert "forecast_next" in report.report_extras["ml_results"]
    for report in (churn_1, churn_2):
        assert report.report_extras["ml_results"]["task"] == "churn_risk"
        assert "roc_auc" in report.report_extras["ml_results"]["metrics"]

    # Same real DB, same task -> the SAME real metrics are expected and
    # correct for two forecast sessions (not a contamination signal) — the
    # actual isolation guarantee is that each session's execution_metadata/
    # report_extras is its own DB row, not shared/overwritten state.
    metadata_forecast_1 = await _get_execution_metadata(id_forecast_1)
    metadata_forecast_2 = await _get_execution_metadata(id_forecast_2)
    metadata_churn_1 = await _get_execution_metadata(id_churn_1)
    metadata_churn_2 = await _get_execution_metadata(id_churn_2)
    for metadata in (metadata_forecast_1, metadata_forecast_2, metadata_churn_1, metadata_churn_2):
        assert metadata["final_status"] == "DONE"
        assert "ml_agent" in metadata["completed_nodes"]

    # Sessions themselves stay distinct rows, correctly attributed.
    session_ids = {id_forecast_1, id_churn_1, id_forecast_2, id_churn_2}
    assert len(session_ids) == 4


@pytest.mark.asyncio
async def test_report_404_before_analysis_exists_and_409_before_done(client):
    unknown_id = uuid.uuid4()
    assert (await client.get(f"/api/v1/analysis/{unknown_id}/report")).status_code == 404
    assert (await client.get(f"/api/v1/analysis/{unknown_id}/status")).status_code == 404
    assert (await client.get(f"/api/v1/analysis/{unknown_id}/charts")).status_code == 404
    assert (await client.get(f"/api/v1/analysis/{unknown_id}")).status_code == 404

    analysis_id = await analysis_service.create_session("still pending")
    report_resp = await client.get(f"/api/v1/analysis/{analysis_id}/report")
    assert report_resp.status_code == 409


@pytest.mark.asyncio
async def test_oversized_question_rejected_with_422(client):
    resp = await client.post("/api/v1/analyze", json={"question": "x" * 2001})
    assert resp.status_code == 422


# --- cross-session isolation ----------------------------------------------


@pytest.mark.asyncio
async def test_two_sessions_data_is_isolated(client):
    id_a = await analysis_service.create_session("Question A")
    id_b = await analysis_service.create_session("Question B")
    await analysis_service.run_analysis(id_a, "Question A", graph=build_graph(llm=_happy_path_llm()))
    # session B never ran -> still PENDING, must not leak A's report/status
    status_b = await client.get(f"/api/v1/analysis/{id_b}/status")
    assert status_b.json()["status"] == "PENDING"
    assert status_b.json()["current_stage"] is None
    assert (await client.get(f"/api/v1/analysis/{id_b}/report")).status_code == 409

    report_a = await client.get(f"/api/v1/analysis/{id_a}/report")
    assert report_a.status_code == 200
    assert report_a.json()["analysis_id"] == str(id_a)


@pytest.mark.asyncio
async def test_question_containing_sql_is_stored_as_opaque_text_never_executed(client):
    """No endpoint accepts or executes raw SQL — a question that LOOKS like
    an injection attempt is just a question string, validated the same as
    any other (min/max length) and stored verbatim as TEXT. Only the
    LLM-planned, sqlglot-validated SQL Agent output (app/tools/database_tools.py,
    untouched by this phase) ever reaches Postgres, over the read-only
    `readonly_analyst` role — never anything derived directly from user input.
    """
    malicious_text = "'; DROP TABLE analytics.orders; --"
    resp = await client.post("/api/v1/analyze", json={"question": malicious_text})
    assert resp.status_code == 202
    analysis_id = uuid.UUID(resp.json()["analysis_id"])
    session = await analysis_service.get_session(analysis_id)
    assert session.question == malicious_text  # stored, never parsed/executed by the API layer


# --- Phase 8 isolation: /analyze must never trigger evaluation -----------


@pytest.mark.asyncio
async def test_run_analysis_never_creates_an_evaluation_run_row():
    """Phase 8 (evaluation) and Phase 11 (analysis API) share nothing at
    runtime — app/services/analysis_service.py never imports or calls
    app/services/evaluation_service.py / app/evaluation/evaluator.py. A
    normal user-facing analysis must never show up as a benchmark run."""
    async with async_session_factory() as db:
        before = len((await db.execute(select(EvaluationRun))).scalars().all())

    analysis_id = await analysis_service.create_session("How many customers?")
    await analysis_service.run_analysis(analysis_id, "How many customers?", graph=build_graph(llm=_happy_path_llm()))

    async with async_session_factory() as db:
        after = len((await db.execute(select(EvaluationRun))).scalars().all())
    assert after == before


# --- concurrency -----------------------------------------------------------


@pytest.mark.asyncio
async def test_two_concurrent_analyses_do_not_interfere():
    id_a = await analysis_service.create_session("Question A")
    id_b = await analysis_service.create_session("Question B")

    await asyncio.gather(
        analysis_service.run_analysis(id_a, "Question A", graph=build_graph(llm=_happy_path_llm())),
        analysis_service.run_analysis(id_b, "Question B", graph=build_graph(llm=_out_of_scope_llm())),
    )

    session_a = await analysis_service.get_session(id_a)
    session_b = await analysis_service.get_session(id_b)
    assert session_a.status == SessionStatus.DONE
    assert session_b.status == SessionStatus.DONE

    report_a = await analysis_service.get_report(id_a)
    report_b = await analysis_service.get_report(id_b)
    assert report_a.executive_summary == "There are customers."
    assert report_b.executive_summary == "I can't answer that from this dataset."

    trace_a = {t.agent_name for t in await analysis_service.get_trace(id_a)}
    trace_b = {t.agent_name for t in await analysis_service.get_trace(id_b)}
    assert trace_a == {
        "supervisor", "sql_agent", "analysis_agent", "ml_agent", "visualization_agent", "critic", "report_agent",
    }
    assert trace_b == {"supervisor"}


def _distinct_llm(label: str) -> ScriptedLLMClient:
    """Each of the 5 concurrent sessions gets an LLM that answers with its
    own unique, distinguishable executive_summary — so any cross-session
    mixup (session A getting session B's report) is immediately, precisely
    detectable, not just "a report showed up"."""
    return ScriptedLLMClient(
        {
            SupervisorPlan: [
                SupervisorPlan(
                    out_of_scope=False, intent="descriptive", target_schema="analytics",
                    steps=["Count total customers"], reasoning=f"plan for {label}",
                )
            ],
            SQLGeneration: [SQLGeneration(sql="SELECT COUNT(*) AS n FROM analytics.customers", purpose="count")],
            SupervisorSynthesis: [
                SupervisorSynthesis(
                    insufficient_evidence=False, executive_summary=f"Result for {label}.",
                    key_findings=[f"finding-{label}"], confidence="High", limitations="",
                )
            ],
        }
    )


@pytest.mark.asyncio
async def test_five_concurrent_analyses_remain_fully_isolated():
    """Phase 13, Section 9: at least 5 concurrent analyses. Synchronization
    is asyncio.gather (a real concurrency primitive — every run_analysis
    coroutine is scheduled together and interleaves on the event loop),
    never a sleep-based race; each session gets a distinguishable answer so
    any mixing is caught precisely, not just detected as "something's
    wrong"."""
    labels = ["A", "B", "C", "D", "E"]
    ids = [await analysis_service.create_session(f"Question {label}") for label in labels]
    assert len(set(ids)) == len(ids)  # unique analysis_ids, guaranteed by uuid4 but asserted anyway

    await asyncio.gather(
        *(
            analysis_service.run_analysis(analysis_id, f"Question {label}", graph=build_graph(llm=_distinct_llm(label)))
            for analysis_id, label in zip(ids, labels)
        )
    )

    for analysis_id, label in zip(ids, labels):
        session = await analysis_service.get_session(analysis_id)
        assert session.status == SessionStatus.DONE

        report = await analysis_service.get_report(analysis_id)
        assert report.executive_summary == f"Result for {label}."  # exactly its own, never another session's
        assert report.key_findings == [f"finding-{label}"]

        trace = await analysis_service.get_trace(analysis_id)
        assert {t.agent_name for t in trace} == {
            "supervisor", "sql_agent", "analysis_agent", "ml_agent", "visualization_agent", "critic", "report_agent",
        }

        metadata = await _get_execution_metadata(analysis_id)
        assert metadata["final_status"] == "DONE"
        assert metadata["error_category"] is None


@pytest.mark.asyncio
async def test_ten_concurrent_analyses_remain_fully_isolated():
    """Phase 14, Issue 9: extends the Phase 13 5-concurrent case to 10 —
    same asyncio.gather-based real concurrency, same per-session
    distinguishable answer so any cross-session mixing is caught precisely
    rather than merely detected as 'something's wrong'.

    Letters only, deliberately (matching the existing 5-concurrent test's
    A-E convention) — a digit-bearing label like "L3" gets its own "3"
    picked up by check_numerical_grounding as a fabricated, ungrounded
    number (a real false-positive discovered while writing this test, not
    a concurrency bug: "Result for L3." reads as citing the value 3.0)."""
    labels = [chr(ord("A") + i) for i in range(10)]
    ids = [await analysis_service.create_session(f"Question {label}") for label in labels]
    assert len(set(ids)) == len(ids)

    await asyncio.gather(
        *(
            analysis_service.run_analysis(analysis_id, f"Question {label}", graph=build_graph(llm=_distinct_llm(label)))
            for analysis_id, label in zip(ids, labels)
        )
    )

    for analysis_id, label in zip(ids, labels):
        session = await analysis_service.get_session(analysis_id)
        assert session.status == SessionStatus.DONE
        report = await analysis_service.get_report(analysis_id)
        assert report.executive_summary == f"Result for {label}."
        assert report.key_findings == [f"finding-{label}"]


@pytest.mark.asyncio
async def test_one_analysis_failing_does_not_affect_a_concurrent_successful_one():
    id_ok = await analysis_service.create_session("Question OK")
    id_bad = await analysis_service.create_session("Question BAD")

    await asyncio.gather(
        analysis_service.run_analysis(id_ok, "Question OK", graph=build_graph(llm=_happy_path_llm())),
        analysis_service.run_analysis(id_bad, "Question BAD", graph=build_graph(llm=_RaisingLLM())),
    )

    session_ok = await analysis_service.get_session(id_ok)
    session_bad = await analysis_service.get_session(id_bad)
    assert session_ok.status == SessionStatus.DONE
    assert session_bad.status == SessionStatus.FAILED
    assert session_bad.error_message is not None
    assert (await analysis_service.get_report(id_ok)) is not None
    assert (await analysis_service.get_report(id_bad)) is None
