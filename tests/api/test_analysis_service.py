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

from app.agents.schemas import SQLGeneration, SupervisorPlan, SupervisorSynthesis
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
        "supervisor", "sql_agent", "analysis_agent", "visualization_agent", "critic", "report_agent",
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
    assert trace_a == {"supervisor", "sql_agent", "analysis_agent", "visualization_agent", "critic", "report_agent"}
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
            "supervisor", "sql_agent", "analysis_agent", "visualization_agent", "critic", "report_agent",
        }

        metadata = await _get_execution_metadata(analysis_id)
        assert metadata["final_status"] == "DONE"
        assert metadata["error_category"] is None


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
