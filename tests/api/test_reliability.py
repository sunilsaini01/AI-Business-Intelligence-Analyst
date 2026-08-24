"""Phase 13, Section 8 — API reliability tests. Deterministic: every LLM
call is scripted or a controlled fake; no live network call anywhere in
this file.
"""

from __future__ import annotations

import uuid

import groq
import httpx
import pytest

from app.agents.schemas import SupervisorPlan
from app.db.models import SessionStatus
from app.graph.workflow import build_graph
from app.services import analysis_service
from tests.fakes import ScriptedLLMClient


class _TimeoutLLM:
    async def complete(self, **kwargs):
        raise RuntimeError("should not be called")

    async def complete_structured(self, **kwargs):
        req = httpx.Request("POST", "http://test")
        raise groq.APITimeoutError(req)


class _RateLimitLLM:
    async def complete(self, **kwargs):
        raise RuntimeError("should not be called")

    async def complete_structured(self, **kwargs):
        resp = httpx.Response(429, request=httpx.Request("POST", "http://test"))
        raise groq.RateLimitError("rate limited", response=resp, body=None)


@pytest.mark.asyncio
async def test_1_unknown_analysis_id_status_is_404(client):
    resp = await client.get(f"/api/v1/analysis/{uuid.uuid4()}/status")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_2_report_before_completion_is_409(client):
    analysis_id = await analysis_service.create_session("still pending")
    resp = await client.get(f"/api/v1/analysis/{analysis_id}/report")
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_3_charts_for_unknown_id_is_404(client):
    resp = await client.get(f"/api/v1/analysis/{uuid.uuid4()}/charts")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_4_invalid_question_type_is_422(client):
    resp = await client.post("/api/v1/analyze", json={"question": 12345})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_4b_missing_question_field_is_422(client):
    resp = await client.post("/api/v1/analyze", json={})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_5_blank_question_is_422(client):
    resp = await client.post("/api/v1/analyze", json={"question": "   "})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_6_oversized_question_is_422(client):
    resp = await client.post("/api/v1/analyze", json={"question": "x" * 2001})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_7_provider_rate_limit_is_correctly_classified():
    analysis_id = await analysis_service.create_session("q")
    await analysis_service.run_analysis(analysis_id, "q", graph=build_graph(llm=_RateLimitLLM()))
    session = await analysis_service.get_session(analysis_id)
    assert session.status == SessionStatus.FAILED
    assert session.execution_metadata["error_category"] == "rate_limit"
    assert "rate-limited or out of quota" in session.error_message


@pytest.mark.asyncio
async def test_8_provider_timeout_is_correctly_classified():
    analysis_id = await analysis_service.create_session("q")
    await analysis_service.run_analysis(analysis_id, "q", graph=build_graph(llm=_TimeoutLLM()))
    session = await analysis_service.get_session(analysis_id)
    assert session.status == SessionStatus.FAILED
    assert session.execution_metadata["error_category"] == "timeout"
    assert "did not respond in time" in session.error_message
    # Never conflated with rate_limit — a different provider condition, a
    # different (still safe) message.
    assert "rate-limited" not in session.error_message


@pytest.mark.asyncio
async def test_9_unexpected_application_error_is_a_generic_internal_error():
    class _BrokenLLM:
        async def complete(self, **kwargs):
            raise RuntimeError("should not be called")

        async def complete_structured(self, **kwargs):
            raise RuntimeError("a genuine unexpected application bug")

    analysis_id = await analysis_service.create_session("q")
    await analysis_service.run_analysis(analysis_id, "q", graph=build_graph(llm=_BrokenLLM()))
    session = await analysis_service.get_session(analysis_id)
    assert session.status == SessionStatus.FAILED
    assert session.execution_metadata["error_category"] == "application_error"
    assert session.error_message == "Analysis failed. Please try again."
    assert "a genuine unexpected application bug" not in session.error_message  # never the raw exception text


@pytest.mark.asyncio
async def test_10_partial_trace_survives_failure():
    class _FailsAfterPlan:
        async def complete(self, **kwargs):
            raise RuntimeError("should not be called")

        async def complete_structured(self, **kwargs):
            if kwargs.get("response_model") is SupervisorPlan:
                return SupervisorPlan(
                    out_of_scope=False, intent="descriptive", target_schema="analytics",
                    steps=["Count total customers"], reasoning="x",
                )
            raise RuntimeError("simulated SQL Agent outage")

    analysis_id = await analysis_service.create_session("q")
    await analysis_service.run_analysis(analysis_id, "q", graph=build_graph(llm=_FailsAfterPlan()))
    session = await analysis_service.get_session(analysis_id)
    assert session.status == SessionStatus.FAILED
    trace = await analysis_service.get_trace(analysis_id)
    assert [t.agent_name for t in trace] == ["supervisor", "supervisor"]  # not empty — survived the later failure


@pytest.mark.asyncio
async def test_11_failed_analysis_response_exposes_no_secrets(client):
    class _LeakyLLM:
        async def complete(self, **kwargs):
            raise RuntimeError("should not be called")

        async def complete_structured(self, **kwargs):
            raise RuntimeError("leaking postgresql://bi_app:supersecret@postgres:5432/bi_agent, sk-ANTHROPIC-FAKE-KEY")

    analysis_id = await analysis_service.create_session("q")
    await analysis_service.run_analysis(analysis_id, "q", graph=build_graph(llm=_LeakyLLM()))

    status_resp = await client.get(f"/api/v1/analysis/{analysis_id}/status")
    detail_resp = await client.get(f"/api/v1/analysis/{analysis_id}")
    combined = str(status_resp.json()) + str(detail_resp.json())
    for marker in ("postgresql://", "supersecret", "sk-anthropic-fake-key", "Traceback"):
        assert marker.lower() not in combined.lower()


@pytest.mark.asyncio
async def test_12a_two_concurrent_requests_get_unique_analysis_ids(client):
    resp_a = await client.post("/api/v1/analyze", json={"question": "Question A"})
    resp_b = await client.post("/api/v1/analyze", json={"question": "Question B"})
    assert resp_a.json()["analysis_id"] != resp_b.json()["analysis_id"]


@pytest.mark.asyncio
async def test_execution_metadata_is_exposed_via_the_detail_endpoint(client):
    analysis_id = await analysis_service.create_session("q")
    await analysis_service.run_analysis(
        analysis_id, "q",
        graph=build_graph(
            llm=ScriptedLLMClient(
                {
                    SupervisorPlan: [
                        SupervisorPlan(
                            out_of_scope=True, intent="out_of_scope", target_schema="analytics",
                            steps=[], reasoning="not a data question",
                        )
                    ]
                }
            )
        ),
    )
    detail_resp = await client.get(f"/api/v1/analysis/{analysis_id}")
    assert detail_resp.status_code == 200
    body = detail_resp.json()
    assert body["execution_metadata"]["final_status"] == "DONE"
    assert body["execution_metadata"]["completed_nodes"] == ["supervisor"]


@pytest.mark.asyncio
async def test_execution_metadata_is_empty_dict_while_still_pending(client):
    analysis_id = await analysis_service.create_session("still pending")
    detail_resp = await client.get(f"/api/v1/analysis/{analysis_id}")
    assert detail_resp.json()["execution_metadata"] == {}
