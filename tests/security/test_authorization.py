"""Phase 14, Issue 6 — the analysis ownership boundary. Every one of these
was reachable by anyone with a UUID before this phase; these tests are the
actual regression guard for that fix. Deterministic: analyses are created
via `analysis_service` directly with a scripted/fake graph where the
distinction matters, or left PENDING where it doesn't (ownership is checked
before status).
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from app.agents.schemas import SupervisorPlan
from app.core.auth import decode_access_token
from app.graph.workflow import build_graph
from app.services import analysis_service
from tests.fakes import ScriptedLLMClient


def _out_of_scope_llm() -> ScriptedLLMClient:
    return ScriptedLLMClient(
        {
            SupervisorPlan: [
                SupervisorPlan(
                    out_of_scope=True, intent="out_of_scope", target_schema="analytics",
                    steps=[], reasoning="not a data question",
                )
            ]
        }
    )


def _current_user_id(authed_client) -> uuid.UUID:
    """No `/auth/me` endpoint exists (kept out of scope — not needed by
    anything) so the owning user's id is decoded straight off the client's
    own bearer token — no extra HTTP round trip, and critically no extra
    session created through POST /analyze, which would schedule its own
    real-graph background run (see the module docstring's note on why
    these tests use analysis_service.create_session directly rather than
    the route whenever they need a SPECIFIC, scripted graph outcome)."""
    token = authed_client.headers["Authorization"].removeprefix("Bearer ")
    return decode_access_token(token)


# --- 401: no token at all ---------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,path_suffix",
    [("POST", None), ("GET", "/status"), ("GET", "/report"), ("GET", "/charts"), ("GET", "")],
)
async def test_unauthenticated_request_is_401(unauthenticated_client, method, path_suffix):
    if path_suffix is None:
        resp = await unauthenticated_client.post("/api/v1/analyze", json={"question": "x"})
    else:
        resp = await unauthenticated_client.get(f"/api/v1/analysis/{uuid.uuid4()}{path_suffix}")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_unauthenticated_reports_listing_is_401(unauthenticated_client):
    resp = await unauthenticated_client.get("/api/v1/reports")
    assert resp.status_code == 401


# --- 404: unknown id, even when authenticated -------------------------------


@pytest.mark.asyncio
async def test_unknown_analysis_id_is_404_not_403_even_when_authenticated(client):
    resp = await client.get(f"/api/v1/analysis/{uuid.uuid4()}/status")
    assert resp.status_code == 404


# --- owner access: 200 -------------------------------------------------------


@pytest.mark.asyncio
async def test_owner_can_read_their_own_analysis(client):
    resp = await client.post("/api/v1/analyze", json={"question": "How many customers?"})
    analysis_id = resp.json()["analysis_id"]
    status_resp = await client.get(f"/api/v1/analysis/{analysis_id}/status")
    assert status_resp.status_code == 200
    detail_resp = await client.get(f"/api/v1/analysis/{analysis_id}")
    assert detail_resp.status_code == 200
    charts_resp = await client.get(f"/api/v1/analysis/{analysis_id}/charts")
    assert charts_resp.status_code == 200


# --- wrong owner: 403, on every endpoint ------------------------------------


@pytest.mark.asyncio
async def test_a_different_authenticated_user_gets_403_on_status(client, second_user_client):
    resp = await client.post("/api/v1/analyze", json={"question": "How many customers?"})
    analysis_id = resp.json()["analysis_id"]
    other = await second_user_client.get(f"/api/v1/analysis/{analysis_id}/status")
    assert other.status_code == 403


@pytest.mark.asyncio
async def test_a_different_authenticated_user_gets_403_on_detail(client, second_user_client):
    resp = await client.post("/api/v1/analyze", json={"question": "How many customers?"})
    analysis_id = resp.json()["analysis_id"]
    other = await second_user_client.get(f"/api/v1/analysis/{analysis_id}")
    assert other.status_code == 403


@pytest.mark.asyncio
async def test_a_different_authenticated_user_gets_403_on_charts(client, second_user_client):
    resp = await client.post("/api/v1/analyze", json={"question": "How many customers?"})
    analysis_id = resp.json()["analysis_id"]
    other = await second_user_client.get(f"/api/v1/analysis/{analysis_id}/charts")
    assert other.status_code == 403


@pytest.mark.asyncio
async def test_a_different_authenticated_user_gets_403_on_report(client, second_user_client):
    analysis_id = await analysis_service.create_session("q", user_id=_current_user_id(client))
    await analysis_service.run_analysis(analysis_id, "q", graph=build_graph(llm=_out_of_scope_llm()))
    other = await second_user_client.get(f"/api/v1/analysis/{analysis_id}/report")
    assert other.status_code == 403


@pytest.mark.asyncio
async def test_403_response_reveals_nothing_about_the_real_owner(client, second_user_client):
    resp = await client.post("/api/v1/analyze", json={"question": "How many customers?"})
    analysis_id = resp.json()["analysis_id"]
    other = await second_user_client.get(f"/api/v1/analysis/{analysis_id}/status")
    assert other.status_code == 403
    body_text = str(other.json()).lower()
    for leak_marker in ("@example.com", "user_id", "owner"):
        assert leak_marker not in body_text


# --- /reports listing is filtered to the caller's own rows ------------------


@pytest.mark.asyncio
async def test_reports_listing_never_includes_another_users_report(client, second_user_client):
    """Sessions are created via analysis_service.create_session directly,
    not via POST /analyze — that route schedules its own real-graph
    background run (app/api/routes/analysis.py's BackgroundTasks), which
    Starlette's test transport runs to completion as part of awaiting the
    response; calling run_analysis a second time on that same id (to get a
    deterministic, scripted outcome instead of a real LLM call) would race
    it and double-insert the report row. Every test in this file that needs
    a specific scripted outcome uses this same direct-creation pattern —
    see tests/api/test_analysis_service.py for the identical convention."""
    mine_id = await analysis_service.create_session("mine", user_id=_current_user_id(client))
    await analysis_service.run_analysis(mine_id, "mine", graph=build_graph(llm=_out_of_scope_llm()))

    theirs_id = await analysis_service.create_session("theirs", user_id=_current_user_id(second_user_client))
    await analysis_service.run_analysis(theirs_id, "theirs", graph=build_graph(llm=_out_of_scope_llm()))

    my_listing = await client.get("/api/v1/reports")
    assert my_listing.status_code == 200
    my_ids = {r["analysis_id"] for r in my_listing.json()}
    assert str(mine_id) in my_ids
    assert str(theirs_id) not in my_ids


# --- a legacy (ownerless) row stays readable by any authenticated user ------


@pytest.mark.asyncio
async def test_a_row_with_no_recorded_owner_is_readable_by_any_authenticated_user(client, second_user_client):
    """Deliberate, documented allowance (app/api/routes/analysis.py::
    _check_ownership) for data that predates Phase 14 — analysis_service.
    create_session's user_id defaults to None precisely so this case is
    representable. Never happens via the real POST /analyze route anymore
    (it always passes the caller's id), only via a direct service call, as
    every pre-Phase-14 test in tests/api/test_analysis_service.py still
    does."""
    analysis_id = await analysis_service.create_session("ownerless")
    resp = await client.get(f"/api/v1/analysis/{analysis_id}/status")
    assert resp.status_code == 200
    resp2 = await second_user_client.get(f"/api/v1/analysis/{analysis_id}/status")
    assert resp2.status_code == 200


# --- multi-user concurrency isolation ---------------------------------------


@pytest.mark.asyncio
async def test_two_different_users_concurrent_analyses_stay_isolated_and_unreadable_by_each_other(
    client, second_user_client
):
    id_a = await analysis_service.create_session("Question A", user_id=_current_user_id(client))
    id_b = await analysis_service.create_session("Question B", user_id=_current_user_id(second_user_client))

    await asyncio.gather(
        analysis_service.run_analysis(id_a, "Question A", graph=build_graph(llm=_out_of_scope_llm())),
        analysis_service.run_analysis(id_b, "Question B", graph=build_graph(llm=_out_of_scope_llm())),
    )

    assert (await client.get(f"/api/v1/analysis/{id_a}/status")).status_code == 200
    assert (await client.get(f"/api/v1/analysis/{id_b}/status")).status_code == 403
    assert (await second_user_client.get(f"/api/v1/analysis/{id_b}/status")).status_code == 200
    assert (await second_user_client.get(f"/api/v1/analysis/{id_a}/status")).status_code == 403
