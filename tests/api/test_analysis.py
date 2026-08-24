"""Deterministic API-layer tests only — status codes, validation, routing.
None of these require the analysis background task to actually *complete*
(that needs the real Supervisor/SQL Agent, which needs ANTHROPIC_API_KEY) —
see tests/api/test_analyze_live_llm.py for the one test that does.
"""

import pytest


@pytest.mark.asyncio
async def test_report_409_before_done(client):
    resp = await client.post("/api/v1/analyze", json={"question": "customers by region"})
    analysis_id = resp.json()["analysis_id"]

    report_resp = await client.get(f"/api/v1/analysis/{analysis_id}/report")
    assert report_resp.status_code in (409, 200)  # 200 only if the background task raced ahead


@pytest.mark.asyncio
async def test_blank_question_rejected(client):
    resp = await client.post("/api/v1/analyze", json={"question": "   "})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_unknown_analysis_id_404s(client):
    resp = await client.get("/api/v1/analysis/00000000-0000-0000-0000-000000000000/status")
    assert resp.status_code == 404
