"""The one API test that exercises the real Supervisor -> SQL Agent -> report
round trip through a live LLM call (whichever provider LLM_PROVIDER selects —
see app/core/llm.py). Skipped automatically without a configured key for the
active provider, so the rest of the suite never depends on an external API
(project rule — see README/docs/architecture.md).

Run explicitly once a key is configured:
    docker compose exec api pytest tests/api/test_analyze_live_llm.py -v
"""

from __future__ import annotations

import asyncio

import pytest

from app.core.config import get_settings

_settings = get_settings()
_active_key = _settings.groq_api_key if _settings.llm_provider == "groq" else _settings.anthropic_api_key

pytestmark = pytest.mark.skipif(
    not _active_key,
    reason=f"No API key configured for LLM_PROVIDER={_settings.llm_provider} — live-LLM test skipped, not failed.",
)


@pytest.mark.asyncio
async def test_analyze_reaches_report_via_real_llm(client):
    resp = await client.post("/api/v1/analyze", json={"question": "How many customers do we have per region?"})
    assert resp.status_code == 202
    analysis_id = resp.json()["analysis_id"]

    status = "PENDING"
    for _ in range(60):
        status_resp = await client.get(f"/api/v1/analysis/{analysis_id}/status")
        status = status_resp.json()["status"]
        if status in ("DONE", "FAILED"):
            break
        await asyncio.sleep(1)

    if status == "FAILED":
        # Phase 11: app/services/analysis_service.py now distinguishes a
        # provider RateLimitError from a genuine code failure via
        # error_message (still SessionStatus.FAILED — no new enum value)
        # specifically so this test can tell "external quota" apart from
        # "a real bug" instead of asserting DONE and failing either way.
        error_message = status_resp.json().get("error_message") or ""
        if "rate-limited or out of quota" in error_message:
            pytest.skip(f"LLM provider rate-limited/quota exhausted, not a code failure: {error_message}")
        pytest.fail(f"analysis failed for a non-quota reason: {error_message}")

    assert status == "DONE", f"analysis did not complete, last status={status}"

    report_resp = await client.get(f"/api/v1/analysis/{analysis_id}/report")
    assert report_resp.status_code == 200
    report = report_resp.json()
    assert report["confidence"] in ("Low", "Medium", "High")
    assert report["executive_summary"]
