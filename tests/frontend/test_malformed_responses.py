"""Phase 13, Section 10 — frontend regression tests for malformed/partial
API responses, empty collections, and the report_view <-> api_client
integration on realistic-but-imperfect data. Complements
test_api_client.py (error status codes), test_report_view.py (section
selection), and test_chart_builder.py (per-chart validation).
"""

from __future__ import annotations

import httpx

from api_client import AnalysisApiClient
from report_view import build_report_sections


def _client(handler) -> AnalysisApiClient:
    return AnalysisApiClient(client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test"))


def test_report_missing_every_optional_phase10_field_still_renders_the_core_sections():
    """A report shaped like the API contract's minimum (only the 6
    Supervisor/Critic-owned fields, none of the 5 Report Generator
    additions) — e.g. a row persisted before migration 0003 — must not
    crash report_view, and must never invent values for what's missing."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "analysis_id": "x", "executive_summary": "Revenue fell.", "key_findings": ["June: 100"],
                "evidence": [], "recommendations": [], "confidence": "Medium", "limitations": "",
                # verified_claims / analysis_explanation / visualizations /
                # technical_details / narrative all absent entirely.
            },
        )

    with _client(handler) as client:
        report = client.get_report("x")

    sections = build_report_sections(report)
    titles = [s.title for s in sections]
    assert titles == ["Executive Summary", "Key Findings", "Confidence"]
    assert "Narrative" not in titles
    assert "Technical Details" not in titles


def test_report_with_empty_visualizations_list_produces_no_chart_section_signal():
    report = {
        "executive_summary": "x", "key_findings": [], "confidence": "Low",
        "visualizations": [],
    }
    sections = build_report_sections(report)
    # build_report_sections doesn't render a "Visualizations" section itself
    # (that's app.py, driven by GET /charts, not the report body) — this
    # just confirms an empty list causes no crash and adds nothing spurious.
    assert all(s.title != "Visualizations" for s in sections)


def test_charts_endpoint_returning_an_empty_list_is_handled_by_the_client_directly():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    with _client(handler) as client:
        charts = client.get_charts("x")
    assert charts == []


def test_status_response_missing_current_stage_is_tolerated():
    """An older/partial status payload without `current_stage` at all
    (rather than `current_stage: null`) — the client just returns the dict
    as received; callers (progress.py) already treat a missing key the
    same as an explicit `None` via `.get()`."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"analysis_id": "x", "status": "ANALYZING"})

    with _client(handler) as client:
        status = client.get_status("x")
    assert status.get("current_stage") is None


def test_detail_response_missing_execution_metadata_is_tolerated():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"analysis_id": "x", "status": "DONE", "trace": []})

    with _client(handler) as client:
        detail = client.get_detail("x")
    assert detail.get("execution_metadata", {}) == {}


def test_fail_exhausted_report_with_low_confidence_and_no_verified_claims_renders_honestly():
    """Integration of api_client + report_view for the exact FAIL-exhausted
    shape app/agents/critic.py::_force_degrade produces — confidence Low,
    limitations present, verified_claims empty."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "analysis_id": "x",
                "executive_summary": "Revenue decreased because customers churned en masse.",
                "key_findings": [], "evidence": [], "recommendations": [],
                "confidence": "Low",
                "limitations": "Automated review found unresolved issues: unsupported claim",
                "verified_claims": [], "analysis_explanation": "", "visualizations": [],
                "technical_details": {"critic_status": "FAIL", "critic_score": 0.2},
                "narrative": None,
            },
        )

    with _client(handler) as client:
        report = client.get_report("x")

    sections = build_report_sections(report)
    titles = [s.title for s in sections]
    assert "Limitations" in titles
    assert "Verified Claims" not in titles  # honestly empty, not fabricated
    assert "Narrative" not in titles
    confidence_section = next(s for s in sections if s.title == "Confidence")
    assert confidence_section.content == "Low"
