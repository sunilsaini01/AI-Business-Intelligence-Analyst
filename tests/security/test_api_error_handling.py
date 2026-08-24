"""Phase 11: API-layer security — an unhandled exception must never leak a
stack trace, DB credentials, or an LLM API key to the client (only to
structured logs), and no route accepts anything resembling raw SQL or a
file path from the client. Complements tests/security/test_sql_injection.py
(the SQL-validator layer) with the FastAPI-integration layer.
"""

from __future__ import annotations

import inspect

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.routes import analysis as analysis_routes
from app.main import app
from app.services import analysis_service


@pytest.mark.asyncio
async def test_unhandled_exception_returns_generic_error_not_internals(monkeypatch):
    """Force a genuine unhandled exception in the request path (not the
    background task) and confirm the global handler (app/main.py) returns
    the generic safe_error_response — never the real exception text, a
    stack trace, or anything resembling a DSN/API key.

    Uses its own client with `raise_server_exceptions=False`: httpx's
    ASGITransport re-raises a route's exception by default (a test-harness
    debugging aid — the shared `client` fixture keeps that default on
    purpose, so a real unhandled-exception bug fails loudly elsewhere in
    the suite). Here the re-raise is exactly what we need to disable to
    inspect the response app/main.py's registered handler actually returns
    to a real client in production.
    """

    async def _boom(question: str):
        raise RuntimeError(
            "leaking postgresql://bi_app:supersecret@postgres:5432/bi_agent and sk-ANTHROPIC-FAKEKEY-1234"
        )

    monkeypatch.setattr(analysis_service, "create_session", _boom)

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/v1/analyze", json={"question": "How many customers do we have?"})

    assert resp.status_code == 500
    body = resp.json()
    assert body["detail"] == "An internal error occurred."
    assert "request_id" in body

    body_text = str(body).lower()
    for leak_marker in ("postgresql://", "supersecret", "sk-anthropic", "runtimeerror", "traceback"):
        assert leak_marker not in body_text


def test_analyze_request_schema_has_no_raw_sql_or_file_path_field():
    """Structural guard: POST /analyze's request model must only ever
    accept a natural-language `question` string — never a `sql`, `query`,
    `path`, or `file` field that could let a client bypass the Supervisor/
    SQL Agent pipeline and the safety layers it goes through."""
    from app.schemas.analysis import AnalyzeRequest

    fields = set(AnalyzeRequest.model_fields.keys())
    assert fields == {"question"}


def test_no_route_in_the_analysis_router_accepts_a_raw_sql_or_path_parameter():
    """Every route handler in app/api/routes/analysis.py takes only
    analysis_id (a UUID) and/or a validated Pydantic body — grep-level
    guard against a future endpoint accidentally exposing raw SQL or
    filesystem access."""
    source = inspect.getsource(analysis_routes)
    for forbidden in ("open(", "os.path", "subprocess", "execute_validated_query", "eval(", "exec("):
        assert forbidden not in source
