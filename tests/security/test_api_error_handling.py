"""Phase 11: API-layer security — an unhandled exception must never leak a
stack trace, DB credentials, or an LLM API key to the client (only to
structured logs), and no route accepts anything resembling raw SQL or a
file path from the client. Complements tests/security/test_sql_injection.py
(the SQL-validator layer) with the FastAPI-integration layer.
"""

from __future__ import annotations

import inspect
import uuid

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

    Phase 14: POST /analyze now requires auth, so this client registers and
    logs in for real (same as the shared `client` fixture in
    tests/conftest.py) before triggering the forced exception — the point
    of this test is what the global handler does with an unhandled
    exception, not the 401 path, so it must genuinely reach the route.
    """

    async def _boom(question: str, user_id=None):
        raise RuntimeError(
            "leaking postgresql://bi_app:supersecret@postgres:5432/bi_agent and sk-ANTHROPIC-FAKEKEY-1234"
        )

    monkeypatch.setattr(analysis_service, "create_session", _boom)

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as raw_client:
        email = f"test-{uuid.uuid4()}@example.com"
        await raw_client.post("/api/v1/auth/register", json={"email": email, "password": "correct-horse-battery"})
        login_resp = await raw_client.post("/api/v1/auth/login", json={"email": email, "password": "correct-horse-battery"})
        token = login_resp.json()["access_token"]
        async with AsyncClient(
            transport=transport, base_url="http://test", headers={"Authorization": f"Bearer {token}"}
        ) as client:
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


def test_cors_is_not_wildcard_and_is_scoped_to_the_configured_frontend_origin():
    """Final deployment phase: CORS must be explicitly scoped to the real
    frontend origin(s) (Settings.frontend_origin), never `allow_origins=["*"]`
    — a wildcard would let any website's browser JS make credentialed
    requests against a logged-in user's bearer-token session."""
    from starlette.middleware.cors import CORSMiddleware as StarletteCORSMiddleware

    cors_middlewares = [m for m in app.user_middleware if m.cls is StarletteCORSMiddleware]
    assert len(cors_middlewares) == 1
    configured_origins = cors_middlewares[0].kwargs["allow_origins"]
    assert configured_origins != ["*"]
    assert "*" not in configured_origins
    assert configured_origins  # never silently empty either


@pytest.mark.asyncio
async def test_cors_preflight_reflects_the_configured_origin_and_rejects_an_unknown_one():
    """A cross-origin browser preflight from the configured frontend origin
    is allowed; one from an arbitrary third-party origin is not."""
    from app.core.config import get_settings

    allowed_origin = get_settings().frontend_origin.split(",")[0].strip()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        allowed_resp = await client.options(
            "/api/v1/health",
            headers={
                "Origin": allowed_origin,
                "Access-Control-Request-Method": "GET",
            },
        )
        assert allowed_resp.headers.get("access-control-allow-origin") == allowed_origin

        blocked_resp = await client.options(
            "/api/v1/health",
            headers={
                "Origin": "https://an-unrelated-third-party-site.example",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert "access-control-allow-origin" not in blocked_resp.headers
