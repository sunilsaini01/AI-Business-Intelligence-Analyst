"""Phase 15, Objective 1 — API key / secret exposure regression tests.
Deterministic: no live LLM call. Uses the REAL configured secret values
(from this environment's own .env) as the thing being searched for, not a
fake stand-in — a test that only checks for a made-up string like
"supersecret" wouldn't actually prove the real ANTHROPIC_API_KEY/
GROQ_API_KEY/SECRET_KEY can't leak.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

import app.core.logging as logging_module
from app.core.config import get_settings
from app.main import app

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _real_secret_values() -> list[str]:
    """Every non-empty secret this environment actually has configured —
    skips any that are blank (e.g. GROQ_API_KEY when only Anthropic is
    configured) since an empty string isn't a meaningful thing to search
    for and would trivially "not be found" everywhere."""
    settings = get_settings()
    candidates = [
        settings.secret_key,
        settings.database_url,
        settings.analytics_database_url,
        settings.readonly_db_password,
        settings.anthropic_api_key,
        settings.groq_api_key,
        settings.previous_secret_key or "",
    ]
    return [v for v in candidates if v]


# --- Settings repr/str never contain secret values --------------------------


def test_settings_repr_never_contains_any_configured_secret_value():
    settings = get_settings()
    rendered = repr(settings) + str(settings)
    for secret in _real_secret_values():
        assert secret not in rendered


def test_settings_repr_omits_the_secret_fields_entirely():
    """Stronger than "the value isn't there" — the FIELD doesn't appear at
    all (Field(repr=False)), so even a secret that happened to collide
    with other visible text couldn't slip through by coincidence."""
    rendered = repr(get_settings())
    for field_name in (
        "database_url",
        "analytics_database_url",
        "readonly_db_password",
        "anthropic_api_key",
        "groq_api_key",
        "secret_key",
        "previous_secret_key",
    ):
        assert f"{field_name}=" not in rendered


def test_direct_attribute_access_to_secrets_still_works():
    """Field(repr=False) only affects repr/str — normal attribute access
    (how every real call site actually uses these values) must be
    completely unaffected."""
    settings = get_settings()
    assert isinstance(settings.secret_key, str) and settings.secret_key
    assert isinstance(settings.database_url, str) and settings.database_url


# --- .env / .env.example hygiene --------------------------------------------


def test_env_is_gitignored():
    gitignore_text = (_REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    lines = {line.strip() for line in gitignore_text.splitlines()}
    assert ".env" in lines


def test_env_example_contains_only_placeholders_for_every_secret():
    """No value in .env.example may look like a real credential — either
    blank, or an obvious placeholder (contains "change_me"/"generate"-
    style wording), never something matching a real provider key shape."""
    example_text = (_REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    secret_var_names = {
        "SECRET_KEY", "PREVIOUS_SECRET_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY",
        "POSTGRES_PASSWORD", "READONLY_DB_PASSWORD",
    }
    real_key_shapes = (
        re.compile(r"sk-ant-"),  # Anthropic
        re.compile(r"gsk_"),  # Groq
        re.compile(r"^[0-9a-f]{32,}$"),  # a bare hex secret someone pasted in directly
    )

    for line in example_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name not in secret_var_names:
            continue
        for pattern in real_key_shapes:
            assert not pattern.search(value), f"{name} in .env.example looks like a real secret: {value!r}"


def test_env_example_never_contains_this_sessions_real_key_values():
    """Direct proof against the specific incident this objective was
    written for: real ANTHROPIC_API_KEY/GROQ_API_KEY values were printed
    into a prior session's transcript by a masking bug in a shell command
    — never written to any tracked file. This asserts .env.example (the
    only credential-shaped file that's actually committed) never picked
    up a real value for any currently-configured secret."""
    example_text = (_REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    for secret in _real_secret_values():
        assert secret not in example_text


# --- API responses / errors never leak secrets -------------------------------


@pytest.mark.asyncio
async def test_health_ready_never_echoes_the_llm_key_value():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/v1/health/ready")
    assert resp.status_code == 200
    body_text = str(resp.json())
    for secret in _real_secret_values():
        assert secret not in body_text
    # only a boolean, never the key itself
    assert isinstance(resp.json()["llm_configured"], bool)


@pytest.mark.asyncio
async def test_an_unhandled_exception_carrying_the_real_secret_key_is_not_echoed(monkeypatch):
    """Forces a genuine unhandled exception whose message contains the
    REAL configured secret_key (not a fake stand-in) and confirms the
    global handler (app/main.py) still returns only the fixed, generic
    safe_error_response — proving "exceptions cannot expose secrets" holds
    even for this environment's actual key, not just a hypothetical one.
    """
    from app.services import analysis_service

    real_secret = get_settings().secret_key

    async def _boom(question: str, user_id=None):
        raise RuntimeError(f"internal failure, secret_key={real_secret}")

    monkeypatch.setattr(analysis_service, "create_session", _boom)

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as raw_client:
        import uuid

        email = f"secret-leak-{uuid.uuid4()}@example.com"
        await raw_client.post("/api/v1/auth/register", json={"email": email, "password": "correct-horse-battery"})
        login_resp = await raw_client.post(
            "/api/v1/auth/login", json={"email": email, "password": "correct-horse-battery"}
        )
        token = login_resp.json()["access_token"]
        async with AsyncClient(
            transport=transport, base_url="http://test", headers={"Authorization": f"Bearer {token}"}
        ) as client:
            resp = await client.post("/api/v1/analyze", json={"question": "test"})

    assert resp.status_code == 500
    assert real_secret not in str(resp.json())
    assert resp.json() == {"detail": "An internal error occurred.", "request_id": resp.json()["request_id"]}


@pytest.mark.asyncio
async def test_login_failure_response_never_contains_any_secret_value(unauthenticated_client):
    resp = await unauthenticated_client.post(
        "/api/v1/auth/login", json={"email": "nobody@example.com", "password": "wrong"}
    )
    body_text = str(resp.json())
    for secret in _real_secret_values():
        assert secret not in body_text


# --- execution_metadata never leaks secrets (extends Phase 14 coverage) -----


@pytest.mark.asyncio
async def test_execution_metadata_never_contains_any_of_this_environments_real_secrets():
    from app.db.models import SessionStatus
    from app.graph.workflow import build_graph
    from app.services import analysis_service

    class _RaisingLLM:
        async def complete(self, **kwargs):
            raise RuntimeError(f"boom, groq_api_key={get_settings().groq_api_key}")

        async def complete_structured(self, **kwargs):
            raise RuntimeError(f"boom, anthropic_api_key={get_settings().anthropic_api_key}")

    analysis_id = await analysis_service.create_session("How many customers?")
    await analysis_service.run_analysis(analysis_id, "How many customers?", graph=build_graph(llm=_RaisingLLM()))

    session = await analysis_service.get_session(analysis_id)
    assert session.status == SessionStatus.FAILED

    metadata_text = str(session.execution_metadata)
    error_message_text = session.error_message or ""
    for secret in _real_secret_values():
        assert secret not in metadata_text
        assert secret not in error_message_text


# --- logging never receives a whole Settings/secret value -------------------


def test_no_source_file_passes_the_settings_object_directly_to_a_logger_call():
    """Static regression guard: grep every app/ source file for a logger
    call receiving `settings` as a bare value (e.g. `logger.info("x",
    settings=settings)`) — the actual failure mode this objective is
    guarding against (an accidental whole-object log). All real call sites
    already only ever pass specific, non-secret fields."""
    pattern = re.compile(r"logger\.\w+\([^)]*\bsettings\b[^)]*\)")
    offenders: list[str] = []
    for path in (_REPO_ROOT / "app").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if pattern.search(text):
            offenders.append(str(path.relative_to(_REPO_ROOT)))
    assert offenders == []


def test_configure_logging_source_never_references_a_secret_field_name():
    source = inspect.getsource(logging_module)
    for field_name in ("secret_key", "api_key", "password", "database_url"):
        assert field_name not in source.lower()
