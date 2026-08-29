"""Phase 14 — POST /auth/register and POST /auth/login. Deterministic,
DB-backed (needs the live Postgres the rest of tests/api/ already assumes),
no LLM call anywhere in this file.
"""

from __future__ import annotations

import uuid

import pytest

_PASSWORD = "correct-horse-battery-staple"


@pytest.mark.asyncio
async def test_register_returns_the_created_user_never_the_password(unauthenticated_client):
    email = f"reg-{uuid.uuid4()}@example.com"
    resp = await unauthenticated_client.post("/api/v1/auth/register", json={"email": email, "password": _PASSWORD})
    assert resp.status_code == 201
    body = resp.json()
    assert body["email"] == email
    assert "password" not in body
    assert "hashed_password" not in body
    assert _PASSWORD not in str(body)


@pytest.mark.asyncio
async def test_register_lowercases_and_strips_the_email(unauthenticated_client):
    raw_email = f"  MixedCase-{uuid.uuid4()}@Example.com  "
    resp = await unauthenticated_client.post("/api/v1/auth/register", json={"email": raw_email, "password": _PASSWORD})
    assert resp.status_code == 201
    assert resp.json()["email"] == raw_email.strip().lower()


@pytest.mark.asyncio
async def test_register_duplicate_email_is_rejected(unauthenticated_client):
    email = f"dup-{uuid.uuid4()}@example.com"
    first = await unauthenticated_client.post("/api/v1/auth/register", json={"email": email, "password": _PASSWORD})
    assert first.status_code == 201
    second = await unauthenticated_client.post("/api/v1/auth/register", json={"email": email, "password": _PASSWORD})
    assert second.status_code == 400


@pytest.mark.asyncio
async def test_register_rejects_a_password_below_the_minimum_length(unauthenticated_client):
    resp = await unauthenticated_client.post(
        "/api/v1/auth/register", json={"email": f"short-{uuid.uuid4()}@example.com", "password": "short"}
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_register_rejects_a_value_with_no_at_sign(unauthenticated_client):
    resp = await unauthenticated_client.post(
        "/api/v1/auth/register", json={"email": "not-an-email", "password": _PASSWORD}
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_login_with_correct_credentials_returns_a_bearer_token(unauthenticated_client):
    email = f"login-{uuid.uuid4()}@example.com"
    await unauthenticated_client.post("/api/v1/auth/register", json={"email": email, "password": _PASSWORD})
    resp = await unauthenticated_client.post("/api/v1/auth/login", json={"email": email, "password": _PASSWORD})
    assert resp.status_code == 200
    body = resp.json()
    assert body["token_type"] == "bearer"
    assert isinstance(body["access_token"], str) and body["access_token"]


@pytest.mark.asyncio
async def test_login_with_wrong_password_is_401(unauthenticated_client):
    email = f"wrongpw-{uuid.uuid4()}@example.com"
    await unauthenticated_client.post("/api/v1/auth/register", json={"email": email, "password": _PASSWORD})
    resp = await unauthenticated_client.post("/api/v1/auth/login", json={"email": email, "password": "not-it"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_login_with_unregistered_email_is_401_not_404(unauthenticated_client):
    """Same status/response shape as a wrong password — an attacker must
    not be able to tell "no such account" from "wrong password" apart."""
    resp = await unauthenticated_client.post(
        "/api/v1/auth/login", json={"email": f"nobody-{uuid.uuid4()}@example.com", "password": _PASSWORD}
    )
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Incorrect email or password."


@pytest.mark.asyncio
async def test_a_fresh_token_actually_authorizes_a_protected_route(unauthenticated_client):
    email = f"e2e-{uuid.uuid4()}@example.com"
    await unauthenticated_client.post("/api/v1/auth/register", json={"email": email, "password": _PASSWORD})
    login_resp = await unauthenticated_client.post("/api/v1/auth/login", json={"email": email, "password": _PASSWORD})
    token = login_resp.json()["access_token"]

    unauthenticated_client.headers["Authorization"] = f"Bearer {token}"
    resp = await unauthenticated_client.post("/api/v1/analyze", json={"question": "How many customers?"})
    assert resp.status_code == 202
