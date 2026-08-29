"""Deterministic tests for app/core/auth.py — password hashing and JWT
issuance/verification. No DB, no network: `get_current_user` (the one
piece that touches the DB) is exercised at the API layer instead (see
tests/api/test_auth.py, tests/security/test_authorization.py).
"""

from __future__ import annotations

import datetime
import uuid

import jwt
import pytest

from app.core.auth import (
    _CREDENTIALS_ERROR,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from app.core.config import get_settings
from fastapi import HTTPException


def test_hash_password_never_stores_the_plaintext():
    hashed = hash_password("correct-horse-battery-staple")
    assert hashed != "correct-horse-battery-staple"
    assert "correct-horse-battery-staple" not in hashed


def test_verify_password_accepts_the_right_password():
    hashed = hash_password("correct-horse-battery-staple")
    assert verify_password("correct-horse-battery-staple", hashed) is True


def test_verify_password_rejects_the_wrong_password():
    hashed = hash_password("correct-horse-battery-staple")
    assert verify_password("wrong-password", hashed) is False


def test_verify_password_tolerates_a_malformed_hash_rather_than_raising():
    assert verify_password("anything", "not-a-real-bcrypt-hash") is False


def test_two_hashes_of_the_same_password_differ_bcrypt_salts_each_call():
    a = hash_password("same-password")
    b = hash_password("same-password")
    assert a != b
    assert verify_password("same-password", a) is True
    assert verify_password("same-password", b) is True


def test_create_and_decode_access_token_round_trips_the_user_id():
    user_id = uuid.uuid4()
    token = create_access_token(user_id)
    assert decode_access_token(token) == user_id


def test_decode_rejects_a_malformed_token():
    with pytest.raises(HTTPException) as exc_info:
        decode_access_token("not-a-real-jwt")
    assert exc_info.value.status_code == 401


def test_decode_rejects_a_token_signed_with_a_different_key():
    settings = get_settings()
    forged = jwt.encode(
        {"sub": str(uuid.uuid4()), "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=5)},
        "a-completely-different-secret",
        algorithm=settings.jwt_algorithm,
    )
    with pytest.raises(HTTPException) as exc_info:
        decode_access_token(forged)
    assert exc_info.value.status_code == 401


def test_decode_rejects_an_expired_token():
    settings = get_settings()
    expired = jwt.encode(
        {
            "sub": str(uuid.uuid4()),
            "exp": datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=1),
        },
        settings.secret_key,
        algorithm=settings.jwt_algorithm,
    )
    with pytest.raises(HTTPException) as exc_info:
        decode_access_token(expired)
    assert exc_info.value.status_code == 401


def test_decode_rejects_a_token_missing_the_subject_claim():
    settings = get_settings()
    no_subject = jwt.encode(
        {"exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=5)},
        settings.secret_key,
        algorithm=settings.jwt_algorithm,
    )
    with pytest.raises(HTTPException) as exc_info:
        decode_access_token(no_subject)
    assert exc_info.value.status_code == 401


def test_every_invalid_token_path_raises_the_same_shared_error_object():
    """A client must not be able to distinguish "expired" from "malformed"
    from "wrong signature" by the error message — that's a fingerprinting
    surface for probing the auth boundary. Every failure path in
    decode_access_token raises this exact same module-level object, not a
    freshly-worded exception per branch."""
    assert _CREDENTIALS_ERROR.detail == "Could not validate credentials."
    assert _CREDENTIALS_ERROR.status_code == 401

    with pytest.raises(HTTPException) as malformed:
        decode_access_token("not-a-real-jwt")
    assert malformed.value is _CREDENTIALS_ERROR


def test_access_token_never_contains_the_word_password_or_a_dsn():
    token = create_access_token(uuid.uuid4())
    for marker in ("password", "postgresql://", "secret_key"):
        assert marker not in token.lower()


# --- Phase 15, Objective 3: SECRET_KEY rotation (kid-based dual-key) -------


@pytest.fixture
def rotation_window(monkeypatch):
    """A simulated in-progress rotation: CURRENT key/id come from the
    already-configured test settings unchanged; a distinct PREVIOUS key/id
    is layered on top. Cleared again on teardown so later tests see the
    normal, non-rotating configuration."""
    monkeypatch.setenv("SECRET_KEY_ID", "key-2026-09")
    monkeypatch.setenv("PREVIOUS_SECRET_KEY", "the-old-secret-key-value")
    monkeypatch.setenv("PREVIOUS_SECRET_KEY_ID", "key-2026-06")
    get_settings.cache_clear()
    yield
    monkeypatch.delenv("SECRET_KEY_ID", raising=False)
    monkeypatch.delenv("PREVIOUS_SECRET_KEY", raising=False)
    monkeypatch.delenv("PREVIOUS_SECRET_KEY_ID", raising=False)
    get_settings.cache_clear()


def test_newly_issued_tokens_are_signed_with_the_current_key_id(rotation_window):
    token = create_access_token(uuid.uuid4())
    header = jwt.get_unverified_header(token)
    assert header["kid"] == get_settings().secret_key_id == "key-2026-09"


def test_token_signed_with_the_current_key_is_valid_during_a_rotation_window(rotation_window):
    user_id = uuid.uuid4()
    token = create_access_token(user_id)
    assert decode_access_token(token) == user_id


def test_token_signed_with_the_previous_key_is_valid_during_the_rotation_window(rotation_window):
    settings = get_settings()
    user_id = uuid.uuid4()
    old_token = jwt.encode(
        {"sub": str(user_id), "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=5)},
        settings.previous_secret_key,
        algorithm=settings.jwt_algorithm,
        headers={"kid": settings.previous_secret_key_id},
    )
    assert decode_access_token(old_token) == user_id


def test_previous_key_token_is_rejected_once_the_grace_period_ends(monkeypatch):
    """Simulates the operator ending the grace period: PREVIOUS_SECRET_KEY
    is set just long enough to mint a token signed with it, then unset
    (the real end-of-rotation action — see docs/security.md) before that
    token is ever presented for verification."""
    monkeypatch.setenv("SECRET_KEY_ID", "key-2026-09")
    monkeypatch.setenv("PREVIOUS_SECRET_KEY", "the-old-secret-key-value")
    monkeypatch.setenv("PREVIOUS_SECRET_KEY_ID", "key-2026-06")
    get_settings.cache_clear()
    settings = get_settings()
    old_token = jwt.encode(
        {
            "sub": str(uuid.uuid4()),
            "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=5),
        },
        settings.previous_secret_key,
        algorithm=settings.jwt_algorithm,
        headers={"kid": settings.previous_secret_key_id},
    )

    monkeypatch.delenv("PREVIOUS_SECRET_KEY", raising=False)
    monkeypatch.delenv("PREVIOUS_SECRET_KEY_ID", raising=False)
    get_settings.cache_clear()
    try:
        with pytest.raises(HTTPException) as exc_info:
            decode_access_token(old_token)
        assert exc_info.value.status_code == 401
    finally:
        monkeypatch.delenv("SECRET_KEY_ID", raising=False)
        get_settings.cache_clear()


def test_a_pre_rotation_kidless_token_still_validates_against_the_current_key(rotation_window):
    """Backward compatibility: a token issued before this rotation feature
    existed (no `kid` header at all) was signed with whatever was THE
    secret_key at the time — for a session that predates any rotation,
    that's exactly today's current key. Rejecting it outright would log
    every existing session out the moment this code ships, which Objective
    3 explicitly forbids."""
    settings = get_settings()
    user_id = uuid.uuid4()
    legacy_token = jwt.encode(
        {"sub": str(user_id), "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=5)},
        settings.secret_key,
        algorithm=settings.jwt_algorithm,
        # deliberately no headers={"kid": ...} — this is the pre-rotation shape
    )
    assert decode_access_token(legacy_token) == user_id


def test_a_token_with_an_unrecognized_kid_is_rejected(rotation_window):
    settings = get_settings()
    token = jwt.encode(
        {
            "sub": str(uuid.uuid4()),
            "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=5),
        },
        settings.secret_key,  # signed correctly...
        algorithm=settings.jwt_algorithm,
        headers={"kid": "some-key-id-nobody-configured"},  # ...but claims an unknown key id
    )
    with pytest.raises(HTTPException) as exc_info:
        decode_access_token(token)
    assert exc_info.value.status_code == 401


def test_rotation_errors_never_expose_either_key_value(rotation_window):
    settings = get_settings()
    forged = jwt.encode(
        {"sub": str(uuid.uuid4()), "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=5)},
        "a-totally-different-forged-key",
        algorithm=settings.jwt_algorithm,
        headers={"kid": settings.previous_secret_key_id},
    )
    with pytest.raises(HTTPException) as exc_info:
        decode_access_token(forged)
    detail_text = str(exc_info.value.detail)
    assert settings.secret_key not in detail_text
    assert settings.previous_secret_key not in detail_text
    assert exc_info.value.detail == "Could not validate credentials."
