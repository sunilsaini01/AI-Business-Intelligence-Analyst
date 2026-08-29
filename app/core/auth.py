"""Minimal auth boundary (Phase 14, Issue 6). Password hashing + a signed,
stateless bearer JWT — no session store, no external identity provider.
`app/db/models.py::User` was already shaped for this ("Auth-ready, not
wired to a login flow yet"); this module is what wires it in.

`get_current_user` is the one dependency every ownership check in
app/api/routes/analysis.py and app/api/routes/reports.py builds on. It
collapses every distinct failure mode (missing header, malformed token,
expired token, bad signature, deleted user) into the same safe 401 —
never a different status/message per failure reason, which would let a
client fingerprint *why* a token was rejected.
"""

from __future__ import annotations

import datetime
import uuid

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select

from app.core.config import Settings, get_settings
from app.db.database import async_session_factory
from app.db.models import User

# tokenUrl is documentation only (drives the /docs "Authorize" prompt) —
# login itself is a plain JSON endpoint (app/api/routes/auth.py), not an
# OAuth2 form, so this scheme is used solely to extract "Authorization:
# Bearer <token>" from the request.
_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)

_CREDENTIALS_ERROR = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials.",
    headers={"WWW-Authenticate": "Bearer"},
)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        # A malformed/foreign hash (e.g. corrupted row) must fail closed,
        # never raise into the caller as an unexpected 500.
        return False


def create_access_token(user_id: uuid.UUID) -> str:
    """Every new token carries the CURRENT key's id in its JWT header
    (Phase 15, Objective 3) — never signed with anything but
    `settings.secret_key`, the previous key (if any) is verify-only, for
    tokens already issued before a rotation."""
    settings = get_settings()
    expires_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
        minutes=settings.access_token_expire_minutes
    )
    payload = {"sub": str(user_id), "exp": expires_at}
    return jwt.encode(
        payload, settings.secret_key, algorithm=settings.jwt_algorithm, headers={"kid": settings.secret_key_id}
    )


def _signing_key_for(kid: str | None, settings: Settings) -> str | None:
    """Which key verifies this token, based on its (unverified) `kid`
    header — never the key material itself, so a caller can log/compare
    the *decision* without ever touching a secret. `kid is None` (every
    token issued before this rotation feature existed) intentionally
    falls back to the CURRENT key: those tokens were always signed with
    whatever was "the" secret_key at issuance time, which — for anyone
    still logged in and no rotation having happened yet — is exactly
    today's secret_key. Treating a missing kid as "reject" would log every
    existing session out the moment this code deploys; Objective 3
    explicitly rules that out ("Do NOT invalidate all users unexpectedly
    during normal operation")."""
    if kid is None or kid == settings.secret_key_id:
        return settings.secret_key
    if kid == settings.previous_secret_key_id and settings.previous_secret_key:
        return settings.previous_secret_key
    return None


def decode_access_token(token: str) -> uuid.UUID:
    """Raises `_CREDENTIALS_ERROR` (never a raw jwt exception) for any
    malformed/expired/mis-signed token — the caller never has to
    distinguish these, and none of jwt's own exception text (which can
    include claim values) ever reaches a response."""
    settings = get_settings()
    try:
        try:
            kid = jwt.get_unverified_header(token).get("kid")
        except jwt.PyJWTError as exc:
            raise _CREDENTIALS_ERROR from exc

        signing_key = _signing_key_for(kid, settings)
        if signing_key is None:
            raise _CREDENTIALS_ERROR

        payload = jwt.decode(token, signing_key, algorithms=[settings.jwt_algorithm])
        subject = payload.get("sub")
        if subject is None:
            raise _CREDENTIALS_ERROR
        return uuid.UUID(subject)
    except (jwt.PyJWTError, ValueError) as exc:
        raise _CREDENTIALS_ERROR from exc


async def get_current_user(token: str | None = Depends(_oauth2_scheme)) -> User:
    if token is None:
        raise _CREDENTIALS_ERROR
    user_id = decode_access_token(token)
    async with async_session_factory() as db:
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
    if user is None:
        raise _CREDENTIALS_ERROR
    return user
