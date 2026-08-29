"""Thin auth service (Phase 14) — same shape as app/services/analysis_service.py:
route handlers stay a thin HTTP layer, this owns the actual DB work.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.core.auth import hash_password, verify_password
from app.db.database import async_session_factory
from app.db.models import User


class EmailAlreadyRegisteredError(Exception):
    pass


async def create_user(email: str, password: str) -> User:
    async with async_session_factory() as db:
        user = User(email=email, hashed_password=hash_password(password))
        db.add(user)
        try:
            await db.commit()
        except IntegrityError as exc:
            await db.rollback()
            raise EmailAlreadyRegisteredError(email) from exc
        await db.refresh(user)
        return user


async def authenticate_user(email: str, password: str) -> User | None:
    """Returns None for both "no such user" and "wrong password" — the
    caller (the login route) must give the client the same generic
    error either way, so this can't be used to enumerate registered
    emails by timing/response-shape."""
    async with async_session_factory() as db:
        result = await db.execute(select(User).where(User.email == email.strip().lower()))
        user = result.scalar_one_or_none()
    if user is None or not verify_password(password, user.hashed_password):
        return None
    return user
