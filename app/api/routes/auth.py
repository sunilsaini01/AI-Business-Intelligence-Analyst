"""Registration + login (Phase 14). Login is a plain JSON endpoint, not an
OAuth2 form — no client here is a browser form post, so there's no reason
to require the `python-multipart` dependency OAuth2PasswordRequestForm
needs. app/core/auth.py's OAuth2PasswordBearer is still used for bearer-
token *extraction* on every protected route; only the login request body
itself is JSON.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from app.core.auth import create_access_token
from app.core.config import get_settings
from app.core.security import get_login_rate_limiter
from app.schemas.auth import Token, UserCreate, UserLogin, UserOut
from app.services import auth_service

router = APIRouter(prefix="/auth", tags=["auth"])

_INVALID_CREDENTIALS_DETAIL = "Incorrect email or password."
_RATE_LIMITED_DETAIL = "Too many login attempts. Please try again later."


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def register(request: UserCreate) -> UserOut:
    try:
        user = await auth_service.create_user(request.email, request.password)
    except auth_service.EmailAlreadyRegisteredError:
        # Deliberately vague: confirms the email is taken, which is an
        # accepted, standard tradeoff for a register endpoint (unlike
        # login, where enumeration is avoided below).
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email is already registered.")
    return UserOut(id=user.id, email=user.email, role=user.role, created_at=user.created_at)


@router.post("/login", response_model=Token)
async def login(request: UserLogin, http_request: Request) -> Token:
    """Phase 15, Objective 2: rate-limited BEFORE the credential check (no
    DB call at all for an over-limit attempt) — keyed by (client IP,
    submitted email), so this can never reveal whether the email is
    registered, and a limit on one victim's email never affects logins to
    a different account. Every attempt counts against the bucket,
    including this one whether it ends up succeeding or failing below —
    see app/core/security.py::LoginRateLimiter for why that matters.
    """
    settings = get_settings()
    if settings.login_rate_limit_enabled:
        client_ip = http_request.client.host if http_request.client else "unknown"
        limiter = get_login_rate_limiter()
        if not limiter.allow(client_ip, request.email):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=_RATE_LIMITED_DETAIL,
                headers={"Retry-After": str(settings.login_rate_limit_window_seconds)},
            )

    user = await auth_service.authenticate_user(request.email, request.password)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=_INVALID_CREDENTIALS_DETAIL)
    return Token(access_token=create_access_token(user.id))
