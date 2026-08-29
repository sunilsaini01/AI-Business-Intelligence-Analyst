"""HTTP client for the FastAPI backend (Phase 12). Pure — no `streamlit`
import — so it's fully testable with pytest + `httpx.MockTransport` (see
tests/frontend/), matching the project's "mock HTTP, never require a live
LLM/backend" testing convention.

Every method raises a typed `ApiError` subclass. `.user_message` on any of
them is always safe to show directly in the UI — built from the response's
`detail` field or a fixed safe string, never the raw exception repr, a
stack trace, or anything from `httpx`'s own error text (which can include
connection internals).

The frontend is a pure HTTP client of the 6 documented endpoints below —
nothing here ever imports psycopg/asyncpg, builds a DSN, or accepts a raw
SQL string (Sec "Phase 4 SQL Safety" — no SQL editor, no direct DB access,
no arbitrary query endpoint).
"""

from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_API_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT_SECONDS = 30.0


class ApiError(Exception):
    """Base class for every error this client raises."""

    def __init__(self, user_message: str, *, status_code: int | None = None) -> None:
        super().__init__(user_message)
        self.user_message = user_message
        self.status_code = status_code


class InvalidRequestError(ApiError):
    """400/422 — bad user input (e.g. blank or oversized question)."""


class NotFoundError(ApiError):
    """404 — no such analysis."""


class NotReadyError(ApiError):
    """409 — the analysis exists but hasn't finished yet. Not a crash."""


class RateLimitedError(ApiError):
    """429 — including the LLM-provider-quota case the API surfaces as a
    FAILED session with a quota-specific error_message (Sec "Known
    rate-limit issue") — see classify_error_message below for that path."""


class AuthenticationError(ApiError):
    """401 — no token, or the token is missing/expired/invalid (Phase 14).
    Also what a wrong email/password on POST /auth/login surfaces as.
    app.py treats this as "go back to the login screen", distinct from
    every other error."""


class ForbiddenError(ApiError):
    """403 — a real, distinct token for a different, authenticated user
    (Phase 14's ownership check) — never reachable through normal use of
    this frontend (it only ever asks for the caller's own analysis_id), but
    typed separately from AuthenticationError since the fix is different:
    this is not a login problem."""


class ServerError(ApiError):
    """5xx — the backend itself hit an internal error."""


class BackendUnavailableError(ApiError):
    """Connection refused / DNS failure / TLS error — the API process
    itself isn't reachable at all."""


class RequestTimeoutError(ApiError):
    """The client-side timeout elapsed before the server responded."""


def default_base_url() -> str:
    return os.environ.get("API_BASE_URL", DEFAULT_API_BASE_URL)


def _safe_detail(resp: httpx.Response, fallback: str) -> str:
    try:
        body = resp.json()
    except ValueError:
        return fallback
    detail = body.get("detail") if isinstance(body, dict) else None
    return detail if isinstance(detail, str) and detail else fallback


def _raise_for_status(resp: httpx.Response) -> None:
    if resp.status_code < 400:
        return
    if resp.status_code in (400, 422):
        raise InvalidRequestError(_safe_detail(resp, "The request was invalid."), status_code=resp.status_code)
    if resp.status_code == 401:
        raise AuthenticationError(_safe_detail(resp, "Please log in again."), status_code=401)
    if resp.status_code == 403:
        raise ForbiddenError(_safe_detail(resp, "You are not authorized to access this analysis."), status_code=403)
    if resp.status_code == 404:
        raise NotFoundError(_safe_detail(resp, "Analysis not found."), status_code=404)
    if resp.status_code == 409:
        raise NotReadyError(_safe_detail(resp, "Analysis is not ready yet."), status_code=409)
    if resp.status_code == 429:
        raise RateLimitedError(
            "The AI provider rate limit was reached. Please try again later.", status_code=429
        )
    if resp.status_code >= 500:
        raise ServerError(
            "The server encountered an internal error. Please try again later.", status_code=resp.status_code
        )
    raise ApiError(_safe_detail(resp, "The request failed."), status_code=resp.status_code)


class AnalysisApiClient:
    """Thin wrapper around the 6 documented endpoints. Pass `client=` (an
    `httpx.Client`, e.g. one built on `httpx.MockTransport`) to inject a
    fake transport in tests — this class never constructs its own
    transport when one is given, and never talks to anything but the
    endpoint paths below.
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        client: httpx.Client | None = None,
        token: str | None = None,
    ) -> None:
        """`token=None` (register/login themselves, and any caller that
        hasn't authenticated yet) sends no Authorization header at all —
        every protected endpoint then correctly responds 401, handled the
        same as any other ApiError. Every other call site in app.py passes
        the token currently held in st.session_state."""
        self._owns_client = client is None
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._client = client or httpx.Client(base_url=base_url or default_base_url(), timeout=timeout, headers=headers)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "AnalysisApiClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            resp = self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise RequestTimeoutError("The request to the server timed out. Please try again.") from exc
        except httpx.ConnectError as exc:
            raise BackendUnavailableError("Could not reach the analysis server. Please try again later.") from exc
        except httpx.RequestError as exc:
            raise BackendUnavailableError(
                "A network error occurred while contacting the analysis server."
            ) from exc
        _raise_for_status(resp)
        return resp.json()

    def register(self, email: str, password: str) -> dict[str, Any]:
        return self._request("POST", "/api/v1/auth/register", json={"email": email, "password": password})

    def login(self, email: str, password: str) -> dict[str, Any]:
        return self._request("POST", "/api/v1/auth/login", json={"email": email, "password": password})

    def analyze(self, question: str) -> dict[str, Any]:
        return self._request("POST", "/api/v1/analyze", json={"question": question})

    def get_status(self, analysis_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/analysis/{analysis_id}/status")

    def get_report(self, analysis_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/analysis/{analysis_id}/report")

    def get_charts(self, analysis_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/api/v1/analysis/{analysis_id}/charts")

    def get_detail(self, analysis_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/analysis/{analysis_id}")

    def health_ready(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/health/ready")
