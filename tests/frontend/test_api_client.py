"""Deterministic tests for frontend/api_client.py — every HTTP call is
mocked via httpx.MockTransport, never a real network call (Sec "Frontend
tests should mock HTTP API responses" / "never dependent on live LLM
calls").
"""

from __future__ import annotations

import httpx
import pytest

from api_client import (
    AnalysisApiClient,
    ApiError,
    BackendUnavailableError,
    InvalidRequestError,
    NotFoundError,
    NotReadyError,
    RateLimitedError,
    RequestTimeoutError,
    ServerError,
)


def _client(handler) -> AnalysisApiClient:
    transport = httpx.MockTransport(handler)
    return AnalysisApiClient(client=httpx.Client(transport=transport, base_url="http://test"))


def test_analyze_success_returns_parsed_json():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/analyze"
        assert request.method == "POST"
        return httpx.Response(202, json={"analysis_id": "abc-123", "status": "PENDING"})

    with _client(handler) as client:
        result = client.analyze("How many customers?")
    assert result == {"analysis_id": "abc-123", "status": "PENDING"}


def test_analyze_sends_only_the_question_field_never_raw_sql():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content)
        return httpx.Response(202, json={"analysis_id": "x", "status": "PENDING"})

    with _client(handler) as client:
        client.analyze("'; DROP TABLE analytics.orders; --")
    assert captured["body"] == {"question": "'; DROP TABLE analytics.orders; --"}


@pytest.mark.parametrize("status_code,exc_type", [(400, InvalidRequestError), (422, InvalidRequestError)])
def test_4xx_bad_request_raises_invalid_request_error(status_code, exc_type):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"detail": "question must not be blank"})

    with _client(handler) as client:
        with pytest.raises(exc_type) as excinfo:
            client.analyze("")
    assert excinfo.value.user_message == "question must not be blank"
    assert excinfo.value.status_code == status_code


def test_404_raises_not_found_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Analysis not found"})

    with _client(handler) as client:
        with pytest.raises(NotFoundError) as excinfo:
            client.get_status("00000000-0000-0000-0000-000000000000")
    assert excinfo.value.status_code == 404


def test_409_raises_not_ready_error_not_a_crash():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"detail": "Analysis not ready (status=ANALYZING)"})

    with _client(handler) as client:
        with pytest.raises(NotReadyError) as excinfo:
            client.get_report("some-id")
    assert "not ready" in excinfo.value.user_message.lower()


def test_429_raises_rate_limited_error_with_a_safe_message():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"detail": "internal rate limiter details"})

    with _client(handler) as client:
        with pytest.raises(RateLimitedError) as excinfo:
            client.get_status("some-id")
    # never passes through the raw backend detail for 429 — always the fixed, user-friendly message
    assert "rate limit" in excinfo.value.user_message.lower()
    assert "internal rate limiter details" not in excinfo.value.user_message


def test_500_raises_server_error_without_leaking_backend_detail():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "An internal error occurred.", "request_id": "abc"})

    with _client(handler) as client:
        with pytest.raises(ServerError) as excinfo:
            client.get_status("some-id")
    assert excinfo.value.status_code == 500
    assert "internal error" in excinfo.value.user_message.lower()


def test_non_json_error_body_falls_back_to_a_safe_message():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"<html>not json</html>")

    with _client(handler) as client:
        with pytest.raises(ServerError) as excinfo:
            client.get_status("some-id")
    assert "<html>" not in excinfo.value.user_message


def test_connection_error_raises_backend_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with _client(handler) as client:
        with pytest.raises(BackendUnavailableError):
            client.get_status("some-id")


def test_timeout_raises_request_timeout_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    with _client(handler) as client:
        with pytest.raises(RequestTimeoutError):
            client.get_status("some-id")


def test_all_errors_are_apierror_subclasses_with_a_safe_user_message():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "boom"})

    with _client(handler) as client:
        with pytest.raises(ApiError) as excinfo:
            client.get_report("x")
    assert isinstance(excinfo.value.user_message, str) and excinfo.value.user_message


def test_client_only_ever_calls_the_six_documented_endpoints():
    seen_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        return httpx.Response(200, json={"ok": True, "status": "DONE", "trace": []})

    with _client(handler) as client:
        client.analyze("q")
        client.get_status("id")
        client.get_report("id")
        client.get_charts("id")
        client.get_detail("id")
        client.health_ready()

    assert seen_paths == [
        "/api/v1/analyze",
        "/api/v1/analysis/id/status",
        "/api/v1/analysis/id/report",
        "/api/v1/analysis/id/charts",
        "/api/v1/analysis/id",
        "/api/v1/health/ready",
    ]
