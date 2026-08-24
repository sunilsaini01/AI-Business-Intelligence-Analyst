import httpx

from api_client import AnalysisApiClient
from health import check_backend_ready


def _client(handler) -> AnalysisApiClient:
    return AnalysisApiClient(client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test"))


def test_ready_backend_reports_ready():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ready": True, "database": True, "llm_provider": "groq", "llm_configured": True})

    with _client(handler) as client:
        ready, message = check_backend_ready(client)
    assert ready is True
    assert message


def test_db_not_connected_is_reported_as_not_ready():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ready": False, "database": False, "llm_provider": "groq", "llm_configured": True})

    with _client(handler) as client:
        ready, message = check_backend_ready(client)
    assert ready is False
    assert "not ready" in message.lower()


def test_backend_unreachable_reports_not_ready_never_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with _client(handler) as client:
        ready, message = check_backend_ready(client)
    assert ready is False
    assert message  # a safe message, never a raw exception repr
    assert "ConnectError" not in message
