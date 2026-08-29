import pytest


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_readiness_reports_database_status(client):
    resp = await client.get("/api/v1/health/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert "ready" in body
    assert "database" in body


@pytest.mark.asyncio
async def test_readiness_returns_503_when_the_database_is_unreachable(client, monkeypatch):
    """Final deployment phase: a status-code-based health check (Render,
    Docker, Kubernetes) can only detect "not ready" via the HTTP status,
    never the JSON body — so an unreachable database must flip the status
    code, not just the `ready` field."""
    from app.api.routes import health as health_routes

    async def _db_down() -> bool:
        return False

    monkeypatch.setattr(health_routes, "db_healthy", _db_down)
    resp = await client.get("/api/v1/health/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["ready"] is False
    assert body["database"] is False
