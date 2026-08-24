"""Pure health-check helper (Phase 12) — takes an already-constructed
`AnalysisApiClient` (dependency injection, same pattern as the backend's
`llm` parameters) so it's testable with a mocked transport. app.py wraps
the call in `st.cache_data(ttl=...)` to avoid hammering
GET /health/ready on every rerun (Sec "Do not repeatedly hammer the health
endpoint") — the caching itself needs `streamlit`, so it stays in app.py;
this module only decides what a health response means.
"""

from __future__ import annotations

from api_client import AnalysisApiClient, ApiError


def check_backend_ready(client: AnalysisApiClient) -> tuple[bool, str]:
    """Returns (ready, message). Never raises — a health check that itself
    fails just means "not ready", not a crash."""
    try:
        data = client.health_ready()
    except ApiError as exc:
        return False, exc.user_message
    ready = bool(data.get("ready"))
    if ready:
        return True, "Backend ready."
    return False, "Backend reachable but not ready (database not connected)."
