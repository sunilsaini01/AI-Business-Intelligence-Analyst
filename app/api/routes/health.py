from fastapi import APIRouter, Response, status

from app.core.config import get_settings
from app.db.database import db_healthy

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready")
async def readiness(response: Response) -> dict[str, object]:
    """Final deployment phase: a status-code-based health check (Render,
    Docker, Kubernetes — none of them parse the response body) can only
    detect "not ready" via a non-2xx status, so this sets 503 whenever
    `ready` is false instead of always answering 200. The JSON body's
    `ready`/`database`/`llm_provider`/`llm_configured` fields are unchanged
    for anything that already reads them."""
    settings = get_settings()
    db_ok = await db_healthy()
    active_key = settings.groq_api_key if settings.llm_provider == "groq" else settings.anthropic_api_key
    llm_configured = bool(active_key)
    ready = db_ok  # LLM not required for Phase 0's graph
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "ready": ready,
        "database": db_ok,
        "llm_provider": settings.llm_provider,
        "llm_configured": llm_configured,
    }
