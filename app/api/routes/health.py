from fastapi import APIRouter

from app.core.config import get_settings
from app.db.database import db_healthy

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready")
async def readiness() -> dict[str, object]:
    settings = get_settings()
    db_ok = await db_healthy()
    active_key = settings.groq_api_key if settings.llm_provider == "groq" else settings.anthropic_api_key
    llm_configured = bool(active_key)
    ready = db_ok  # LLM not required for Phase 0's graph
    return {
        "ready": ready,
        "database": db_ok,
        "llm_provider": settings.llm_provider,
        "llm_configured": llm_configured,
    }
