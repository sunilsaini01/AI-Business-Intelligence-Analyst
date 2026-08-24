from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import analysis, evaluation, health, reports
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.core.security import RateLimitMiddleware, safe_error_response
from app.db.database import close_analytics_pool, init_analytics_pool

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_analytics_pool()
    yield
    await close_analytics_pool()


app = FastAPI(title="AI Business Intelligence Analyst", version="0.1.0", lifespan=lifespan)

app.add_middleware(RateLimitMiddleware, enabled=False)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = str(uuid.uuid4())
    logger.error("unhandled_exception", request_id=request_id, path=str(request.url), error=str(exc))
    return JSONResponse(status_code=500, content=safe_error_response(request_id))


app.include_router(health.router, prefix="/api/v1")
app.include_router(analysis.analyze_router, prefix="/api/v1")
app.include_router(analysis.router, prefix="/api/v1")
app.include_router(reports.router, prefix="/api/v1")
app.include_router(evaluation.router, prefix="/api/v1")


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "ai-business-intelligence-agent", "status": "running"}
