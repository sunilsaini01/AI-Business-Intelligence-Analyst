from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class AnalyzeRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v.strip()


class AnalyzeAccepted(BaseModel):
    analysis_id: uuid.UUID
    status: str = "PENDING"


class AnalysisStatusResponse(BaseModel):
    analysis_id: uuid.UUID
    status: str
    current_stage: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class TraceEventOut(BaseModel):
    node: str
    event: str
    timestamp: str
    duration_ms: float | None = None


class AnalysisDetailResponse(AnalysisStatusResponse):
    question: str
    trace: list[TraceEventOut] = []
    # Phase 13, Objective D — structured execution observability (see
    # app.db.models.AnalysisSession.execution_metadata). Populated once the
    # run reaches a terminal state; {} while still PENDING/ANALYZING (never
    # fabricated ahead of time).
    execution_metadata: dict[str, Any] = {}
