from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any


from pydantic import BaseModel


class EvaluationRunRequest(BaseModel):
    label: str = "manual-run"
    dataset_path: str = "evaluation/datasets/benchmark.json"


class EvaluationRunAccepted(BaseModel):
    run_id: uuid.UUID
    status: str = "RUNNING"


class EvaluationCaseResult(BaseModel):
    case_id: str
    scores: dict[str, Any]
    judge_reasoning: str | None = None
    latency_ms: int | None = None
    passed: bool


class EvaluationRunResult(BaseModel):
    run_id: uuid.UUID
    label: str
    model_name: str
    started_at: datetime
    finished_at: datetime | None
    aggregate_scores: dict[str, Any]
    results: list[EvaluationCaseResult] = []
