"""Sec 7 evaluation endpoints. `run` fires the benchmark runner
(app/evaluation/evaluator.py, against the real production graph) as a
background task and returns immediately — same fire-and-poll shape as
POST /analyze (app/api/routes/analysis.py). Poll GET /results (or GET
/results?run_id=...) for the finished aggregate/case scores.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, BackgroundTasks
from sqlalchemy import select

from app.db.database import async_session_factory
from app.db.models import EvaluationResult, EvaluationRun
from app.schemas.evaluation import (
    EvaluationCaseResult,
    EvaluationRunAccepted,
    EvaluationRunRequest,
    EvaluationRunResult,
)
from app.services import evaluation_service

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post("/run", response_model=EvaluationRunAccepted, status_code=202)
async def run_evaluation(request: EvaluationRunRequest, background_tasks: BackgroundTasks) -> EvaluationRunAccepted:
    run_id = await evaluation_service.create_run(request.label, request.dataset_path)
    background_tasks.add_task(evaluation_service.start_evaluation_run, run_id, request.label, request.dataset_path)
    return EvaluationRunAccepted(run_id=run_id, status="RUNNING")


@router.get("/results", response_model=list[EvaluationRunResult])
async def list_results(run_id: uuid.UUID | None = None) -> list[EvaluationRunResult]:
    async with async_session_factory() as db:
        stmt = select(EvaluationRun)
        if run_id is not None:
            stmt = stmt.where(EvaluationRun.id == run_id)
        runs = (await db.execute(stmt)).scalars().all()

        out: list[EvaluationRunResult] = []
        for run in runs:
            results = (
                (await db.execute(select(EvaluationResult).where(EvaluationResult.run_id == run.id)))
                .scalars()
                .all()
            )
            out.append(
                EvaluationRunResult(
                    run_id=run.id,
                    label=run.label,
                    model_name=run.model_name,
                    started_at=run.started_at,
                    finished_at=run.finished_at,
                    aggregate_scores=run.aggregate_scores,
                    results=[
                        EvaluationCaseResult(
                            case_id=r.case_id,
                            scores=r.scores,
                            judge_reasoning=r.judge_reasoning,
                            latency_ms=r.latency_ms,
                            passed=r.passed,
                        )
                        for r in results
                    ],
                )
            )
        return out
