"""Orchestrates a benchmark run: creates the EvaluationRun row, invokes
app/evaluation/evaluator.py::run_benchmark (which runs the SAME production
graph as the API, per case), persists EvaluationResult rows. Mirrors
app/services/analysis_service.py's create-then-background-execute shape.
"""

from __future__ import annotations

import datetime
import uuid

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.database import async_session_factory
from app.db.models import EvaluationResult, EvaluationRun
from app.evaluation.evaluator import run_benchmark
from app.evaluation.models import CaseEvaluation

logger = get_logger(__name__)


def _current_model_name() -> str:
    settings = get_settings()
    return settings.groq_model_strong if settings.llm_provider == "groq" else settings.llm_model_strong


def _case_scores(case: CaseEvaluation) -> dict:
    return {
        "status": case.status,
        "sql_correct": case.sql_correct,
        "answer_correct": case.answer_correct,
        "analysis_correct": case.analysis_correct,
        "visualization_correct": case.visualization_correct,
        "ml_correct": case.ml_correct,
        "critic_correct": case.critic_correct,
        "critic_effectiveness_correct": case.critic_effectiveness_correct,
        "report_completeness_correct": case.report_completeness_correct,
        "grounded": case.grounded,
        "hallucination_detected": case.hallucination_detected,
        "first_failing_level": case.first_failing_level,
        "token_usage": case.token_usage,
        "errors": case.errors,
        "notes": case.notes,
    }


async def create_run(label: str, dataset_path: str) -> uuid.UUID:
    async with async_session_factory() as db:
        run_row = EvaluationRun(label=label, model_name=_current_model_name(), aggregate_scores={})
        db.add(run_row)
        await db.commit()
        await db.refresh(run_row)
        return run_row.id


async def start_evaluation_run(run_id: uuid.UUID, label: str, dataset_path: str) -> None:
    """Background task body. Never raises out of this function — a failure
    must land in the run row (empty aggregate_scores, finished_at set, error
    noted), not crash the background task silently (Sec 9's pattern, same as
    analysis_service.run_analysis)."""
    try:
        summary = await run_benchmark(dataset_path=dataset_path, label=label)
    except Exception as exc:  # noqa: BLE001 — must always resolve the run row
        logger.error("evaluation_run_failed", run_id=str(run_id), error=str(exc))
        async with async_session_factory() as db:
            run_row = await db.get(EvaluationRun, run_id)
            if run_row is not None:
                run_row.finished_at = datetime.datetime.utcnow()
                run_row.aggregate_scores = {"error": str(exc)}
                await db.commit()
        return

    async with async_session_factory() as db:
        run_row = await db.get(EvaluationRun, run_id)
        if run_row is None:
            logger.error("evaluation_run_row_not_found", run_id=str(run_id))
            return
        run_row.finished_at = datetime.datetime.utcnow()
        run_row.aggregate_scores = summary.aggregate_scores

        for case in summary.case_results:
            db.add(
                EvaluationResult(
                    run_id=run_id,
                    case_id=case.case_id,
                    scores=_case_scores(case),
                    judge_reasoning=None,
                    latency_ms=int(case.latency_ms) if case.latency_ms is not None else None,
                    passed=case.status == "PASSED",
                )
            )

        await db.commit()
