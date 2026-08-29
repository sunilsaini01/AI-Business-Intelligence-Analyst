"""Internal, deterministic result types for the evaluation framework (Sec 6,
Phase 8). Dataclasses — matching this project's established convention
(app/tools/analysis_tools.py, app/tools/critic_checks.py) that Pydantic
BaseModel is for LLM-facing/API-facing structures (app/schemas/evaluation.py)
and dataclasses are for internally-computed results. `evaluator.py` builds
these; `evaluation_service.py` converts them to the existing Pydantic
schemas/DB rows at the API/persistence boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

EvalStatus = Literal["PASSED", "FAILED", "ERROR", "SKIPPED_QUOTA"]


@dataclass
class LevelResult:
    """One evaluation layer's verdict (Sec 8's "Level 1-5", used to localize
    a failure: SQL wrong? Analysis wrong? Visualization wrong? Critic
    failed? Synthesis wrong?). `correct=None` means this level doesn't apply
    to this particular case (e.g. no chart was expected)."""

    level: Literal["sql", "analysis", "visualization", "ml", "critic", "end_to_end"]
    correct: bool | None
    details: str = ""


@dataclass
class CaseEvaluation:
    case_id: str
    question: str
    status: EvalStatus

    expected: dict[str, Any]
    actual: dict[str, Any]

    sql_correct: bool | None = None
    answer_correct: bool | None = None
    analysis_correct: bool | None = None
    visualization_correct: bool | None = None
    critic_correct: bool | None = None
    critic_effectiveness_correct: bool | None = None
    report_completeness_correct: bool | None = None
    # Phase 16 — None whenever the case's final state has no ml_results at
    # all (not a predictive question, e.g. all 6 pre-Phase-15 benchmark
    # cases: see app/evaluation/metrics.py::evaluate_forecast_quality/
    # evaluate_churn_quality's shared "not applicable" convention with
    # every other *_correct field above). Set (and part of end_to_end) only
    # for a case whose question actually reached the ML Agent.
    ml_correct: bool | None = None

    grounded: bool = False
    hallucination_detected: bool = False

    levels: list[LevelResult] = field(default_factory=list)
    first_failing_level: str | None = None

    latency_ms: float | None = None
    token_usage: dict[str, int] | None = None

    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class EvaluationRunSummary:
    label: str
    model_name: str
    total_cases: int
    passed: int
    failed: int
    errored: int
    skipped: int
    end_to_end_success_rate: float
    hallucination_rate: float
    mean_latency_ms: float | None
    aggregate_scores: dict[str, float]
    case_results: list[CaseEvaluation] = field(default_factory=list)
