"""Loads/validates evaluation/datasets/*.json against the Sec 6 case schema,
extended in Phase 8 to cover the Analysis/Visualization layers that didn't
exist when Sec 6 was first drafted. All new fields are `total=False` so the
original bi-004-shaped case (and any future case that doesn't need them)
keeps working unchanged — this loader has never validated field types at
runtime, only that the file parses as JSON, so widening the shape here is
backward compatible with the one pre-existing seed case.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict


class AcceptanceCriteria(TypedDict, total=False):
    min_groundedness: float
    min_relevance: int
    must_mention_segments: bool


class Tolerance(TypedDict, total=False):
    abs: float
    rel: float


class GroundTruth(TypedDict, total=False):
    """Type-discriminated on `type` — app/evaluation/metrics.py dispatches
    comparison logic on this field. One of: category_values, top_category,
    period_comparison_with_contribution, trend_bounds. Deliberately loose
    (total=False, extra keys allowed via the loader's plain dict passthrough)
    since each type uses a different subset of keys."""

    type: str


class BenchmarkCase(TypedDict, total=False):
    id: str
    question: str
    expected_intent: str
    expected_tools: list[str]
    expected_tables: list[str]
    reference_query: str
    expected_key_facts: list[str]
    acceptance_criteria: AcceptanceCriteria

    # Added Phase 8 — optional, only present on cases that exercise the
    # Analysis/Visualization/Critic layers.
    expected_schema: str  # "analytics" | "olist"
    expected_analysis_type: str  # contribution | trend | diagnostic | distribution
    expected_chart_types: list[str]
    ground_truth: GroundTruth
    tolerance: Tolerance
    expected_limitations: bool
    expected_behavior: str  # "answerable" | "out_of_scope" | "insufficient_evidence"


def load_benchmark(path: str) -> list[BenchmarkCase]:
    data: list[dict[str, Any]] = json.loads(Path(path).read_text(encoding="utf-8"))
    for case in data:
        if "id" not in case or "question" not in case:
            raise ValueError(f"Benchmark case missing required 'id'/'question': {case}")
    return [BenchmarkCase(**case) for case in data]  # type: ignore[typeddict-item]
