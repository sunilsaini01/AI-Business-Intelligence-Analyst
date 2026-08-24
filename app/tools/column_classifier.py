"""Heuristic column-role detection for arbitrary SQL result rows (Phase 6).

The SQL Agent's queries are LLM-generated per question, so there's no fixed
schema for the Analysis Agent to rely on — this infers which column is a
time period, which are categorical dimensions, and which is the numeric
metric to analyze, from column names and value shapes. Deliberately
conservative: when nothing looks like a metric, callers get back
`(None, [], None)` and skip analysis for that query rather than guessing.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

_ID_COL = re.compile(r"(^id$|_id$)", re.IGNORECASE)
_PERIOD_NAME_HINT = re.compile(r"month|period|quarter|week|^date$|_date$", re.IGNORECASE)
_METRIC_NAME_HINT = re.compile(r"revenue|amount|total|count|score|value|spend|price|^n$", re.IGNORECASE)
_YYYY_MM = re.compile(r"^\d{4}-\d{2}(-\d{2})?$")

# A "dimension" with more distinct values than this is almost certainly an
# ID-like or free-text column, not a meaningful grouping — skip it rather
# than produce a degenerate, uninformative breakdown.
_MAX_DIMENSION_CARDINALITY = 50


def _looks_like_period_values(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(20)
    if sample.empty:
        return False
    return bool(sample.map(lambda v: bool(_YYYY_MM.match(v))).all())


def _classify(rows: list[dict[str, Any]]) -> tuple[str | None, list[str], list[str]]:
    """Internal: (period_col, dimension_cols, numeric_candidates) — full
    numeric candidate list, before picking just one "the" value column.
    """
    if not rows:
        return None, [], []
    df = pd.DataFrame(rows)

    period_col: str | None = None
    numeric_candidates: list[str] = []
    dimension_cols: list[str] = []

    for col in df.columns:
        if _ID_COL.search(col):
            continue
        if period_col is None and (_PERIOD_NAME_HINT.search(col) or _looks_like_period_values(df[col])):
            period_col = col
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().mean() > 0.9:
            numeric_candidates.append(col)
        elif df[col].nunique(dropna=True) <= _MAX_DIMENSION_CARDINALITY:
            dimension_cols.append(col)

    return period_col, dimension_cols, numeric_candidates


def classify_columns(rows: list[dict[str, Any]]) -> tuple[str | None, list[str], str | None]:
    """Returns (period_col, dimension_cols, value_col) — any may be None/empty
    if nothing in `rows` looks like that role.
    """
    period_col, dimension_cols, numeric_candidates = _classify(rows)
    return period_col, dimension_cols, _pick_value_col(numeric_candidates)


def _pick_value_col(candidates: list[str]) -> str | None:
    if not candidates:
        return None
    for c in candidates:
        if _METRIC_NAME_HINT.search(c):
            return c
    return candidates[0]


def find_scatter_candidate(rows: list[dict[str, Any]]) -> tuple[str, str] | None:
    """Phase 7: (x_col, y_col) when `rows` has 2+ numeric columns and no
    period/dimension role — i.e. two independent numeric measurements per
    row, the shape a scatter plot is for. Returns None otherwise (including
    when a period/dimension column exists — that data has a more specific,
    better-fitting chart type already, e.g. a trend or a bar breakdown).
    """
    period_col, dimension_cols, numeric_candidates = _classify(rows)
    if period_col or dimension_cols or len(numeric_candidates) < 2:
        return None
    return numeric_candidates[0], numeric_candidates[1]
