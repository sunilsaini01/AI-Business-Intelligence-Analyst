"""Deterministic pandas/NumPy analysis functions for the Analysis Agent
(Sec 5, Phase 6). Pure functions only — no LLM, no I/O beyond the rows
passed in, no DB access. This is what "LLM never does arithmetic" looks
like in code: period comparison, contribution/ranking, trend, and
distribution stats all happen here, not in a prompt.

Every function returns a dataclass with `ok`/`insufficient_evidence` rather
than raising on bad input, and never guesses or fills a missing value —
see each function's docstring for exactly which edge cases it handles.

A dimension/period value absent from one period's rows is treated as a
genuine 0 for that period, not a fabrication: these functions consume
already-aggregated `SUM(...) GROUP BY ...` results, and SQL's GROUP BY
naturally omits a group with no matching rows — that omission *is* the zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd


def compute_revenue(order_items: pd.DataFrame) -> pd.Series:
    """revenue = quantity * unit_price * (1 - discount), per Sec 3 note on order_items."""
    return order_items["quantity"] * order_items["unit_price"] * (1 - order_items["discount"])


def _to_numeric_clean(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    return df.dropna(subset=[value_col])


# ---------------------------------------------------------------------------
# Period comparison
# ---------------------------------------------------------------------------


@dataclass
class PeriodComparison:
    ok: bool
    period_col: str = ""
    value_col: str = ""
    baseline_period: str | None = None
    current_period: str | None = None
    baseline_value: float | None = None
    current_value: float | None = None
    absolute_change: float | None = None
    percentage_change: float | None = None  # None only when baseline is 0 (undefined)
    direction: Literal["increase", "decrease", "no_change"] | None = None
    note: str | None = None
    insufficient_evidence: bool = False
    reason: str | None = None


def compare_periods(rows: list[dict[str, Any]], period_col: str, value_col: str) -> PeriodComparison:
    """Compares the two most recent periods present in `rows` (sorted by
    period label). Duplicate rows for the same period are summed first.

    Edge cases: empty rows -> insufficient_evidence. Missing column ->
    insufficient_evidence. Fewer than 2 distinct periods -> insufficient_evidence
    (can't compare with only a baseline or nothing). More than 2 periods ->
    compares the latest two, notes the rest were present. Zero baseline ->
    percentage_change is None (undefined), not a fabricated number.
    """
    if not rows:
        return PeriodComparison(ok=False, insufficient_evidence=True, reason="No rows provided.")

    df = pd.DataFrame(rows)
    if period_col not in df.columns or value_col not in df.columns:
        return PeriodComparison(
            ok=False,
            insufficient_evidence=True,
            reason=f"Expected columns '{period_col}' and '{value_col}' were not both present.",
        )

    df = _to_numeric_clean(df[[period_col, value_col]], value_col)
    if df.empty:
        return PeriodComparison(ok=False, insufficient_evidence=True, reason="No numeric values to compare.")

    grouped = df.groupby(period_col, as_index=False)[value_col].sum().sort_values(period_col)

    if len(grouped) < 2:
        return PeriodComparison(
            ok=False,
            insufficient_evidence=True,
            reason=f"Only {len(grouped)} period(s) present ({list(grouped[period_col])}); need at least 2 to compare.",
        )

    note = None
    if len(grouped) > 2:
        note = (
            f"{len(grouped)} periods present; compared the two most recent "
            f"({grouped[period_col].iloc[-2]!s} vs {grouped[period_col].iloc[-1]!s})."
        )

    baseline_row = grouped.iloc[-2]
    current_row = grouped.iloc[-1]
    baseline_value = float(baseline_row[value_col])
    current_value = float(current_row[value_col])
    absolute_change = current_value - baseline_value
    percentage_change = None if baseline_value == 0 else (absolute_change / baseline_value) * 100

    if absolute_change > 0:
        direction: Literal["increase", "decrease", "no_change"] = "increase"
    elif absolute_change < 0:
        direction = "decrease"
    else:
        direction = "no_change"

    return PeriodComparison(
        ok=True,
        period_col=period_col,
        value_col=value_col,
        baseline_period=str(baseline_row[period_col]),
        current_period=str(current_row[period_col]),
        baseline_value=baseline_value,
        current_value=current_value,
        absolute_change=absolute_change,
        percentage_change=percentage_change,
        direction=direction,
        note=note,
    )


# ---------------------------------------------------------------------------
# Trend analysis
# ---------------------------------------------------------------------------


@dataclass
class TrendPoint:
    period: str
    value: float
    pct_change_from_prior: float | None


@dataclass
class TrendAnalysis:
    ok: bool
    period_col: str = ""
    value_col: str = ""
    points: list[TrendPoint] = field(default_factory=list)
    min_value: float | None = None
    max_value: float | None = None
    mean_value: float | None = None
    direction: Literal["increasing", "decreasing", "flat"] | None = None
    insufficient_evidence: bool = False
    reason: str | None = None


def analyze_trend(rows: list[dict[str, Any]], period_col: str, value_col: str) -> TrendAnalysis:
    """Requires at least 3 distinct periods — a trend cannot be inferred from
    1-2 observations (that is compare_periods' job, not this one).

    Direction is decided from a linear fit's slope relative to the series'
    own spread (a small, robust deterministic rule) rather than just
    comparing the first and last point, which a single outlier could flip.
    """
    if not rows:
        return TrendAnalysis(ok=False, insufficient_evidence=True, reason="No rows provided.")

    df = pd.DataFrame(rows)
    if period_col not in df.columns or value_col not in df.columns:
        return TrendAnalysis(
            ok=False,
            insufficient_evidence=True,
            reason=f"Expected columns '{period_col}' and '{value_col}' were not both present.",
        )

    df = _to_numeric_clean(df[[period_col, value_col]], value_col)
    grouped = df.groupby(period_col, as_index=False)[value_col].sum().sort_values(period_col)

    if len(grouped) < 3:
        return TrendAnalysis(
            ok=False,
            insufficient_evidence=True,
            reason=f"Only {len(grouped)} period(s) present; need at least 3 to assess a trend.",
        )

    values = grouped[value_col].to_numpy(dtype=float)
    periods = grouped[period_col].astype(str).tolist()

    pct_changes: list[float | None] = [None]
    for i in range(1, len(values)):
        prev = values[i - 1]
        pct_changes.append(None if prev == 0 else float((values[i] - prev) / prev * 100))

    points = [
        TrendPoint(period=p, value=float(v), pct_change_from_prior=c)
        for p, v, c in zip(periods, values, pct_changes)
    ]

    x = np.arange(len(values))
    slope = float(np.polyfit(x, values, 1)[0])
    value_range = float(values.max() - values.min())
    total_effect = abs(slope * (len(values) - 1))
    if value_range == 0 or total_effect < 0.02 * value_range:
        direction: Literal["increasing", "decreasing", "flat"] = "flat"
    elif slope > 0:
        direction = "increasing"
    else:
        direction = "decreasing"

    return TrendAnalysis(
        ok=True,
        period_col=period_col,
        value_col=value_col,
        points=points,
        min_value=float(values.min()),
        max_value=float(values.max()),
        mean_value=float(values.mean()),
        direction=direction,
    )


# ---------------------------------------------------------------------------
# Contribution / ranking
# ---------------------------------------------------------------------------


@dataclass
class Contributor:
    group: str
    current_value: float
    prior_value: float | None
    change: float | None
    pct_change: float | None
    pct_of_total_current: float
    # This group's share of the change among groups moving the SAME direction
    # as the overall total (e.g. share of the total decline, only counting
    # other groups that also declined) — None if this group moved opposite
    # the overall trend (it isn't part of "the decline"), or if there's no
    # prior period, or if no group moved with the trend. Deliberately NOT
    # change/total_change: offsetting movements across groups can shrink
    # total_change to near zero, which blows that ratio up into a
    # meaningless number (observed live: -3000%) even though every input is
    # correct — see analyze_contribution's docstring.
    pct_of_total_change: float | None
    rank: int


@dataclass
class ContributionAnalysis:
    ok: bool
    dimension_col: str = ""
    value_col: str = ""
    total_current: float | None = None
    total_prior: float | None = None
    total_change: float | None = None
    # Labels for whatever baseline/current periods the caller used to split
    # current_rows/prior_rows — carried through so diagnose_decline can build
    # a "total went from X to Y" fact straight from this object's own totals,
    # guaranteed to describe the same two periods as its `contributors`
    # (a separately-run period comparison query can drift to a different
    # window — see diagnose_decline's docstring).
    baseline_period: str | None = None
    current_period: str | None = None
    contributors: list[Contributor] = field(default_factory=list)
    insufficient_evidence: bool = False
    reason: str | None = None


def analyze_contribution(
    current_rows: list[dict[str, Any]],
    dimension_col: str,
    value_col: str,
    *,
    prior_rows: list[dict[str, Any]] | None = None,
    baseline_period: str | None = None,
    current_period: str | None = None,
) -> ContributionAnalysis:
    """Ranks groups by contribution to `current_rows`' total. If `prior_rows`
    (same dimension, a comparison period) is given, also computes each
    group's change vs prior and its share of the TOTAL change across all
    groups — i.e. "who drove the change." `baseline_period`/`current_period`
    are pure labels (this function doesn't use them for anything but
    carrying them through to the result) — pass the period values the caller
    used to build current_rows/prior_rows so the result can describe itself.

    Edge cases: empty/missing-column input -> insufficient_evidence. Zero
    total_current -> pct_of_total_current is 0.0 for every group (not a
    crash). Zero total_change -> pct_of_total_change stays None (undefined,
    not fabricated). A group present in one period only gets 0 for the
    other — aggregation semantics, see module docstring.
    """
    if not current_rows:
        return ContributionAnalysis(ok=False, insufficient_evidence=True, reason="No rows provided.")

    df = pd.DataFrame(current_rows)
    if dimension_col not in df.columns or value_col not in df.columns:
        return ContributionAnalysis(
            ok=False,
            insufficient_evidence=True,
            reason=f"Expected columns '{dimension_col}' and '{value_col}' were not both present.",
        )
    df = _to_numeric_clean(df, value_col)
    if df.empty:
        return ContributionAnalysis(ok=False, insufficient_evidence=True, reason="No numeric values to analyze.")

    current = df.groupby(dimension_col)[value_col].sum()

    prior = None
    if prior_rows:
        pdf = pd.DataFrame(prior_rows)
        if dimension_col in pdf.columns and value_col in pdf.columns:
            pdf = _to_numeric_clean(pdf, value_col)
            if not pdf.empty:
                prior = pdf.groupby(dimension_col)[value_col].sum()

    total_current = float(current.sum())
    total_prior = float(prior.sum()) if prior is not None else None
    total_change = (total_current - total_prior) if total_prior is not None else None

    all_groups = set(current.index) | (set(prior.index) if prior is not None else set())

    # "Share of the total change" is only numerically meaningful measured
    # against groups moving the *same direction* as the overall change — the
    # net total_change can be tiny (or flip sign) purely because increases in
    # some groups offset decreases in others, which blows up change/total_change
    # into a nonsensical percentage (seen live: -3000%) even though every
    # individual number involved is correct. A group moving opposite to the
    # overall trend gets pct_of_total_change=None (it isn't "contributing to
    # the decline" at all, it's fighting it) rather than a distorted number.
    raw_changes: dict[Any, float | None] = {}
    for g in all_groups:
        cur_v = float(current.get(g, 0.0))
        prior_v = float(prior.get(g, 0.0)) if prior is not None else None
        raw_changes[g] = (cur_v - prior_v) if prior_v is not None else None

    same_direction_total: float | None = None
    if total_change is not None and total_change != 0:
        same_direction = [
            c for c in raw_changes.values()
            if c is not None and ((total_change < 0 and c < 0) or (total_change > 0 and c > 0))
        ]
        same_direction_total = sum(same_direction) if same_direction else None

    contributors: list[Contributor] = []
    for g in all_groups:
        cur_v = float(current.get(g, 0.0))
        prior_v = float(prior.get(g, 0.0)) if prior is not None else None
        change = raw_changes[g]
        pct_change = None if (prior_v is None or prior_v == 0) else (change / prior_v * 100)
        pct_of_total_current = (cur_v / total_current * 100) if total_current != 0 else 0.0

        pct_of_total_change = None
        if change is not None and total_change is not None and same_direction_total not in (None, 0):
            moves_with_total = (total_change < 0 and change < 0) or (total_change > 0 and change > 0)
            if moves_with_total:
                pct_of_total_change = change / same_direction_total * 100

        contributors.append(
            Contributor(
                group=str(g),
                current_value=cur_v,
                prior_value=prior_v,
                change=change,
                pct_change=pct_change,
                pct_of_total_current=pct_of_total_current,
                pct_of_total_change=pct_of_total_change,
                rank=0,
            )
        )

    # Rank by |change| when we have a prior period — that's what "contribution
    # to the change" means; otherwise rank by current value.
    if total_change is not None:
        contributors.sort(key=lambda c: abs(c.change or 0.0), reverse=True)
    else:
        contributors.sort(key=lambda c: c.current_value, reverse=True)
    for i, c in enumerate(contributors, start=1):
        c.rank = i

    return ContributionAnalysis(
        ok=True,
        dimension_col=dimension_col,
        value_col=value_col,
        total_current=total_current,
        total_prior=total_prior,
        total_change=total_change,
        baseline_period=baseline_period,
        current_period=current_period,
        contributors=contributors,
    )


def top_n(
    rows: list[dict[str, Any]], dimension_col: str, value_col: str, n: int = 5, ascending: bool = False
) -> list[dict[str, Any]]:
    """Deterministic top/bottom-N by summed value_col per dimension_col group."""
    if not rows or dimension_col not in (rows[0] if rows else {}) or value_col not in (rows[0] if rows else {}):
        return []
    df = pd.DataFrame(rows)
    if dimension_col not in df.columns or value_col not in df.columns:
        return []
    df = _to_numeric_clean(df, value_col)
    if df.empty:
        return []
    grouped = df.groupby(dimension_col, as_index=False)[value_col].sum()
    grouped = grouped.sort_values(value_col, ascending=ascending).head(n)
    return grouped.to_dict("records")


# ---------------------------------------------------------------------------
# Distribution
# ---------------------------------------------------------------------------


@dataclass
class DistributionStats:
    ok: bool
    column: str = ""
    count: int = 0
    mean: float | None = None
    median: float | None = None
    min: float | None = None
    max: float | None = None
    std: float | None = None
    q25: float | None = None
    q75: float | None = None
    insufficient_evidence: bool = False
    reason: str | None = None


def distribution_stats(rows: list[dict[str, Any]], value_col: str) -> DistributionStats:
    """count/mean/median/min/max/std/quantiles over value_col. std is None
    for fewer than 2 values (sample std is undefined for n<2, not 0)."""
    if not rows:
        return DistributionStats(ok=False, insufficient_evidence=True, reason="No rows provided.")
    df = pd.DataFrame(rows)
    if value_col not in df.columns:
        return DistributionStats(ok=False, insufficient_evidence=True, reason=f"Column '{value_col}' not present.")
    series = pd.to_numeric(df[value_col], errors="coerce").dropna()
    if series.empty:
        return DistributionStats(ok=False, insufficient_evidence=True, reason="No numeric values to analyze.")

    return DistributionStats(
        ok=True,
        column=value_col,
        count=int(series.count()),
        mean=float(series.mean()),
        median=float(series.median()),
        min=float(series.min()),
        max=float(series.max()),
        std=float(series.std()) if series.count() >= 2 else None,
        q25=float(series.quantile(0.25)),
        q75=float(series.quantile(0.75)),
    )


# ---------------------------------------------------------------------------
# Diagnostic composite — fact / interpretation / limitation
# ---------------------------------------------------------------------------


@dataclass
class DiagnosticResult:
    ok: bool
    facts: list[str] = field(default_factory=list)
    interpretations: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    insufficient_evidence: bool = False
    reason: str | None = None


def diagnose_decline(
    comparison: PeriodComparison,
    contributions: list[ContributionAnalysis],
    *,
    contribution_threshold_pct: float = 20.0,
) -> DiagnosticResult:
    """Turns an already-computed period comparison plus zero or more
    contribution breakdowns into explicit fact / interpretation / limitation
    statements. Never recomputes anything from raw rows — only ever
    re-derives direction/pct from totals `compare_periods`/`analyze_contribution`
    already computed.

    Discovered live: a standalone overall period_comparison query and a
    separate dimension-breakdown query can each independently pick a
    slightly different 2-period window out of a wider result set (e.g. one
    lands on Aug-vs-Sep, the other on Jun-vs-Jul) — both individually
    correct, but stating a "fact" from one and an "interpretation" from the
    other would describe two different changes as if they were the same
    one. So: when a contribution breakdown with valid totals exists, the
    fact is built from *that* breakdown's own total_prior/total_current
    (which is guaranteed to describe the exact same two periods as the
    interpretations derived from it) — `comparison` is only used as a
    fallback when no contribution breakdown is available at all.

    A contributor is only called out as an interpretation when its share of
    the total change clears `contribution_threshold_pct` — below that, no
    single driver dominates and saying so would overstate the evidence.
    """
    usable = [c for c in contributions if c.ok and c.total_change is not None and c.contributors]

    if usable:
        primary = usable[0]
        value_col = primary.value_col
        baseline_value, current_value = primary.total_prior, primary.total_current
        baseline_period = primary.baseline_period or "the prior period"
        current_period = primary.current_period or "the current period"
    elif comparison.ok:
        value_col = comparison.value_col
        baseline_value, current_value = comparison.baseline_value, comparison.current_value
        baseline_period, current_period = comparison.baseline_period, comparison.current_period
    else:
        return DiagnosticResult(
            ok=False,
            insufficient_evidence=True,
            reason=comparison.reason or "No valid period comparison available.",
            limitations=["Could not establish whether the metric actually changed between periods."],
        )

    absolute_change = current_value - baseline_value  # type: ignore[operator]
    percentage_change = None if baseline_value == 0 else (absolute_change / baseline_value) * 100
    if absolute_change > 0:
        direction: Literal["increase", "decrease", "no_change"] = "increase"
    elif absolute_change < 0:
        direction = "decrease"
    else:
        direction = "no_change"

    pct_text = f" ({percentage_change:+.1f}%)" if percentage_change is not None else ""
    facts = [
        f"{value_col} went from {baseline_value:,.2f} in {baseline_period} to {current_value:,.2f} "
        f"in {current_period} ({direction}{pct_text})."
    ]

    if direction == "no_change":
        return DiagnosticResult(ok=True, facts=facts, interpretations=[], limitations=[])

    interpretations: list[str] = []
    limitations: list[str] = []

    if not usable:
        limitations.append(
            "No segment/region/category breakdown with matching prior-period data was available "
            "to explain what drove the change."
        )
        return DiagnosticResult(
            ok=True,
            facts=facts,
            interpretations=[],
            limitations=limitations,
            insufficient_evidence=True,
            reason="A change was confirmed, but no contribution breakdown could be computed.",
        )

    for contrib in usable:
        top = contrib.contributors[0]
        if top.pct_of_total_change is not None and abs(top.pct_of_total_change) >= contribution_threshold_pct:
            interpretations.append(
                f"By {contrib.dimension_col}, '{top.group}' appears to be the dominant contributor, "
                f"accounting for approximately {top.pct_of_total_change:.1f}% of the total change "
                f"({top.change:+,.2f})."
            )
        else:
            limitations.append(
                f"By {contrib.dimension_col}, no single group dominates the change "
                f"(largest: '{top.group}')."
            )

    return DiagnosticResult(ok=True, facts=facts, interpretations=interpretations, limitations=limitations)
