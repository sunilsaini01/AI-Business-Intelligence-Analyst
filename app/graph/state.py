"""Sec 1: the single typed state threaded through every graph node.

Nodes read what they need and append their own output — never mutate another
node's fields. The final state *is* the audit log (Sec 9 observability).
"""

from __future__ import annotations

import datetime
from typing import Any, Literal, TypedDict


class SQLQueryRecord(TypedDict):
    text: str
    validated_ok: bool
    rejection_reason: str | None
    rows: list[dict[str, Any]]
    row_count: int
    exec_ms: float


class ChartRecord(TypedDict):
    """Phase 7. `path` is kept (always "") for compatibility with the
    existing Chart DB model / ChartOut API schema, which were built assuming
    a rendered file — Plotly specs travel as JSON instead (Sec 2 tech
    decision), no file rendering happens. `spec_summary` carries the full
    structured spec (see app/tools/chart_selector.py::VisualizationSpec) for
    DB/API serialization.
    """

    chart_type: str  # bar | horizontal_bar | line | area | pie | scatter | kpi | table
    title: str
    subtitle: str | None
    x_axis: str | None
    y_axis: str | None
    group_by: str | None
    sort: str | None
    data: list[dict[str, Any]]
    units: str | None
    source_analysis: str
    reason: str
    limitations: list[str]
    path: str
    spec_summary: dict[str, Any]


class CriticFinding(TypedDict):
    """One issue the Critic found. `category` groups findings for filtering
    (a caller wanting just "errors" or "chart_errors" filters this list by
    severity/category rather than the state carrying five near-duplicate
    parallel lists that could drift out of sync with each other)."""

    severity: Literal["ERROR", "WARNING", "INFO"]
    category: str  # numerical | period_consistency | chart_consistency | contribution_arithmetic | missing_evidence | causal_claim | semantic
    message: str


class CriticVerdict(TypedDict):
    """Phase 9. `status` has three levels (blueprint Sec 1's original PASS/FAIL
    gained WARN): FAIL routes back for revision (bounded by max_retries, see
    app/agents/critic.py); WARN proceeds like PASS but the issue stays visible
    in `findings`/the final report's limitations, it just isn't blocking."""

    status: Literal["PASS", "WARN", "FAIL"]
    score: float  # 0.0-1.0, see app/tools/critic_checks.py::summarize_findings
    findings: list[CriticFinding]
    verified_claims: list[str]
    unsupported_claims: list[str]
    recommendations: list[str]
    target_agent: str | None  # which node to route back to on FAIL
    # Phase 13, Objective A: set only when the ONE semantic LLM call failed
    # (app/agents/critic.py::_semantic_check) — None when it succeeded, was
    # skipped (nothing to compare against), or wasn't reached at all. A
    # failure here NEVER changes `status` (still an INFO finding, still
    # "infra failure isn't a content failure") — this field exists purely
    # so the orchestration layer/observability can tell WHICH kind of
    # failure degraded the semantic check, without re-parsing message text.
    # See app/core/errors.py::ErrorCategory for the possible values.
    semantic_check_error_category: str | None


class TraceEvent(TypedDict):
    node: str
    event: Literal["enter", "exit"]
    timestamp: str
    duration_ms: float | None


class BusinessReport(TypedDict):
    """The 6 fields below this comment are owned by the Supervisor's
    synthesis step (app/agents/supervisor.py) and the Critic (Phase 9,
    app/agents/critic.py, which may force `confidence` down and append to
    `limitations` on an unresolved FAIL) — the Report Generator (Phase 10,
    app/agents/report_agent.py) never touches them, only reads them.

    The 5 fields below THAT comment are owned by the Report Generator: pure
    presentation/finalization additions, each traceable verbatim to
    already-computed state (critic_feedback / analysis_results / charts) —
    see report_agent.py's module docstring for the "never invents" contract.
    Populated once report_agent_node runs; `[]`/`""`/`{}`/`None` beforehand
    (see app/agents/supervisor.py's two BusinessReport construction sites).
    """

    executive_summary: str
    key_findings: list[str]
    evidence: list[dict[str, Any]]
    recommendations: list[str]
    confidence: Literal["Low", "Medium", "High"]
    limitations: str

    verified_claims: list[str]
    analysis_explanation: str
    # Phase 15, Objective 4 — deterministic rendering of state["ml_results"]
    # (app/agents/ml_agent.py::_format_ml_summary), same "presentation
    # only, never re-derives" ownership as analysis_explanation. "" when
    # ml_results is None (not a predictive question) or before
    # report_agent_node has run.
    ml_summary: str
    visualizations: list[dict[str, Any]]
    technical_details: dict[str, Any]
    narrative: str | None


class AgentState(TypedDict):
    question: str
    plan: list[str]
    intent: Literal["descriptive", "diagnostic", "predictive", "comparative", "out_of_scope"]
    required_tools: list[str]
    target_schema: Literal["analytics", "olist"] | None

    sql_queries: list[SQLQueryRecord]
    analysis_results: dict[str, Any]
    ml_results: dict[str, Any] | None
    charts: list[ChartRecord]

    critic_feedback: CriticVerdict | None
    retry_count: int
    max_retries: int

    report: BusinessReport | None
    trace: list[TraceEvent]


def new_state(question: str, *, max_retries: int = 2) -> AgentState:
    return AgentState(
        question=question,
        plan=[],
        intent="descriptive",
        required_tools=[],
        target_schema=None,
        sql_queries=[],
        analysis_results={},
        ml_results=None,
        charts=[],
        critic_feedback=None,
        retry_count=0,
        max_retries=max_retries,
        report=None,
        trace=[],
    )


def trace_event(node: str, event: Literal["enter", "exit"], duration_ms: float | None = None) -> TraceEvent:
    return TraceEvent(
        node=node,
        event=event,
        timestamp=datetime.datetime.utcnow().isoformat(),
        duration_ms=duration_ms,
    )
