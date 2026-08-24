"""Orchestrates one graph run against the app-schema session state (Sec 7).

Kept out of the route module so the FastAPI layer stays thin and this logic
is unit-testable without spinning up ASGI.
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any

from sqlalchemy import select

from app.core.config import get_settings
from app.core.errors import SAFE_MESSAGE_BY_CATEGORY, classify_exception
from app.core.llm import get_llm_client
from app.core.logging import get_logger
from app.db.database import async_session_factory
from app.db.models import AnalysisReport, AnalysisSession, AnalysisStep, Chart, SessionStatus
from app.graph.state import new_state
from app.graph.workflow import get_graph

# Canonical pipeline order (Phase 11/13 observability only — never used for
# control flow, the graph's own routing in app/graph/workflow.py is the
# only thing that decides what actually runs next). Used to derive
# `current_stage`/`failed_node` for execution_metadata the same honest way
# frontend/progress.py derives its checklist: "last node confirmed done",
# never "currently executing".
_PIPELINE_STAGES = ["supervisor", "sql_agent", "analysis_agent", "visualization_agent", "critic", "report_agent"]

logger = get_logger(__name__)


async def create_session(question: str) -> uuid.UUID:
    async with async_session_factory() as db:
        session_row = AnalysisSession(question=question, status=SessionStatus.PENDING)
        db.add(session_row)
        await db.commit()
        await db.refresh(session_row)
        return session_row.id


async def _persist_new_trace_events(analysis_id: uuid.UUID, trace: list[dict], already_persisted: int) -> int:
    """Writes only the trace events not yet in the DB (`trace[already_persisted:]`)
    in one short-lived session/transaction, then returns the new total —
    called after every graph super-step (see `run_analysis`'s `astream`
    loop), so `GET /analysis/{id}` and the derived `current_stage` (Sec 8)
    reflect real progress from a still-running analysis, not just the
    finished one. Each call is its own commit — no transaction is held open
    across node executions, let alone the whole graph run.
    """
    new_events = trace[already_persisted:]
    if not new_events:
        return already_persisted
    async with async_session_factory() as db:
        for i, event in enumerate(new_events, start=already_persisted):
            db.add(
                AnalysisStep(
                    session_id=analysis_id,
                    agent_name=event["node"],
                    step_order=i,
                    input_json={},
                    output_json=dict(event),
                    status=event["event"],
                    duration_ms=int(event["duration_ms"]) if event["duration_ms"] else None,
                )
            )
        await db.commit()
    return len(trace)


def _completed_nodes(trace: list[dict]) -> list[str]:
    """Distinct node names with a recorded `exit` event, in first-seen
    order — the same "confirmed done" signal `get_current_stage` already
    uses, just the full list instead of only the latest one."""
    seen: dict[str, None] = {}
    for event in trace:
        if event.get("event") == "exit":
            seen[event["node"]] = None
    return list(seen.keys())


def _infer_failed_node(completed_nodes: list[str]) -> str | None:
    """Best-effort observability hint only — NEVER used for control flow.
    The first canonical-order stage not yet confirmed complete is *probably*
    where execution was when it failed, but this is an inference from
    `completed_nodes`, not a fact observed directly (a node that raises
    mid-execution leaves no trace event at all — see run_analysis's
    docstring on why `astream` can't see partial node state). Returns None
    once every stage is accounted for (nothing left to infer)."""
    for stage in _PIPELINE_STAGES:
        if stage not in completed_nodes:
            return stage
    return None


def _build_execution_metadata(
    *,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    final_status: str,
    trace: list[dict],
    error_category: str | None,
    retry_count: int,
    token_usage: dict[str, int] | None,
    narrative_enabled: bool,
    report_generated: bool,
) -> dict[str, Any]:
    """Phase 13, Objective D. One JSON bundle per run — see
    app.db.models.AnalysisSession.execution_metadata (migration
    0004_execution_metadata). Never contains secrets: node names drawn from
    app/graph/state.py::trace_event, counts, timestamps, booleans, and
    token counts only — no request/response bodies, no headers, no
    exception text.
    """
    completed_nodes = _completed_nodes(trace)
    return {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "total_duration_ms": (end_time - start_time).total_seconds() * 1000,
        "final_status": final_status,
        # Same semantic as GET /analysis/{id}/status's current_stage (Sec
        # 8) — the most recently COMPLETED node, never "currently
        # executing" (see frontend/progress.py's identical honesty
        # constraint on the presentation side).
        "current_stage": completed_nodes[-1] if completed_nodes else None,
        "completed_nodes": completed_nodes,
        "failed_node": _infer_failed_node(completed_nodes) if final_status == "FAILED" else None,
        "error_category": error_category,
        "retry_count": retry_count,
        "token_usage": token_usage,
        "narrative_enabled": narrative_enabled,
        "report_generated": report_generated,
    }


async def _persist_execution_metadata(analysis_id: uuid.UUID, metadata: dict[str, Any]) -> None:
    async with async_session_factory() as db:
        session_row = await db.get(AnalysisSession, analysis_id)
        if session_row is not None:
            session_row.execution_metadata = metadata
            await db.commit()


async def run_analysis(analysis_id: uuid.UUID, question: str, graph: Any = None) -> None:
    """Runs the compiled graph and persists trace/report. Never raises out of this
    function — a failure must land in the session row (status=FAILED), not crash
    the background task silently (Sec 9).

    Drives the graph via `astream(..., stream_mode="values")` rather than
    `ainvoke` so each node's trace events are persisted as they happen
    (`_persist_new_trace_events`), not only in one batch after the entire
    pipeline finishes — this is what lets a FAILED run keep whatever partial
    trace it reached instead of none at all, and lets `GET /analysis/{id}`
    reflect real progress on a still-ANALYZING session. No change to the
    graph itself (same compiled `StateGraph`, same nodes, same routing) —
    purely how this orchestration layer consumes it.

    `graph=None` (every real call — app/api/routes/analysis.py's
    `background_tasks.add_task` never passes it) uses the real production
    singleton (`app.graph.workflow.get_graph()`). Tests pass a
    `build_graph(llm=ScriptedLLMClient(...))` instance directly — same
    dependency-injection pattern as `llm` on the node functions — so
    concurrency/incremental-persistence behavior is testable without a live
    LLM call or touching the process-wide compiled-graph singleton.

    Phase 13, Objective A: exception classification (app/core/errors.py)
    replaces the old binary "RateLimitError vs everything else" check with
    the full rate_limit/timeout/provider_error/validation_error/
    application_error set — SessionStatus is still just DONE/FAILED (no new
    enum value), only `error_message` (safe, fixed per category) and
    `execution_metadata.error_category` (structured, for observability)
    change with it.
    """
    start_time = datetime.datetime.utcnow()
    async with async_session_factory() as db:
        session_row = await db.get(AnalysisSession, analysis_id)
        if session_row is None:
            logger.error("session_not_found", analysis_id=str(analysis_id))
            return
        session_row.status = SessionStatus.ANALYZING
        await db.commit()

    graph = graph or get_graph()
    max_retries = get_settings().critic_max_retries
    result_state: dict | None = None
    persisted = 0
    trace_so_far: list[dict] = []

    # Token-usage delta (Sec 8's "token_usage if already available"), same
    # before/after-snapshot pattern already established and reviewed in
    # app/evaluation/evaluator.py::run_case_live — `total_usage` on
    # LLMClient/GroqLLMClient is a cumulative-since-process-start counter
    # (a cached singleton, see app/core/llm.py::get_llm_client), so a raw
    # snapshot would leak every OTHER request's usage into this one's
    # metadata; the delta is this run's usage only. A test-injected fake
    # graph doesn't call the real singleton at all, so the delta is
    # harmlessly 0 in that case, not wrong.
    usage_client = get_llm_client()
    usage_before = dict(getattr(usage_client, "total_usage", {}) or {})

    try:
        async for state_chunk in graph.astream(new_state(question, max_retries=max_retries), stream_mode="values"):
            result_state = state_chunk
            trace_so_far = state_chunk.get("trace", [])
            persisted = await _persist_new_trace_events(analysis_id, trace_so_far, persisted)
    except Exception as exc:  # noqa: BLE001 — must always resolve the session row
        error_category = classify_exception(exc)
        logger.error(
            "analysis_failed", analysis_id=str(analysis_id), error_category=error_category, error=str(exc)
        )
        async with async_session_factory() as db:
            session_row = await db.get(AnalysisSession, analysis_id)
            session_row.status = SessionStatus.FAILED
            session_row.error_message = SAFE_MESSAGE_BY_CATEGORY[error_category]
            await db.commit()
        usage_after = getattr(usage_client, "total_usage", None)
        await _persist_execution_metadata(
            analysis_id,
            _build_execution_metadata(
                start_time=start_time,
                end_time=datetime.datetime.utcnow(),
                final_status="FAILED",
                trace=trace_so_far,
                error_category=error_category,
                retry_count=(result_state or {}).get("retry_count", 0),
                token_usage=(
                    {k: usage_after.get(k, 0) - usage_before.get(k, 0) for k in usage_after}
                    if usage_after is not None
                    else None
                ),
                narrative_enabled=get_settings().report_narrative_enabled,
                report_generated=False,
            ),
        )
        return

    async with async_session_factory() as db:
        session_row = await db.get(AnalysisSession, analysis_id)
        session_row.status = SessionStatus.DONE

        report = (result_state or {}).get("report")
        if report is not None:
            db.add(
                AnalysisReport(
                    session_id=analysis_id,
                    executive_summary=report["executive_summary"],
                    key_findings=report["key_findings"],
                    evidence=report["evidence"],
                    recommendations=report["recommendations"],
                    confidence=report["confidence"],
                    limitations=report["limitations"],
                    # Phase 10 (Report Generator) fields — `.get()` defensively:
                    # the out_of_scope path (app/agents/supervisor.py) sets
                    # these to their empty defaults directly since
                    # report_agent_node never runs on that path, but `.get()`
                    # costs nothing and protects against any future path that
                    # forgets to.
                    report_extras={
                        "verified_claims": report.get("verified_claims", []),
                        "analysis_explanation": report.get("analysis_explanation", ""),
                        "visualizations": report.get("visualizations", []),
                        "technical_details": report.get("technical_details", {}),
                        "narrative": report.get("narrative"),
                    },
                )
            )

        for chart in (result_state or {}).get("charts", []):
            db.add(
                Chart(
                    session_id=analysis_id,
                    chart_type=chart["chart_type"],
                    title=chart["title"],
                    storage_path=chart["path"],
                    spec_json=chart["spec_summary"],
                )
            )

        await db.commit()

    usage_after = getattr(usage_client, "total_usage", None)
    await _persist_execution_metadata(
        analysis_id,
        _build_execution_metadata(
            start_time=start_time,
            end_time=datetime.datetime.utcnow(),
            final_status="DONE",
            trace=trace_so_far,
            error_category=None,
            retry_count=(result_state or {}).get("retry_count", 0),
            token_usage=(
                {k: usage_after.get(k, 0) - usage_before.get(k, 0) for k in usage_after}
                if usage_after is not None
                else None
            ),
            narrative_enabled=get_settings().report_narrative_enabled,
            report_generated=report is not None,
        ),
    )


async def get_session(analysis_id: uuid.UUID) -> AnalysisSession | None:
    async with async_session_factory() as db:
        return await db.get(AnalysisSession, analysis_id)


async def get_report(analysis_id: uuid.UUID) -> AnalysisReport | None:
    async with async_session_factory() as db:
        result = await db.execute(
            select(AnalysisReport).where(AnalysisReport.session_id == analysis_id)
        )
        return result.scalar_one_or_none()


async def get_charts(analysis_id: uuid.UUID) -> list[Chart]:
    async with async_session_factory() as db:
        result = await db.execute(select(Chart).where(Chart.session_id == analysis_id))
        return list(result.scalars().all())


async def get_trace(analysis_id: uuid.UUID) -> list[AnalysisStep]:
    async with async_session_factory() as db:
        result = await db.execute(
            select(AnalysisStep)
            .where(AnalysisStep.session_id == analysis_id)
            .order_by(AnalysisStep.step_order)
        )
        return list(result.scalars().all())


async def get_current_stage(analysis_id: uuid.UUID) -> str | None:
    """The most recently COMPLETED node's name (Sec 8 status persistence) —
    not "currently executing": a node's enter+exit are both appended before
    it returns (app/graph/state.py::trace_event), so there's no observable
    in-between state to report, only "last node that finished". `None`
    before the first node has completed (PENDING, or ANALYZING that hasn't
    reached its first `_persist_new_trace_events` write yet) — never
    fabricated, only ever a stage that genuinely ran (see
    app/services/analysis_service.py::run_analysis's incremental persistence).
    """
    async with async_session_factory() as db:
        result = await db.execute(
            select(AnalysisStep)
            .where(AnalysisStep.session_id == analysis_id)
            .order_by(AnalysisStep.step_order.desc())
            .limit(1)
        )
        latest = result.scalar_one_or_none()
        return latest.agent_name if latest is not None else None
