from __future__ import annotations

import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.db.models import SessionStatus
from app.schemas.analysis import (
    AnalysisDetailResponse,
    AnalysisStatusResponse,
    AnalyzeAccepted,
    AnalyzeRequest,
    TraceEventOut,
)
from app.schemas.reports import ChartOut, ReportResponse
from app.services import analysis_service

router = APIRouter(prefix="/analysis", tags=["analysis"])
analyze_router = APIRouter(tags=["analysis"])


@analyze_router.post("/analyze", response_model=AnalyzeAccepted, status_code=202)
async def analyze(request: AnalyzeRequest, background_tasks: BackgroundTasks) -> AnalyzeAccepted:
    analysis_id = await analysis_service.create_session(request.question)
    background_tasks.add_task(analysis_service.run_analysis, analysis_id, request.question)
    return AnalyzeAccepted(analysis_id=analysis_id)


@router.get("/{analysis_id}/status", response_model=AnalysisStatusResponse)
async def get_status(analysis_id: uuid.UUID) -> AnalysisStatusResponse:
    session = await analysis_service.get_session(analysis_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    current_stage = await analysis_service.get_current_stage(analysis_id)
    return AnalysisStatusResponse(
        analysis_id=session.id,
        status=session.status,
        current_stage=current_stage,
        error_message=session.error_message,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


@router.get("/{analysis_id}/report", response_model=ReportResponse)
async def get_report(analysis_id: uuid.UUID) -> ReportResponse:
    session = await analysis_service.get_session(analysis_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    if session.status != SessionStatus.DONE:
        raise HTTPException(status_code=409, detail=f"Analysis not ready (status={session.status})")

    report = await analysis_service.get_report(analysis_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")

    extras = report.report_extras or {}
    return ReportResponse(
        analysis_id=analysis_id,
        executive_summary=report.executive_summary,
        key_findings=report.key_findings,
        evidence=report.evidence,
        recommendations=report.recommendations,
        confidence=report.confidence,
        limitations=report.limitations,
        verified_claims=extras.get("verified_claims", []),
        analysis_explanation=extras.get("analysis_explanation", ""),
        visualizations=extras.get("visualizations", []),
        technical_details=extras.get("technical_details", {}),
        narrative=extras.get("narrative"),
    )


@router.get("/{analysis_id}/charts", response_model=list[ChartOut])
async def get_charts(analysis_id: uuid.UUID) -> list[ChartOut]:
    session = await analysis_service.get_session(analysis_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    charts = await analysis_service.get_charts(analysis_id)
    return [
        ChartOut(
            chart_type=c.chart_type,
            title=c.title,
            storage_path=c.storage_path,
            spec_json=c.spec_json,
        )
        for c in charts
    ]


@router.get("/{analysis_id}", response_model=AnalysisDetailResponse)
async def get_detail(analysis_id: uuid.UUID) -> AnalysisDetailResponse:
    session = await analysis_service.get_session(analysis_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    trace = await analysis_service.get_trace(analysis_id)
    current_stage = trace[-1].agent_name if trace else None
    return AnalysisDetailResponse(
        analysis_id=session.id,
        status=session.status,
        current_stage=current_stage,
        error_message=session.error_message,
        created_at=session.created_at,
        updated_at=session.updated_at,
        question=session.question,
        trace=[
            TraceEventOut(
                node=t.agent_name,
                event=t.status,
                timestamp=t.created_at.isoformat(),
                duration_ms=t.duration_ms,
            )
            for t in trace
        ],
        execution_metadata=session.execution_metadata or {},
    )
