"""Cross-session report listing — distinct from the single-session report
endpoint in analysis.py (`/analysis/{id}/report`). Useful for a frontend
"recent analyses" panel once one exists.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select

from app.core.auth import get_current_user
from app.db.database import async_session_factory
from app.db.models import AnalysisReport, AnalysisSession, User
from app.schemas.reports import ReportResponse

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("", response_model=list[ReportResponse])
async def list_reports(limit: int = 20, current_user: User = Depends(get_current_user)) -> list[ReportResponse]:
    """Phase 14: strictly the caller's own reports — unlike the single-
    session endpoints (app/api/routes/analysis.py), this is a listing, not
    an id-addressed lookup, so there's no equivalent "ownerless row" case to
    special-case: pre-Phase-14 rows have no owner and simply never appear
    here for anyone (they remain reachable directly by id, per
    _check_ownership's documented allowance)."""
    async with async_session_factory() as db:
        result = await db.execute(
            select(AnalysisReport, AnalysisSession.id)
            .join(AnalysisSession, AnalysisReport.session_id == AnalysisSession.id)
            .where(AnalysisSession.user_id == current_user.id)
            .order_by(AnalysisReport.created_at.desc())
            .limit(limit)
        )
        rows = result.all()
        return [
            ReportResponse(
                analysis_id=session_id,
                executive_summary=report.executive_summary,
                key_findings=report.key_findings,
                evidence=report.evidence,
                recommendations=report.recommendations,
                confidence=report.confidence,
                limitations=report.limitations,
                verified_claims=(report.report_extras or {}).get("verified_claims", []),
                analysis_explanation=(report.report_extras or {}).get("analysis_explanation", ""),
                ml_summary=(report.report_extras or {}).get("ml_summary", ""),
                ml_results=(report.report_extras or {}).get("ml_results"),
                visualizations=(report.report_extras or {}).get("visualizations", []),
                technical_details=(report.report_extras or {}).get("technical_details", {}),
                narrative=(report.report_extras or {}).get("narrative"),
            )
            for report, session_id in rows
        ]
