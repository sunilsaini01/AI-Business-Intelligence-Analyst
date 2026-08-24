from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel


class EvidenceItem(BaseModel):
    source: str
    row_count: int | None = None


class ReportResponse(BaseModel):
    analysis_id: uuid.UUID
    executive_summary: str
    key_findings: list[str]
    evidence: list[dict[str, Any]]
    recommendations: list[str]
    confidence: Literal["Low", "Medium", "High"]
    limitations: str
    # Phase 10 (Report Generator) — presentation-only additions, sourced
    # from AnalysisReport.report_extras. Defaulted so a report persisted
    # before this migration (report_extras == {}) still serializes cleanly.
    verified_claims: list[str] = []
    analysis_explanation: str = ""
    visualizations: list[dict[str, Any]] = []
    technical_details: dict[str, Any] = {}
    narrative: str | None = None


class ChartOut(BaseModel):
    chart_type: str
    title: str
    storage_path: str
    spec_json: dict[str, Any]
