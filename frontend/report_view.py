"""Pure report-section builder (Phase 12) — no `streamlit` import. Turns a
`ReportResponse` dict (app/schemas/reports.py, served by
GET /analysis/{id}/report) into an ordered list of sections to display.
app.py/components iterate this and call `st.*` to actually draw it — this
module only decides WHAT to show and in what order, never HOW, which is
what makes "narrative omitted when null", "confidence never upgraded",
"limitations never dropped" testable without mocking Streamlit widgets.

Every field is read verbatim from the given dict — this module performs no
calculation, no chart selection, and no rewriting of any value (Sec
"Phase 6/7/9/10" — the frontend displays what the API already computed and
the Critic already validated, it does not recompute or restate it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

SectionKind = Literal["text", "list", "warning", "json"]


@dataclass(frozen=True)
class ReportSection:
    title: str
    kind: SectionKind
    content: Any


def build_report_sections(report: dict[str, Any]) -> list[ReportSection]:
    """Sections that have nothing to show (empty list, blank string, null)
    are simply omitted — never rendered as an empty heading, and never
    replaced with placeholder/invented text (Sec "Do not invent missing
    values")."""
    sections: list[ReportSection] = []

    sections.append(ReportSection("Executive Summary", "text", report.get("executive_summary") or ""))

    narrative = report.get("narrative")
    if narrative:
        sections.append(ReportSection("Narrative", "text", narrative))

    key_findings = list(report.get("key_findings") or [])
    if key_findings:
        sections.append(ReportSection("Key Findings", "list", key_findings))

    evidence = report.get("evidence") or []
    if evidence:
        sections.append(ReportSection("Evidence", "json", evidence))

    verified_claims = list(report.get("verified_claims") or [])
    if verified_claims:
        sections.append(ReportSection("Verified Claims", "list", verified_claims))

    analysis_explanation = report.get("analysis_explanation")
    if analysis_explanation:
        sections.append(ReportSection("Analysis", "text", analysis_explanation))

    # Confidence is always shown, verbatim, exactly as the Critic/Report
    # Generator left it — never upgraded, downgraded, or reworded here.
    sections.append(ReportSection("Confidence", "text", report.get("confidence") or ""))

    limitations = report.get("limitations")
    if limitations:
        sections.append(ReportSection("Limitations", "warning", limitations))

    recommendations = list(report.get("recommendations") or [])
    if recommendations:
        sections.append(ReportSection("Recommendations", "list", recommendations))

    technical_details = report.get("technical_details") or {}
    if technical_details:
        sections.append(ReportSection("Technical Details", "json", technical_details))

    return sections
