"""Report Generator (Phase 10): one additive JSONB column on
app.analysis_reports holding the 5 presentation-only fields the Report
Generator adds (verified_claims, analysis_explanation, visualizations,
technical_details, narrative) — see app/graph/state.py::BusinessReport and
app/agents/report_agent.py. Nullable-safe default of '{}' so every existing
row (from before this migration) reads back as an empty dict, not NULL.

No change to the readonly_analyst role or the analytics/olist schemas —
this only touches the app-owned `app.analysis_reports` table.

Revision ID: 0003_report_extras
Revises: 0002_olist_schema
Create Date: 2026-08-25
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0003_report_extras"
down_revision: Union[str, None] = "0002_olist_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "analysis_reports",
        sa.Column("report_extras", JSONB, nullable=False, server_default="{}"),
        schema="app",
    )


def downgrade() -> None:
    op.drop_column("analysis_reports", "report_extras", schema="app")
