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

from alembic import op

revision: str = "0003_report_extras"
down_revision: Union[str, None] = "0002_olist_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Raw SQL with IF NOT EXISTS, not op.add_column: migration 0001's
    # Base.metadata.create_all(bind=bind) reads the CURRENT app/db/models.py,
    # which has carried this column permanently since it was added there —
    # so a database migrated from scratch already has it by the time this
    # migration runs, and a plain add_column duplicate-column errors. A
    # database that went through 0001 before this column existed in the
    # model (every real deployment so far) still gets it added here exactly
    # as before — this only changes behavior for the from-scratch replay
    # case, a real defect found via the Phase 14 migration audit.
    op.execute(
        "ALTER TABLE app.analysis_reports ADD COLUMN IF NOT EXISTS report_extras JSONB NOT NULL DEFAULT '{}'"
    )


def downgrade() -> None:
    op.drop_column("analysis_reports", "report_extras", schema="app")
