"""Phase 13 (Objective D — structured execution observability): one
additive JSONB column on app.analysis_sessions holding a per-run metadata
bundle (start/end time, duration, completed_nodes, failed_node,
error_category, retry_count, token_usage, narrative_enabled,
report_generated — see app/services/analysis_service.py::
_build_execution_metadata). Same additive-JSONB-column shape as migration
0003_report_extras, applied to a different table. Nullable-safe default of
'{}' so every existing row reads back as an empty dict, not NULL.

No change to the readonly_analyst role or the analytics/olist schemas —
this only touches the app-owned `app.analysis_sessions` table.

Revision ID: 0004_execution_metadata
Revises: 0003_report_extras
Create Date: 2026-08-26
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0004_execution_metadata"
down_revision: Union[str, None] = "0003_report_extras"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "analysis_sessions",
        sa.Column("execution_metadata", JSONB, nullable=False, server_default="{}"),
        schema="app",
    )


def downgrade() -> None:
    op.drop_column("analysis_sessions", "execution_metadata", schema="app")
