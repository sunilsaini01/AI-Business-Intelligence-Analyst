"""Olist schema: real Brazilian e-commerce data, separate from `analytics`
(see app/db/models_olist.py and docs/architecture.md for why). Extends the
Sec 4 read-only role rather than introducing a new one — same security
boundary, one more schema it's allowed to SELECT from.

Revision ID: 0002_olist_schema
Revises: 0001_initial
Create Date: 2026-08-23
"""

from typing import Sequence, Union

from alembic import op

from app.db.models import Base
from app.db.models_olist import (  # noqa: F401 — import registers the tables on Base.metadata
    OlistCustomer,
    OlistGeolocation,
    OlistOrder,
    OlistOrderItem,
    OlistOrderPayment,
    OlistOrderReview,
    OlistProduct,
    OlistProductCategoryTranslation,
    OlistSeller,
)

revision: str = "0002_olist_schema"
down_revision: Union[str, None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_OLIST_TABLES = [
    "customers",
    "sellers",
    "product_category_name_translation",
    "products",
    "orders",
    "order_items",
    "order_payments",
    "order_reviews",
    "geolocation",
]


def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS olist")

    bind = op.get_bind()
    olist_tables = [
        Base.metadata.tables[f"olist.{name}"] for name in _OLIST_TABLES
    ]
    Base.metadata.create_all(bind=bind, tables=olist_tables)

    op.execute("GRANT USAGE ON SCHEMA olist TO readonly_analyst")
    op.execute("GRANT SELECT ON ALL TABLES IN SCHEMA olist TO readonly_analyst")
    op.execute(
        "ALTER DEFAULT PRIVILEGES IN SCHEMA olist GRANT SELECT ON TABLES TO readonly_analyst"
    )


def downgrade() -> None:
    op.execute("REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA olist FROM readonly_analyst")
    op.execute("DROP SCHEMA IF EXISTS olist CASCADE")
