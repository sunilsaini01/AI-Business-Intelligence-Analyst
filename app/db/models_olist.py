"""Olist schema (real Brazilian e-commerce marketplace data, Kaggle).

Architecture decision — see docs/architecture.md "Olist integration" for the
full reasoning. Short version: this is a *separate* schema from `analytics`,
not a mapping into it, because:

- Olist has no `region`/`segment` concept (only customer_state/city — a real
  Brazilian state code, not a fabricated business region) and no marketing
  or activity data at all. Force-fitting it into the synthetic schema's
  shape would mean inventing columns the source data doesn't have, which
  breaks the "never fabricate data" rule.
- `customer_id` in Olist is *order-scoped*, not person-scoped — the same
  human can have several `customer_id`s (one per order) but a stable
  `customer_unique_id`. That's a real, load-bearing distinction the
  synthetic `customers` table's shape doesn't have room for.
- Column names/types here mirror the source CSVs as-is. No FK constraints:
  this is externally-sourced data of unverified referential cleanliness,
  and the DB role is read-only anyway — indexes on the join columns cover
  the query patterns without asserting integrity we haven't verified.

Loaded by scripts/load_olist.py from data/raw/*.csv. The existing synthetic
`analytics` schema (app/db/models.py) is untouched and stays the eval
ground-truth dataset (app/tools/schema_tools.py::SCHEMA_DESCRIPTIONS).
"""

from __future__ import annotations

import datetime

from sqlalchemy import Index, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.models import Base


class OlistCustomer(Base):
    __tablename__ = "customers"
    __table_args__ = ({"schema": "olist"},)

    customer_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    customer_unique_id: Mapped[str] = mapped_column(String(32), index=True)
    customer_zip_code_prefix: Mapped[int]
    customer_city: Mapped[str] = mapped_column(String(100))
    customer_state: Mapped[str] = mapped_column(String(2), index=True)


class OlistSeller(Base):
    __tablename__ = "sellers"
    __table_args__ = ({"schema": "olist"},)

    seller_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    seller_zip_code_prefix: Mapped[int]
    seller_city: Mapped[str] = mapped_column(String(100))
    seller_state: Mapped[str] = mapped_column(String(2), index=True)


class OlistProductCategoryTranslation(Base):
    __tablename__ = "product_category_name_translation"
    __table_args__ = ({"schema": "olist"},)

    product_category_name: Mapped[str] = mapped_column(String(100), primary_key=True)
    product_category_name_english: Mapped[str] = mapped_column(String(100))


class OlistProduct(Base):
    __tablename__ = "products"
    __table_args__ = ({"schema": "olist"},)

    product_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    product_category_name: Mapped[str | None] = mapped_column(String(100), index=True, nullable=True)
    product_name_lenght: Mapped[int | None]
    product_description_lenght: Mapped[int | None]
    product_photos_qty: Mapped[int | None]
    product_weight_g: Mapped[int | None]
    product_length_cm: Mapped[int | None]
    product_height_cm: Mapped[int | None]
    product_width_cm: Mapped[int | None]


class OlistOrder(Base):
    __tablename__ = "orders"
    __table_args__ = (
        Index("ix_olist_orders_customer_id", "customer_id"),
        Index("ix_olist_orders_purchase_ts", "order_purchase_timestamp"),
        {"schema": "olist"},
    )

    order_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    customer_id: Mapped[str] = mapped_column(String(32))
    order_status: Mapped[str] = mapped_column(String(20), index=True)
    order_purchase_timestamp: Mapped[datetime.datetime]
    order_approved_at: Mapped[datetime.datetime | None]
    order_delivered_carrier_date: Mapped[datetime.datetime | None]
    order_delivered_customer_date: Mapped[datetime.datetime | None]
    order_estimated_delivery_date: Mapped[datetime.datetime]


class OlistOrderItem(Base):
    """No `quantity`/`discount` columns — unlike analytics.order_items, Olist
    lists one row per physical unit (order_item_id increments per unit), so
    revenue for a line is just `price`, not `price * quantity`."""

    __tablename__ = "order_items"
    __table_args__ = (
        Index("ix_olist_order_items_product_id", "product_id"),
        Index("ix_olist_order_items_seller_id", "seller_id"),
        {"schema": "olist"},
    )

    order_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    order_item_id: Mapped[int] = mapped_column(primary_key=True)
    product_id: Mapped[str] = mapped_column(String(32))
    seller_id: Mapped[str] = mapped_column(String(32))
    shipping_limit_date: Mapped[datetime.datetime]
    price: Mapped[float] = mapped_column(Numeric(10, 2))
    freight_value: Mapped[float] = mapped_column(Numeric(10, 2))


class OlistOrderPayment(Base):
    __tablename__ = "order_payments"
    __table_args__ = (
        Index("ix_olist_order_payments_order_id", "order_id"),
        {"schema": "olist"},
    )

    order_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    payment_sequential: Mapped[int] = mapped_column(primary_key=True)
    payment_type: Mapped[str] = mapped_column(String(20))
    payment_installments: Mapped[int]
    payment_value: Mapped[float] = mapped_column(Numeric(10, 2))


class OlistOrderReview(Base):
    """`review_id` is not unique in the source data (duplicates exist) — a
    synthetic surrogate key backs this table instead."""

    __tablename__ = "order_reviews"
    __table_args__ = (
        Index("ix_olist_order_reviews_order_id", "order_id"),
        {"schema": "olist"},
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    review_id: Mapped[str] = mapped_column(String(32), index=True)
    order_id: Mapped[str] = mapped_column(String(32))
    review_score: Mapped[int]
    review_comment_title: Mapped[str | None] = mapped_column(String(200), nullable=True)
    review_comment_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    review_creation_date: Mapped[datetime.datetime]
    review_answer_timestamp: Mapped[datetime.datetime]


class OlistGeolocation(Base):
    """No natural key — the source has many duplicate rows per zip prefix
    (multiple lat/lng samples). Surrogate key + index on the zip prefix,
    which is the actual join column back to customers/sellers."""

    __tablename__ = "geolocation"
    __table_args__ = (
        Index("ix_olist_geolocation_zip", "geolocation_zip_code_prefix"),
        {"schema": "olist"},
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    geolocation_zip_code_prefix: Mapped[int]
    geolocation_lat: Mapped[float]
    geolocation_lng: Mapped[float]
    geolocation_city: Mapped[str] = mapped_column(String(100))
    geolocation_state: Mapped[str] = mapped_column(String(2))
