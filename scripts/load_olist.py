"""Loads the raw Olist CSVs (data/raw/*.csv, downloaded from Kaggle by hand —
not committed, see .gitignore) into the `olist` schema via asyncpg COPY.

Faithful load: no fabricated columns, no forcing into the `analytics` shape
(see app/db/models_olist.py, docs/architecture.md). Empty CSV fields become
SQL NULL wherever the column is nullable.

Usage: `python scripts/load_olist.py` (connects with the app role — DATABASE_URL
— since loading needs write access; the readonly_analyst role never writes).
"""

from __future__ import annotations

import asyncio
import csv
import datetime
import sys
from pathlib import Path
from typing import Any, Callable

import asyncpg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def _int(v: str) -> int | None:
    return int(v) if v != "" else None


def _float(v: str) -> float | None:
    return float(v) if v != "" else None


def _str_or_none(v: str) -> str | None:
    return v if v != "" else None


def _ts(v: str) -> datetime.datetime | None:
    return datetime.datetime.fromisoformat(v) if v != "" else None


# column name -> caster. Column names are unique-enough across the Olist
# files that one global map is unambiguous; anything not listed defaults to
# `str` (never empty in this dataset's non-nullable text columns).
_CASTERS: dict[str, Callable[[str], Any]] = {
    "customer_zip_code_prefix": _int,
    "seller_zip_code_prefix": _int,
    "geolocation_zip_code_prefix": _int,
    "geolocation_lat": _float,
    "geolocation_lng": _float,
    "order_item_id": _int,
    "price": _float,
    "freight_value": _float,
    "payment_sequential": _int,
    "payment_installments": _int,
    "payment_value": _float,
    "review_score": _int,
    "product_category_name": _str_or_none,
    "product_name_lenght": _int,
    "product_description_lenght": _int,
    "product_photos_qty": _int,
    "product_weight_g": _int,
    "product_length_cm": _int,
    "product_height_cm": _int,
    "product_width_cm": _int,
    "review_comment_title": _str_or_none,
    "review_comment_message": _str_or_none,
    "order_purchase_timestamp": _ts,
    "order_approved_at": _ts,
    "order_delivered_carrier_date": _ts,
    "order_delivered_customer_date": _ts,
    "order_estimated_delivery_date": _ts,
    "shipping_limit_date": _ts,
    "review_creation_date": _ts,
    "review_answer_timestamp": _ts,
}


def _caster(column: str) -> Callable[[str], Any]:
    return _CASTERS.get(column, str)


# (csv filename, table name, [csv columns to load, in DB column order])
# order_reviews/geolocation have a serial surrogate `id` not present in the
# CSV — omitted from `columns` so Postgres assigns it.
_LOAD_PLAN: list[tuple[str, str, list[str]]] = [
    ("olist_customers_dataset.csv", "customers", [
        "customer_id", "customer_unique_id", "customer_zip_code_prefix",
        "customer_city", "customer_state",
    ]),
    ("olist_sellers_dataset.csv", "sellers", [
        "seller_id", "seller_zip_code_prefix", "seller_city", "seller_state",
    ]),
    ("product_category_name_translation.csv", "product_category_name_translation", [
        "product_category_name", "product_category_name_english",
    ]),
    ("olist_products_dataset.csv", "products", [
        "product_id", "product_category_name", "product_name_lenght",
        "product_description_lenght", "product_photos_qty", "product_weight_g",
        "product_length_cm", "product_height_cm", "product_width_cm",
    ]),
    ("olist_orders_dataset.csv", "orders", [
        "order_id", "customer_id", "order_status", "order_purchase_timestamp",
        "order_approved_at", "order_delivered_carrier_date",
        "order_delivered_customer_date", "order_estimated_delivery_date",
    ]),
    ("olist_order_items_dataset.csv", "order_items", [
        "order_id", "order_item_id", "product_id", "seller_id",
        "shipping_limit_date", "price", "freight_value",
    ]),
    ("olist_order_payments_dataset.csv", "order_payments", [
        "order_id", "payment_sequential", "payment_type",
        "payment_installments", "payment_value",
    ]),
    ("olist_order_reviews_dataset.csv", "order_reviews", [
        "review_id", "order_id", "review_score", "review_comment_title",
        "review_comment_message", "review_creation_date", "review_answer_timestamp",
    ]),
    ("olist_geolocation_dataset.csv", "geolocation", [
        "geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng",
        "geolocation_city", "geolocation_state",
    ]),
]

_TRUNCATE_ORDER = [
    "order_reviews", "order_payments", "order_items", "orders",
    "products", "product_category_name_translation", "geolocation",
    "sellers", "customers",
]


def _to_app_role_dsn(database_url: str) -> str:
    return database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


async def load() -> None:
    settings = get_settings()
    conn = await asyncpg.connect(dsn=_to_app_role_dsn(settings.database_url))

    try:
        qualified = ", ".join(f"olist.{t}" for t in _TRUNCATE_ORDER)
        await conn.execute(f"TRUNCATE {qualified} RESTART IDENTITY CASCADE")

        for csv_filename, table, columns in _LOAD_PLAN:
            csv_path = RAW_DIR / csv_filename
            if not csv_path.exists():
                print(f"skip {table}: {csv_path} not found")
                continue

            casters = [_caster(c) for c in columns]
            with csv_path.open(newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                records = [
                    tuple(cast(row[c]) for c, cast in zip(columns, casters))
                    for row in reader
                ]

            if not records:
                continue

            await conn.copy_records_to_table(
                table, schema_name="olist", columns=columns, records=records
            )
            print(f"loaded {len(records):>8} rows -> olist.{table}")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(load())
