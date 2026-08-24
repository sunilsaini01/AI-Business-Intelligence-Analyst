"""Loads data/seeds/*.csv into the analytics schema via asyncpg COPY (fast bulk load).

Run `python scripts/generate_data.py` first if data/seeds/ is empty.
Usage: `python scripts/seed_database.py` (uses ANALYTICS_DATABASE_URL's *app*
counterpart — seeding needs write access, so it connects with DATABASE_URL,
not the readonly role).
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

SEED_DIR = Path(__file__).resolve().parent.parent / "data" / "seeds"

# csv.DictReader hands back strings for everything — cast to the type asyncpg's
# binary COPY protocol expects, keyed by column name (unambiguous across tables).
_INT_COLUMNS = {
    "region_id", "customer_id", "order_id", "product_id", "order_item_id",
    "quantity", "payment_id", "campaign_id", "activity_id",
}
_FLOAT_COLUMNS = {"unit_price", "cost", "discount", "amount", "spend"}
_DATE_COLUMNS = {"signup_date", "order_date", "start_date", "end_date", "activity_date"}
_DATETIME_COLUMNS = {"paid_at"}


def _caster(column: str) -> Callable[[str], Any]:
    if column in _INT_COLUMNS:
        return int
    if column in _FLOAT_COLUMNS:
        return float
    if column in _DATE_COLUMNS:
        return datetime.date.fromisoformat
    if column in _DATETIME_COLUMNS:
        return datetime.datetime.fromisoformat
    return str

# FK-safe load order.
TABLES = [
    ("regions", ["region_id", "name"]),
    ("customers", ["customer_id", "name", "segment", "region_id", "signup_date", "status"]),
    ("products", ["product_id", "name", "category", "unit_price", "cost"]),
    ("orders", ["order_id", "customer_id", "region_id", "order_date", "channel", "status"]),
    ("order_items", ["order_item_id", "order_id", "product_id", "quantity", "unit_price", "discount"]),
    ("payments", ["payment_id", "order_id", "amount", "method", "paid_at"]),
    ("marketing_campaigns", ["campaign_id", "name", "region_id", "start_date", "end_date", "spend", "channel"]),
    ("customer_activity", ["activity_id", "customer_id", "activity_date", "activity_type"]),
]


def _to_app_role_url(analytics_url: str, database_url: str) -> str:
    """asyncpg needs a plain postgresql:// DSN; DATABASE_URL is postgresql+asyncpg://."""
    return database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


async def seed() -> None:
    settings = get_settings()
    dsn = _to_app_role_url(settings.analytics_database_url, settings.database_url)
    conn = await asyncpg.connect(dsn=dsn)

    try:
        await conn.execute(
            "TRUNCATE analytics.customer_activity, analytics.marketing_campaigns, "
            "analytics.payments, analytics.order_items, analytics.orders, "
            "analytics.products, analytics.customers, analytics.regions RESTART IDENTITY CASCADE"
        )

        for table, columns in TABLES:
            csv_path = SEED_DIR / f"{table}.csv"
            if not csv_path.exists():
                print(f"skip {table}: {csv_path} not found (run generate_data.py first)")
                continue

            casters = [_caster(c) for c in columns]
            with csv_path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                records = [tuple(cast(row[c]) for c, cast in zip(columns, casters)) for row in reader]

            if not records:
                continue

            await conn.copy_records_to_table(
                table, schema_name="analytics", columns=columns, records=records
            )
            print(f"loaded {len(records):>7} rows -> analytics.{table}")

        # Sync sequences after COPYing explicit ids.
        for table, columns in TABLES:
            pk = columns[0]
            await conn.execute(
                f"SELECT setval(pg_get_serial_sequence('analytics.{table}', '{pk}'), "
                f"COALESCE((SELECT MAX({pk}) FROM analytics.{table}), 1))"
            )
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(seed())
