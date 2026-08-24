"""Synthetic analytical data generator (Sec 3 sizing, Sec 12 Phases 1-3).

Deliberately not uniform: bakes in a July 2026 Enterprise/North revenue dip
so the benchmark's diagnostic case (Sec 6, bi-004 — "why did revenue decrease
in July?") has a real, findable answer rather than a hand-picked one. Also
adds plain seasonality (Nov/Dec bump, Jan dip) so trend/forecast questions
have something to bite into.

The benchmark month is a FIXED constant (BENCHMARK_DIP_YEAR/MONTH below), not
derived from datetime.now()/today() — earlier versions computed it as an
offset from the data's end date, which was itself anchored to "today," so
regenerating on a different day silently moved the dip to a different month
(e.g. May 2026 instead of July 2026), quietly invalidating the bi-004
benchmark and the "why did revenue decrease in July?" demo question. The data
window itself is now anchored to the benchmark month instead of the other way
around, so `same code + same seed + same config = same benchmark` regardless
of generation date.

Writes CSVs to data/seeds/ — scripts/seed_database.py loads them into Postgres.
Run standalone: `python scripts/generate_data.py --customers 5000 --months 20`
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

# Single seed for every source of randomness in this module — do not add a
# second, unrelated seed elsewhere; reusing this one is what makes
# `same code + same seed + same config = same benchmark` hold.
SEED = 42
fake = Faker()
Faker.seed(SEED)
RNG = np.random.default_rng(SEED)

REGIONS = ["North", "South", "East", "West", "Central", "Northeast"]
SEGMENTS = ["SMB", "Mid-Market", "Enterprise"]
SEGMENT_WEIGHTS = [0.6, 0.3, 0.1]
CATEGORIES = [
    "Software", "Hardware", "Services", "Support Plans",
    "Training", "Cloud Storage", "Analytics Add-ons", "Integrations",
]
CHANNELS = ["web", "mobile", "sales_rep", "partner"]
PAYMENT_METHODS = ["credit_card", "ach", "wire", "invoice"]
ACTIVITY_TYPES = ["login", "support_ticket", "cart_abandon"]

# Fixed diagnostic-benchmark scenario (Sec 6, bi-004) — deterministic, never
# derived from the current date. Changing these changes the benchmark; do not
# recompute them from datetime.now()/today() or any "months before end" offset.
BENCHMARK_DIP_YEAR = 2026
BENCHMARK_DIP_MONTH = 7
BENCHMARK_DIP_SEGMENT = "Enterprise"
BENCHMARK_DIP_REGION = "North"

# How many months of data to keep *after* the dip month, so trend/recovery
# questions have something to look at — preserves the original generator's
# "dip sits 2 months before the data's end date" shape, just anchored to the
# fixed benchmark month now instead of to today's date.
DIP_TRAILING_MONTHS = 2


def _seasonality_multiplier(month: int) -> float:
    if month in (11, 12):
        return 1.35
    if month == 1:
        return 0.75
    return 1.0


def generate(seed_dir: Path, n_customers: int, months: int, n_products: int) -> None:
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Anchor the whole date window to the fixed benchmark month (not to
    # datetime.now()) — end_date is DIP_TRAILING_MONTHS after the dip month,
    # start_date is `months` months before that. No current-date input here.
    dip_period = pd.Period(year=BENCHMARK_DIP_YEAR, month=BENCHMARK_DIP_MONTH, freq="M")
    end_date = (dip_period + DIP_TRAILING_MONTHS).end_time.date()
    start_date = (end_date.replace(day=1) - pd.DateOffset(months=months - 1)).date()

    dip_year, dip_month = BENCHMARK_DIP_YEAR, BENCHMARK_DIP_MONTH

    # --- regions ---
    regions_df = pd.DataFrame({"region_id": range(1, len(REGIONS) + 1), "name": REGIONS})
    dip_region_id = int(regions_df.loc[regions_df.name == BENCHMARK_DIP_REGION, "region_id"].iloc[0])

    # --- customers ---
    region_ids = RNG.choice(regions_df.region_id, size=n_customers)
    segments = RNG.choice(SEGMENTS, size=n_customers, p=SEGMENT_WEIGHTS)
    signup_days = RNG.integers(0, (end_date - start_date).days, size=n_customers)
    customers_df = pd.DataFrame(
        {
            "customer_id": range(1, n_customers + 1),
            "name": [fake.company() for _ in range(n_customers)],
            "segment": segments,
            "region_id": region_ids,
            "signup_date": [start_date + datetime.timedelta(days=int(d)) for d in signup_days],
            "status": RNG.choice(["active", "active", "active", "churned"], size=n_customers),
        }
    )

    # --- products ---
    products_df = pd.DataFrame(
        {
            "product_id": range(1, n_products + 1),
            "name": [fake.catch_phrase() for _ in range(n_products)],
            "category": RNG.choice(CATEGORIES, size=n_products),
        }
    )
    cost = RNG.uniform(10, 400, size=n_products).round(2)
    products_df["cost"] = cost
    products_df["unit_price"] = (cost * RNG.uniform(1.3, 2.2, size=n_products)).round(2)

    # --- orders + order_items (month by month, with seasonality + the July dip) ---
    orders_rows = []
    order_items_rows = []
    order_id = 1
    order_item_id = 1
    customers_by_id = customers_df.set_index("customer_id")

    months_range = pd.period_range(start=start_date, end=end_date, freq="M")
    base_orders_per_month = max(20, n_customers // 8)

    for period in months_range:
        month_start = period.start_time.date()
        days_in_month = period.days_in_month
        multiplier = _seasonality_multiplier(period.month)
        n_orders_this_month = int(base_orders_per_month * multiplier)

        order_customer_ids = RNG.choice(customers_df.customer_id, size=n_orders_this_month)

        for cust_id in order_customer_ids:
            cust = customers_by_id.loc[cust_id]
            is_dip_case = (
                period.year == dip_year
                and period.month == dip_month
                and cust["segment"] == BENCHMARK_DIP_SEGMENT
                and cust["region_id"] == dip_region_id
            )
            # Enterprise/North orders in the dip month are suppressed ~55% of the time.
            if is_dip_case and RNG.random() < 0.55:
                continue

            order_day = int(RNG.integers(0, days_in_month))
            order_date = month_start + datetime.timedelta(days=order_day)
            orders_rows.append(
                {
                    "order_id": order_id,
                    "customer_id": int(cust_id),
                    "region_id": int(cust["region_id"]),
                    "order_date": order_date,
                    "channel": RNG.choice(CHANNELS),
                    "status": "completed",
                }
            )

            n_items = int(RNG.integers(1, 5))
            item_products = RNG.choice(products_df.product_id, size=n_items, replace=False)
            for pid in item_products:
                unit_price = float(products_df.loc[products_df.product_id == pid, "unit_price"].iloc[0])
                # Enterprise/North dip month: surviving orders also skew smaller.
                qty_high = 2 if is_dip_case else 6
                order_items_rows.append(
                    {
                        "order_item_id": order_item_id,
                        "order_id": order_id,
                        "product_id": int(pid),
                        "quantity": int(RNG.integers(1, qty_high)),
                        "unit_price": unit_price,
                        "discount": round(float(RNG.choice([0, 0, 0, 0.05, 0.1, 0.15])), 3),
                    }
                )
                order_item_id += 1
            order_id += 1

    orders_df = pd.DataFrame(orders_rows)
    order_items_df = pd.DataFrame(order_items_rows)

    # --- payments (settled 0-5 days after order, ~97% of orders) ---
    items_revenue = order_items_df.assign(
        line_total=order_items_df.quantity * order_items_df.unit_price * (1 - order_items_df.discount)
    )
    order_totals = items_revenue.groupby("order_id")["line_total"].sum().reset_index()
    paid_orders = order_totals.sample(frac=0.97, random_state=SEED)
    payments_df = paid_orders.merge(orders_df[["order_id", "order_date"]], on="order_id")
    payments_df["payment_id"] = range(1, len(payments_df) + 1)
    payments_df["amount"] = payments_df["line_total"].round(2)
    payments_df["method"] = RNG.choice(PAYMENT_METHODS, size=len(payments_df))
    lag_days = RNG.integers(0, 6, size=len(payments_df))
    payments_df["paid_at"] = [
        datetime.datetime.combine(d, datetime.time(12, 0)) + datetime.timedelta(days=int(lag))
        for d, lag in zip(payments_df["order_date"], lag_days)
    ]
    payments_df = payments_df[["payment_id", "order_id", "amount", "method", "paid_at"]]

    # --- marketing_campaigns ---
    n_campaigns = len(months_range) // 2 * len(REGIONS)
    campaign_starts = RNG.choice(
        [p.start_time.date() for p in months_range], size=n_campaigns
    )
    campaigns_df = pd.DataFrame(
        {
            "campaign_id": range(1, n_campaigns + 1),
            "name": [f"{fake.bs().title()} Campaign" for _ in range(n_campaigns)],
            "region_id": RNG.choice(regions_df.region_id, size=n_campaigns),
            "start_date": campaign_starts,
            "channel": RNG.choice(CHANNELS, size=n_campaigns),
            "spend": RNG.uniform(1000, 50000, size=n_campaigns).round(2),
        }
    )
    campaigns_df["end_date"] = [d + datetime.timedelta(days=int(RNG.integers(14, 45))) for d in campaigns_df["start_date"]]

    # --- customer_activity (logins/tickets/cart-abandons; dip cohort goes quiet in the dip month) ---
    activity_rows = []
    activity_id = 1
    for cust_id, cust in customers_by_id.iterrows():
        n_events = int(RNG.integers(5, 40))
        for _ in range(n_events):
            offset_days = int(RNG.integers(0, (end_date - start_date).days))
            event_date = start_date + datetime.timedelta(days=offset_days)
            if (
                cust["segment"] == BENCHMARK_DIP_SEGMENT
                and cust["region_id"] == dip_region_id
                and event_date.year == dip_year
                and event_date.month == dip_month
                and RNG.random() < 0.5
            ):
                continue  # activity drop-off mirrors the revenue dip
            activity_rows.append(
                {
                    "activity_id": activity_id,
                    "customer_id": int(cust_id),
                    "activity_date": event_date,
                    "activity_type": RNG.choice(ACTIVITY_TYPES, p=[0.7, 0.15, 0.15]),
                }
            )
            activity_id += 1
    activity_df = pd.DataFrame(activity_rows)

    for name, df in {
        "regions": regions_df,
        "customers": customers_df,
        "products": products_df,
        "orders": orders_df,
        "order_items": order_items_df,
        "payments": payments_df,
        "marketing_campaigns": campaigns_df,
        "customer_activity": activity_df,
    }.items():
        df.to_csv(seed_dir / f"{name}.csv", index=False)
        print(f"wrote {len(df):>7} rows -> {name}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--customers", type=int, default=5000)
    parser.add_argument("--months", type=int, default=20)
    parser.add_argument("--products", type=int, default=120)
    parser.add_argument("--out", type=str, default="data/seeds")
    args = parser.parse_args()

    generate(Path(args.out), args.customers, args.months, args.products)
