"""Schema introspection across every schema the SQL Agent is allowed to read.

Two schemas as of Phase 3 (see docs/architecture.md "Olist integration"):

- `analytics` — synthetic B2B SaaS data (regions/segments/customers/orders/...).
  Controlled ground truth: the July Enterprise/North revenue dip is baked in
  on purpose so diagnostic questions have a known, checkable answer (Sec 6
  eval case bi-004). Keep using this for anything eval-related.
- `olist`     — real Brazilian e-commerce marketplace data (2016-09 to
  2018-10), loaded as-is from Kaggle's Olist dataset via scripts/load_olist.py.
  For realistic, messier business-analysis demos.

Both schemas can have tables with the same name (both have a `customers`,
an `orders`, ...) with *different* columns, so every consumer of this module
keys by `"schema.table"`, never by bare table name — this is also why
database_tools.py now requires SQL to schema-qualify every table reference.

Feeds two consumers:
1. The Supervisor/SQL Agent's prompt (Sec 0: static per session, cache it).
2. app/tools/database_tools.py's Layer 3 allow-list check (Sec 4) — the
   *enforcement* copy, fetched independently of whatever the agent claims.

Never trust an agent-supplied schema description for the allow-list check;
always re-introspect live.
"""

from __future__ import annotations

from app.db.database import analytics_readonly_connection

# The only schemas readonly_analyst has USAGE+SELECT on (migrations
# 0001_initial, 0002_olist_schema). Adding a schema here without also
# granting it in a migration just means queries against it get rejected at
# Layer 3 before they ever reach Postgres — safe failure mode either way.
ALLOWED_SCHEMAS: tuple[str, ...] = ("analytics", "olist")

SCHEMA_DESCRIPTIONS: dict[str, str] = {
    "analytics": "Synthetic B2B SaaS dataset (regions, segments, campaigns, "
    "customer activity) — controlled ground truth, includes a known July "
    "Enterprise/North revenue dip for diagnostic questions.",
    "olist": "Real Brazilian e-commerce marketplace orders, Sep 2016 - Oct "
    "2018 (Kaggle's Olist dataset). No 'region' or 'segment' columns — use "
    "customer_state/customer_city directly. No marketing or activity data.",
}

_INTROSPECTION_QUERY = """
    SELECT table_schema, table_name, column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = ANY($1::text[])
    ORDER BY table_schema, table_name, ordinal_position
"""


async def get_analytics_schema() -> dict[str, dict[str, str]]:
    """{"schema.table": {column_name: data_type}} across every allowed schema."""
    async with analytics_readonly_connection() as conn:
        records = await conn.fetch(_INTROSPECTION_QUERY, list(ALLOWED_SCHEMAS))

    schema: dict[str, dict[str, str]] = {}
    for row in records:
        key = f"{row['table_schema']}.{row['table_name']}"
        schema.setdefault(key, {})[row["column_name"]] = row["data_type"]
    return schema


def format_schema_for_prompt(schema: dict[str, dict[str, str]], *, only_schema: str | None = None) -> str:
    """Compact `schema.table(col type, col type, ...)` listing for an agent's
    system prompt. Pass `only_schema` to keep the prompt small once the
    Supervisor has already decided which dataset a question targets.
    """
    lines = []
    for description_schema, description in SCHEMA_DESCRIPTIONS.items():
        if only_schema and description_schema != only_schema:
            continue
        lines.append(f"# {description_schema}: {description}")
        for qualified_table, columns in sorted(schema.items()):
            if not qualified_table.startswith(f"{description_schema}."):
                continue
            col_list = ", ".join(f"{name} {dtype}" for name, dtype in columns.items())
            lines.append(f"{qualified_table}({col_list})")
        lines.append("")
    return "\n".join(lines).strip()
