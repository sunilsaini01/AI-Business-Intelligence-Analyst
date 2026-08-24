"""Sec 4: the SQL safety pipeline. Every layer here can fail open only to
"reject and retry" — never to "execute anyway" (Fig. 4). The DB role grant
(migrations 0001_initial, 0002_olist_schema — readonly_analyst has SELECT on
analytics.* and olist.* and nothing else) is the layer that has to hold even
if everything below is buggy; this module is fail-fast-with-a-good-error, not
the last line of defense.

Phase 3 generalizes this from one schema to N (ALLOWED_SCHEMAS in
schema_tools.py) — every table reference must now be schema-qualified
(`analytics.customers`, not bare `customers`), because two schemas can share
a table name with different columns. That's a stricter allow-list than
before, not a weaker one.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

import asyncpg
import sqlglot
from sqlglot import exp

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.database import analytics_readonly_connection
from app.tools.schema_tools import ALLOWED_SCHEMAS, get_analytics_schema

logger = get_logger(__name__)

# Layer 2: node types that must never appear, anywhere in the tree — including
# nested inside a CTE. A single disallowed node anywhere rejects the whole query.
_DISALLOWED_NODE_TYPES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Alter,
    exp.Create,
    exp.TruncateTable,
    exp.Copy,
    exp.Command,  # catches COPY/VACUUM/etc. sqlglot doesn't model explicitly
    exp.Into,  # SELECT ... INTO — writes a new table
    exp.Grant,
)

# Layer 2: function calls that are read-only in name but not in effect.
_DISALLOWED_FUNCTIONS = {"pg_sleep", "dblink", "dblink_connect", "pg_read_file", "lo_import", "lo_export"}


@dataclass
class ValidationResult:
    ok: bool
    sql: str  # rewritten SQL (LIMIT injected/clamped) if ok, else the original
    rejection_reason: str | None = None
    tables_referenced: list[str] = field(default_factory=list)  # "schema.table" form


def _reject(original_sql: str, reason: str) -> ValidationResult:
    # Layer 6 audit trail: rejections matter as much as executions — this is
    # the record that a retry actually happened and why, not just that a
    # query eventually succeeded.
    logger.info(
        "sql_rejected",
        reason=reason,
        query_hash=hashlib.sha256(original_sql.encode()).hexdigest()[:16],
    )
    return ValidationResult(ok=False, sql=original_sql, rejection_reason=reason)


async def validate_sql(raw_sql: str) -> ValidationResult:
    """Layers 2-4: AST validation, schema allow-list, LIMIT cap. Read-only,
    no DB connection needed except to fetch the current allow-list.
    """
    settings = get_settings()

    try:
        statements = sqlglot.parse(raw_sql, dialect="postgres")
    except Exception as e:
        return _reject(raw_sql, f"SQL failed to parse: {e}")

    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        return _reject(raw_sql, f"Expected exactly one statement, got {len(statements)}.")

    tree = statements[0]
    if not isinstance(tree, exp.Select):
        return _reject(raw_sql, f"Only SELECT statements are allowed, got {type(tree).__name__}.")

    for node in tree.walk():
        node_expr = node[0] if isinstance(node, tuple) else node
        if isinstance(node_expr, _DISALLOWED_NODE_TYPES):
            return _reject(raw_sql, f"Disallowed statement type in query: {type(node_expr).__name__}.")
        if isinstance(node_expr, exp.Anonymous) and node_expr.name.lower() in _DISALLOWED_FUNCTIONS:
            return _reject(raw_sql, f"Disallowed function call: {node_expr.name}.")

    # Layer 3: schema allow-list — re-introspected live, never trusted from the agent.
    allowed_schema = await get_analytics_schema()

    # CTE names (`WITH monthly AS (...)`) are local aliases, not real tables —
    # referencing them unqualified later in the query is legitimate. The CTE's
    # own body is still walked by this same loop, so every *real* table it
    # selects from still goes through the qualification + allow-list check.
    cte_names = {cte.alias_or_name for cte in tree.find_all(exp.CTE)}

    qualified_tables_referenced: list[str] = []
    for table in tree.find_all(exp.Table):
        if not table.db:
            if table.name in cte_names:
                continue
            return _reject(
                raw_sql,
                f"Table '{table.name}' must be schema-qualified, e.g. analytics.{table.name} "
                f"or olist.{table.name}.",
            )
        if table.db not in ALLOWED_SCHEMAS:
            return _reject(raw_sql, f"Unknown or disallowed schema: {table.db}")

        qualified_name = f"{table.db}.{table.name}"
        if qualified_name not in allowed_schema:
            return _reject(raw_sql, f"Unknown or disallowed table: {qualified_name}")
        qualified_tables_referenced.append(qualified_name)

    if not qualified_tables_referenced:
        return _reject(raw_sql, "Query references no tables from an allowed schema.")

    # Column check scoped to the tables actually referenced — not a global
    # union across every schema, so a column hallucinated from the *other*
    # schema's table of the same name still gets caught. Output aliases
    # (`SUM(x) AS total`, then `ORDER BY total` or a CTE column read by the
    # outer query) are real, legitimate column-shaped identifiers that don't
    # exist in any table's schema — exempt anything defined as an alias
    # anywhere in the tree. This only widens what's accepted; Layer 1 (the DB
    # role) and Layer 3's table allow-list are still the actual boundary, and
    # Postgres itself rejects any reference that's genuinely bogus.
    defined_aliases = {a.alias_or_name for a in tree.find_all(exp.Alias) if a.alias_or_name}
    allowed_columns = {
        col for t in set(qualified_tables_referenced) for col in allowed_schema[t]
    } | defined_aliases
    for column in tree.find_all(exp.Column):
        if column.name == "*":
            continue
        if column.name not in allowed_columns:
            return _reject(raw_sql, f"Unknown or disallowed column: {column.name}")

    # Layer 4: inject or clamp LIMIT.
    limit_node = tree.args.get("limit")
    if limit_node is None:
        tree = tree.limit(settings.sql_row_limit_default)
    else:
        try:
            current_limit = int(limit_node.expression.this)
        except (AttributeError, ValueError, TypeError):
            current_limit = None
        if current_limit is None or current_limit > settings.sql_row_limit_default:
            tree = tree.copy()
            tree.set("limit", None)
            tree = tree.limit(settings.sql_row_limit_default)

    return ValidationResult(
        ok=True,
        sql=tree.sql(dialect="postgres"),
        tables_referenced=sorted(set(qualified_tables_referenced)),
    )


async def execute_validated_query(validated_sql: str, session_id: str | None = None) -> tuple[list[dict[str, Any]], float]:
    """Layers 1/5: executes as readonly_analyst inside a READ ONLY txn with a
    statement timeout (Sec 4). Assumes `validated_sql` already passed
    `validate_sql` — this function does not re-validate.
    """
    started = time.perf_counter()
    async with analytics_readonly_connection() as conn:
        records = await conn.fetch(validated_sql)
    exec_ms = (time.perf_counter() - started) * 1000

    rows = [dict(r) for r in records]

    # Layer 6: audit log — query hash + table set + row count + duration, never the rows themselves.
    query_hash = hashlib.sha256(validated_sql.encode()).hexdigest()[:16]
    logger.info(
        "sql_executed",
        session_id=session_id,
        query_hash=query_hash,
        row_count=len(rows),
        exec_ms=round(exec_ms, 2),
    )

    return rows, exec_ms


@dataclass
class QueryToolResult:
    """Structured, agent-facing result — the SQL Agent (Phase 5) reads this,
    never raw exceptions or raw asyncpg rows. `ok=False` covers both
    validation rejection and execution failure; `rejection_reason` is always
    safe to show the LLM (never a raw DB error / stack trace — Sec 9).
    """

    ok: bool
    sql: str | None = None
    columns: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    exec_ms: float = 0.0
    is_empty: bool = False
    tables_referenced: list[str] = field(default_factory=list)
    rejection_reason: str | None = None


async def run_query(raw_sql: str, *, session_id: str | None = None) -> QueryToolResult:
    """The single entry point agents should use: validate -> execute ->
    normalize, in one call. Never raises — DB errors (timeout, transient
    connection failure) are caught and returned as a rejection reason so a
    calling agent can decide whether to retry, not crash the graph run.
    """
    validation = await validate_sql(raw_sql)
    if not validation.ok:
        return QueryToolResult(ok=False, sql=raw_sql, rejection_reason=validation.rejection_reason)

    try:
        rows, exec_ms = await execute_validated_query(validation.sql, session_id=session_id)
    except (asyncpg.PostgresError, TimeoutError, OSError) as e:
        logger.error("sql_execution_failed", session_id=session_id, error_type=type(e).__name__)
        return QueryToolResult(
            ok=False,
            sql=validation.sql,
            tables_referenced=validation.tables_referenced,
            rejection_reason=f"Query failed during execution ({type(e).__name__}). Try a simpler query.",
        )

    columns = list(rows[0].keys()) if rows else []
    return QueryToolResult(
        ok=True,
        sql=validation.sql,
        columns=columns,
        rows=rows,
        row_count=len(rows),
        exec_ms=exec_ms,
        is_empty=len(rows) == 0,
        tables_referenced=validation.tables_referenced,
    )
