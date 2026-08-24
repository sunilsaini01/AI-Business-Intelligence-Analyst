# Security

## SQL safety layers (blueprint Sec 4)

| # | Layer | Where |
|---|---|---|
| 1 | Read-only DB role, no write/DDL grants, SELECT-only on `analytics.*` and `olist.*` | `migrations/versions/0001_initial.py`, `0002_olist_schema.py` |
| 2 | AST validation (single SELECT, no disallowed nodes) | `app/tools/database_tools.py::validate_sql` |
| 3 | Schema-qualified allow-list across both schemas (live-introspected, never agent-supplied) | `app/tools/schema_tools.py`, `database_tools.py` |
| 4 | LIMIT injection/clamp | `database_tools.py::validate_sql` |
| 5 | `statement_timeout`, `READ ONLY` transaction | `app/db/database.py::analytics_readonly_connection` |
| 6 | Audit log (query hash, table set, row count, duration — never rows) | `database_tools.py::execute_validated_query` |

Layer 1 is the boundary that has to hold even if 2-6 all have bugs. Layers
2-6 exist so a bad query fails fast with a structured reason, not so the
system is "safe" without the role grant.

## Required security tests (blueprint Sec 10 MUST-HAVE)

`tests/security/` must cover, at minimum:
- Stacked query: `SELECT 1; DROP TABLE orders;`
- Comment-obfuscated keyword: `SEL/**/ECT` or similar
- A write disguised as a CTE
- A request for a table outside the allow-list

All four pass against the live `readonly_analyst` role as of Phase 3
(`tests/security/test_sql_injection.py`) — re-run this suite after any change
to `database_tools.py`'s validation logic before trusting it again.

## Secrets

`.env` is gitignored; only `.env.example` (no real values) is committed.
Never log SQL text or raw DB errors to the API response — see
`app/core/security.py::safe_error_response` and the global exception handler
in `app/main.py`.
