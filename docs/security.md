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

## Secrets (Phase 15, Objective 1)

`.env` is gitignored; only `.env.example` (no real values) is committed —
both verified directly by `tests/security/test_secret_exposure.py`, not
just asserted in prose. Never log SQL text or raw DB errors to the API
response — see `app/core/security.py::safe_error_response` and the global
exception handler in `app/main.py`.

Every secret-bearing `Settings` field (`database_url`,
`analytics_database_url`, `readonly_db_password`, `anthropic_api_key`,
`groq_api_key`, `secret_key`, `previous_secret_key`) is declared with
`Field(repr=False)` (`app/core/config.py`) — pydantic excludes those
fields entirely from the model's `__repr__`/`__str__`. This closes a real
accidental-exposure path found during development: a plain `Settings()`
repr is pulled into any log line, exception message, or (as actually
happened once, harmlessly, in this project's own pytest output) an
assertion-failure trace that happens to mention `settings`. Direct
attribute access (`settings.secret_key`, the only way the app itself ever
uses these values) is completely unaffected.

**Rotating `GROQ_API_KEY` / `ANTHROPIC_API_KEY`** (manual — this project
never rotates a key automatically):
1. Generate a new key in the provider's console (console.groq.com /
   console.anthropic.com); do not delete the old one yet.
2. Update `GROQ_API_KEY`/`ANTHROPIC_API_KEY` in the deployment's `.env`
   (or secret store) to the new value.
3. Restart the API process — `Settings` is loaded once and cached
   (`get_settings()`'s `@lru_cache`), so a running process never picks up
   an env var change without a restart.
4. Confirm `GET /api/v1/health/ready`'s `llm_configured` is `true` and a
   real analysis completes successfully.
5. Revoke the OLD key in the provider's console once the new one is
   confirmed working.

**If a key was ever exposed** (e.g. printed to a terminal, committed, or
pasted somewhere it shouldn't have been — this happened harmlessly during
this project's own Phase 14 development, to this session's terminal
output only, never to a committed file): treat it as compromised and
follow the rotation steps above immediately. Never attempt to recover or
re-display an exposed key from logs/history — only revoke and replace it.

## CORS (final deployment phase)

`app/main.py` applies `CORSMiddleware` scoped to `Settings.frontend_origin`
(comma-separated for more than one allowed origin) — never
`allow_origins=["*"]`. This is defense-in-depth, not load-bearing: the
Streamlit frontend calls this API server-to-server
(`frontend/api_client.py` runs inside the frontend's own container, never
in a visitor's browser), so the primary user flow doesn't depend on CORS
at all. It's still scoped to the real frontend origin so a browser-based
client from an unrelated third-party site can't make a credentialed
cross-origin request against a logged-in user's bearer-token session — see
`tests/security/test_api_error_handling.py`'s CORS tests.

## Authentication & authorization (Phase 14)

Every analysis is owned by the authenticated user who created it
(`app/core/auth.py`, `app/api/routes/auth.py`) — see docs/api.md's "Auth &
ownership" table for the exact 401/403/404 semantics. Passwords are bcrypt-
hashed (`app/services/auth_service.py`), never stored or logged in plain
text; access tokens are signed JWTs (`SECRET_KEY`, required, no default —
see `.env.example`) and are never logged (`app/core/logging.py`'s
structured events only ever carry `analysis_id`/stage/duration/category,
the same safe-list `execution_metadata` already used, extended to cover
the auth code path — see `tests/api/test_analysis_service.py::
test_execution_metadata_never_contains_secrets` and the auth-specific
regression tests in `tests/security/test_authorization.py` and
`tests/api/test_auth.py`). Login intentionally returns the same `401` and
message for "no such user" and "wrong password" — it never lets a client
enumerate registered emails.

`GET /evaluation/*` stays unauthenticated by design: it's benchmark/dev
tooling over system-level aggregate data, not a per-user analysis.

## Login rate limiting (Phase 15, Objective 2)

`POST /auth/login` is rate-limited by `app/core/security.py::
LoginRateLimiter` — a token bucket keyed by `(client IP, submitted
email)`, separate from the general `RateLimitMiddleware` above (which
stays IP-only, off by default, and is never applied to `/auth/login`
specially — this is a dedicated mechanism, not a reuse of that
middleware's routing). Every login attempt consumes one token, success or
failure, so a successful login never resets or bypasses the limit. Over
the limit: `429` with a fixed `Retry-After` header (the configured window,
not a precise remaining-time computation — deliberately, so the response
never reveals the bucket's exact internal state) and a generic detail
string, identical whether or not the submitted email is actually
registered.

Configurable via `Settings` / environment:

| Variable | Default | Purpose |
|---|---|---|
| `LOGIN_RATE_LIMIT_ENABLED` | `true` | Master on/off switch |
| `LOGIN_RATE_LIMIT_MAX_ATTEMPTS` | `5` | Attempts allowed per window, per (IP, email) |
| `LOGIN_RATE_LIMIT_WINDOW_SECONDS` | `300` | Window length in seconds |

In-memory, per-process — same documented tradeoff `RateLimitMiddleware`
already carries for a portfolio-scale deploy (no Redis/shared store;
multiple API replicas would each track their own bucket). See
`tests/security/test_login_rate_limit.py`.

## SECRET_KEY rotation (Phase 15, Objective 3)

JWTs carry a `kid` (key id) header (`app/core/auth.py::create_access_token`)
naming which key signed them. `decode_access_token` accepts a token signed
with either the CURRENT key (`SECRET_KEY`/`SECRET_KEY_ID`) or, during a
rotation window, the PREVIOUS one (`PREVIOUS_SECRET_KEY`/
`PREVIOUS_SECRET_KEY_ID`) — new tokens are always signed with the current
key only; the previous key is verify-only.

**Rotation procedure:**
1. Generate a new `SECRET_KEY` (`python -c "import secrets;
   print(secrets.token_hex(32))"`) and pick a new `SECRET_KEY_ID` (any
   short label, e.g. a date: `2026-09`).
2. Set `PREVIOUS_SECRET_KEY` to the OLD `SECRET_KEY` value, and
   `PREVIOUS_SECRET_KEY_ID` to the OLD `SECRET_KEY_ID`.
3. Set `SECRET_KEY`/`SECRET_KEY_ID` to the new values. Deploy/restart.
   Tokens issued before the rotation keep validating (against the
   previous key); every new token uses the new key.
4. Once you're satisfied every pre-rotation token has either expired
   (`ACCESS_TOKEN_EXPIRE_MINUTES`, default 24h) or you're willing to force
   those remaining sessions to re-login, unset `PREVIOUS_SECRET_KEY` and
   `PREVIOUS_SECRET_KEY_ID` and redeploy — tokens bearing the old `kid`
   are rejected (`401`) from that point on.

**Design tradeoff, stated plainly:** the "grace period" is operator-
controlled (however long you leave `PREVIOUS_SECRET_KEY` configured), not
a self-expiring timer tracked inside the app — the simplest mechanism that
fits a stateless-JWT design without adding a token-revocation database
table or a scheduled job. A token issued before this rotation feature
existed at all (no `kid` header) is still accepted against the CURRENT
key — treating a missing `kid` as an automatic rejection would log every
existing session out the moment this code deploys, which was explicitly
ruled out. See `tests/unit/test_auth.py`'s rotation tests.
