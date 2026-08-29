# Deployment (Render)

Final deployment phase. Two Docker-based Render web services + one Render
PostgreSQL instance — no Redis, no Kafka, no Kubernetes, no additional
microservices. `render.yaml` at the repo root is the Blueprint; this file
covers the parts a declarative blueprint can't express.

```
Render
│
├── PostgreSQL (aibia-postgres)
│     app schema (r/w, bi_app role) + analytics schema (readonly_analyst role)
│
├── aibia-api        (Docker web service, ./Dockerfile)
│     FastAPI · LangGraph agent graph · SQL tools · ML Agent · evaluation
│     healthCheckPath: /api/v1/health/ready
│
└── aibia-frontend   (Docker web service, ./frontend/Dockerfile)
      Streamlit — talks to aibia-api over HTTPS via API_BASE_URL
      healthCheckPath: /_stcore/health
```

## Deploy order

1. **Create the Blueprint** from `render.yaml` (Render dashboard -> New ->
   Blueprint, point at this repo). This provisions `aibia-postgres`,
   `aibia-api`, and `aibia-frontend` together, but the two `sync: false`
   database env vars below still need a manual edit before `aibia-api` can
   start — Render will deploy it in a crash-looping state until you set
   them (see "Database must be created first" below; this is expected,
   not a bug).
2. **Set the two database URLs** (see "Database URLs" below).
3. **Set `GROQ_API_KEY`** (or `ANTHROPIC_API_KEY` + `LLM_PROVIDER=anthropic`).
4. **Run migrations** — either automatically via `preDeployCommand`
   (paid instance types only) or manually (free tier) — see "Migrations"
   below. The database has zero tables until this runs.
5. **Confirm the two placeholder URLs** in `render.yaml`
   (`FRONTEND_ORIGIN` on the backend, `API_BASE_URL` on the frontend)
   match the actual `*.onrender.com` hostnames Render assigned — they're
   only guessable in advance if you keep the service names in
   `render.yaml` unchanged.
6. **Verify** with the smoke test below.

## Database URLs

Render's managed Postgres only exposes one connection string
(`postgresql://user:pass@host/db`), but this app needs it in two different
forms because it uses two different DB roles through two different
drivers (`app/db/database.py` — this split is deliberate, see
docs/security.md's SQL safety layers, and is unchanged by this phase):

| Variable | Derived from Render's connection string | Why it differs |
|---|---|---|
| `DATABASE_URL` | Change the scheme from `postgresql://` to `postgresql+asyncpg://`; keep everything else (user, password, host, port, db) | SQLAlchemy's async engine dispatches on the URL scheme and requires this exact form |
| `ANALYTICS_DATABASE_URL` | Same host/port/db; replace the user with `readonly_analyst` and the password with the `READONLY_DB_PASSWORD` value Render generated | Raw `asyncpg` driver, no SQLAlchemy involved — plain `postgresql://` scheme is correct here |

The `readonly_analyst` role itself doesn't exist until migration `0001`
runs once against `DATABASE_URL` (it creates the role with
`READONLY_DB_PASSWORD` as its password — see `app/db/migrations/versions/
0001_initial.py`). Until then, `ANALYTICS_DATABASE_URL` will correctly
fail to authenticate — that's expected on a brand-new database, not a
deployment bug.

## Migrations

`preDeployCommand: python -m alembic upgrade head` in `render.yaml` runs
this automatically before each new deploy starts serving traffic — **but
Render's free instance types don't support `preDeployCommand`.** On the
free tier, run it manually once per deploy via the Render Shell tab on the
`aibia-api` service:

```
python -m alembic upgrade head
```

The chain is `0001_initial -> 0002_olist_schema -> 0003_report_extras ->
0004_execution_metadata`, verified in this phase with a real
upgrade -> downgrade -> upgrade round-trip against a disposable Postgres
container (never the dev database) — see the Final Report for the exact
result. Never run a fresh `upgrade head` against a database that already
holds real session/report data without first confirming which migrations
it's already at (`alembic current`).

## Seed data

`scripts/generate_data.py` / `scripts/load_olist.py` populate the
`analytics`/`olist` schemas the evaluation benchmark and demo questions
rely on. Run these manually via the Render Shell **once**, after
migrations, only on a database that doesn't already have this data —
re-running them is not idempotent-safe against a database you care about,
and neither script is part of the deploy pipeline (`preDeployCommand` runs
only the migration, deliberately — seeding is a one-time operator action,
never something that runs automatically against a possibly-existing
production database).

## Environment variables

See `.env.example` (reorganized this phase into Required / Optional) and
`render.yaml`'s inline comments for the authoritative list — this file
doesn't duplicate it. Every variable name is one `app/core/config.py`
already read before this phase; none were invented for Render.

## CORS

`app/main.py` now applies `CORSMiddleware` scoped to `Settings.
frontend_origin` (new this phase — previously there was no CORS
middleware at all). This is defense-in-depth, not load-bearing: the
Streamlit frontend calls this API server-to-server
(`frontend/api_client.py` runs inside the frontend's own container, never
in the visitor's browser), so the primary user flow works with any CORS
policy. It's still scoped to the real frontend origin, never `"*"`, in
case anything ever calls the API directly from a browser (e.g. `/docs`,
or a future direct client). Comma-separate `FRONTEND_ORIGIN` if you need
to allow more than one origin (e.g. local dev plus the deployed Render
frontend) at once.

## Health checks

- `GET /api/v1/health` — liveness only, always `200 {"status": "ok"}` once
  the process is up.
- `GET /api/v1/health/ready` — checks real DB connectivity
  (`app/db/database.py::db_healthy`). **This phase fixed a real gap**: it
  previously always returned HTTP `200` regardless of the `ready` field's
  value, which made it unusable as an infra-level health check — Render
  (like Docker and Kubernetes) gates traffic/restarts on the HTTP status
  code, never the response body. It now returns `503` when not ready,
  `200` when ready; the JSON body shape is unchanged for anything that
  already reads it. `render.yaml` points `aibia-api`'s `healthCheckPath`
  at this endpoint (not the plain `/health`) so Render won't route traffic
  to an instance that's up but can't reach its database.
- Streamlit's own built-in `/_stcore/health` is used for `aibia-frontend`
  (no code change — a stock Streamlit endpoint).

## Startup commands

Both Dockerfiles now use `CMD exec <command> --port ${PORT:-8000|8501}`
(shell form, not the previous JSON exec-form array) so Render's injected
`PORT` env var is honored — Render's Docker runtime requires the
container to bind to whatever port it assigns, which is not always 8000/
8501. The leading `exec` matters: without it, the shell (not the actual
process) is PID 1 and doesn't forward `SIGTERM` on to uvicorn/Streamlit,
which delays graceful shutdown during a Render deploy. Local
`docker-compose.yml` is unaffected — it already supplies its own explicit
`command:` per service, which overrides the Dockerfile `CMD` entirely, and
never sets `PORT`, so the `${PORT:-8000}`/`${PORT:-8501}` fallback keeps
local behavior byte-identical to before this phase. Streamlit's frontend
`CMD` now also passes `--server.headless=true` explicitly rather than
relying on Streamlit's no-TTY auto-detection.

## What this phase deliberately did not change

Per this phase's explicit scope: no new agents, no new ML models, no
Redis, no Kafka, no Kubernetes, no Grafana/Prometheus/Alertmanager, no
MCP, no vector database, no password reset, no email verification, no
additional microservices. The in-memory login rate limiter
(`app/core/security.py::LoginRateLimiter`) still does not coordinate
across multiple Render instances if you ever scale `aibia-api` beyond one
instance — documented, not solved, per this phase's explicit instruction
not to add Redis merely to remove that limitation.
