# API

Base path: `/api/v1`. Full endpoint table: blueprint Sec 7. Interactive docs
at `/docs` (FastAPI's auto-generated Swagger UI) once the API is running.

| Endpoint | Status |
|---|---|
| `GET /health` | done |
| `GET /health/ready` | done — checks DB connectivity; LLM key presence is reported but not required (Phase 0 has no LLM calls). Returns HTTP `503` when `ready` is `false`, `200` when `true` (fixed in the final deployment phase — a status-code-based health check, which is what Render/Docker/Kubernetes all actually use to gate traffic, previously couldn't distinguish the two states since this always answered `200`) |
| `POST /auth/register` | done (Phase 14) — `{email, password}` -> `201 {id, email, role, created_at}`. `400` if the email is already registered. |
| `POST /auth/login` | done (Phase 14) — `{email, password}` -> `200 {access_token, token_type: "bearer"}`. `401` for any wrong email/password combination (never distinguishes "no such user" from "wrong password"). Rate-limited (Phase 15) — `429` with `Retry-After` past `LOGIN_RATE_LIMIT_MAX_ATTEMPTS` per (IP, email); see docs/security.md. |
| `POST /analyze` | done — **requires** `Authorization: Bearer <token>` (Phase 14). Runs the full agent graph (Supervisor -> SQL -> Analysis -> Visualization -> Supervisor -> Critic -> Report Generator); returns `202 {analysis_id, status}`. The session is owned by the calling user. |
| `GET /analysis/{id}/status` | done — requires auth + ownership (Phase 14, see below). Includes `current_stage` (Phase 11): the most recently *completed* node's name, derived from incrementally-persisted trace events, `null` until the first node finishes |
| `GET /analysis/{id}/report` | done — requires auth + ownership. `409` until `status == DONE`; includes Phase 10's Report Generator fields (`verified_claims`, `analysis_explanation`, `visualizations`, `technical_details`, `narrative`) plus Phase 15's `ml_summary`/`ml_results` |
| `GET /analysis/{id}/charts` | done — requires auth + ownership. Populated by the Visualization Agent (Phase 7); `[]` when no chart was warranted for the question |
| `GET /analysis/{id}` | done — requires auth + ownership. Full session + trace + `execution_metadata` (Phase 13: start/end time, duration, completed_nodes, error_category, retry_count, token_usage, narrative_enabled, report_generated — `{}` until the run reaches a terminal state), backs the "show agent trace" panel |
| `GET /reports` | done — requires auth (Phase 14); cross-session listing filtered to the caller's own reports only |
| `POST /evaluation/run` | done (Phase 8) — fires the benchmark runner as a background task, returns `202 {run_id, status}`. Unauthenticated: benchmark/dev tooling over system-level data, not per-user analyses. |
| `GET /evaluation/results` | done — filter by `run_id`, or list all runs. Unauthenticated, same reasoning as `POST /evaluation/run`. |

## Async pattern

Fire-and-poll (Sec 7 Fig. 5), no external queue — `POST /analyze` schedules
the graph run via FastAPI `BackgroundTasks` and returns immediately. Poll
`/analysis/{id}/status` (never `/report`/`/charts`/`/analysis/{id}` — those
are fetched once, only after `status == DONE`) until `DONE` or `FAILED`,
then fetch `/report`. The reference frontend (`frontend/polling.py`) polls
every 2s with a 180s bound; past that it reports "taking longer than
expected" rather than either failing or polling forever.

## Error classification (Phase 13)

A `FAILED` session's `error_message` is one of 5 fixed, safe strings keyed
by `app/core/errors.py::ErrorCategory` (`rate_limit` / `timeout` /
`provider_error` / `validation_error` / `application_error`) — never the
raw exception text. The same category is recorded structurally in
`execution_metadata.error_category` for programmatic use. `SessionStatus`
itself is unchanged (`PENDING`/`ANALYZING`/`DONE`/`FAILED`) — the
classification changes what's reported about a failure, never the state
machine.

## Auth & ownership (Phase 14)

`app/core/auth.py`: password hashing (bcrypt) + a signed, stateless bearer
JWT (`pyjwt`, `Settings.secret_key`/`SECRET_KEY` — required, no default).
No session store, no external identity provider, no refresh tokens —
`ACCESS_TOKEN_EXPIRE_MINUTES` (default 1440) is the only expiry knob.

Every `/analysis/*` endpoint and `GET /reports` require
`Authorization: Bearer <token>` via the `get_current_user` dependency:

| Condition | Response |
|---|---|
| No/malformed/expired/mis-signed token | `401`, always the same generic detail — never reveals *why* the token was rejected |
| Valid token, unknown `analysis_id` | `404` |
| Valid token, `analysis_id` exists but owned by a different user | `403` |
| Valid token, `analysis_id` owned by the caller (or has no recorded owner — see below) | normal response |

A session with `user_id IS NULL` (only possible for rows created before
Phase 14 shipped — `POST /analyze` always sets an owner now) is readable
by any authenticated user, a deliberate, documented allowance for pre-auth
data rather than a gap — see `app/api/routes/analysis.py::_check_ownership`.
`GET /reports` has no equivalent case: it's a listing filtered strictly to
`user_id == caller`, so ownerless rows simply never appear there (they
remain reachable directly by id).

## ML Agent (Phase 15, Objective 4)

`app/agents/ml_agent.py` runs in every graph execution (same "always in
the chain, no-op if not applicable" pattern as the Analysis/Visualization
agents) but only does real work when the Supervisor classified the
question `intent == "predictive"`. Two supported tasks, chosen by keyword
match against the question (never guessed by an LLM — this module makes
zero LLM calls):

- **forecasting** ("forecast", "predict", "projection", "next month/
  quarter/year", "trend", "expected/future revenue") — a linear-trend
  baseline (`app/tools/ml_tools.py::evaluate_and_forecast`) over monthly
  revenue, evaluated with a TIME-AWARE train/test split (the held-out
  points are always the most recent ones, never shuffled).
- **churn_risk** ("churn", "risk", "retention", "attrition", "at risk",
  "likely to leave/cancel") — logistic regression
  (`fit_churn_classifier`) over deterministic per-customer features
  (order history + `analytics.customer_activity` counts); churn is
  DEFINED as no order within 180 days of the dataset's most recent order
  — a modeling choice, stated in the result's own `limitations`, not a
  fact reported by customers.

Both tasks reach the database exclusively through
`app/tools/database_tools.py::run_query` — the fixed, reviewed SQL this
module uses still goes through the exact same AST validation/schema
allow-list/LIMIT clamp/readonly-role pipeline as every LLM-generated
query; nothing here is exempt.

`GET /analysis/{id}/report` exposes the result two ways:
- `ml_summary` (string) — a deterministic, no-LLM one-line rendering
  (`app/agents/report_agent.py::_format_ml_summary`), `""` if the
  question wasn't predictive.
- `ml_results` (object or `null`) — the full structured result:
  `task`, `target`, `features`, `model_name`, `train_size`, `test_size`,
  `metrics`, `sample_predictions`/`forecast_next`, `feature_importance`,
  `limitations`, `confidence`. On "not appropriate" or "insufficient
  data" (`ok: false`), only `status`/`task`/`reason` are populated — no
  fabricated metrics.

The Supervisor's synthesis prompt is given the same result (so it can
legitimately cite a real metric), and the Critic's numerical-grounding
check accepts exactly those same values as evidence (`app/tools/
critic_checks.py::_collect_known_values`) — a genuine ML number is never
flagged as fabricated, and a number that ISN'T actually in the ML result
still is.

**Limitations:** the forecast is a simple linear trend, not seasonal or
causal — it will not anticipate promotions or one-off events. Churn
feature importance reflects statistical association in this dataset, not
proven cause; neither the model nor the synthesis prompt is permitted to
phrase a prediction as a certainty or a causal claim.

**Evaluation (Phase 16):** the same `ml_results` this endpoint returns is
now a measurable regression contract, not just a manually-eyeballed
number — `app/evaluation/metrics.py::evaluate_ml_quality` gates real
forecast/churn output against fixed thresholds (MAPE/MAE, ROC-AUC/
accuracy/precision/recall) as part of the existing evaluation framework
(no new endpoint, no second evaluation system). See docs/evaluation.md's
"ML Agent evaluation" section for the exact thresholds and rationale.
