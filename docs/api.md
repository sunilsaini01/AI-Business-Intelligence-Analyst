# API

Base path: `/api/v1`. Full endpoint table: blueprint Sec 7. Interactive docs
at `/docs` (FastAPI's auto-generated Swagger UI) once the API is running.

| Endpoint | Status |
|---|---|
| `GET /health` | done |
| `GET /health/ready` | done — checks DB connectivity; LLM key presence is reported but not required (Phase 0 has no LLM calls) |
| `POST /analyze` | done — runs the full agent graph (Supervisor -> SQL -> Analysis -> Visualization -> Supervisor -> Critic -> Report Generator); returns `202 {analysis_id, status}` |
| `GET /analysis/{id}/status` | done — includes `current_stage` (Phase 11): the most recently *completed* node's name, derived from incrementally-persisted trace events, `null` until the first node finishes |
| `GET /analysis/{id}/report` | done — `409` until `status == DONE`; includes Phase 10's Report Generator fields (`verified_claims`, `analysis_explanation`, `visualizations`, `technical_details`, `narrative`) |
| `GET /analysis/{id}/charts` | done — populated by the Visualization Agent (Phase 7); `[]` when no chart was warranted for the question |
| `GET /analysis/{id}` | done — full session + trace + `execution_metadata` (Phase 13: start/end time, duration, completed_nodes, error_category, retry_count, token_usage, narrative_enabled, report_generated — `{}` until the run reaches a terminal state), backs the "show agent trace" panel |
| `GET /reports` | done — cross-session listing, not in the original Sec 7 table but useful for a "recent analyses" panel |
| `POST /evaluation/run` | done (Phase 8) — fires the benchmark runner as a background task, returns `202 {run_id, status}` |
| `GET /evaluation/results` | done — filter by `run_id`, or list all runs |

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
