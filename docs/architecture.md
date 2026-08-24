# Architecture

Full design rationale lives in `BI_AGENT_BLUEPRINT_1.md` at the repo root —
this file tracks how the *implementation* maps onto it and records decisions
that need to survive independent of that document.

## Production hardening (Phase 13)

Four additive objectives — no rewrite of any existing agent, no change to
the LangGraph workflow, no change to business-analysis mathematics.

**A. Error classification** (`app/core/errors.py::classify_exception`) —
a shared, provider-agnostic classifier (`rate_limit` / `timeout` /
`provider_error` / `validation_error` / `application_error`), verified
against the real `anthropic`/`groq` SDK exception hierarchies (both mirror
each other: `RateLimitError`/`AuthenticationError` etc. as `APIStatusError`
subclasses, `APITimeoutError` as an `APIConnectionError` subclass) rather
than assumed. Used in three places, each WITHOUT changing what already
happened on failure — only what gets *recorded*:
- `app/agents/critic.py` — a failed semantic check is still just an INFO
  finding, never a content FAIL; `critic_feedback.semantic_check_error_
  category` now records which kind of failure it was.
- `app/agents/report_agent.py` — a failed narrative call still degrades to
  `narrative: null`; `report.technical_details.narrative_error_category`
  records which kind.
- `app/services/analysis_service.py::run_analysis` — a whole-run failure
  is still `SessionStatus.FAILED` (no new enum value), with a safe,
  category-specific `error_message` and a structured `error_category` in
  `execution_metadata`.

This also finally makes the live-test quota-skip path *reachable*: earlier
phases' audits found that `critic_node`/`report_agent_node` catch
`RateLimitError` internally (by design — Sec 9's "infra failure isn't a
content failure" rule), which meant a `try/except RateLimitError: skip()`
wrapped around either of them was unreachable dead code. Phase 13's live
tests check the new classification field instead of relying on the
exception propagating — see `tests/agents/test_critic_live_llm.py`.

**B. Browser E2E testing** (`tests/e2e/`, Playwright) — a real browser
driving the real, running Streamlit frontend against the real FastAPI
backend and the real seeded database. Made deterministic (no live-quota
dependency) via a third, non-default LLM provider value,
`LLM_PROVIDER=fake` (`app/core/fake_llm.py`) — a canned-response stand-in
used ONLY when explicitly configured for an E2E run (`docker-compose.e2e.yml`
overrides just the `api` service's env var; `.env`'s own `LLM_PROVIDER`,
and therefore every normal `docker compose up`, is untouched). The fake
client's synthesis text deliberately cites no numbers, so nothing it
returns can ever be flagged by the Critic's real, unweakened numeric-
grounding check — every number in the resulting report still comes from
the real Analysis Agent running against real query results.

Actually run and verified in a real Chromium instance (not just written)
— see `tests/e2e/test_analysis_journey.py`'s module docstring for the
exact commands, including the apt package list Playwright's own
`--with-deps` couldn't install (a handful of its Ubuntu-20.04-fallback
font package names don't exist on this image's Debian trixie base;
installing the actual shared libraries directly worked). Doing this run
for real — not just trusting the design — caught two genuine bugs before
they could ship, both fixed and covered by regression tests:
1. Streamlit's `text_input` only syncs its value to the Python session on
   blur/Enter — the test's own `.fill()` call wasn't triggering that, so
   the Analyze button stayed disabled. Fixed in the test (a real user
   tabs/clicks away too), not in `app.py`, which was already correct.
2. `FakeLLMClient`'s keyword matching was checking the LLM call's full
   `system` prompt (which embeds the database schema description, and
   therefore always mentions "customers"/"regions" regardless of the
   actual question) instead of only the real user question in `messages`
   — "What's the weather like today?" was answered as if it were the
   customers-by-region question. Fixed in `app/core/fake_llm.py` to match
   against `messages` only; regression test in `tests/unit/test_fake_llm.py`.

**C. Live-test reliability** — `test_critic_live_llm.py` was rewritten
after a live run surfaced a genuine ambiguity: its fixture's wording
("driven mainly by Enterprise") was less hedged than its own
interpretation ("Enterprise *appears to be* the dominant contributor"),
and the semantic check's own system prompt says restating a hedged
interpretation as a plain fact counts as unsupported — so an occasional
strict FAIL from the live model was a defensible read, not a bug. The
fixture now mirrors its interpretation's hedge level exactly (removing the
ambiguity), and a second, genuinely unambiguous adversarial case (a
fabricated root cause naming no evidence at all) was added so the suite
still has real discriminative power — a genuine semantic miss must still
fail the test, never silently become a skip.

**D. Structured execution observability** — one additive JSONB column,
`AnalysisSession.execution_metadata` (migration `0004_execution_metadata`,
same shape as Phase 10's `report_extras`), populated once per run
(DONE or FAILED) with: start/end time, duration, `current_stage`,
`completed_nodes`, an inferred `failed_node` (best-effort only, from the
canonical pipeline order — never used for control flow), `error_category`,
`retry_count`, `token_usage` (delta-based, not cumulative — see Phase 8's
token-tracking fix), `narrative_enabled`, `report_generated`. Exposed via
`GET /analysis/{id}` (not `/status`, which stays lean for polling). Never
contains secrets — node names, counts, timestamps, booleans, token counts
only.

## Critic Agent (Phase 9)

`supervisor(synthesize) -> critic -> {END, or back to supervisor for revision}`.
Almost entirely deterministic (`app/tools/critic_checks.py`): numeric
grounding (every number in the report text must trace to a real value in
`analysis_results`/evidence, within tolerance), period/category label
consistency between charts and their source analysis, contribution
arithmetic re-verification (independently re-checks Phase 6 bug #4/#5 stay
fixed), evidence-sufficiency vs claimed confidence, and a causal-claim
heuristic (entity + dominant-contributor matching). The one LLM call
(`CriticSemanticCheck`, `app/agents/schemas.py`) is isolated to the single
question deterministic code can't answer — does the wording overstate what
the facts/interpretations support — and sees only the executive summary,
key findings, and the Analysis Agent's own facts/interpretations, never raw
SQL rows. An LLM/infra error there degrades to an INFO finding, not a FAIL
(Sec 9's "infra failure isn't a content failure" rule).

Two real bugs found and fixed while building this: (1) a bare year mention
("July 2026") was being extracted as a claimed numeric value needing
evidence-grounding — fixed by excluding bare 4-digit integers in the
1900-2100 range; (2) percentages are commonly stated unsigned in prose ("a
6.7% decline", not "-6.7%") since the direction word already carries the
sign — the grounding check now matches both signed and unsigned forms.

Retry loop reuses the exact `report is None` signal `_route_after_supervisor`
already keyed off for the sql_agent branch — `critic_node` clears `report`
back to `None` on FAIL with `retry_count < max_retries` (which naturally
re-enters `_synthesize()`, now with `critic_feedback` added to its prompt so
the revision addresses what actually failed), and leaves `report` set
(confidence forced to "Low", findings appended to limitations) once retries
are exhausted or the verdict is PASS/WARN. `CRITIC_MAX_RETRIES` is now wired
through from settings into `new_state()` (previously declared but unused,
since nothing consumed `max_retries` before the Critic existed).

## LLM reliability — bounded retry with backoff (Phase 6 correction)

An earlier version of this doc said neither client retries transient
failures and flagged it as an unimplemented gap. That was wrong — verified
live (see a real 429 hit during Phase 6 testing) that both `anthropic` and
`groq`'s official SDKs already retry transient errors (429/5xx/connection)
internally with exponential backoff by default (`max_retries=2` each,
confirmed via `inspect.signature`), and correctly do *not* retry non-transient
errors (400 bad request, 401 auth) — exactly the bounded/transient-only/
no-duplicate-agent-logic behavior a hand-written retry layer would have had
to implement anyway, already handled at the one chokepoint both agents go
through. The only real gap was that this was an invisible SDK default rather
than a documented, tunable setting — fixed by adding `LLM_MAX_RETRIES` (env,
default 2) and passing it explicitly to both `AsyncAnthropic`/`AsyncGroq`
constructors in `app/core/llm.py`. No custom retry/backoff code was written;
none was needed.

## LLM provider

`app/core/llm.py` builds either `LLMClient` (Anthropic, default) or
`GroqLLMClient` based on `LLM_PROVIDER` — both implement `LLMClientProtocol`,
so no agent code branches on which one is active. Added because live
Phase 4/5 validation happened while the Anthropic account was out of credit;
switching back is a one-line env change, not a code change. Groq model IDs
must be confirmed against `client.models.list()` before use — the catalog
has changed over time (no `llama-3.x` models were available at integration
time; `openai/gpt-oss-{20b,120b}` were).

## Findings from live-LLM validation (Phase 4/5, Groq)

Real bugs a real model surfaced that the fake-LLM tests couldn't, each fixed
and re-verified against actual Postgres data:

1. **`ORDER BY`/CTE-alias false rejections** — found in Phase 3 tooling before
   any agent existed; see the SQL safety section above.
2. **Wrong discount formula** — the SQL Agent generated `unit_price - discount`
   (treating the fraction as currency) instead of `unit_price * (1 - discount)`.
   Ranking was coincidentally right, the dollar figure was ~4.8% off. Fixed
   with explicit column-semantics guidance in `sql_agent.py`'s system prompt.
3. **Blind sequential steps** — each plan step generated SQL with no memory of
   earlier steps' results, so a diagnostic question spanning multiple years
   led to later steps hardcoding a guessed year (2023) that didn't exist in
   the data at all, returning empty results for the wrong reason. Fixed by
   threading a running evidence summary into each subsequent step's prompt
   (`sql_agent_node`'s `prior_evidence_lines`).
4. **Schema/provider mismatch crash** — `SupervisorPlan.steps` had
   `min_length=1` as an infinite-loop guard. A model that correctly reasoned
   "no such column exists -> out_of_scope, steps=[]" got its entire tool call
   rejected server-side by Groq (400, schema violation) before the code ever
   saw it — an unhandled exception. Fixed by moving the invariant into a
   Pydantic `model_validator` (post-parse, this code's own logic) instead of
   the JSON schema sent to the provider, since "steps required unless
   out_of_scope" is conditional and `min_length` can't express that.
5. **Oversized synthesis/evidence prompts** — capping evidence by row count
   alone isn't enough (a 4-table Olist join with review text is far wider per
   row than a 2-column analytics aggregate); a wide join blew through Groq's
   free-tier 8000 TPM limit. Fixed with `app/agents/prompt_utils.py::compact_rows_json`
   — caps by row count AND serialized character length.
6. **Overly granular plans** — the Supervisor defaulted to one step per join
   ("join A+B", "join +C", "aggregate") instead of one step per complete
   aggregate query, wasting LLM calls, DB round trips, and tokens (directly
   contributed to #5). Fixed with explicit "one step = one query" guidance in
   the planning prompt.

Also found, not an agent bug: `scripts/generate_data.py`'s deliberate revenue
dip landed relative to the data-generation date, not a fixed calendar month —
regenerating on a different day silently moved it (e.g. May 2026 instead of
July 2026), quietly invalidating benchmark case `bi-004`. **Fixed**: the dip
month is now the fixed constants `BENCHMARK_DIP_YEAR`/`BENCHMARK_DIP_MONTH`
(2026-07) in `generate_data.py`, with the whole date window anchored to that
month instead of to `datetime.now()`, and a single named `SEED = 42` used
everywhere randomness is drawn. Verified directly in Postgres after
regenerating: June 2026 revenue $161,445.80 -> July 2026 $150,633.02 (real
-6.7% decline); Enterprise segment's decline alone accounts for 98% of the
total decline ($10,610.84 of $10,812.78); North+Enterprise specifically
collapsed from $2,021.57 (June) to $0 (July).

## Graph

Supervisor-routed LangGraph `StateGraph` (`app/graph/workflow.py`), state
shape in `app/graph/state.py`. See blueprint Sec 1 for the full node diagram
and the Critic retry loop.

**Current status (Phase 10):** default graph is
`supervisor (plan) -> sql_agent -> analysis_agent -> visualization_agent ->
supervisor (synthesize) -> critic -> report_agent -> END`, with a bounded
Critic-triggered revision loop (`critic -> supervisor -> ... -> critic`,
`CRITIC_MAX_RETRIES`) and an `out_of_scope` short-circuit straight to END
from the first Supervisor visit (skips both critic and report_agent — a
fixed decline message has nothing to review or finalize).
`app/graph/workflow.py::build_phase0_graph()` preserves the original
`fetch -> respond` plumbing-only graph for its own test
(`tests/integration/test_phase0_workflow.py`) but is no longer the default.
`app/agents/ml_agent.py` is the one remaining `NotImplementedError` stub —
not wired into the default graph at all (see README's Status table).

`app/agents/report_agent.py` (Phase 10) is a presentation/finalization
layer, not a second synthesis step: it runs only after the Critic has
produced a terminal verdict (PASS, WARN, or a FAIL that force-degraded the
report at `max_retries`) and never touches the 6 fields the Supervisor/
Critic already own (`executive_summary`, `key_findings`, `evidence`,
`recommendations`, `confidence`, `limitations`). It adds 5 presentation-only
fields — `verified_claims` (verbatim from `critic_feedback`),
`analysis_explanation` (deterministic prose built from
`analysis_results`, reusing `diagnose_decline`'s own already-stakeholder-
readable facts/interpretations when present), `visualizations` (a
lightweight reference to `charts`, not a re-derivation), `technical_details`
(critic status/score/retry count), and one optional LLM-polished
`narrative` — re-validated against the same
`check_numerical_grounding` the Critic uses and discarded (`None`) if it
introduces anything not already in the evidence, or if the call fails/hits
a provider quota. Persisted via one additive JSONB column,
`AnalysisReport.report_extras` (migration `0003_report_extras`).

Every LLM call in the Supervisor and SQL Agent goes through
`app/core/llm.py::LLMClient.complete_structured` — forced Anthropic tool-use
so the reply is validated, structured data (`app/agents/schemas.py`), never
free text parsed with regex. Both agent node functions accept an injectable
`llm: LLMClientProtocol | None` parameter (default: the real client) so
tests run against `tests/fakes.py::ScriptedLLMClient` with no network call
and no `ANTHROPIC_API_KEY` — see `docs/evaluation.md`-adjacent note below on
which tests are live vs deterministic.

**Live vs deterministic tests:** only
`tests/api/test_analyze_live_llm.py` calls the real Claude API, and it
self-skips (not fails) when `ANTHROPIC_API_KEY` is unset. Every other test —
including the full graph round trip in `tests/integration/test_workflow.py`
and the retry-on-rejected-SQL path in `tests/agents/test_sql_agent.py` —
runs with a `ScriptedLLMClient` against the real, seeded Postgres, so the
SQL safety pipeline, schema allow-list, and graph routing are genuinely
exercised without needing a key.

## "No SciPy" interpretation

scikit-learn, XGBoost, SHAP, and statsmodels all pull in SciPy transitively —
it will appear in `pip freeze` and that's expected. The constraint we're
actually holding to: **no hand-rolled `scipy.stats` calls inside agent/tool
code as a shortcut for real analysis.** Anywhere NumPy/pandas arithmetic is
clearer and auditable, that's what's used (see `app/tools/analysis_tools.py`,
`app/tools/ml_tools.py`). Reviewers should read this as a coding-style rule,
not a dependency ban.

## SQL safety

The read-only DB role (`readonly_analyst`, granted in
`app/db/migrations/versions/0001_initial.py` and extended to the `olist`
schema in `0002_olist_schema.py`) is the actual security boundary —
everything in `app/tools/database_tools.py` (sqlglot AST validation, schema
allow-list, LIMIT cap) exists to fail fast with a good error message, not as
the last line of defense. Full detail: `docs/security.md`.

Phase 3 generalized the allow-list from one schema to N
(`app/tools/schema_tools.py::ALLOWED_SCHEMAS`). Because two schemas can have
a table with the same name and different columns, every table reference must
now be schema-qualified (`analytics.customers`, not bare `customers`) — a
stricter check than Phase 0's, not a weaker one. Two known-safe relaxations
were needed to avoid false rejections of legitimate SQL, both proven not to
weaken the boundary in `tests/integration/test_database_tools.py`:
- **Output aliases** (`SUM(x) AS total`, then `ORDER BY total`, or an outer
  query reading a CTE's computed column) are exempted from the column
  allow-list — they're not real table columns, and Postgres itself rejects
  anything genuinely bogus.
- **CTE names** (`WITH monthly AS (...)`) are exempted from the
  schema-qualification requirement when referenced as a pseudo-table later in
  the query — the CTE's own body is still fully walked and every real table
  it selects from still goes through the full check.

## Olist integration (Phase 3)

Decision: **`olist` is a separate schema from `analytics`**
(`app/db/models_olist.py`, migration `0002_olist_schema.py`), not a mapping
into the existing tables. Reasoning, from actually inspecting the CSVs
(`data/raw/*.csv`, Kaggle's Brazilian E-Commerce Public Dataset):

- No `region`/`segment` concept exists in the source data — only a real
  `customer_state`/`customer_city`. Forcing a `segment` column onto Olist
  customers would mean inventing a value the data doesn't support, which
  breaks the project's "never fabricate data" rule outright.
- `customer_id` in Olist is **order-scoped**, not person-scoped: the same
  human gets a new `customer_id` per order but a stable
  `customer_unique_id` (99,441 customer_id rows, 96,096 unique people). The
  synthetic `analytics.customers` table has no room for that distinction —
  mapping Olist into it would silently lose or misrepresent it.
- No marketing-campaign or activity/login data exists in Olist at all — the
  `analytics.marketing_campaigns`/`analytics.customer_activity` tables have
  no honest source data to map from.
- No product cost — Olist's `products` table has physical dimensions, not
  cost, so margin can't be computed the way `analytics` does.

The `analytics` schema is untouched and stays the eval ground-truth dataset
(the deliberate July Enterprise/North dip backing benchmark case `bi-004`,
Sec 6). `olist` is for realistic, messier demo questions where the "region"
concept is genuinely a Brazilian state, not a fabricated one.

No FK constraints on the Olist tables (see `app/db/models_olist.py` docstring)
— externally-sourced data of unverified referential cleanliness, loaded
faithfully via `scripts/load_olist.py` (asyncpg COPY, ~1.3M rows across 9
tables, real column names/types, empty CSV fields become SQL NULL only where
the column is genuinely nullable in the source).

## Critic retry loop

Hard cap at `CRITIC_MAX_RETRIES` (default 2, `.env`). Two failed corrections
force-exit to the Report Agent with confidence downgraded to `Low` and the
Limitations section disclosing that verification did not fully pass — the
graph never loops a third time, and never drops that disclosure. See
`app/agents/critic.py` and blueprint Sec 1 Fig. 2.
