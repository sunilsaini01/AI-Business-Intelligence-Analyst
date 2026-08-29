# Evaluation

Full metric definitions: blueprint Sec 6, extended in Phase 8 to cover the
Analysis/Visualization/Critic layers that postdate the original blueprint
(Phases 6/7/9). Deterministic metrics (SQL correctness, answer correctness,
analysis correctness, visualization correctness, critic correctness/
effectiveness, evidence groundedness, hallucination detection, latency,
end-to-end success) require **no LLM call** and are what `run_benchmark`
computes by default. The blueprint's original 0.30 LLM-judge weight
(relevance, recommendation quality — `app/evaluation/judges.py`) is isolated,
optional, and provider-agnostic: it is **not** invoked by a default run.

## Architecture

```
evaluation/datasets/benchmark.json   — 7 cases (5 Phase 8 + 2 Phase 16 ML
                                        cases), real verified ground truth
        |
app/evaluation/benchmark.py          — load_benchmark(): parses/validates cases
        |
app/evaluation/evaluator.py
    run_case_live(case)              — runs the SAME production graph
                                        (app.graph.workflow.get_graph) that
                                        the API uses; no second pipeline
        |
    evaluate_case_from_state(...)    — pure, deterministic scoring of the
                                        resulting AgentState
        |
app/evaluation/metrics.py            — the actual comparison logic per level
        |
app/evaluation/models.py             — CaseEvaluation / EvaluationRunSummary
                                        (internal dataclasses, not the API's
                                        Pydantic schemas — see file docstring)
        |
app/services/evaluation_service.py   — persists EvaluationRun/EvaluationResult
                                        rows (app schema), mirrors
                                        analysis_service.py's create+background
                                        shape
        |
POST /api/v1/evaluation/run          — fire-and-poll, same as POST /analyze
GET  /api/v1/evaluation/results
```

`scripts/run_evaluation.py` runs the same `run_benchmark` from the CLI and
writes a timestamped JSON report to `evaluation/reports/` — no DB required.

## Ground truth

`evaluation/datasets/benchmark.json` has 7 cases, each with a type-
discriminated `ground_truth` (`category_values` | `top_category` |
`period_comparison_with_contribution` | `trend_bounds` | `ml_forecast_quality`
| `ml_churn_quality`) that `app/evaluation/metrics.py`'s
`evaluate_answer_correctness` / `evaluate_analysis_correctness` dispatch on.
All values were re-verified via direct `psql` queries against the seeded
benchmark data at the time these cases were written, matching the fixed
July 2026 Enterprise/North revenue dip (`scripts/generate_data.py`). Ground
truth lives ONLY in this dataset — it is never referenced from `app/`
application code; a case's `tolerance` (`abs`/`rel`) governs how close a
live-run value must be, reusing the same tolerance rule the Critic itself
uses (`app/tools/critic_checks.py::values_are_close`). The two `ml_*`
ground-truth types (`ml-001`/`ml-002`, Phase 16) exist purely for schema
uniformity — `evaluate_ml_quality` reads `final_state["ml_results"]`
directly rather than consulting `ground_truth`'s contents, since the actual
quality gate is a fixed regression threshold, not a per-case expected value
(see "ML Agent evaluation" below).

## Levels (failure localization)

Each case is scored at 6 levels (`app/evaluation/models.py::LevelResult`):
`sql`, `analysis`, `visualization`, `ml`, `critic`, `end_to_end`. A case's
`first_failing_level` says where the pipeline first went wrong —
`app/evaluation/failure_analysis.py::summarize_failures` buckets a whole
run's failures by level so a regression is diagnosable ("3 cases now fail at
the SQL level"), not just a number that dropped. The `ml` level is `None`
("not applicable", the same convention every other level uses) for any
case whose question isn't predictive — it only activates when
`final_state["ml_results"]` is actually populated.

## ML Agent evaluation (Phase 16)

`app/agents/ml_agent.py`'s real, deterministic output (Phase 15) is now a
measurable regression contract, not just a manually-eyeballed number.
`app/evaluation/metrics.py::evaluate_forecast_quality` /
`evaluate_churn_quality` (dispatched via `evaluate_ml_quality`) gate the ML
Agent's actual measured QUALITY against fixed thresholds — never whether it
merely produced *a* result:

| Metric | Threshold | Why this number |
|---|---|---|
| Forecast MAPE | ≤ 25% | Phase 15 observed ~11% on the current ~20-month seeded revenue series with the linear-trend baseline; 25% absorbs ordinary month-to-month noise while still catching a genuine regression (e.g. a broken time alignment would push error far higher, not by a few points). |
| Forecast MAE | ≤ 20% of the held-out period's mean actual value | Dataset-scale-independent — a hard dollar ceiling would silently stop meaning anything if the seed data were ever regenerated at a different revenue scale. |
| Churn ROC-AUC | ≥ 0.65 | The PRIMARY signal (explicit instruction: accuracy alone is misleading under class imbalance). Phase 15 observed ~0.82; 0.65 is meaningfully above the 0.5 "coin flip" floor with room for legitimate split-to-split variance. |
| Churn accuracy / precision / recall | ≥ 0.60 / 0.55 / 0.55 | Phase 15 observed ~75% / ~76% / ~73%. All three floors sit well below the observed baseline — they exist to catch a genuinely broken model (e.g. a feature/label swap collapsing one class), not to lock in today's exact numbers. |

A graceful degradation (`ml_results["ok"] is False` — not appropriate /
insufficient data / an unexpected computation error, see "Failure handling"
below) is deliberately scored `correct: None` ("not applicable"), never a
quality failure — that would incorrectly punish the safe-degradation path
the ML Agent is supposed to take. An `ok: True` result missing an expected
metric key (malformed/fabricated-looking output that should never occur
from a real `ml_tools.py` call) is scored `correct: False`, not skipped —
the two failure modes stay distinguishable.

**Determinism**: `evaluate_and_forecast`'s time-aware split and
`fit_churn_classifier`'s `train_test_split`/`LogisticRegression` both use
fixed random seeds (`random_state=42`); running the same scenario twice
against the same data produces byte-identical metrics
(`tests/unit/test_ml_tools.py`, `tests/agents/test_ml_agent.py`) and the
same evaluation verdict (`tests/evaluation/test_ml_evaluation.py`'s "Case 5"
tests) — no LLM call is required for any of this (`ml_agent.py` never
imports `app.core.llm`), so none of it touches Groq/Anthropic quota.

**Data leakage audit**: forecasting's held-out test points are always the
MOST RECENT ones (never shuffled — `tests/unit/test_ml_tools.py::
test_forecast_never_uses_the_held_out_tail_to_fit_the_evaluated_model`
proves the model fit on history alone can't "see" a deliberately planted
spike in the held-out point). Churn's label (`churned`) is derived from
`days_since_last_order`/`last_order_date`, and neither of those — nor
`churned` itself — is ever among the columns actually fed to
`LogisticRegression.fit` (`feature_columns_for`'s explicit column list,
verified by `test_feature_columns_for_never_leaks_the_columns_the_label_is_
derived_from` and `test_fitted_churn_model_coefficients_never_include_a_
leaking_column`). Audited, no leakage found — nothing here was rewritten.

**Failure handling**: insufficient data (too few historical periods / too
few customers / too few examples of one churn class) degrades to a
structured `{"ok": False, "status": "insufficient_data", "reason": ...}`
result (unchanged from Phase 15). Phase 16 added one adjacent case: an
UNEXPECTED exception raised inside the model computation itself (as
opposed to a query-layer rejection, which was already handled) now also
degrades gracefully — `{"ok": False, "status": "error", "reason": "ML
computation failed unexpectedly (<ExceptionType>)."}`, never the raw
exception message — instead of crashing the entire analysis. See
`app/agents/ml_agent.py::_ml_error` and
`tests/agents/test_ml_agent.py`'s "unexpected computation error" tests.

**Critic / Report integration**: ML metrics are grounded exactly like any
other evidence — `app/tools/critic_checks.py::_collect_known_values`
accepts real `ml_results` metrics/predictions/feature-importance values as
citable numbers (`tests/unit/test_critic_checks.py`'s ML grounding tests,
Phase 15), and a fabricated ML number is still rejected even with real
`ml_results` present. A causal claim tied to ML language (e.g. "revenue is
declining *because* customers with low order counts are churning") has no
path to satisfy `check_causal_claims` (feature importance is never treated
as a "dominant contributor") and is rejected the same conservative way an
unsupported SQL-evidence causal claim would be — verified, not modified,
in `tests/unit/test_critic_checks.py`'s "ML causal-claim boundary" tests.
`app/agents/report_agent.py` only ever FORMATS an already-computed
`ml_results` into `report.ml_summary` (deterministic, zero LLM calls,
`_format_ml_summary`) — it never recalculates a metric and never enables
narrative generation based on whether ML was involved (`REPORT_NARRATIVE_
ENABLED` stays the only gate, still `false` by default).

**Model quality vs. business correctness**: clearing these thresholds
means the model performs measurably better than chance on THIS seeded
dataset, evaluated the way it was trained/split — it is not evidence that
the model's predictions are causally correct, that churn is genuinely
*caused* by the features it weights most heavily, or that the forecast
will hold in the real world. `ml_agent.py`'s own `limitations` field (and
the synthesis prompt's explicit rule against phrasing feature importance
as causation) exists precisely because a good benchmark score is a
statistical association claim, not a business-causality claim.

## Critic effectiveness

Measured by mutation testing against the REAL deterministic Critic checks
(`app/tools/critic_checks.py::run_all_deterministic_checks`), not a
re-implementation: a real, already-produced report is mutated with a
fabricated number and with an unsupported causal claim, and the Critic's
checks must escalate the verdict on both
(`app/evaluation/metrics.py::evaluate_critic_effectiveness`). Deterministic —
runs on every case, quota or not.

## LLM provider / quota handling

`run_case_live` invokes the real production graph, which means it makes
real LLM calls. A `RateLimitError` from either provider (`anthropic` or
`groq` — see `app/core/llm.py`'s provider abstraction) is caught per case
and recorded honestly as `status="SKIPPED_QUOTA"`, never hidden and never
counted as a failure — it does not abort the rest of the run. Deterministic
evaluation (everything except `judges.py`) still requires an LLM, since the
graph itself does; there is currently no LLM-free path through the real
pipeline (the Supervisor plans and synthesizes via the LLM). What Phase 8
does guarantee is: the metrics themselves are pure arithmetic/string
matching (unit-testable with a hand-built `AgentState` and zero LLM calls —
see `tests/evaluation/test_metrics.py` and `test_evaluator.py`), and a
quota failure during a live run is recorded, not disguised as a bug.

## Regression tracking

Every `EvaluationResult` row (`app.evaluation_results`) records
case_id/scores/latency/passed per run; `EvaluationRun.aggregate_scores`
holds the run-level summary — that's what makes "last week 0.94, today
0.71" diffable instead of anecdotal.
