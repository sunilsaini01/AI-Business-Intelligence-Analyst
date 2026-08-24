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
evaluation/datasets/benchmark.json   — 5 cases, real verified ground truth
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

`evaluation/datasets/benchmark.json` has 5 cases, each with a type-
discriminated `ground_truth` (`category_values` | `top_category` |
`period_comparison_with_contribution` | `trend_bounds`) that
`app/evaluation/metrics.py`'s `evaluate_answer_correctness` /
`evaluate_analysis_correctness` dispatch on. All values were re-verified via
direct `psql` queries against the seeded benchmark data at the time these
cases were written, matching the fixed July 2026 Enterprise/North revenue
dip (`scripts/generate_data.py`). Ground truth lives ONLY in this dataset —
it is never referenced from `app/` application code; a case's `tolerance`
(`abs`/`rel`) governs how close a live-run value must be, reusing the same
tolerance rule the Critic itself uses
(`app/tools/critic_checks.py::values_are_close`).

## Levels (failure localization)

Each case is scored at 5 levels (`app/evaluation/models.py::LevelResult`):
`sql`, `analysis`, `visualization`, `critic`, `end_to_end`. A case's
`first_failing_level` says where the pipeline first went wrong —
`app/evaluation/failure_analysis.py::summarize_failures` buckets a whole
run's failures by level so a regression is diagnosable ("3 cases now fail at
the SQL level"), not just a number that dropped.

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
