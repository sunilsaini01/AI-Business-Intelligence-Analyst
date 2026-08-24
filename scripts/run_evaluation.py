"""CLI entry point for the benchmark runner (Sec 6, Sec 12).

Runs the same production graph (app.graph.workflow.get_graph) that the API
uses against every case in the given dataset, prints a summary, and writes a
timestamped JSON report to evaluation/reports/ (see
app/evaluation/evaluator.py::run_benchmark). Does not touch the app DB — for
a DB-persisted run (EvaluationRun/EvaluationResult rows queryable via
GET /api/v1/evaluation/results), use the API endpoint instead
(POST /api/v1/evaluation/run), which wraps the same run_benchmark call.

Usage:
    docker compose exec api python -m scripts.run_evaluation
    docker compose exec api python -m scripts.run_evaluation --label nightly --dataset evaluation/datasets/benchmark.json
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from app.evaluation.evaluator import run_benchmark


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BI agent evaluation benchmark.")
    parser.add_argument("--dataset", default="evaluation/datasets/benchmark.json", help="Path to a benchmark JSON file")
    parser.add_argument("--label", default="cli-run", help="Label for this run, used in the report filename")
    return parser.parse_args(argv)


async def _main_async(args: argparse.Namespace) -> int:
    summary = await run_benchmark(dataset_path=args.dataset, label=args.label)

    print(f"Evaluation run: {summary.label} ({summary.model_name})")
    print(f"  cases: {summary.total_cases} total, {summary.passed} passed, {summary.failed} failed, "
          f"{summary.errored} errored, {summary.skipped} skipped (quota/rate-limit)")
    print(f"  end_to_end_success_rate: {summary.end_to_end_success_rate:.2f}")
    print(f"  hallucination_rate: {summary.hallucination_rate:.2f}")
    if summary.mean_latency_ms is not None:
        print(f"  mean_latency_ms: {summary.mean_latency_ms:.0f}")
    print("  aggregate_scores:")
    for key, value in summary.aggregate_scores.items():
        print(f"    {key}: {value:.3f}")

    if summary.skipped:
        print(
            f"\n{summary.skipped} case(s) skipped due to LLM provider quota/rate limits — "
            "not a code failure. Re-run when quota is available for full coverage.",
            file=sys.stderr,
        )

    return 1 if summary.errored else 0


def main() -> int:
    args = _parse_args(sys.argv[1:])
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
