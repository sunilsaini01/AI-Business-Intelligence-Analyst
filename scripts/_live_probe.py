"""One-off manual diagnostic for live-LLM validation (not part of the app,
not imported by anything). Runs the real Supervisor -> SQL Agent graph for a
single question and prints full instrumentation: intent, target schema,
generated SQL per step, validation result, row counts, retry evidence,
timing, and the final synthesized answer. Never logs the API key.

Usage: docker compose exec api python scripts/_live_probe.py "question text"
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.graph.state import new_state  # noqa: E402
from app.graph.workflow import get_graph  # noqa: E402


async def main(question: str) -> None:
    graph = get_graph()
    started = time.perf_counter()
    result = await graph.ainvoke(new_state(question))
    elapsed = time.perf_counter() - started

    print("=" * 70)
    print("QUESTION:", question)
    print("=" * 70)
    print("intent:", result["intent"])
    print("target_schema:", result["target_schema"])
    print("plan steps:")
    for s in result["plan"]:
        print("  -", s)
    print()
    print(f"sql_queries ({len(result['sql_queries'])}):")
    for i, q in enumerate(result["sql_queries"]):
        print(f"  [{i}] validated_ok={q['validated_ok']} row_count={q['row_count']} exec_ms={q['exec_ms']:.1f}")
        print(f"      sql: {q['text']}")
        if not q["validated_ok"]:
            print(f"      rejection_reason: {q['rejection_reason']}")
        else:
            print(f"      rows (up to 10): {json.dumps(q['rows'][:10], default=str)}")
    print()
    print("ANALYSIS_RESULTS (Phase 6, deterministic pandas — 0 LLM calls):")
    print(json.dumps(result["analysis_results"], indent=2, default=str))
    print()
    print(f"CHARTS (Phase 7, deterministic selection — 0 LLM calls, {len(result['charts'])} produced):")
    for i, c in enumerate(result["charts"]):
        print(f"  [{i}] chart_type={c['chart_type']} title={c['title']!r} source_analysis={c['source_analysis']}")
        print(f"      x_axis={c['x_axis']} y_axis={c['y_axis']} sort={c['sort']}")
        print(f"      reason: {c['reason']}")
        if c["limitations"]:
            print(f"      limitations: {c['limitations']}")
        print(f"      data: {json.dumps(c['data'][:8], default=str)}")
    print()
    print("trace node sequence:", [t["node"] for t in result["trace"]])
    print()
    print("REPORT:")
    print(json.dumps(result["report"], indent=2, default=str))
    print()
    print(f"total wall time: {elapsed:.2f}s")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
