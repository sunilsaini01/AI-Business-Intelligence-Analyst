"""SQL / Retrieval Agent (Sec 1, Sec 4, Sec 5; user Phase 5 spec).

Generates one query per Supervisor plan step (capped — see _MAX_STEPS), each
independently validated and executed through app/tools/database_tools.py.
The LLM never touches Postgres directly: `run_query()` is the only path from
a generated string to the database, and it re-validates from scratch
regardless of what the LLM claims about its own SQL.

On rejection: one regeneration attempt with the rejection reason fed back
(Sec 9 "one regeneration attempt, then escalate"). If still invalid, the
query is recorded with `validated_ok=False` and the Supervisor's synthesis
step sees the rejection reason as part of its evidence — it does not crash
the graph run.

`llm` is dependency-injected — see app/agents/supervisor.py's docstring and
tests/agents/test_sql_agent.py for the fake-LLM test pattern.
"""

from __future__ import annotations

import time

from app.agents.prompt_utils import compact_rows_json
from app.agents.schemas import SQLGeneration
from app.core.llm import LLMClientProtocol, ModelTier, get_llm_client
from app.graph.state import AgentState, trace_event
from app.tools.database_tools import run_query
from app.tools.schema_tools import format_schema_for_prompt, get_analytics_schema

_MAX_STEPS = 4  # bounds LLM calls + query count per request

# Discovered live: a complex diagnostic query (CTE + long WHERE) with the
# default 2048-token completion budget can get cut off mid-JSON-string,
# which Groq's server then rejects outright (400, "failed to parse tool call
# arguments") rather than returning a partial/salvageable response — there's
# nothing to catch and retry, the call itself errors. More headroom avoids
# hitting the cutoff in the first place.
_SQL_GENERATION_MAX_TOKENS = 4096

_GENERATION_SYSTEM_PROMPT = """You write a single read-only PostgreSQL SELECT query to answer one
step of a business-analysis plan.

Rules:
- Every table reference MUST be schema-qualified: {target_schema}.<table>, e.g.
  {target_schema}.orders. Unqualified table names are rejected.
- Only reference tables/columns that appear in the schema below — never invent one.
- Single SELECT statement only (CTEs with `WITH ... AS (SELECT ...)` are fine). No INSERT/
  UPDATE/DELETE/DDL — you don't have permission to write, and it will be rejected.
- Prefer aggregation (SUM/COUNT/AVG with GROUP BY) over fetching raw rows when the plan step
  asks for a total, comparison, or breakdown.
- Do not add your own LIMIT unless the step specifically asks for "top N" — the system adds a
  safety cap automatically.
- Write the SQL on as few lines as reasonable (avoid heavy indentation/newline formatting) —
  it doesn't need to be pretty, and a shorter response is less likely to get cut off.

Column semantics to get right (verify this kind of thing against the schema/data instead of
guessing when a column name is ambiguous):
- Any column named `discount` is a FRACTION off (0.0-1.0), not a dollar amount. Revenue/total
  from a line item is `quantity * unit_price * (1 - discount)` — multiply, never subtract
  `discount` directly as if it were currency.

Dates: NEVER guess a literal year for "this month" / "last month" / a specific month name unless
the question or the evidence below already pins one down. The data may span multiple years, so a
bare month name (e.g. "July") is ambiguous. Prefer deriving the relevant year from the evidence
already gathered (below) — if an earlier step already found which year's July is the one in
question, use that exact year. If this is the first query and no year is given, use a subquery
against MAX(the relevant date column) to find the most recent occurrence instead of hardcoding one.
When the question names one specific period (e.g. "July"), scope your WHERE clause to exactly
that period and the one immediately before it (e.g. June and July only) — do not widen the window
to extra surrounding months unless the plan step explicitly asks for a longer trend; an extra
month changes which two periods "the decrease" refers to.

Shape of comparison/breakdown results: when a step asks to compare two periods (e.g. June vs
July) broken down by a dimension (segment, region, category), put the PERIOD AS A ROW VALUE via
GROUP BY — one row per (dimension, period) pair, e.g. columns (segment, month, revenue) — not as
separate columns per period (e.g. avoid revenue_june/revenue_july as two different output
columns). The row-per-period shape is what downstream analysis consumes; a column-per-period
shape cannot be compared automatically even though the numbers are correct.

Original question (for context): {question}

Evidence gathered so far from earlier plan steps (empty if this is the first step — use it to
stay consistent, e.g. don't re-guess a year a previous step already established):
{prior_evidence}

Schema ({target_schema} only):
{schema_text}
"""

_RETRY_SYSTEM_PROMPT = (
    _GENERATION_SYSTEM_PROMPT
    + """
Your previous attempt was rejected: {rejection_reason}
Previous SQL: {previous_sql}
Fix it and try again.
"""
)


async def sql_agent_node(state: AgentState, llm: LLMClientProtocol | None = None) -> AgentState:
    llm = llm or get_llm_client()
    state["trace"].append(trace_event("sql_agent", "enter"))
    node_started = time.perf_counter()

    target_schema = state["target_schema"] or "analytics"
    schema = await get_analytics_schema()
    schema_text = format_schema_for_prompt(schema, only_schema=target_schema)

    prior_evidence_lines: list[str] = []
    for step in state["plan"][:_MAX_STEPS]:
        record = await _generate_and_run(
            question=state["question"],
            step=step,
            target_schema=target_schema,
            schema_text=schema_text,
            prior_evidence="\n".join(prior_evidence_lines) if prior_evidence_lines else "(none yet)",
            llm=llm,
        )
        state["sql_queries"].append(record)

        if record["validated_ok"]:
            prior_evidence_lines.append(
                f"- Step '{step}' -> {record['row_count']} rows: {compact_rows_json(record['rows'])}"
            )
        else:
            prior_evidence_lines.append(f"- Step '{step}' -> query rejected: {record['rejection_reason']}")

    state["trace"].append(
        trace_event("sql_agent", "exit", duration_ms=(time.perf_counter() - node_started) * 1000)
    )
    return state


async def _generate_and_run(
    *,
    question: str,
    step: str,
    target_schema: str,
    schema_text: str,
    prior_evidence: str,
    llm: LLMClientProtocol,
) -> dict:
    system = _GENERATION_SYSTEM_PROMPT.format(
        target_schema=target_schema, question=question, schema_text=schema_text, prior_evidence=prior_evidence
    )
    generation = await llm.complete_structured(
        tier=ModelTier.FAST,
        system=system,
        messages=[{"role": "user", "content": f"Plan step: {step}"}],
        response_model=SQLGeneration,
        max_tokens=_SQL_GENERATION_MAX_TOKENS,
    )

    started = time.perf_counter()
    result = await run_query(generation.sql)

    if not result.ok:
        retry_system = _RETRY_SYSTEM_PROMPT.format(
            target_schema=target_schema,
            question=question,
            schema_text=schema_text,
            prior_evidence=prior_evidence,
            rejection_reason=result.rejection_reason,
            previous_sql=generation.sql,
        )
        retry_generation = await llm.complete_structured(
            tier=ModelTier.FAST,
            system=retry_system,
            messages=[{"role": "user", "content": f"Plan step: {step}"}],
            response_model=SQLGeneration,
            max_tokens=_SQL_GENERATION_MAX_TOKENS,
        )
        result = await run_query(retry_generation.sql)

    exec_ms = (time.perf_counter() - started) * 1000
    return {
        "text": result.sql or generation.sql,
        "validated_ok": result.ok,
        "rejection_reason": result.rejection_reason,
        "rows": result.rows,
        "row_count": result.row_count,
        "exec_ms": result.exec_ms if result.ok else exec_ms,
    }
