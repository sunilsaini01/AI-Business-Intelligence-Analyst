"""StateGraph wiring (Sec 1; user Phase 4/5/6/7/9/10 spec).

Default graph as of Phase 15:

    supervisor (plan) -> sql_agent -> analysis_agent -> ml_agent
        -> visualization_agent -> supervisor (synthesize) -> critic
            -> PASS/WARN, or FAIL at max_retries -> report_agent -> END
            -> FAIL with retries left -> supervisor (revise) -> critic -> ...
    supervisor (out_of_scope) -> END directly (skips critic AND report_agent
        — a fixed "I can't answer that" message never touches analysis/
        evidence, so there's nothing to review or finalize)

ml_agent (app/agents/ml_agent.py, Phase 15 Objective 4) is always in the
chain, same as analysis_agent/visualization_agent — it decides internally
whether `state["intent"] == "predictive"` before doing any real work
(fitting a model, running its own fixed/reviewed queries through the same
safety pipeline as sql_agent), so a non-predictive question pays only the
cost of one fast no-op node, not a conditional graph edge.

Three-way fork after the Supervisor (`_route_after_supervisor`): `report is
None` -> still gathering evidence, go to sql_agent. `report is not None` and
`intent == "out_of_scope"` -> done, skip the critic entirely. Otherwise a
real synthesis just happened -> critic reviews it.

The retry loop is driven entirely by `report`, reusing the exact signal
`_route_after_supervisor` already keys off — app/agents/critic.py clears
`report` back to None on a FAIL with retries remaining (which naturally
re-enters `_synthesize()`, now with `critic_feedback` in state) and leaves it
set (with confidence force-degraded) once retries are exhausted or the
verdict is PASS/WARN. `_route_after_critic` just checks `report is None`
again — no separate retry-tracking routing logic needed, `state["retry_count"]`
vs `state["max_retries"]` bounds it (see critic.py), so this can't loop
forever. Every terminal state the Critic hands off (`report is not None`,
whether that's a clean PASS/WARN or a force-degraded FAIL-exhausted report)
now goes to `report_agent` (Phase 10, app/agents/report_agent.py) — a
presentation/finalization pass, never a second analysis or a chance to
override what the Critic decided — before reaching END.

`build_phase0_graph()` is kept as-is for tests/integration/test_phase0_workflow.py
— it's superseded as the default but still exercises the original
hard-coded-query plumbing check from Sec 12 Phase 0.
"""

from __future__ import annotations

import functools

from langgraph.graph import END, StateGraph

from app.agents.analysis_agent import analysis_agent_node
from app.agents.critic import critic_node
from app.agents.ml_agent import ml_agent_node
from app.agents.report_agent import report_agent_node
from app.agents.sql_agent import sql_agent_node
from app.agents.supervisor import supervisor_node
from app.agents.visualization_agent import visualization_agent_node
from app.core.llm import LLMClientProtocol
from app.graph.nodes import fetch_node, respond_node
from app.graph.state import AgentState


def _route_after_supervisor(state: AgentState) -> str:
    if state["report"] is None:
        return "sql_agent"
    if state["intent"] == "out_of_scope":
        return "end"
    return "critic"


def _route_after_critic(state: AgentState) -> str:
    # `report is None` -> Critic FAILed with retries left, cleared `report`
    # to trigger a Supervisor revision. Otherwise the Critic is done with
    # this attempt (PASS, WARN, or a FAIL that force-degraded the report at
    # max_retries — see app/agents/critic.py) -> hand off to the Report
    # Generator (Phase 10) for presentation/finalization, never straight to
    # END — that's the one thing that changed here from Phase 9.
    return "supervisor" if state["report"] is None else "report_agent"


def build_graph(llm: LLMClientProtocol | None = None):
    """`llm=None` (production): each LLM-backed node lazily grabs the real
    app.core.llm.get_llm_client(). Tests pass a ScriptedLLMClient here so the
    whole graph runs deterministically with no network/API key — see
    tests/fakes.py and tests/integration/test_workflow.py. analysis_agent and
    visualization_agent never take an `llm` argument (Sec 5: 0 LLM calls
    each), so neither is affected by this parameter; critic_node does take
    one (for its single isolated semantic check) and is wired the same way
    as supervisor/sql_agent.
    """
    supervisor = functools.partial(supervisor_node, llm=llm) if llm else supervisor_node
    sql_agent = functools.partial(sql_agent_node, llm=llm) if llm else sql_agent_node
    critic = functools.partial(critic_node, llm=llm) if llm else critic_node
    report_agent = functools.partial(report_agent_node, llm=llm) if llm else report_agent_node

    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("sql_agent", sql_agent)
    graph.add_node("analysis_agent", analysis_agent_node)
    graph.add_node("ml_agent", ml_agent_node)
    graph.add_node("visualization_agent", visualization_agent_node)
    graph.add_node("critic", critic)
    graph.add_node("report_agent", report_agent)

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges(
        "supervisor", _route_after_supervisor, {"sql_agent": "sql_agent", "critic": "critic", "end": END}
    )
    graph.add_edge("sql_agent", "analysis_agent")
    # Phase 15, Objective 4: ml_agent sits between analysis_agent and
    # visualization_agent — after the deterministic analysis it may build
    # features from, before the chart selection that (like ml_agent
    # itself) never runs LLM-invented logic. Always in the linear chain,
    # same pattern as analysis_agent/visualization_agent: it decides
    # internally whether it has anything to do (state["intent"] ==
    # "predictive") rather than the graph routing around it — a no-op
    # here is just as fast as a conditional edge would be, and this keeps
    # the graph topology unchanged everywhere else (retry loop, out_of_
    # scope short-circuit, Critic authority all untouched).
    graph.add_edge("analysis_agent", "ml_agent")
    graph.add_edge("ml_agent", "visualization_agent")
    graph.add_edge("visualization_agent", "supervisor")
    graph.add_conditional_edges(
        "critic", _route_after_critic, {"supervisor": "supervisor", "report_agent": "report_agent"}
    )
    graph.add_edge("report_agent", END)

    return graph.compile()


def build_phase0_graph():
    graph = StateGraph(AgentState)
    graph.add_node("fetch", fetch_node)
    graph.add_node("respond", respond_node)

    graph.set_entry_point("fetch")
    graph.add_edge("fetch", "respond")
    graph.add_edge("respond", END)

    return graph.compile()


_compiled_graph = None
_compiled_phase0_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def get_phase0_graph():
    global _compiled_phase0_graph
    if _compiled_phase0_graph is None:
        _compiled_phase0_graph = build_phase0_graph()
    return _compiled_phase0_graph
