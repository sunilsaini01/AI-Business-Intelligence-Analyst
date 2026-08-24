"""ML Agent (Sec 1, Sec 2, Sec 5). Phase 8. Only runs when Analysis flags
`needs_prediction` — never on every question (Sec 1: "why a supervisor").

Reads: analysis_results, raw rows for feature building
Writes: ml_results
LLM calls: 0 for fit/predict — deterministic sklearn/XGBoost. The Report Agent
explains the output; this agent never narrates it.

Sec 5 RULE: this module must never import app.core.llm (same CI grep as
analysis_agent.py).

Sec 2 JUDGMENT CALL: revenue forecasting starts with a linear-trend /
seasonal-naive baseline (NumPy polyfit or sklearn LinearRegression) — reach
for XGBoost only if the baseline demonstrably underfits and there's enough
history to validate it. Churn classification is XGBoost/SHAP's actual fit
here (row-per-customer, real feature interactions).

On too-few-rows-to-fit: return ml_results=None with a reason (Sec 9) — the
Report Agent omits the prediction section rather than fabricating one.
"""

from __future__ import annotations

from app.graph.state import AgentState


async def ml_agent_node(state: AgentState) -> AgentState:
    raise NotImplementedError("ML Agent — Sec 12 Phase 8, baseline forecast first, XGBoost only if justified")
