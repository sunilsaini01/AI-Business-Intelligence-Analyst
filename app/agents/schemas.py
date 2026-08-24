"""Pydantic models for structured LLM output (Sec 5: "the Supervisor should
produce structured state, not uncontrolled text"). Shared between the
Supervisor and SQL Agent so both sides of the handoff agree on shape.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SupervisorPlan(BaseModel):
    """First-pass Supervisor output: classify + plan, before any SQL runs.

    `steps` has no `min_length` in the JSON schema on purpose — a hard
    minimum is a *conditional* rule (required when in-scope, meaningless
    when out_of_scope) that provider-side structured-output validation
    enforces unconditionally. Discovered live against Groq: a model that
    correctly reasoned "no such column exists -> out_of_scope=True,
    steps=[]" got its whole tool call rejected server-side with a 400
    because `steps` was empty, before this code ever saw it — the model was
    right and the schema was wrong. The invariant is enforced here instead,
    after parsing, where a violation can be handled instead of crashing.
    """

    out_of_scope: bool = Field(
        description="True if the question cannot be answered from either dataset at all "
        "(not a business/data question, or asks for something neither schema could contain)."
    )
    intent: Literal["descriptive", "diagnostic", "predictive", "comparative", "out_of_scope"]
    target_schema: Literal["analytics", "olist"] = Field(
        description="Which dataset answers this question. 'analytics' = synthetic B2B SaaS "
        "data (regions, segments, campaigns). 'olist' = real Brazilian e-commerce marketplace "
        "orders (states/cities, no segments, no campaigns)."
    )
    steps: list[str] = Field(
        max_length=6,
        description="1-6 short, concrete steps describing what evidence to gather, e.g. "
        "'Retrieve June and July revenue by month', 'Break down July revenue by region'. "
        "Leave empty only when out_of_scope is true.",
    )
    reasoning: str = Field(description="One or two sentences on why this plan answers the question.")

    @model_validator(mode="after")
    def _empty_steps_implies_out_of_scope(self) -> "SupervisorPlan":
        """Belt-and-suspenders: if the model left steps empty without
        marking out_of_scope, treat it as out_of_scope rather than letting
        app/agents/supervisor.py loop on an empty plan (see workflow.py's
        routing note on why that can't infinite-loop, but it can still waste
        calls) — a plan with zero steps can't do anything else useful.
        """
        if not self.steps and not self.out_of_scope:
            self.out_of_scope = True
        return self


class SupervisorSynthesis(BaseModel):
    """Second-pass Supervisor output: turn gathered SQL evidence into an answer.
    Every number must come from the evidence passed in the prompt — this model
    doesn't enforce that itself (the Critic Agent, Phase 9, checks that
    deterministically after the fact — see app/tools/critic_checks.py), but
    the system prompt instructs it strictly and `insufficient_evidence` gives
    the model an explicit, structured way to decline instead of inventing a
    cause.
    """

    insufficient_evidence: bool = Field(
        description="True if the gathered evidence does not clearly support a conclusion — "
        "e.g. a diagnostic question where the query results don't isolate a cause."
    )
    executive_summary: str = Field(
        description="1-3 sentences answering the question. If insufficient_evidence is true, "
        "this must be exactly: 'Insufficient evidence to determine the cause.' (or the "
        "appropriate non-diagnostic equivalent, e.g. 'Insufficient evidence to answer this "
        "question.')"
    )
    key_findings: list[str] = Field(
        description="Each finding must cite a specific number that appears in the evidence."
    )
    confidence: Literal["Low", "Medium", "High"]
    limitations: str = Field(default="")


class SQLGeneration(BaseModel):
    """SQL Agent's output for one query. `sql` must be schema-qualified,
    single-statement SELECT — app/tools/database_tools.py::validate_sql is
    the actual enforcement; this is just what the model is asked to produce.
    """

    sql: str
    purpose: str = Field(description="One short sentence: what business question this query answers.")


class CriticSemanticCheck(BaseModel):
    """Critic Agent (Phase 9), the ONE genuinely-semantic check — everything
    else the Critic does is deterministic Python (app/tools/critic_checks.py):
    numeric grounding, period/chart consistency, and contribution arithmetic
    are all mechanically checkable, but "does this wording overstate what the
    facts support" needs language understanding. Kept deliberately narrow —
    the model sees only the executive summary, key findings, and the
    Analysis Agent's own facts/interpretations, never raw SQL rows.
    """

    supported: bool = Field(
        description="True if every claim in the executive summary and key findings is backed "
        "by the given facts/interpretations, with no invented claims and no causal/certainty "
        "language stronger than the evidence supports."
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Direct quotes (or close paraphrases) of specific phrases that go beyond "
        "what the facts/interpretations support. Empty if supported=true.",
    )
    reasoning: str = Field(description="One or two sentences explaining the verdict.")


class ReportNarrative(BaseModel):
    """Report Generator (Phase 10), the ONE optional LLM call — a pure
    wording pass over an already-Critic-approved executive summary and key
    findings, for a non-technical stakeholder audience. Never asked to add
    information: app/agents/report_agent.py re-validates the result against
    the same numeric-grounding check the Critic uses
    (app/tools/critic_checks.py::check_numerical_grounding) and discards it
    if it introduces anything not already present in the source text.
    """

    narrative: str = Field(
        description="A short (2-4 sentence), stakeholder-friendly paragraph reorganizing/clarifying "
        "the given executive summary and key findings. Must not introduce any number, entity, "
        "percentage, period, or claim that isn't already present verbatim in the given text."
    )
