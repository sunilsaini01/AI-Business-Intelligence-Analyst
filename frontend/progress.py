"""Pure pipeline-progress logic (Phase 12) — no `streamlit` import, fully
unit-testable. Turns the API's `current_stage` (Phase 11,
app/services/analysis_service.py::get_current_stage — the most recently
COMPLETED node, never "currently executing") into an honest checklist
against the canonical pipeline order.

`current_stage` is a single snapshot per poll — it does not, by itself,
distinguish "the pipeline is progressing normally" from "the Critic just
sent it back to the Supervisor for a revision" (Sec 1 Fig. 2's bounded
retry loop can make `current_stage` report "supervisor" again AFTER
"critic" already completed once). `advance_progress` takes the caller's
running `max_index_so_far` (Streamlit session_state persists it across
polls) and only ever moves it forward, so a stage already confirmed
complete stays checked through a retry instead of "un-completing".
"""

from __future__ import annotations

from dataclasses import dataclass

PIPELINE_STAGES: list[str] = [
    "supervisor",
    "sql_agent",
    "analysis_agent",
    "ml_agent",
    "visualization_agent",
    "critic",
    "report_agent",
]

STAGE_LABELS: dict[str, str] = {
    "supervisor": "Supervisor",
    "sql_agent": "SQL Agent",
    "analysis_agent": "Analysis Agent",
    "ml_agent": "ML Agent",
    "visualization_agent": "Visualization",
    "critic": "Critic",
    "report_agent": "Report Agent",
}


@dataclass(frozen=True)
class StageStatus:
    name: str
    label: str
    completed: bool
    is_next: bool


def advance_progress(current_stage: str | None, max_index_so_far: int) -> tuple[list[StageStatus], int]:
    """Returns (checklist, new_max_index). `completed=True` means the API
    has, at some point, reported this stage (or a later one) as the most
    recently finished node — never a claim about what's executing right
    now. `is_next` marks only the single stage immediately after the
    furthest one confirmed so far, for an honest "up next" hint — not a
    claim that it has started.
    """
    if current_stage is not None and current_stage in PIPELINE_STAGES:
        max_index_so_far = max(max_index_so_far, PIPELINE_STAGES.index(current_stage))

    statuses = [
        StageStatus(
            name=name,
            label=STAGE_LABELS[name],
            completed=i <= max_index_so_far,
            is_next=i == max_index_so_far + 1,
        )
        for i, name in enumerate(PIPELINE_STAGES)
    ]
    return statuses, max_index_so_far
