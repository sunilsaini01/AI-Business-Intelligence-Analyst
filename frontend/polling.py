"""Pure bounded-polling policy (Phase 12) — no `streamlit`, no real
sleep/HTTP, so the *decision* logic (not the I/O) is unit-testable.
Deciding whether to keep polling, and recognizing DONE/FAILED/timeout, is
what actually needs to be correct; the sleep/HTTP calls around it are
thin glue in app.py.
"""

from __future__ import annotations

from typing import Literal

PollDecision = Literal["continue", "done", "failed", "timeout"]

DEFAULT_POLL_INTERVAL_SECONDS = 2.0
DEFAULT_POLL_TIMEOUT_SECONDS = 180.0

_TERMINAL_STATUSES = frozenset({"DONE", "FAILED"})


def next_poll_decision(
    status: str, elapsed_seconds: float, timeout_seconds: float = DEFAULT_POLL_TIMEOUT_SECONDS
) -> PollDecision:
    """`status` is whatever `AnalysisSession.status` (app/db/models.py)
    currently is — PENDING/ANALYZING/DONE/FAILED, the one existing enum,
    never a second one. Timeout is checked AFTER the terminal statuses so a
    slow-but-just-finished response never gets misreported as a timeout.
    """
    if status == "DONE":
        return "done"
    if status == "FAILED":
        return "failed"
    if elapsed_seconds >= timeout_seconds:
        return "timeout"
    return "continue"
