"""Shared helper for putting query evidence into a prompt without blowing
through a provider's per-request token limit.

Discovered live against Groq's free tier (8000 TPM): capping by row *count*
alone isn't enough — a 4-table Olist join with review text is far wider per
row than a 2-column analytics aggregate, so the same row cap can be fine for
one and still oversized for the other. Cap by serialized character length
too, as a hard backstop independent of schema shape.

Also discovered live: a small row cap combined with `ORDER BY month, ...` can
silently drop an entire month from what the model sees — e.g. a June+July
breakdown sorted alphabetically puts all of June before any July row, so a
10-row cap showed only June and the Supervisor correctly (from its own view)
reported July data as "not provided" even though the SQL Agent's full result
had it. Callers whose evidence doesn't compound across multiple steps (the
Supervisor's synthesis sees each query's result exactly once) can afford a
higher `max_rows` than callers that accumulate evidence across several steps
(the SQL Agent's per-step prior-evidence, included in *every* subsequent
step's prompt) — see the different caps each passes.
"""

from __future__ import annotations

import json
from typing import Any

_DEFAULT_MAX_ROWS = 10
_MAX_CHARS = 3000


def compact_rows_json(rows: list[dict[str, Any]], *, max_rows: int = _DEFAULT_MAX_ROWS) -> str:
    text = json.dumps(rows[:max_rows], default=str)
    if len(text) > _MAX_CHARS:
        text = text[:_MAX_CHARS] + f"... (truncated, {len(rows)} rows total)"
    return text
