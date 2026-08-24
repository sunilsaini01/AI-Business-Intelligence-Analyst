"""Groups failed/errored/skipped benchmark cases by which of the 5 levels
(sql, analysis, visualization, critic, end_to_end) first went wrong, or by
ERROR/SKIPPED_QUOTA when the case never reached scoring at all — so a
regression is diagnosable ("3 cases now fail at the SQL level") rather than
just a number that dropped. Reuses `CaseEvaluation.first_failing_level`,
computed once in app/evaluation/evaluator.py::evaluate_case_from_state,
rather than re-deriving it from raw scores here.
"""

from __future__ import annotations

from collections import defaultdict

from app.evaluation.models import CaseEvaluation


def summarize_failures(results: list[CaseEvaluation]) -> dict[str, list[str]]:
    """Returns {bucket: [case_id, ...]} for every case that did not pass —
    a case with status="PASSED" contributes nothing. Buckets are the 5
    evaluation levels plus "error" (pipeline exception) and "skipped_quota"
    (live LLM call was rate-limited/quota-exhausted, not a code failure)."""
    buckets: dict[str, list[str]] = defaultdict(list)
    for case in results:
        if case.status == "PASSED":
            continue
        if case.status == "ERROR":
            buckets["error"].append(case.case_id)
        elif case.status == "SKIPPED_QUOTA":
            buckets["skipped_quota"].append(case.case_id)
        elif case.first_failing_level is not None:
            buckets[case.first_failing_level].append(case.case_id)
        else:
            buckets["unknown"].append(case.case_id)
    return dict(buckets)
