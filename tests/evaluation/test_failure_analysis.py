from app.evaluation.failure_analysis import summarize_failures
from app.evaluation.models import CaseEvaluation


def _case(case_id: str, status: str, first_failing_level: str | None = None) -> CaseEvaluation:
    return CaseEvaluation(
        case_id=case_id, question="q", status=status, expected={}, actual={},
        first_failing_level=first_failing_level,
    )


def test_summarize_failures_buckets_by_first_failing_level():
    results = [
        _case("a", "PASSED"),
        _case("b", "FAILED", first_failing_level="sql"),
        _case("c", "FAILED", first_failing_level="sql"),
        _case("d", "FAILED", first_failing_level="visualization"),
        _case("e", "ERROR"),
        _case("f", "SKIPPED_QUOTA"),
    ]
    buckets = summarize_failures(results)
    assert buckets["sql"] == ["b", "c"]
    assert buckets["visualization"] == ["d"]
    assert buckets["error"] == ["e"]
    assert buckets["skipped_quota"] == ["f"]
    assert "a" not in [c for v in buckets.values() for c in v]


def test_summarize_failures_all_passed_is_empty():
    results = [_case("a", "PASSED"), _case("b", "PASSED")]
    assert summarize_failures(results) == {}
