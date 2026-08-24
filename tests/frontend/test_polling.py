from polling import next_poll_decision


def test_pending_before_timeout_continues():
    assert next_poll_decision("PENDING", elapsed_seconds=1.0, timeout_seconds=180.0) == "continue"


def test_analyzing_before_timeout_continues():
    assert next_poll_decision("ANALYZING", elapsed_seconds=50.0, timeout_seconds=180.0) == "continue"


def test_done_stops_polling_regardless_of_elapsed_time():
    assert next_poll_decision("DONE", elapsed_seconds=0.0, timeout_seconds=180.0) == "done"


def test_failed_stops_polling_regardless_of_elapsed_time():
    assert next_poll_decision("FAILED", elapsed_seconds=0.0, timeout_seconds=180.0) == "failed"


def test_timeout_when_elapsed_exceeds_limit_and_still_pending():
    assert next_poll_decision("ANALYZING", elapsed_seconds=181.0, timeout_seconds=180.0) == "timeout"


def test_exactly_at_timeout_boundary_times_out():
    assert next_poll_decision("PENDING", elapsed_seconds=180.0, timeout_seconds=180.0) == "timeout"


def test_done_takes_priority_over_timeout_even_if_both_conditions_true():
    """A slow-but-just-finished response must never be misreported as a timeout."""
    assert next_poll_decision("DONE", elapsed_seconds=999.0, timeout_seconds=180.0) == "done"


def test_never_loops_forever_bounded_by_timeout_only_four_possible_decisions():
    decisions = {
        next_poll_decision("PENDING", 0, 10),
        next_poll_decision("ANALYZING", 5, 10),
        next_poll_decision("DONE", 0, 10),
        next_poll_decision("FAILED", 0, 10),
        next_poll_decision("PENDING", 11, 10),
    }
    assert decisions == {"continue", "done", "failed", "timeout"}
