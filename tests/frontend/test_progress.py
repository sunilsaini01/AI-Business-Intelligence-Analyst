from progress import PIPELINE_STAGES, advance_progress


def test_no_stage_yet_nothing_completed():
    statuses, max_idx = advance_progress(None, -1)
    assert max_idx == -1
    assert all(not s.completed for s in statuses)
    assert statuses[0].is_next  # supervisor is up next


def test_first_stage_completed_marks_only_that_one_done():
    statuses, max_idx = advance_progress("supervisor", -1)
    assert max_idx == 0
    assert statuses[0].completed is True
    assert statuses[1].completed is False
    assert statuses[1].is_next is True


def test_later_stage_marks_all_earlier_ones_done_too():
    statuses, max_idx = advance_progress("visualization_agent", -1)
    assert max_idx == PIPELINE_STAGES.index("visualization_agent")
    for stage in statuses[: max_idx + 1]:
        assert stage.completed is True
    for stage in statuses[max_idx + 1 :]:
        assert stage.completed is False


def test_report_agent_completes_the_whole_checklist():
    statuses, max_idx = advance_progress("report_agent", -1)
    assert all(s.completed for s in statuses)
    assert not any(s.is_next for s in statuses)  # nothing left to be "next"


def test_unknown_stage_name_is_ignored_not_a_crash():
    statuses, max_idx = advance_progress("some_future_node", 2)
    assert max_idx == 2  # unchanged — unrecognized name doesn't move the checklist
    assert statuses[2].completed is True
    assert statuses[3].completed is False


def test_progress_is_monotonic_across_a_critic_retry_loop():
    """The Critic's bounded retry loop (Sec 1 Fig. 2) can report
    "supervisor" again AFTER "critic" already completed once — the
    checklist must not un-complete critic just because the API's latest
    snapshot says "supervisor"."""
    statuses1, max_idx = advance_progress("critic", -1)
    assert statuses1[PIPELINE_STAGES.index("critic")].completed is True

    # Critic FAILed with retries left -> back to supervisor for a revision
    statuses2, max_idx = advance_progress("supervisor", max_idx)
    assert max_idx == PIPELINE_STAGES.index("critic")  # never regresses
    assert statuses2[PIPELINE_STAGES.index("critic")].completed is True
    assert statuses2[PIPELINE_STAGES.index("supervisor")].completed is True


def test_only_one_stage_is_ever_marked_next():
    statuses, _ = advance_progress("sql_agent", -1)
    next_stages = [s for s in statuses if s.is_next]
    assert len(next_stages) == 1
    assert next_stages[0].name == "analysis_agent"


def test_never_claims_a_stage_is_currently_executing():
    """StageStatus has exactly two boolean signals — completed and is_next
    — never a third "running" state; is_next's label text is the caller's
    job (app.py renders it as "(next)", never "(running)")."""
    statuses, _ = advance_progress("sql_agent", -1)
    for stage in statuses:
        assert set(vars(stage).keys()) == {"name", "label", "completed", "is_next"}
