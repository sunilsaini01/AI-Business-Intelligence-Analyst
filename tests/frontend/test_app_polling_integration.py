"""Phase 14, Issue 4 — drives the REAL frontend/app.py script via Streamlit's
AppTest harness (streamlit.testing.v1), not just polling.py's pure decision
function in isolation (see tests/frontend/test_polling.py). Only
`AnalysisApiClient`'s HTTP methods are mocked — session_state transitions,
the actual `_handle_status_response` control flow, and app.py's own script
structure all run for real, proving the polling loop is genuinely bounded
end to end and that a malformed/missing/erroring response can't crash the
script or spin forever.

Login is bypassed by pre-seeding session_state (access_token/user_email)
before the first `.run()` — the login/register form itself has its own
widget-interaction tests below, separate from the polling scenarios.

`time.sleep` is patched to a no-op in every scenario that reaches the
"continue polling" branch, so a multi-tick test doesn't really wait
DEFAULT_POLL_INTERVAL_SECONDS (2s) per tick.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import api_client
from streamlit.testing.v1 import AppTest

_APP_PATH = str(Path(__file__).resolve().parents[2] / "frontend" / "app.py")
_TOKEN = "fake-token-for-apptest"


def _new_app() -> AppTest:
    return AppTest.from_file(_APP_PATH, default_timeout=15)


def _authenticated(at: AppTest, *, analysis_id: str = "abc-123", elapsed_seconds: float = 0.0) -> AppTest:
    """Seeds session_state past the login gate and with an in-flight
    analysis already set up, mirroring what _reset_for_new_analysis would
    have produced — this file is about the POLLING loop, not re-driving the
    login form or the initial POST /analyze for every scenario."""
    at.session_state["access_token"] = _TOKEN
    at.session_state["user_email"] = "test@example.com"
    at.session_state["analysis_id"] = analysis_id
    at.session_state["terminal_status"] = None
    at.session_state["poll_started_at"] = time.monotonic() - elapsed_seconds
    return at


def _run_authenticated(get_status_return=None, get_status_side_effect=None, **kwargs) -> AppTest:
    at = _authenticated(_new_app(), **kwargs)
    with patch.object(api_client.AnalysisApiClient, "health_ready", return_value={"ready": True}), patch.object(
        api_client.AnalysisApiClient, "get_status", return_value=get_status_return, side_effect=get_status_side_effect
    ), patch("time.sleep"):
        at.run()
    return at


# --- bounded polling: the required PENDING -> ANALYZING -> ... -> timeout --


def test_pending_then_analyzing_then_done_progresses_and_completes():
    """The PENDING -> ANALYZING -> ... -> DONE progression Issue 4 asks for.
    Deliberately does NOT patch time.sleep here: app.py's real "continue"
    branch calls st.rerun() immediately after a real DEFAULT_POLL_INTERVAL_
    SECONDS sleep — under AppTest that rerun restarts the script in the same
    call, so with a static always-ANALYZING mock (and sleep patched away)
    it never reaches a stopped state at all (confirmed: that combination
    genuinely times out AppTest's own runner, not a bug in app.py, just an
    unrealistic test setup). A short side_effect sequence resolving to DONE
    after one real ANALYZING tick is what "still polling, not yet terminal"
    actually looks like end to end."""
    at = _authenticated(_new_app(), elapsed_seconds=0.0)
    responses = [
        {"status": "PENDING", "current_stage": None, "error_message": None},
        {"status": "ANALYZING", "current_stage": "sql_agent", "error_message": None},
        {"status": "DONE", "current_stage": "report_agent", "error_message": None},
    ]
    report = {
        "executive_summary": "Done.", "key_findings": [], "evidence": [],
        "recommendations": [], "confidence": "High", "limitations": "",
    }
    with patch.object(api_client.AnalysisApiClient, "health_ready", return_value={"ready": True}), patch.object(
        api_client.AnalysisApiClient, "get_status", side_effect=responses
    ), patch.object(api_client.AnalysisApiClient, "get_report", return_value=report), patch.object(
        api_client.AnalysisApiClient, "get_charts", return_value=[]
    ), patch.object(api_client.AnalysisApiClient, "get_detail", return_value={"trace": []}):
        at.run()

    assert len(at.exception) == 0
    assert at.session_state["terminal_status"] == "DONE"
    assert at.session_state["analysis_id"] == "abc-123"  # never lost across the multi-tick poll
    assert at.session_state["report"]["executive_summary"] == "Done."


def test_elapsed_past_the_bound_stops_polling_with_timeout_not_an_infinite_loop():
    """The actual proof of boundedness: even with the analysis still
    ANALYZING (never DONE/FAILED), once elapsed_seconds >= the 180s bound
    the very next poll tick resolves to a terminal TIMEOUT — it does not
    keep calling st.rerun() forever."""
    at = _run_authenticated(
        get_status_return={"status": "ANALYZING", "current_stage": "sql_agent", "error_message": None},
        elapsed_seconds=300.0,
    )
    assert len(at.exception) == 0
    assert at.session_state["terminal_status"] == "TIMEOUT"
    assert "taking longer than expected" in at.warning[0].value
    # analysis_id must be PRESERVED after a timeout — the user can refresh
    # and it's still there to inspect/retry, never silently cleared.
    assert at.session_state["analysis_id"] == "abc-123"


def test_a_terminal_status_does_not_poll_again_on_a_later_rerun():
    """Once terminal, app.py's own `if terminal_status is None` guard skips
    the polling block entirely — this is what actually prevents unbounded
    polling in the app, not just next_poll_decision's own logic."""
    at = _authenticated(_new_app(), elapsed_seconds=5.0)
    at.session_state["terminal_status"] = "TIMEOUT"  # already resolved before this run
    with patch.object(api_client.AnalysisApiClient, "health_ready", return_value={"ready": True}), patch.object(
        api_client.AnalysisApiClient, "get_status"
    ) as get_status_mock, patch("time.sleep"):
        at.run()
    get_status_mock.assert_not_called()


# --- terminal outcomes ------------------------------------------------------


def test_backend_returns_done_report_is_fetched_and_rendered():
    at = _new_app()
    _authenticated(at)
    report = {
        "executive_summary": "Revenue grew.", "key_findings": ["June: 100"],
        "evidence": [], "recommendations": [], "confidence": "High", "limitations": "",
    }
    with patch.object(api_client.AnalysisApiClient, "health_ready", return_value={"ready": True}), patch.object(
        api_client.AnalysisApiClient, "get_status", return_value={"status": "DONE", "current_stage": "report_agent", "error_message": None}
    ), patch.object(api_client.AnalysisApiClient, "get_report", return_value=report), patch.object(
        api_client.AnalysisApiClient, "get_charts", return_value=[]
    ), patch.object(
        api_client.AnalysisApiClient, "get_detail", return_value={"trace": []}
    ):
        at.run()
    assert len(at.exception) == 0
    assert at.session_state["terminal_status"] == "DONE"
    assert at.session_state["report"]["executive_summary"] == "Revenue grew."
    assert "Revenue grew." in [t.value for t in at.markdown] or "Revenue grew." in str(at.main)


def test_backend_returns_failed_error_is_shown_not_crashed():
    at = _run_authenticated(
        get_status_return={"status": "FAILED", "current_stage": "sql_agent", "error_message": "Analysis failed. Please try again."}
    )
    assert len(at.exception) == 0
    assert at.session_state["terminal_status"] == "FAILED"
    assert at.session_state["error_message"] == "Analysis failed. Please try again."
    assert at.error[0].value == "Analysis failed. Please try again."


def test_backend_unavailable_during_polling_fails_safely():
    at = _run_authenticated(get_status_side_effect=api_client.BackendUnavailableError("Could not reach the analysis server."))
    assert len(at.exception) == 0
    assert at.session_state["terminal_status"] == "FAILED"
    assert "Could not reach" in at.session_state["error_message"]


def test_404_unknown_analysis_during_polling_fails_safely():
    at = _run_authenticated(get_status_side_effect=api_client.NotFoundError("Analysis not found."))
    assert len(at.exception) == 0
    assert at.session_state["terminal_status"] == "FAILED"
    assert at.session_state["error_message"] == "Analysis not found."


def test_rate_limit_during_polling_fails_safely_with_the_rate_limit_message():
    at = _run_authenticated(
        get_status_side_effect=api_client.RateLimitedError("The AI provider rate limit was reached. Please try again later.")
    )
    assert len(at.exception) == 0
    assert at.session_state["terminal_status"] == "FAILED"
    assert "rate limit" in at.session_state["error_message"].lower()


def test_malformed_status_response_missing_the_status_key_fails_safely():
    """Phase 14 regression guard: this exact shape (`{}`, no "status" key)
    used to crash the script on `status_data["status"]` — a real bug found
    while writing this test, fixed in app.py's _handle_status_response."""
    at = _run_authenticated(get_status_return={})
    assert len(at.exception) == 0
    assert at.session_state["terminal_status"] == "FAILED"
    assert "unexpected response" in at.session_state["error_message"].lower()


def test_malformed_status_response_with_an_unrecognized_status_value_fails_safely():
    at = _run_authenticated(get_status_return={"status": "SOMETHING_NEW", "current_stage": None, "error_message": None})
    assert len(at.exception) == 0
    assert at.session_state["terminal_status"] == "FAILED"


def test_report_unavailable_before_done_is_handled_not_crashed():
    """A 409 fetching the report (status flipped to DONE without the report
    persisted yet, per _fetch_terminal_result's own docstring on why this
    "shouldn't happen but is handled") must not crash the script."""
    at = _new_app()
    _authenticated(at)
    with patch.object(api_client.AnalysisApiClient, "health_ready", return_value={"ready": True}), patch.object(
        api_client.AnalysisApiClient, "get_status", return_value={"status": "DONE", "current_stage": "report_agent", "error_message": None}
    ), patch.object(api_client.AnalysisApiClient, "get_report", side_effect=api_client.NotReadyError("Analysis is not ready yet.")):
        at.run()
    assert len(at.exception) == 0
    assert at.session_state["report"] is None
    assert at.session_state["error_message"] == "Analysis is not ready yet."


# --- auth: an expired/invalid token mid-poll bounces to the login gate -----


def test_expired_token_during_polling_logs_out_and_shows_the_login_gate():
    at = _new_app()
    _authenticated(at)
    with patch.object(api_client.AnalysisApiClient, "health_ready", return_value={"ready": True}), patch.object(
        api_client.AnalysisApiClient, "get_status", side_effect=api_client.AuthenticationError("Please log in again.")
    ):
        at.run()
    assert len(at.exception) == 0
    assert at.session_state["access_token"] is None
    assert at.session_state["analysis_id"] is None
    assert any("Log in" in t.value for t in at.subheader)


# --- login gate itself (widget-driven, not just pre-seeded state) ----------


def test_login_gate_is_shown_when_not_authenticated():
    at = _new_app()
    with patch.object(api_client.AnalysisApiClient, "health_ready", return_value={"ready": True}):
        at.run()
    assert len(at.exception) == 0
    assert any("Log in" in t.value for t in at.subheader)
    assert not any(t.value == "Business Question" for t in at.subheader)


def test_successful_login_through_the_real_form_reaches_the_main_ui():
    at = _new_app()
    with patch.object(api_client.AnalysisApiClient, "health_ready", return_value={"ready": True}):
        at.run()

    at.tabs[0].text_input(key="login_email").set_value("user@example.com")
    at.tabs[0].text_input(key="login_password").set_value("correct-horse-battery")
    with patch.object(api_client.AnalysisApiClient, "health_ready", return_value={"ready": True}), patch.object(
        api_client.AnalysisApiClient, "login", return_value={"access_token": _TOKEN, "token_type": "bearer"}
    ):
        at.tabs[0].button(key="login_submit").click().run()

    assert len(at.exception) == 0
    assert at.session_state["access_token"] == _TOKEN
    assert any(t.value == "Business Question" for t in at.subheader)


def test_login_with_wrong_credentials_shows_an_error_and_stays_on_the_login_gate():
    at = _new_app()
    with patch.object(api_client.AnalysisApiClient, "health_ready", return_value={"ready": True}):
        at.run()

    at.tabs[0].text_input(key="login_email").set_value("user@example.com")
    at.tabs[0].text_input(key="login_password").set_value("wrong-password")
    with patch.object(api_client.AnalysisApiClient, "health_ready", return_value={"ready": True}), patch.object(
        api_client.AnalysisApiClient, "login", side_effect=api_client.AuthenticationError("Incorrect email or password.")
    ):
        at.tabs[0].button(key="login_submit").click().run()

    assert len(at.exception) == 0
    assert at.session_state["access_token"] is None
    assert at.error[0].value == "Incorrect email or password."
