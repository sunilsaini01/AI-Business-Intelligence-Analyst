"""Streamlit frontend (Phase 12) — a pure client of the FastAPI backend.

    Streamlit -> FastAPI -> Analysis Service -> LangGraph -> Agents -> PostgreSQL

This file never touches PostgreSQL, LangGraph, or any agent directly — it
only calls the 6 documented endpoints via api_client.AnalysisApiClient, and
only ever renders what those endpoints return (Sec "Final engineering
principle": Streamlit owns presentation, nothing else). The pure logic
(HTTP/error handling, polling policy, progress checklist, report-section
ordering, chart validation) lives in sibling modules with no `streamlit`
import — see frontend/{api_client,polling,progress,report_view,
chart_builder,health}.py and tests/frontend/ — this file is the thin glue
that calls `st.*` to actually draw what those modules decide.

Polling uses one status check per script run + `st.rerun()`, not a
blocking while-loop: each run stays fast and bounded, and everything that
must survive a rerun (analysis_id, question, poll start time, the fetched
report/charts/trace) lives in `st.session_state` — never credentials,
never a DB/LLM client, never raw LangGraph state (Sec "Session state").
Caching the terminal result in session_state also means an unrelated
rerun (e.g. the user opening an expander) redraws the same finished report
instead of losing it.
"""

from __future__ import annotations

import time

import streamlit as st

from api_client import AnalysisApiClient, ApiError
from components.charts import render_chart
from health import check_backend_ready
from polling import DEFAULT_POLL_INTERVAL_SECONDS, DEFAULT_POLL_TIMEOUT_SECONDS, next_poll_decision
from progress import advance_progress
from report_view import build_report_sections

st.set_page_config(page_title="AI Business Intelligence Analyst", layout="centered")

_DEFAULTS = {
    "analysis_id": None,
    "question": "",
    "max_stage_index": -1,
    "poll_started_at": None,
    "terminal_status": None,  # None (still polling) | "DONE" | "FAILED" | "TIMEOUT"
    "error_message": None,
    "report": None,
    "charts": None,
    "trace": None,
}
for _key, _default in _DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default


def _reset_for_new_analysis(analysis_id: str, question: str) -> None:
    st.session_state.analysis_id = analysis_id
    st.session_state.question = question
    st.session_state.max_stage_index = -1
    st.session_state.poll_started_at = time.monotonic()
    st.session_state.terminal_status = None
    st.session_state.error_message = None
    st.session_state.report = None
    st.session_state.charts = None
    st.session_state.trace = None


@st.cache_data(ttl=30, show_spinner=False)
def _cached_backend_status() -> tuple[bool, str]:
    with AnalysisApiClient() as client:
        return check_backend_ready(client)


def _render_backend_status() -> None:
    ready, message = _cached_backend_status()
    if ready:
        st.sidebar.success(message)
    else:
        st.sidebar.error(f"Backend unavailable: {message}")


def _render_progress(status: str, stage_statuses) -> None:
    st.subheader("Analysis Progress")
    st.write(f"Status: {status}")
    for stage in stage_statuses:
        if stage.completed:
            st.write(f"✓ {stage.label}")
        elif stage.is_next:
            st.write(f"• {stage.label} (next)")
        else:
            st.write(f"○ {stage.label}")
    st.caption(
        "✓ = confirmed complete by the API. • = next stage in the typical pipeline order "
        "(not a claim that it is currently running). Not every question uses every stage."
    )


def _render_report_section(section) -> None:
    if section.kind == "text":
        st.subheader(section.title)
        st.write(section.content)
    elif section.kind == "list":
        st.subheader(section.title)
        for item in section.content:
            st.write(f"- {item}")
    elif section.kind == "warning":
        st.subheader(section.title)
        st.warning(section.content)
    elif section.kind == "json":
        with st.expander(section.title):
            st.json(section.content)


def _fetch_terminal_result(client: AnalysisApiClient, analysis_id: str) -> None:
    """Called exactly once per completed analysis (guarded by
    `st.session_state.report is None`) — GET /report, /charts, and the
    full detail (for trace) are never re-fetched on a later rerun."""
    try:
        st.session_state.report = client.get_report(analysis_id)
    except ApiError as exc:
        # A 409 here would mean status flipped to DONE without the report
        # persisted yet — analysis_service.py commits both in the same
        # transaction, so this shouldn't happen, but it's handled as "not
        # ready", not a crash (Sec "409 on report retrieval").
        st.session_state.error_message = exc.user_message
        return

    try:
        st.session_state.charts = client.get_charts(analysis_id)
    except ApiError:
        st.session_state.charts = []

    try:
        detail = client.get_detail(analysis_id)
        st.session_state.trace = detail.get("trace") or []
    except ApiError:
        st.session_state.trace = []


def _render_final_report() -> None:
    report = st.session_state.report
    if report is None:
        st.error(st.session_state.error_message or "The report could not be retrieved.")
        return

    st.header("Final Report")
    for section in build_report_sections(report):
        _render_report_section(section)

    charts = st.session_state.charts or []
    if charts:
        st.subheader("Visualizations")
        for chart in charts:
            render_chart(chart)

    trace = st.session_state.trace or []
    if trace:
        with st.expander("Agent Trace"):
            st.dataframe(trace, use_container_width=True)


st.title("AI Business Intelligence Analyst")
st.caption("Supervisor -> SQL Agent -> Analysis Agent -> Visualization Agent -> Critic -> Report Generator.")
_render_backend_status()

st.subheader("Business Question")
question_input = st.text_input(
    "Business Question",
    value=st.session_state.question,
    placeholder="e.g. What happened to revenue last month?",
    label_visibility="collapsed",
)
submitted = st.button("Analyze", disabled=not question_input.strip())

if submitted:
    with AnalysisApiClient() as api_client:
        try:
            accepted = api_client.analyze(question_input.strip())
        except ApiError as exc:
            st.error(exc.user_message)
        else:
            _reset_for_new_analysis(accepted["analysis_id"], question_input.strip())
            st.rerun()

if st.session_state.analysis_id:
    analysis_id = st.session_state.analysis_id

    if st.session_state.terminal_status is None:
        with AnalysisApiClient() as api_client:
            try:
                status_data = api_client.get_status(analysis_id)
            except ApiError as exc:
                st.session_state.terminal_status = "FAILED"
                st.session_state.error_message = exc.user_message
            else:
                status = status_data["status"]
                stage_statuses, st.session_state.max_stage_index = advance_progress(
                    status_data.get("current_stage"), st.session_state.max_stage_index
                )
                _render_progress(status, stage_statuses)

                elapsed = time.monotonic() - st.session_state.poll_started_at
                decision = next_poll_decision(status, elapsed, DEFAULT_POLL_TIMEOUT_SECONDS)
                if decision == "done":
                    st.session_state.terminal_status = "DONE"
                elif decision == "failed":
                    st.session_state.terminal_status = "FAILED"
                    st.session_state.error_message = status_data.get("error_message")
                elif decision == "timeout":
                    st.session_state.terminal_status = "TIMEOUT"
                else:
                    time.sleep(DEFAULT_POLL_INTERVAL_SECONDS)
                    st.rerun()

    if st.session_state.terminal_status == "DONE":
        if st.session_state.report is None:
            with AnalysisApiClient() as api_client:
                _fetch_terminal_result(api_client, analysis_id)
        _render_final_report()
    elif st.session_state.terminal_status == "FAILED":
        st.error(st.session_state.error_message or "Analysis failed. Please try again.")
    elif st.session_state.terminal_status == "TIMEOUT":
        st.warning("Analysis is taking longer than expected. You can refresh and check again.")
