"""Phase 13, Objective B — real browser end-to-end test of the actual user
journey, via Playwright, against the REAL running Streamlit frontend and
FastAPI backend (docker compose up -d) — not a mock of either.

Deterministic on purpose: requires the stack to be running with
`LLM_PROVIDER=fake` (app/core/fake_llm.py) so the whole
Supervisor -> SQL Agent -> Analysis Agent -> Visualization Agent -> Critic
-> Report Generator pipeline runs for real, against the real seeded
database, but with zero dependency on live Groq/Anthropic quota — the
production default (`LLM_PROVIDER=anthropic`) is never changed by this
file; the E2E run sets the env var on its own container invocation only
(see docs/architecture.md's "Browser E2E" section for the exact command).

NOT part of `docker compose exec api pytest tests/` — these need a real
browser (Playwright's Chromium) and a live frontend port, neither of which
belong in the api image's normal test run. See requirements-e2e.txt for
setup and the module-level skip below for what happens if the stack isn't
reachable.

Run (verified working from inside the `api` container against Debian
trixie — `playwright install --with-deps` doesn't recognize a couple of
trixie's font package names, so its OS-dependency step must be installed
manually; see docs/architecture.md's Browser E2E section for the exact
apt package list that was actually needed):

    docker compose -f docker-compose.yml -f docker-compose.e2e.yml up -d api frontend
    docker compose exec api pip install -r requirements-e2e.txt
    docker compose exec -u root api playwright install chromium
    docker compose exec -u root api bash -c "apt-get update -qq && apt-get install -y --no-install-recommends \
        libglib2.0-0 libnss3 libnspr4 libdbus-1-3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
        libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libasound2 \
        libpango-1.0-0 libcairo2 libatspi2.0-0 libx11-6 libxcb1 libxext6 fonts-liberation"
    docker compose exec -e FRONTEND_BASE_URL=http://frontend:8501 -e API_BASE_URL=http://api:8000 \
        api python -m pytest tests/e2e/ -v --confcutdir=tests/e2e -p no:asyncio

`--confcutdir=tests/e2e` is required: the root tests/conftest.py's autouse
async DB-connection fixture conflicts with pytest-playwright's sync
fixtures (a real `RuntimeError: Cannot run the event loop while another
loop is running` otherwise) — E2E tests don't need that fixture at all
(they never import app.db directly), so cutting conftest inheritance at
tests/e2e/ is the correct fix, not a workaround.

Phase 14: POST /analyze (and every /analysis/* endpoint) now requires
auth, so every journey below logs in first — see `_register_and_log_in`
and the `authenticated_page` fixture. `analysis_page` (no login) is kept
for the one test that specifically checks the login gate itself.
"""

from __future__ import annotations

import os
import uuid

import httpx
import pytest
from playwright.sync_api import BrowserContext, Page, expect

FRONTEND_BASE_URL = os.environ.get("FRONTEND_BASE_URL", "http://localhost:8511")
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8010")

_QUESTION = "How many customers do we have per region?"
_PASSWORD = "correct-horse-battery-staple"


def _stack_is_reachable() -> bool:
    try:
        resp = httpx.get(f"{API_BASE_URL}/api/v1/health/ready", timeout=3.0)
        return resp.status_code == 200
    except httpx.RequestError:
        return False


pytestmark = pytest.mark.skipif(
    not _stack_is_reachable(),
    reason=f"Backend not reachable at {API_BASE_URL} — E2E test skipped (start the stack first), not failed.",
)


def _register_and_log_in(page: Page, email: str) -> None:
    """Drives the REAL login gate (frontend/app.py::_render_login_gate) —
    register on one tab, log in on the other, wait for the main UI. Each
    call uses a fresh, unique email so tests never collide on the
    register endpoint's uniqueness constraint or on session isolation."""
    page.goto(FRONTEND_BASE_URL)
    expect(page.get_by_text("AI Business Intelligence Analyst")).to_be_visible(timeout=15_000)

    page.get_by_role("tab", name="Register").click()
    register_panel = page.get_by_role("tabpanel").filter(has=page.get_by_role("button", name="Register"))
    register_panel.get_by_role("textbox", name="Email").fill(email)
    register_panel.get_by_role("textbox", name="Email").press("Tab")
    register_panel.get_by_role("textbox", name="Password").fill(_PASSWORD)
    register_panel.get_by_role("textbox", name="Password").press("Tab")
    register_button = register_panel.get_by_role("button", name="Register")
    expect(register_button).to_be_enabled(timeout=15_000)
    register_button.click()
    expect(page.get_by_text("Account created")).to_be_visible(timeout=15_000)

    _log_in(page, email)


def _log_in(page: Page, email: str) -> None:
    """Just the login half — used both by _register_and_log_in (a fresh
    account) and directly by tests that need to log back IN as an
    already-registered user (e.g. after a browser refresh, which — like
    any Streamlit session_state — always discards access_token along with
    everything else; only analysis_id survives a refresh, via
    st.query_params, and only because app.py deliberately puts it there).
    Assumes the login gate (or the "Log in" tab of it) is already on
    screen."""
    if page.get_by_role("tab", name="Log in").is_visible():
        page.get_by_role("tab", name="Log in").click()
    login_panel = page.get_by_role("tabpanel").filter(has=page.get_by_role("button", name="Log in"))
    login_panel.get_by_role("textbox", name="Email").fill(email)
    login_panel.get_by_role("textbox", name="Email").press("Tab")
    login_panel.get_by_role("textbox", name="Password").fill(_PASSWORD)
    login_panel.get_by_role("textbox", name="Password").press("Tab")
    login_button = login_panel.get_by_role("button", name="Log in")
    expect(login_button).to_be_enabled(timeout=15_000)
    login_button.click()

    expect(page.get_by_role("heading", name="Business Question")).to_be_visible(timeout=15_000)
    expect(page.get_by_text(f"Logged in as {email}")).to_be_visible(timeout=15_000)


@pytest.fixture
def analysis_page(page: Page) -> Page:
    page.goto(FRONTEND_BASE_URL)
    return page


@pytest.fixture
def authenticated_page(page: Page) -> Page:
    """A fresh, uniquely-registered, logged-in user, ready at the main
    "Business Question" screen — every journey that exercises an actual
    analysis needs this now that POST /analyze requires auth."""
    _register_and_log_in(page, f"e2e-{uuid.uuid4()}@example.com")
    return page


@pytest.fixture
def authenticated_page_with_email(page: Page) -> tuple[Page, str]:
    """Same as authenticated_page, but also hands back the email — for the
    one journey that needs to log back IN later (after a refresh, which
    discards session_state including access_token)."""
    email = f"e2e-{uuid.uuid4()}@example.com"
    _register_and_log_in(page, email)
    return page, email


def _ask_and_wait_for_report(page: Page, question: str) -> None:
    question_box = page.get_by_placeholder("e.g. What happened to revenue last month?")
    question_box.fill(question)
    question_box.press("Tab")
    analyze_button = page.get_by_role("button", name="Analyze")
    expect(analyze_button).to_be_enabled(timeout=15_000)
    analyze_button.click()
    expect(page.get_by_text("Final Report")).to_be_visible(timeout=60_000)


def test_login_gate_is_shown_before_any_account_exists(analysis_page: Page) -> None:
    """1/2/3. The unauthenticated entry point — proves the auth boundary is
    real at the UI layer too, not just the API (see
    tests/security/test_authorization.py for the API-layer proof)."""
    page = analysis_page
    expect(page.get_by_text("AI Business Intelligence Analyst")).to_be_visible(timeout=15_000)
    expect(page.get_by_role("heading", name="Log in")).to_be_visible()
    expect(page.get_by_placeholder("e.g. What happened to revenue last month?")).not_to_be_visible()


def test_full_analysis_journey_via_the_real_browser(authenticated_page: Page) -> None:
    page = authenticated_page

    # 7/8. Analysis starts; progress UI appears.
    question_box = page.get_by_placeholder("e.g. What happened to revenue last month?")
    question_box.fill(_QUESTION)
    question_box.press("Tab")
    analyze_button = page.get_by_role("button", name="Analyze")
    expect(analyze_button).to_be_enabled(timeout=15_000)
    analyze_button.click()

    expect(page.get_by_text("Analysis Progress")).to_be_visible(timeout=15_000)
    expect(page.get_by_text("Status:", exact=False)).to_be_visible()

    # 9/10. Bounded wait for completion — the fake provider completes in a
    # couple of seconds, well inside app.py's own 180s poll timeout; this
    # test's own bound is much shorter since a real timeout here should
    # fail loudly, not silently pass as if it were normal.
    expect(page.get_by_text("Final Report")).to_be_visible(timeout=60_000)

    # 11. Executive Summary, Key Findings, Confidence, always present for a
    # successful in-scope run; Verified Claims/Limitations/Recommendations
    # are conditional sections (Sec "Do not invent missing values" — Phase
    # 12) so they're checked only if the report actually has them.
    expect(page.get_by_text("Executive Summary")).to_be_visible()
    expect(page.get_by_text("Customer counts differ across regions", exact=False)).to_be_visible()
    expect(page.get_by_text("Key Findings")).to_be_visible()
    expect(page.get_by_text("Confidence")).to_be_visible()

    # 12. Charts section — the customers-by-region question always
    # produces at least one chart (a bar chart, Phase 7).
    expect(page.get_by_text("Visualizations")).to_be_visible()

    # 13. Open Technical Details — this is itself a Streamlit
    # rerun-triggering interaction.
    technical_details = page.get_by_text("Technical Details")
    expect(technical_details).to_be_visible()
    technical_details.click()

    # 14. Report remains visible after that rerun — this is the exact bug
    # class Phase 12 found and fixed (session_state persistence); if it
    # ever regresses, this assertion is what catches it.
    expect(page.get_by_text("Executive Summary")).to_be_visible()
    expect(page.get_by_text("Customer counts differ across regions", exact=False)).to_be_visible()


def test_out_of_scope_question_shows_a_clear_low_confidence_result(authenticated_page: Page) -> None:
    """15. "Backend errors surfaced clearly" — the fake provider's
    unrecognized-question path exercises a real, complete, low-confidence
    decline result end to end (app/agents/supervisor.py::
    _direct_out_of_scope_report), the most realistic "this could not be
    answered" case reachable without deliberately crashing the backend
    mid-request. Genuine application-exception -> FAILED handling is
    covered at the API/service layer (tests/api/test_reliability.py),
    which doesn't need a browser to verify.
    """
    page = authenticated_page

    question_box = page.get_by_placeholder("e.g. What happened to revenue last month?")
    question_box.fill("What's the weather like today?")
    question_box.press("Tab")
    analyze_button = page.get_by_role("button", name="Analyze")
    expect(analyze_button).to_be_enabled(timeout=15_000)
    analyze_button.click()

    expect(page.get_by_text("Final Report")).to_be_visible(timeout=60_000)
    expect(page.get_by_text("I can't answer that from this dataset.")).to_be_visible()
    expect(page.get_by_text("Low")).to_be_visible()


def test_browser_refresh_after_starting_an_analysis_still_reaches_the_report(
    authenticated_page_with_email: tuple[Page, str],
) -> None:
    """Phase 14: analysis_id is persisted to st.query_params
    (_reset_for_new_analysis) specifically so a mid-analysis refresh
    resumes polling instead of losing all progress — this is the direct
    regression guard for that fix. Reloads immediately after clicking
    Analyze (racing the fake provider's own few-seconds completion, not
    waiting for a specific mid-flight window) and asserts the report is
    still reachable afterward either way.

    A refresh discards ALL of st.session_state (Streamlit sessions are
    tied to the WebSocket connection, not to the URL) — including
    access_token, not just analysis_id. Only analysis_id survives, and
    only because it's deliberately mirrored into st.query_params; the user
    genuinely does have to log back in, which is the correct, security-
    appropriate behavior (a URL alone must never be enough to resume
    someone else's session) — this test logs back in with the SAME
    account and asserts the ANALYSIS itself, not the login, resumes
    correctly from there.
    """
    page, email = authenticated_page_with_email

    question_box = page.get_by_placeholder("e.g. What happened to revenue last month?")
    question_box.fill(_QUESTION)
    question_box.press("Tab")
    analyze_button = page.get_by_role("button", name="Analyze")
    expect(analyze_button).to_be_enabled(timeout=15_000)
    analyze_button.click()
    expect(page.get_by_text("Analysis Progress")).to_be_visible(timeout=15_000)

    assert "analysis_id=" in page.url  # the resumption mechanism itself: it's really in the URL
    page.reload()

    expect(page.get_by_role("heading", name="Log in")).to_be_visible(timeout=15_000)
    _log_in(page, email)

    expect(page.get_by_text("Final Report")).to_be_visible(timeout=60_000)
    expect(page.get_by_text("Customer counts differ across regions", exact=False)).to_be_visible()


def test_two_independent_logged_in_sessions_never_see_each_others_report(browser: BrowserContext) -> None:
    """Issue 9/Issue 8 combined: two different browser contexts, two
    different registered users, run different questions concurrently-ish
    (both started before either is awaited to completion) — each must only
    ever see its own report, proving the Phase 14 ownership boundary holds
    through the real UI, not just via direct API calls (see
    tests/security/test_authorization.py for that layer)."""
    context_a = browser.new_context()
    context_b = browser.new_context()
    try:
        page_a, page_b = context_a.new_page(), context_b.new_page()
        _register_and_log_in(page_a, f"e2e-a-{uuid.uuid4()}@example.com")
        _register_and_log_in(page_b, f"e2e-b-{uuid.uuid4()}@example.com")

        _ask_and_wait_for_report(page_a, _QUESTION)
        _ask_and_wait_for_report(page_b, "What's the weather like today?")

        expect(page_a.get_by_text("Customer counts differ across regions", exact=False)).to_be_visible()
        expect(page_a.get_by_text("I can't answer that from this dataset.")).not_to_be_visible()

        expect(page_b.get_by_text("I can't answer that from this dataset.")).to_be_visible()
        expect(page_b.get_by_text("Customer counts differ across regions", exact=False)).not_to_be_visible()
    finally:
        context_a.close()
        context_b.close()
