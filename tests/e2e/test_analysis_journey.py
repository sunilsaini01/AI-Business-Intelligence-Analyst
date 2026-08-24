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
"""

from __future__ import annotations

import os

import httpx
import pytest
from playwright.sync_api import Page, expect

FRONTEND_BASE_URL = os.environ.get("FRONTEND_BASE_URL", "http://localhost:8511")
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8010")

_QUESTION = "How many customers do we have per region?"


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


@pytest.fixture
def analysis_page(page: Page) -> Page:
    page.goto(FRONTEND_BASE_URL)
    return page


def test_full_analysis_journey_via_the_real_browser(analysis_page: Page) -> None:
    page = analysis_page

    # 4. Page loads.
    expect(page.get_by_text("AI Business Intelligence Analyst")).to_be_visible(timeout=15_000)

    # 5. Enter a valid business question. Streamlit's text_input only syncs
    # its value to the Python session (and reruns, enabling the button) on
    # blur/Enter — `.fill()` alone sets the DOM value without firing that,
    # so a real user's next action (pressing Tab/clicking elsewhere) has to
    # be simulated explicitly, not skipped.
    question_box = page.get_by_placeholder("e.g. What happened to revenue last month?")
    question_box.fill(_QUESTION)
    question_box.press("Tab")

    # 6. Click Analyze.
    analyze_button = page.get_by_role("button", name="Analyze")
    expect(analyze_button).to_be_enabled(timeout=15_000)
    analyze_button.click()

    # 7/8. Analysis starts; progress UI appears.
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


def test_out_of_scope_question_shows_a_clear_low_confidence_result(analysis_page: Page) -> None:
    """15. "Backend errors surfaced clearly" — the fake provider's
    unrecognized-question path exercises a real, complete, low-confidence
    decline result end to end (app/agents/supervisor.py::
    _direct_out_of_scope_report), the most realistic "this could not be
    answered" case reachable without deliberately crashing the backend
    mid-request. Genuine application-exception -> FAILED handling is
    covered at the API/service layer (tests/api/test_reliability.py),
    which doesn't need a browser to verify.
    """
    page = analysis_page
    expect(page.get_by_text("AI Business Intelligence Analyst")).to_be_visible(timeout=15_000)

    question_box = page.get_by_placeholder("e.g. What happened to revenue last month?")
    question_box.fill("What's the weather like today?")
    question_box.press("Tab")
    analyze_button = page.get_by_role("button", name="Analyze")
    expect(analyze_button).to_be_enabled(timeout=15_000)
    analyze_button.click()

    expect(page.get_by_text("Final Report")).to_be_visible(timeout=60_000)
    expect(page.get_by_text("I can't answer that from this dataset.")).to_be_visible()
    expect(page.get_by_text("Low")).to_be_visible()
