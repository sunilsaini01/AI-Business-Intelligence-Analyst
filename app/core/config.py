from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central config. Every field is env-driven — no hard-coded secrets or hosts.

    Security hardening (Phase 15, Objective 1): every secret-bearing field
    below is declared with `repr=False` — pydantic excludes those fields
    entirely from the model's `__repr__`/`__str__`. This is not cosmetic:
    a real Settings object's default repr was observed, during Phase 14
    development, to leak straight through pytest's own assertion-rewrite
    output (any `assert x == settings.some_field` failure prints the whole
    `settings` object to explain the comparison) — the exact accidental-
    exposure path this closes. It also protects against the same object
    ever being logged or included in an exception/traceback by mistake.
    `get_settings()` itself is still the only way to reach these values;
    nothing about how the rest of the app *uses* them changes.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # App DB (read/write, app schema only) — DSN embeds a password, repr=False.
    database_url: str = Field(repr=False)

    # Analytical DB (read-only role, analytics schema only — Sec 4 safety layer)
    analytics_database_url: str = Field(repr=False)
    readonly_db_user: str = "readonly_analyst"
    readonly_db_password: str = Field(default="", repr=False)
    sql_row_limit_default: int = 5000
    sql_statement_timeout_ms: int = 8000

    # LLM — see app/core/llm.py for the provider abstraction. "fake" (Phase
    # 13, Objective B) is a deterministic, offline stand-in
    # (app/core/fake_llm.py) used ONLY for the browser E2E test — never the
    # default, and nothing in production code ever sets it implicitly.
    llm_provider: Literal["anthropic", "groq", "fake"] = "anthropic"

    anthropic_api_key: str = Field(default="", repr=False)
    llm_model_fast: str = "claude-haiku-4-5-20251001"
    llm_model_strong: str = "claude-sonnet-5"

    groq_api_key: str = Field(default="", repr=False)
    groq_model_fast: str = "openai/gpt-oss-20b"
    groq_model_strong: str = "openai/gpt-oss-120b"

    # Bounded retry w/ exponential backoff for transient errors (429/5xx/connection) —
    # both anthropic and groq SDKs implement this internally and skip retrying
    # non-transient errors (400 bad request, 401 auth) on their own; this just
    # makes the count explicit/tunable instead of relying on each SDK's
    # undocumented default. See app/core/llm.py.
    llm_max_retries: int = 2

    # Critic loop (Sec 1, Fig. 2)
    critic_max_retries: int = 2

    # Report Generator (Phase 10) — the one optional LLM call (a grounding-
    # checked stakeholder-narrative rewrite, see app/agents/report_agent.py).
    # Off by default: every PASS/WARN report already has a Critic-validated
    # executive_summary, so the narrative is a bonus polish pass the report
    # doesn't depend on, not something worth an extra LLM call (and Groq
    # quota) on every single successful analysis by default. Flip to true
    # once quota headroom allows it.
    report_narrative_enabled: bool = False

    # Auth (Phase 14) — see app/core/auth.py. No default: signing every
    # session's access token with an empty/well-known key would make the
    # whole boundary decorative, so (like database_url) the app must be
    # given a real value via .env. Generate one with, e.g.,
    # `python -c "import secrets; print(secrets.token_hex(32))"`.
    secret_key: str = Field(repr=False)
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24

    # SECRET_KEY rotation (Phase 15, Objective 3) — see app/core/auth.py's
    # kid-based dual-key verification. secret_key_id labels whichever key
    # is CURRENT (every newly issued token carries it in the JWT header);
    # previous_secret_key/previous_secret_key_id are both None outside a
    # rotation window. To rotate: generate a new SECRET_KEY, move the old
    # value to PREVIOUS_SECRET_KEY, move the old SECRET_KEY_ID to
    # PREVIOUS_SECRET_KEY_ID, set a new SECRET_KEY_ID, deploy. Tokens
    # already issued under the old key keep validating (grace period) for
    # as long as PREVIOUS_SECRET_KEY stays configured; unset both
    # PREVIOUS_* fields and redeploy to end the grace period and reject
    # them. See docs/security.md's "SECRET_KEY rotation" section.
    secret_key_id: str = "default"
    previous_secret_key: str | None = Field(default=None, repr=False)
    previous_secret_key_id: str | None = None

    # Login brute-force protection (Phase 15, Objective 2) — see
    # app/core/security.py::LoginRateLimiter, applied only to
    # POST /auth/login (never the general RateLimitMiddleware below, and
    # never any authenticated endpoint). Token-bucket: at most
    # login_rate_limit_max_attempts attempts per login_rate_limit_window_
    # seconds, per (client IP, email) pair — every attempt counts, whether
    # it succeeds or fails.
    login_rate_limit_enabled: bool = True
    login_rate_limit_max_attempts: int = 5
    login_rate_limit_window_seconds: int = 300

    # CORS (final deployment phase) — the Streamlit frontend talks to this
    # API server-to-server (frontend/api_client.py runs inside the
    # frontend's own container, never in the visitor's browser), so CORS
    # is not on the critical path for the app to function. It's still set
    # explicitly, scoped to the one real frontend origin rather than "*",
    # for any direct browser-based access to the API (e.g. someone hitting
    # /docs or calling the API from a separate client). Comma-separated so
    # a local dev origin and a deployed Render origin can both be allowed
    # at once without code changes.
    frontend_origin: str = "http://localhost:8511"


@lru_cache
def get_settings() -> Settings:
    return Settings()
