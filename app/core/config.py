from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central config. Every field is env-driven — no hard-coded secrets or hosts."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # App DB (read/write, app schema only)
    database_url: str

    # Analytical DB (read-only role, analytics schema only — Sec 4 safety layer)
    analytics_database_url: str
    readonly_db_user: str = "readonly_analyst"
    readonly_db_password: str = ""
    sql_row_limit_default: int = 5000
    sql_statement_timeout_ms: int = 8000

    # LLM — see app/core/llm.py for the provider abstraction. "fake" (Phase
    # 13, Objective B) is a deterministic, offline stand-in
    # (app/core/fake_llm.py) used ONLY for the browser E2E test — never the
    # default, and nothing in production code ever sets it implicitly.
    llm_provider: Literal["anthropic", "groq", "fake"] = "anthropic"

    anthropic_api_key: str = ""
    llm_model_fast: str = "claude-haiku-4-5-20251001"
    llm_model_strong: str = "claude-sonnet-5"

    groq_api_key: str = ""
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


@lru_cache
def get_settings() -> Settings:
    return Settings()
