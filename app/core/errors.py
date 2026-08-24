"""Shared, provider-agnostic exception classification (Phase 13).

Used wherever an LLM call's failure needs to be distinguished for
observability WITHOUT changing whether that failure is treated as fatal —
this module only answers "what kind of failure was this", never "what
should happen because of it". That policy stays exactly where it already
lived:

- app/agents/critic.py::_semantic_check — a failure here is still an INFO
  finding, never a content FAIL (Sec 9's "infra failure isn't a content
  failure" rule, unchanged).
- app/agents/report_agent.py::_try_narrative — a failure here still
  degrades the narrative to `None`, never blocks the report.
- app/services/analysis_service.py::run_analysis — a failure anywhere
  else in the graph still marks the whole session FAILED (unchanged
  SessionStatus enum); the classification only changes WHICH safe
  `error_message` gets stored, and what gets recorded in
  `execution_metadata.error_category` for observability.

Both `anthropic` and `groq`'s SDKs mirror the same exception hierarchy
(`APIError` -> `APIStatusError`/`APIConnectionError`, `RateLimitError` and
`AuthenticationError` as `APIStatusError` subclasses, `APITimeoutError` as
an `APIConnectionError` subclass) — confirmed live via
`inspect`/`dir(anthropic)`/`dir(groq)` rather than assumed, since guessing
SDK exception names would itself be exactly the kind of bug this module
exists to avoid introducing.
"""

from __future__ import annotations

from typing import Literal

import anthropic
import groq

ErrorCategory = Literal["rate_limit", "timeout", "provider_error", "validation_error", "application_error"]

_RATE_LIMIT_ERRORS = (anthropic.RateLimitError, groq.RateLimitError)
_TIMEOUT_ERRORS = (anthropic.APITimeoutError, groq.APITimeoutError)
# Every other provider-side status/connection problem (bad request, auth,
# 5xx, network) — RateLimitError/APITimeoutError are checked first since
# they're subclasses of these broader types.
_PROVIDER_ERRORS = (anthropic.APIStatusError, groq.APIStatusError, anthropic.APIConnectionError, groq.APIConnectionError)
_VALIDATION_ERRORS = (ValueError, TypeError, KeyError, LookupError)


def classify_exception(exc: BaseException) -> ErrorCategory:
    """Never raises, never returns anything but one of the 5 documented
    categories — an unrecognized exception type is "application_error",
    the safe default, not a crash in the classifier itself."""
    if isinstance(exc, _RATE_LIMIT_ERRORS):
        return "rate_limit"
    if isinstance(exc, _TIMEOUT_ERRORS):
        return "timeout"
    if isinstance(exc, _PROVIDER_ERRORS):
        return "provider_error"
    if isinstance(exc, _VALIDATION_ERRORS):
        return "validation_error"
    return "application_error"


# Fixed, safe (no exception text, no stack trace) messages per category —
# shared so every layer that surfaces one to a user says the same honest
# thing about the same kind of failure (Sec 11 security: sanitized API
# responses only).
SAFE_MESSAGE_BY_CATEGORY: dict[ErrorCategory, str] = {
    "rate_limit": (
        "Analysis could not complete: the configured LLM provider is rate-limited or "
        "out of quota. This is an external/infrastructure condition, not an application "
        "error — please try again later."
    ),
    "timeout": (
        "Analysis could not complete: the configured LLM provider did not respond in "
        "time. This is an external/infrastructure condition, not an application error — "
        "please try again."
    ),
    "provider_error": (
        "Analysis could not complete: the configured LLM provider returned an error. "
        "This is an external/infrastructure condition, not an application error — "
        "please try again later."
    ),
    "validation_error": "Analysis failed due to an internal data validation error. Please try again.",
    "application_error": "Analysis failed. Please try again.",
}


def safe_message_for(exc: BaseException) -> str:
    return SAFE_MESSAGE_BY_CATEGORY[classify_exception(exc)]
