"""Deterministic tests for app/core/errors.py (Phase 13, Objective A). No
live LLM calls — every exception is constructed directly.
"""

from __future__ import annotations

import anthropic
import groq
import httpx
import pytest

from app.core.errors import SAFE_MESSAGE_BY_CATEGORY, classify_exception, safe_message_for

_RESP = httpx.Response(429, request=httpx.Request("POST", "http://test"))


def _status_error(cls, status_code: int):
    resp = httpx.Response(status_code, request=httpx.Request("POST", "http://test"))
    return cls("msg", response=resp, body=None)


@pytest.mark.parametrize("cls", [anthropic.RateLimitError, groq.RateLimitError])
def test_rate_limit_errors_classified_as_rate_limit(cls):
    assert classify_exception(_status_error(cls, 429)) == "rate_limit"


@pytest.mark.parametrize("cls", [anthropic.APITimeoutError, groq.APITimeoutError])
def test_timeout_errors_classified_as_timeout(cls):
    req = httpx.Request("POST", "http://test")
    assert classify_exception(cls(req)) == "timeout"


@pytest.mark.parametrize(
    "cls,status",
    [
        (anthropic.AuthenticationError, 401),
        (anthropic.BadRequestError, 400),
        (anthropic.InternalServerError, 500),
        (groq.AuthenticationError, 401),
        (groq.InternalServerError, 500),
    ],
)
def test_other_provider_status_errors_classified_as_provider_error(cls, status):
    assert classify_exception(_status_error(cls, status)) == "provider_error"


def test_connection_error_classified_as_provider_error():
    req = httpx.Request("POST", "http://test")
    assert classify_exception(anthropic.APIConnectionError(request=req)) == "provider_error"


@pytest.mark.parametrize("exc", [ValueError("bad"), TypeError("bad"), KeyError("missing")])
def test_python_value_errors_classified_as_validation_error(exc):
    assert classify_exception(exc) == "validation_error"


def test_unrecognized_exception_defaults_to_application_error():
    assert classify_exception(RuntimeError("something else broke")) == "application_error"


def test_classify_exception_never_raises_on_an_odd_input():
    class _Weird(Exception):
        pass

    assert classify_exception(_Weird()) == "application_error"


def test_every_category_has_a_safe_message_with_no_exception_text():
    for category, message in SAFE_MESSAGE_BY_CATEGORY.items():
        assert isinstance(message, str) and message
        assert "Traceback" not in message


def test_safe_message_for_rate_limit_matches_the_rate_limit_category():
    exc = _status_error(groq.RateLimitError, 429)
    assert safe_message_for(exc) == SAFE_MESSAGE_BY_CATEGORY["rate_limit"]


def test_safe_message_never_echoes_the_original_exception_message():
    exc = _status_error(anthropic.BadRequestError, 400)
    exc.args = ("leaking postgresql://user:pw@host/db",)
    message = safe_message_for(exc)
    assert "postgresql://" not in message
    assert "leaking" not in message
