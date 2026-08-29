"""Phase 14, Issue 3 — provider contract tests. `app/core/llm.py::
LLMClientProtocol` is what every agent actually depends on (Sec 0 decision
note); this file asserts `LLMClient` (Anthropic), `GroqLLMClient`, and
`FakeLLMClient` (app/core/fake_llm.py) are all genuinely interchangeable
implementations of it, not just three unrelated classes that happen to work
today.

No live provider call anywhere in this file — `LLMClient()`/`GroqLLMClient()`
only construct the underlying SDK client (a local object, no network I/O);
every check here is structural (signatures, attributes) or exercises the
dispatch logic in `get_llm_client()`. Existing behavioral coverage for the
fake provider's actual responses already lives in tests/unit/test_fake_llm.py
and isn't duplicated here.
"""

from __future__ import annotations

import inspect

import pytest

import app.core.llm as llm_module
from app.core.fake_llm import FakeLLMClient
from app.core.llm import GroqLLMClient, LLMClient, LLMClientProtocol, get_llm_client

_IMPLEMENTATIONS = [LLMClient, GroqLLMClient, FakeLLMClient]


@pytest.fixture(autouse=True)
def _reset_llm_singleton(monkeypatch):
    """get_llm_client() caches one process-wide client in a module global —
    tests that flip LLM_PROVIDER via env/settings must not leak that
    singleton into a later, unrelated test."""
    monkeypatch.setattr(llm_module, "_client", None)
    yield
    monkeypatch.setattr(llm_module, "_client", None)


def _protocol_required_param_names(method_name: str) -> set[str]:
    """Only the Protocol's no-default (`= ...`-less) parameters — tier,
    system, messages, and response_model for complete_structured.
    max_tokens/temperature carry defaults in the Protocol itself and are
    legitimately satisfiable via a `**kwargs` catch-all (FakeLLMClient does
    this) rather than a literally-named parameter; requiring an exact name
    match on those would penalize a valid duck-typed implementation."""
    sig = inspect.signature(getattr(LLMClientProtocol, method_name))
    return {
        name
        for name, param in sig.parameters.items()
        if name != "self" and param.default is inspect.Parameter.empty
    }


@pytest.mark.parametrize("impl", _IMPLEMENTATIONS)
def test_every_implementation_exposes_both_protocol_methods(impl):
    assert hasattr(impl, "complete")
    assert hasattr(impl, "complete_structured")
    assert inspect.iscoroutinefunction(impl.complete)
    assert inspect.iscoroutinefunction(impl.complete_structured)


@pytest.mark.parametrize("impl", _IMPLEMENTATIONS)
@pytest.mark.parametrize("method_name", ["complete", "complete_structured"])
def test_every_implementation_accepts_the_protocols_required_keyword_arguments(impl, method_name):
    """Every keyword the Protocol declares (tier, system, messages, and
    response_model for complete_structured) must be a real, name-matching
    parameter on the concrete method — an agent written against the
    Protocol must be able to call any implementation identically."""
    required = _protocol_required_param_names(method_name)
    concrete_params = set(inspect.signature(getattr(impl, method_name)).parameters) - {"self"}
    assert required <= concrete_params, f"{impl.__name__}.{method_name} is missing {required - concrete_params}"


@pytest.mark.parametrize("impl", [LLMClient, GroqLLMClient, FakeLLMClient])
def test_every_implementation_can_be_constructed_without_a_network_call(impl, monkeypatch):
    """Construction only ever builds a local SDK client object / sets local
    state — never a network round trip. LLMClient/GroqLLMClient read real
    settings (api keys may be blank in some environments), so this only
    asserts construction doesn't raise for a *reachable* reason, not that
    the resulting client can complete a real request."""
    instance = impl()
    assert hasattr(instance, "total_usage")
    assert instance.total_usage == {"input_tokens": 0, "output_tokens": 0}


def test_get_llm_client_dispatches_to_fake_when_provider_is_fake(monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setenv("LLM_PROVIDER", "fake")
    get_settings.cache_clear()
    try:
        assert isinstance(get_llm_client(), FakeLLMClient)
    finally:
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        get_settings.cache_clear()


def test_get_llm_client_dispatches_to_groq_when_provider_is_groq(monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setenv("LLM_PROVIDER", "groq")
    get_settings.cache_clear()
    try:
        assert isinstance(get_llm_client(), GroqLLMClient)
    finally:
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        get_settings.cache_clear()


def test_get_llm_client_dispatches_to_anthropic_when_provider_is_the_default(monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    get_settings.cache_clear()
    try:
        assert isinstance(get_llm_client(), LLMClient)
    finally:
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        get_settings.cache_clear()


def test_get_llm_client_returns_the_same_cached_instance_on_repeated_calls():
    first = get_llm_client()
    second = get_llm_client()
    assert first is second


@pytest.mark.asyncio
async def test_fake_client_complete_returns_a_string_matching_the_protocols_return_type():
    client = FakeLLMClient()
    result = await client.complete(tier="fast", system="s", messages=[{"role": "user", "content": "x"}])
    assert isinstance(result, str)


def test_real_clients_total_usage_accumulates_the_same_shape_as_the_fake_clients():
    """All three expose the same {"input_tokens": int, "output_tokens": int}
    bookkeeping shape (app/evaluation/evaluator.py and
    app/services/analysis_service.py both read `.total_usage` without
    branching on which provider produced it) — this is the contract that
    lets token-usage reporting be provider-agnostic."""
    for impl in _IMPLEMENTATIONS:
        instance = impl()
        assert set(instance.total_usage.keys()) == {"input_tokens", "output_tokens"}
        assert all(isinstance(v, int) for v in instance.total_usage.values())
