"""Unit tests for the additive token-usage bookkeeping added to
app/core/llm.py for Phase 8 ("LLM usage/cost where available"). Only tests
the accounting helper directly (`_record_usage`) — no network call, no API
key required, and no change to any existing call site's behavior.
"""

from __future__ import annotations

from app.core.llm import GroqLLMClient, LLMClient


def test_llm_client_accumulates_usage_across_calls():
    client = LLMClient()
    assert client.total_usage == {"input_tokens": 0, "output_tokens": 0}
    client._record_usage(100, 50)
    client._record_usage(10, 5)
    assert client.total_usage == {"input_tokens": 110, "output_tokens": 55}


class _FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


def test_groq_client_accumulates_usage_from_openai_shaped_usage_object():
    client = GroqLLMClient()
    client._record_usage(_FakeUsage(prompt_tokens=20, completion_tokens=10))
    client._record_usage(_FakeUsage(prompt_tokens=5, completion_tokens=2))
    assert client.total_usage == {"input_tokens": 25, "output_tokens": 12}


def test_groq_client_tolerates_a_missing_usage_object():
    client = GroqLLMClient()
    client._record_usage(None)
    assert client.total_usage == {"input_tokens": 0, "output_tokens": 0}
