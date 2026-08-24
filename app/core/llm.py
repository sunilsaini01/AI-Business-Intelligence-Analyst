"""Single adapter for all LLM calls (Sec 0 decision note).

Every agent that needs the LLM imports from here, never `anthropic`/`groq`
directly — swapping providers touches this file only, never the seven agent
files. Two model tiers: `fast` for SQL generation / routing, `strong` for the
Critic and Report agents where reasoning quality matters most.

Two REAL providers as of the Groq addition (Anthropic account temporarily
out of credit): `LLM_PROVIDER=anthropic` (default) uses Claude via forced
Messages-API tool-use; `LLM_PROVIDER=groq` uses Groq's OpenAI-compatible
chat-completions API with forced function-calling. Same `LLMClientProtocol`
either way — no agent code branches on provider. Swap back by changing one
env var.

A third, non-production value, `LLM_PROVIDER=fake` (Phase 13, Objective B —
see app/core/fake_llm.py), returns deterministic canned responses instead
of calling any network API. It exists solely so the browser E2E test can
drive the real backend without live provider quota; the default stays
`anthropic`, so nothing about normal operation changes.

Per Sec 5's "0 LLM calls" rule: app/agents/analysis_agent.py and
app/agents/ml_agent.py must never import this module.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Protocol, TypeVar

import anthropic
import groq
from pydantic import BaseModel

from app.core.config import get_settings

T = TypeVar("T", bound=BaseModel)


class ModelTier(str, Enum):
    FAST = "fast"
    STRONG = "strong"


class LLMClientProtocol(Protocol):
    """What an agent actually depends on — narrow enough that tests can hand
    in a fake without importing anthropic or touching ANTHROPIC_API_KEY.
    """

    async def complete(
        self, *, tier: ModelTier, system: str, messages: list[dict[str, Any]],
        max_tokens: int = ..., temperature: float = ...,
    ) -> str: ...

    async def complete_structured(
        self, *, tier: ModelTier, system: str, messages: list[dict[str, Any]],
        response_model: type[T], max_tokens: int = ...,
    ) -> T: ...


class LLMClient:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key, max_retries=settings.llm_max_retries
        )
        self._models = {
            ModelTier.FAST: settings.llm_model_fast,
            ModelTier.STRONG: settings.llm_model_strong,
        }
        # Cumulative token usage across every call made through this instance —
        # additive-only bookkeeping (Phase 8's evaluator reads it for the
        # "cost where available" metric); no existing call site's signature
        # or return value changes because of this.
        self.total_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

    def _record_usage(self, input_tokens: int, output_tokens: int) -> None:
        self.total_usage["input_tokens"] += input_tokens
        self.total_usage["output_tokens"] += output_tokens

    async def complete(
        self,
        *,
        tier: ModelTier,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> str:
        """One non-streaming completion. Transient errors (429/5xx/connection) are
        retried automatically by the underlying SDK client (bounded, exponential
        backoff — see `max_retries` in __init__); non-transient errors (400, 401)
        are not retried. Callers don't need their own retry loop."""
        response = await self._client.messages.create(
            model=self._models[tier],
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=messages,
        )
        self._record_usage(response.usage.input_tokens, response.usage.output_tokens)
        return "".join(block.text for block in response.content if block.type == "text")

    async def complete_structured(
        self,
        *,
        tier: ModelTier,
        system: str,
        messages: list[dict[str, Any]],
        response_model: type[T],
        max_tokens: int = 2048,
    ) -> T:
        """Forces the model to reply as a single tool_use call whose input
        matches `response_model`'s JSON schema — the standard way to get
        reliable structured output from Claude, no ad hoc JSON-parsing of a
        text reply. Used by the Supervisor and SQL Agent (Sec 5): plans and
        generated SQL are structured data, not prose to regex out.
        """
        tool_name = f"emit_{response_model.__name__.lower()}"
        response = await self._client.messages.create(
            model=self._models[tier],
            max_tokens=max_tokens,
            temperature=0.0,
            system=system,
            messages=messages,
            tools=[
                {
                    "name": tool_name,
                    "description": f"Return the result as {response_model.__name__}.",
                    "input_schema": response_model.model_json_schema(),
                }
            ],
            tool_choice={"type": "tool", "name": tool_name},
        )
        self._record_usage(response.usage.input_tokens, response.usage.output_tokens)
        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                return response_model.model_validate(block.input)
        raise ValueError(f"LLM did not return the expected {tool_name} tool call.")


class GroqLLMClient:
    """Same contract as LLMClient, against Groq's OpenAI-compatible API.
    Forced tool-calling stands in for Anthropic's forced tool_use: Groq's
    `tool_choice={"type": "function", "function": {"name": ...}}` makes the
    model call exactly the one function whose parameters schema is the
    Pydantic model, same guarantee — a validated structured object, not
    prose to regex out of a text reply.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._client = groq.AsyncGroq(api_key=settings.groq_api_key, max_retries=settings.llm_max_retries)
        self._models = {
            ModelTier.FAST: settings.groq_model_fast,
            ModelTier.STRONG: settings.groq_model_strong,
        }
        self.total_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

    def _record_usage(self, usage: Any) -> None:
        # Groq's OpenAI-compatible response may omit `usage` entirely on some
        # error/edge paths — tolerate that rather than raising here, since
        # token accounting must never be the thing that breaks a real call.
        if usage is None:
            return
        self.total_usage["input_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        self.total_usage["output_tokens"] += getattr(usage, "completion_tokens", 0) or 0

    async def complete(
        self,
        *,
        tier: ModelTier,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> str:
        response = await self._client.chat.completions.create(
            model=self._models[tier],
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "system", "content": system}, *messages],
        )
        self._record_usage(response.usage)
        return response.choices[0].message.content or ""

    async def complete_structured(
        self,
        *,
        tier: ModelTier,
        system: str,
        messages: list[dict[str, Any]],
        response_model: type[T],
        max_tokens: int = 2048,
    ) -> T:
        tool_name = f"emit_{response_model.__name__.lower()}"
        response = await self._client.chat.completions.create(
            model=self._models[tier],
            max_tokens=max_tokens,
            temperature=0.0,
            messages=[{"role": "system", "content": system}, *messages],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": f"Return the result as {response_model.__name__}.",
                        "parameters": response_model.model_json_schema(),
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": tool_name}},
        )
        self._record_usage(response.usage)
        message = response.choices[0].message
        tool_calls = message.tool_calls or []
        for call in tool_calls:
            if call.function.name == tool_name:
                return response_model.model_validate(json.loads(call.function.arguments))
        raise ValueError(
            f"LLM did not return the expected {tool_name} tool call. "
            f"Raw content: {message.content!r}"
        )


_client: LLMClientProtocol | None = None


def get_llm_client() -> LLMClientProtocol:
    global _client
    if _client is None:
        settings = get_settings()
        if settings.llm_provider == "groq":
            _client = GroqLLMClient()
        elif settings.llm_provider == "fake":
            # Phase 13, Objective B — deterministic, offline stand-in for
            # the browser E2E test only. Requires an explicit
            # LLM_PROVIDER=fake; the default ("anthropic") never reaches
            # this branch, so production behavior is unchanged.
            from app.core.fake_llm import FakeLLMClient

            _client = FakeLLMClient()
        else:
            _client = LLMClient()
    return _client
