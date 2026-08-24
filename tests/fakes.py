"""Fake LLM client for tests — implements app.core.llm.LLMClientProtocol
without touching ANTHROPIC_API_KEY or the network. Per the project's testing
rules: never fabricate a *live* LLM result, but deterministic tests of the
scaffolding (prompt construction, plan/response wiring, routing) shouldn't
depend on an external API either.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from pydantic import BaseModel


class ScriptedLLMClient:
    """`responses[SomeModel]` is a list consumed in order — one entry per call
    for that response_model type. Raises if a type is requested with nothing
    left scripted, so a test can't silently pass on an unexpected extra call.
    """

    def __init__(self, responses: dict[type[BaseModel], list[BaseModel]]) -> None:
        self._queues: dict[type[BaseModel], list[BaseModel]] = defaultdict(list)
        for model_type, values in responses.items():
            self._queues[model_type] = list(values)
        self.calls: list[dict[str, Any]] = []

    async def complete(self, *, tier, system: str, messages: list[dict[str, Any]], **kwargs) -> str:
        raise NotImplementedError("ScriptedLLMClient only supports complete_structured")

    async def complete_structured(
        self, *, tier, system: str, messages: list[dict[str, Any]], response_model, **kwargs
    ):
        self.calls.append({"tier": tier, "system": system, "messages": messages, "response_model": response_model})
        queue = self._queues[response_model]
        if not queue:
            raise AssertionError(f"ScriptedLLMClient: no more responses queued for {response_model.__name__}")
        return queue.pop(0)
