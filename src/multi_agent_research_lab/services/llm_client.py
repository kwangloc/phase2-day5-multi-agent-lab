"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

import logging
from dataclasses import dataclass
from math import ceil
from typing import Any, cast

from tenacity import Retrying, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client.

    Uses OpenAI when configured and falls back to a deterministic offline response.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = None

        if self._settings.openai_api_key:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self._settings.openai_api_key)
            except Exception:
                self._client = None

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, ceil(len(text) / 4))

    @staticmethod
    def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
        # Approximate $/1M token rates for quick benchmarking.
        rates = {
            "gpt-4o-mini": (0.15, 0.60),
            "gpt-4.1-mini": (0.40, 1.60),
        }
        if model not in rates:
            return None
        in_rate, out_rate = rates[model]
        return (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate

    def _offline_complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        combined = f"{system_prompt}\n\n{user_prompt}".strip()
        input_tokens = self._estimate_tokens(combined)
        content = (
            "[offline-mode] Generated without external LLM provider.\n"
            f"Request summary: {user_prompt[:280].strip()}"
        )
        output_tokens = self._estimate_tokens(content)
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=0.0,
        )

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion.
        """

        if not user_prompt.strip():
            return LLMResponse(content="", input_tokens=0, output_tokens=0, cost_usd=0.0)

        if self._client is None:
            logger.info("LLM fallback mode: no provider client configured")
            return self._offline_complete(system_prompt, user_prompt)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        with trace_span("llm.complete", {"model": self._settings.openai_model}) as span:
            for attempt in Retrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=1, max=8),
                reraise=True,
            ):
                with attempt:
                    response = self._client.chat.completions.create(
                        model=self._settings.openai_model,
                        messages=cast(Any, messages),
                        timeout=self._settings.timeout_seconds,
                    )

        message = response.choices[0].message.content if response.choices else ""
        content = message or ""

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)

        if input_tokens is None:
            input_tokens = self._estimate_tokens(f"{system_prompt}\n{user_prompt}")
        if output_tokens is None:
            output_tokens = self._estimate_tokens(content)

        logger.info(
            "LLM completion model=%s input_tokens=%s output_tokens=%s duration=%.3fs",
            self._settings.openai_model,
            input_tokens,
            output_tokens,
            float(span["duration_seconds"] or 0.0),
        )

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=self._estimate_cost(self._settings.openai_model, input_tokens, output_tokens),
        )
