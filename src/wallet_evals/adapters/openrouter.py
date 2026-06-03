"""OpenRouter adapter (OpenAI-compatible chat completions with tool calls).

Sampler mirrors production (RecognitionRunner.swift): temperature 0.2,
bounded max tokens. Built first so the eval runs before any local model infra.
"""
from __future__ import annotations

import os
from typing import Any

from wallet_evals.parsing import parse_turn
from wallet_evals.schema import ParsedTurn
from wallet_evals.tools import SYSTEM_PROMPT, TOOLS

_MAX_TOKENS = 512
_TEMPERATURE = 0.2
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _default_client() -> Any:
    from openai import OpenAI

    return OpenAI(base_url=_OPENROUTER_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"])


class OpenRouterAdapter:
    def __init__(self, *, model: str, client: Any | None = None, seed: int = 0xC0DEFEED):
        self.model = model
        self._client = client if client is not None else _default_client()
        self._seed = seed

    def run(self, user_message: str) -> ParsedTurn:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            tools=TOOLS,
            temperature=_TEMPERATURE,
            max_tokens=_MAX_TOKENS,
            seed=self._seed,
        )
        message = response.choices[0].message
        native = None
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            native = [{"name": tc.function.name, "arguments": tc.function.arguments} for tc in tool_calls]
        content = getattr(message, "content", None)
        return parse_turn(content=content, native_tool_calls=native, raw_text=content or "")
