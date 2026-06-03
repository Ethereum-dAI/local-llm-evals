"""Normalize a raw model response into a ParsedTurn.

Prefers native OpenAI tool calls (used by OpenRouter and llama-server with
tool-calling). Falls back to the Gemma DSL parser for models that emit the DSL
in plain text. In both cases the result is a list of ParsedToolCall with a
positional `args` list.
"""
from __future__ import annotations

import json
from typing import Any

from wallet_evals.gemma_dsl import parse_gemma_tool_calls
from wallet_evals.schema import ParsedToolCall, ParsedTurn


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _coerce_args(value: Any) -> list[Any]:
    """args may arrive as a real list (native) or a JSON-encoded string (DSL)."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (ValueError, TypeError):
            return []
        return decoded if isinstance(decoded, list) else []
    return []


def _build_call(name: str, fields: dict[str, Any]) -> ParsedToolCall:
    return ParsedToolCall(
        name=name,
        chainId=_as_str(fields.get("chainId")),
        to=_as_str(fields.get("to")),
        value=_as_str(fields.get("value")),
        function=_as_str(fields.get("function")),
        args=_coerce_args(fields.get("args")),
        currencyIn=_as_str(fields.get("currencyIn")),
        currencyOut=_as_str(fields.get("currencyOut")),
        amountIn=_as_str(fields.get("amountIn")),
        amountOutMinimum=_as_str(fields.get("amountOutMinimum")),
        recipient=_as_str(fields.get("recipient")),
    )


def parse_turn(
    *,
    content: str | None,
    native_tool_calls: list[dict[str, Any]] | None,
    raw_text: str,
) -> ParsedTurn:
    calls: list[ParsedToolCall] = []

    if native_tool_calls:
        for tc in native_tool_calls:
            name = tc.get("name", "")
            args_blob = tc.get("arguments", "{}")
            try:
                fields = json.loads(args_blob) if isinstance(args_blob, str) else dict(args_blob)
            except (ValueError, TypeError):
                fields = {}
            calls.append(_build_call(name, fields))
    else:
        for name, fields in parse_gemma_tool_calls(raw_text):
            calls.append(_build_call(name, fields))

    return ParsedTurn(content=content, tool_calls=calls)
