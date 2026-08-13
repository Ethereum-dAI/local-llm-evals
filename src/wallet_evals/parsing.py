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


def _normalize_function(value: Any) -> str | None:
    """A no-calldata call has function `null`. Some models emit the literal
    string "null"/"none" or "" instead of JSON null — treat those as None so a
    native transfer still matches the gold."""
    s = _as_str(value)
    if s is not None and s.strip().lower() in ("", "null", "none"):
        return None
    return s


# The quote sequences a Gemma DSL dialect can wrap a value in. Gemma-4's own
# chat template serializes an ARRAY argument element-wise in these markers
# (`format_argument` -> `[<|"|>a<|"|>,<|"|>b<|"|>]`), so they appear inside the
# bracketed substring `gemma_dsl._parse_args` hands over intact.
_DSL_QUOTES = ('<|"|>', "<escape>")


def _coerce_args(value: Any) -> list[Any]:
    """args may arrive as a real list (native) or a JSON-encoded string (DSL).

    The DSL string is not always valid JSON. The stock on-device Gemma-4 emits the
    element-quoted form its own template documents:

        args:[<|"|>0xrecipient<|"|>,<|"|>3000000<|"|>]

    which `json.loads` rejects. Returning [] there scores a byte-perfect call as
    wrong — and it is not a symmetric mistake: on the Jul-20 run the stock model
    emitted this form 95 times and the fine-tune (trained to emit a bare JSON
    array) zero, so the failure only ever cost the baseline. Decoding the markers
    erases an ENCODING difference, not a capability difference: the model still
    has to get the recipient and the base-unit amount right to score.
    """
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (ValueError, TypeError):
            decoded = _decode_dsl_quoted_array(value)
        return decoded if isinstance(decoded, list) else []
    return []


def _decode_dsl_quoted_array(value: str) -> list[Any] | None:
    """Decode `[<|"|>a<|"|>,<|"|>b<|"|>]` into ["a", "b"].

    Splits on the marker rather than rewriting it to a JSON quote: the GEMMA4
    marker <|"|> itself contains a double quote, so any escape-then-substitute
    pass mangles it. Splitting is also immune to commas and quotes inside a
    value, since element boundaries are the markers themselves — for
    `[<|"|>a<|"|>,<|"|>b<|"|>]` the split yields ['', 'a', ',', 'b', ''] and the
    values are exactly the odd positions.

    Returns None when the text is not a marker-quoted array, leaving the
    caller's [].
    """
    text = value.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return None
    inner = text[1:-1].strip()
    if not inner:
        return None
    for quote in _DSL_QUOTES:
        if quote not in inner:
            continue
        parts = inner.split(quote)
        # A well-formed run is marker-delimited, so the split must have an odd
        # length (empty head and tail) and the even slots must be only the commas
        # between elements. Anything else is malformed — refuse rather than guess.
        if len(parts) % 2 == 0:
            return None
        if any(p.strip(" \t\r\n,") for p in parts[::2]):
            return None
        return parts[1::2]
    return None


def _build_call(name: str, fields: dict[str, Any]) -> ParsedToolCall:
    return ParsedToolCall(
        name=name,
        chainId=_as_str(fields.get("chainId")),
        to=_as_str(fields.get("to")),
        value=_as_str(fields.get("value")),
        function=_normalize_function(fields.get("function")),
        args=_coerce_args(fields.get("args")),
        currencyIn=_as_str(fields.get("currencyIn")),
        currencyOut=_as_str(fields.get("currencyOut")),
        amountIn=_as_str(fields.get("amountIn")),
        amountOutMinimum=_as_str(fields.get("amountOutMinimum")),
        recipient=_as_str(fields.get("recipient")),
        amount=_as_str(fields.get("amount")),
        token=_as_str(fields.get("token")),
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
