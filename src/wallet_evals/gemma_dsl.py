"""Port of the Swift Gemma4FallbackParser.

Format (verbatim from Gemma4FallbackParser.swift):
    <|tool_call>call:NAME{key:<|"|>value<|"|>,key:<|"|>value<|"|>,...}<tool_call|>
Quoted values are delimited by the 5-char sequence <|"|>; bare values run to the
next comma. Returns a list of (name, flat_fields) tuples.
"""
from __future__ import annotations

_OPENER = "<|tool_call>"
_CLOSER = "<tool_call|>"
_QUOTE = '<|"|>'


def _parse_args(body: str) -> dict[str, str]:
    out: dict[str, str] = {}
    rest = body
    while rest:
        rest = rest.lstrip(" \t\r\n,")
        if not rest:
            break
        colon = rest.find(":")
        if colon == -1:
            break
        key = rest[:colon].strip()
        rest = rest[colon + 1 :].lstrip(" \t\r\n")
        if rest.startswith(_QUOTE):
            after_open = rest[len(_QUOTE) :]
            end = after_open.find(_QUOTE)
            if end == -1:
                break
            out[key] = after_open[:end]
            rest = after_open[end + len(_QUOTE) :]
        else:
            comma = rest.find(",")
            if comma == -1:
                comma = len(rest)
            value = rest[:comma].strip()
            if value:
                out[key] = value
            rest = rest[comma:]
    return out


def parse_gemma_tool_calls(raw: str) -> list[tuple[str, dict[str, str]]]:
    """Extract all Gemma-DSL tool calls from raw model output."""
    calls: list[tuple[str, dict[str, str]]] = []
    search_from = 0
    while True:
        open_idx = raw.find(_OPENER, search_from)
        if open_idx == -1:
            break
        close_idx = raw.find(_CLOSER, open_idx)
        if close_idx == -1:
            break
        inner = raw[open_idx + len(_OPENER) : close_idx]
        search_from = close_idx + len(_CLOSER)

        inner = inner.strip()
        if not inner.startswith("call:"):
            continue
        inner = inner[len("call:") :]
        brace = inner.find("{")
        if brace == -1:
            name, body = inner.strip(), ""
        else:
            name = inner[:brace].strip()
            body = inner[brace + 1 :]
            if body.endswith("}"):
                body = body[:-1]
        calls.append((name, _parse_args(body)))
    return calls
