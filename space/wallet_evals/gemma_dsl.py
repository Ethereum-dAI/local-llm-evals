"""Port of the Swift Gemma4FallbackParser, generalized over delimiter dialects.

The shipped on-device Gemma 4 emits:
    <|tool_call>call:NAME{key:<|"|>value<|"|>,key:<|"|>value<|"|>,...}<tool_call|>

FunctionGemma-270m emits the same call:NAME{...} body with different delimiters:
    <start_function_call>call:NAME{key:<escape>value<escape>,...}<end_function_call>
(https://ai.google.dev/gemma/docs/functiongemma/function-calling-with-hf).

Quoted values are delimited by the dialect's quote sequence; bare values run to
the next comma. Returns a list of (name, flat_fields) tuples.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Dialect:
    """The three delimiters that distinguish a Gemma tool-call DSL variant."""

    opener: str
    closer: str
    quote: str


GEMMA4 = Dialect(opener="<|tool_call>", closer="<tool_call|>", quote='<|"|>')
FUNCTIONGEMMA = Dialect(
    opener="<start_function_call>", closer="<end_function_call>", quote="<escape>"
)

# String → Dialect, so a promptfoo provider can select a dialect from YAML config
# (`config.dialect: gemma4`). Keys are the stable public names.
DIALECTS: dict[str, Dialect] = {"gemma4": GEMMA4, "functiongemma": FUNCTIONGEMMA}


def _parse_args(body: str, quote: str) -> dict[str, str]:
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
        if rest.startswith(quote):
            after_open = rest[len(quote) :]
            end = after_open.find(quote)
            if end == -1:
                break
            out[key] = after_open[:end]
            rest = after_open[end + len(quote) :]
        elif rest[:1] in "[{":
            # A bare (un-quoted) JSON array/object: read to the matching bracket
            # so internal commas don't split the value. FunctionGemma-270m emits
            # `args` this way (e.g. args:["0xrecipient","1000000"]).
            end = _match_bracket(rest)
            out[key] = rest[:end]
            rest = rest[end:]
        else:
            comma = rest.find(",")
            if comma == -1:
                comma = len(rest)
            value = rest[:comma].strip()
            if value:
                out[key] = value
            rest = rest[comma:]
    return out


def _match_bracket(s: str) -> int:
    """Index just past the bracket that closes the one opening at s[0]. Falls
    back to len(s) if unbalanced."""
    pairs = {"[": "]", "{": "}"}
    stack: list[str] = []
    for i, ch in enumerate(s):
        if ch in pairs:
            stack.append(pairs[ch])
        elif stack and ch == stack[-1]:
            stack.pop()
            if not stack:
                return i + 1
    return len(s)


def parse_gemma_tool_calls(
    raw: str, dialect: Dialect = GEMMA4
) -> list[tuple[str, dict[str, str]]]:
    """Extract all Gemma-DSL tool calls from raw model output."""
    calls: list[tuple[str, dict[str, str]]] = []
    search_from = 0
    while True:
        open_idx = raw.find(dialect.opener, search_from)
        if open_idx == -1:
            break
        close_idx = raw.find(dialect.closer, open_idx)
        if close_idx == -1:
            break
        inner = raw[open_idx + len(dialect.opener) : close_idx]
        search_from = close_idx + len(dialect.closer)

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
        calls.append((name, _parse_args(body, dialect.quote)))
    return calls
