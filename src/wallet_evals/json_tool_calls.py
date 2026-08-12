"""Parse tool calls that a local GGUF emits as JSON inside plain text.

The Gemma family emits its own `call:NAME{...}` DSL (see `gemma_dsl`), but Qwen3
— and most non-Gemma models — write a JSON object instead, wrapped in a marker
their chat template documents:

    Qwen3 / Hermes   <tool_call>{"name": "executeTx", "arguments": {...}}</tool_call>

The other wrappers below (``functools[...]``, fenced JSON, a bare object) are
shapes models reach for when they improvise; accepting them costs nothing and
keeps a formatting quirk from being scored as a capability gap.

`llama_cpp` renders `tools` into the GGUF's own chat template but does NOT parse
the calls back out — `message.tool_calls` stays empty and the call arrives as
text — so the provider needs this to reach the same shape `pf/assert.py` scores.

What is deliberately NOT done here: no repair of malformed JSON, no regex
scraping of key/value pairs out of prose. A model that cannot emit its own
documented call format has failed the task, and papering over that would erase a
real capability gap — the same line `scorer._norm_scalar` holds.
"""
from __future__ import annotations

import json
import re
from typing import Any

# Marker pairs, most specific first. A model often echoes the tag it was shown.
_TAGS: tuple[tuple[str, str], ...] = (
    ("<tool_call>", "</tool_call>"),      # Hermes / Qwen style
    ("<|tool_call|>", "<|/tool_call|>"),  # tagged variant
    ("```json", "```"),                   # models that fence the call
)
# Some models prefix a bare JSON array with `functools` instead of tagging it.
_PREFIX_RE = re.compile(r"\bfunctools\s*(?=\[)")


def _match_bracket(s: str) -> int:
    """Index just past the bracket closing the one at s[0], honouring strings.

    Unlike `gemma_dsl._match_bracket`, this must ignore brackets inside JSON
    string values — a recipient like "0x[..]" is not real, but an ENS name or a
    prose `content` field alongside the call can carry one.
    """
    pairs = {"[": "]", "{": "}"}
    stack: list[str] = []
    in_string = False
    escaped = False
    for i, ch in enumerate(s):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in pairs:
            stack.append(pairs[ch])
        elif stack and ch == stack[-1]:
            stack.pop()
            if not stack:
                return i + 1
    return len(s)


def _candidates(raw: str) -> list[str]:
    """Every substring of `raw` that might be a JSON tool call, in priority order."""
    out: list[str] = []
    for opener, closer in _TAGS:
        start = 0
        while True:
            i = raw.find(opener, start)
            if i == -1:
                break
            j = raw.find(closer, i + len(opener))
            if j == -1:
                # Truncated by max_tokens, or the model never closed the tag:
                # take the rest and let the JSON decode decide.
                out.append(raw[i + len(opener):])
                break
            out.append(raw[i + len(opener):j])
            start = j + len(closer)
    if out:
        return out

    m = _PREFIX_RE.search(raw)
    if m:
        rest = raw[m.end():]
        return [rest[: _match_bracket(rest)]]

    # Unmarked: the first balanced JSON value in the text.
    for i, ch in enumerate(raw):
        if ch in "[{":
            return [raw[i:][: _match_bracket(raw[i:])]]
    return []


# Keys that name the call rather than being an argument to it. Everything else in
# a flat emission is an argument — `function` included, since in this tool schema
# it holds the solidity signature ("transfer(address,uint256)").
_NAME_KEYS = ("name", "tool_name", "type")


def _args_from(obj: dict[str, Any]) -> dict[str, Any]:
    """The argument dict, under whichever key this model chose for it."""
    for key in ("arguments", "parameters", "args"):
        value = obj.get(key)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                continue
        if isinstance(value, dict):
            return value
    # Flat emission: the arguments sit beside the name. `args` stays a field here
    # (it is a real tool argument — the positional calldata list), which is why
    # the loop above only accepts it when it decodes to a dict.
    return {k: v for k, v in obj.items() if k not in _NAME_KEYS}


def _as_call(obj: Any) -> tuple[str, dict[str, Any]] | None:
    """One JSON object -> (tool name, arguments), across the wrappers models pick.

    A model may answer with its own nesting, e.g.

        {"type": "executeTx",
         "function": {"name": "executeTx", "parameters": {"chainId": "1", ...}}}

    Reading only `arguments` there yields an empty-arg call and would score a
    fully-populated answer as wrong. Unwrapping is the same class of fix as
    `parsing._decode_dsl_quoted_array`: it erases a wrapper the model was never
    told to avoid, while every value still has to be right to score.
    """
    if not isinstance(obj, dict):
        return None
    inner = obj.get("function")
    if isinstance(inner, dict):  # OpenAI-style nesting
        name = inner.get("name") or next(
            (obj[k] for k in _NAME_KEYS if isinstance(obj.get(k), str)), None
        )
        args = _args_from(inner)
    else:
        name = next((obj[k] for k in _NAME_KEYS if isinstance(obj.get(k), str)), None)
        args = _args_from(obj)
    if not isinstance(name, str) or not name:
        return None
    return name, args


def parse_json_tool_calls(raw: str) -> list[tuple[str, dict[str, Any]]]:
    """Extract JSON tool calls from raw model text. Empty list if there are none."""
    calls: list[tuple[str, dict[str, Any]]] = []
    for chunk in _candidates(raw):
        chunk = chunk.strip().strip("`").strip()
        if chunk[:4] == "json":  # leftover ```json fence label
            chunk = chunk[4:].strip()
        try:
            decoded = json.loads(chunk)
        except ValueError:
            continue
        for item in decoded if isinstance(decoded, list) else [decoded]:
            call = _as_call(item)
            if call:
                calls.append(call)
    return calls
