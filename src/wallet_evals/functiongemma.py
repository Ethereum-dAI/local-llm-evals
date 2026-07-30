"""Pure translation logic for the local Gemma-family GGUF promptfoo provider.

Gemma-family models (FunctionGemma-270m, the shipped Gemma-4 E4B) are local GGUFs
served in-process via llama-cpp-python. They emit tool calls in a Gemma DSL as
plain text rather than OpenAI `tool_calls`; the delimiter dialect differs per
model (see `gemma_dsl.Dialect`). This module turns promptfoo's prompt into chat
messages and the model's raw text into something `pf/assert.py` already scores —
so the scorer, tools.json, and dataset stay unchanged.
"""
from __future__ import annotations

import json
import re
from typing import Any

from wallet_evals.gemma_dsl import FUNCTIONGEMMA, Dialect, parse_gemma_tool_calls

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def decode_prompt(prompt: str, system_role: str = "developer") -> list[dict[str, str]]:
    """promptfoo hands the rendered prompt as a JSON conversation (our
    prompt.py:render returns a message list) or a plain string. Normalize to a
    message list and remap the `system` role to `system_role`.

    FunctionGemma requires `developer` to activate function calling (the default);
    the stock Gemma-4 chat template has no `developer` role, so serve it with
    `system_role="system"`."""
    messages: list[dict[str, str]]
    stripped = prompt.strip()
    if stripped[:1] in ("[", "{"):
        try:
            decoded = json.loads(stripped)
        except ValueError:
            decoded = None
        if isinstance(decoded, list):
            messages = [dict(m) for m in decoded if isinstance(m, dict)]
        elif isinstance(decoded, dict):
            messages = [dict(decoded)]
        else:
            messages = [{"role": "user", "content": prompt}]
    else:
        messages = [{"role": "user", "content": prompt}]

    for m in messages:
        if m.get("role") == "system":
            m["role"] = system_role
    return messages


def raw_output_to_scoreable(raw: str, dialect: Dialect = FUNCTIONGEMMA) -> str:
    """Convert the model's raw text into a value `pf/assert.py` understands.

    If the text contains Gemma-DSL tool calls (in `dialect`), return them as an
    OpenAI-shaped JSON string (name + JSON-string arguments) so assert.py's native
    path scores them. Otherwise return the text verbatim so refusals / clarifying
    questions (expected_calls == []) score and their prose surfaces in the reason.

    A leading `<think>…</think>` reasoning block (the fine-tune is trained to emit
    one before the call) is stripped first, so it never leaks into a refusal's
    surfaced prose. It never wraps the call itself, so parsing is unaffected."""
    raw = _THINK_RE.sub("", raw, count=1)
    parsed = parse_gemma_tool_calls(raw, dialect)
    if not parsed:
        return raw
    calls: list[dict[str, Any]] = [
        {"name": name, "arguments": json.dumps(fields)} for name, fields in parsed
    ]
    return json.dumps(calls)


def tool_calls_to_scoreable(tool_calls: list[dict[str, Any]]) -> str:
    """Convert llama-cpp native `message.tool_calls` into the same OpenAI-shaped
    JSON string `raw_output_to_scoreable` produces. Some Gemma GGUF chat templates
    surface tool calls as structured `tool_calls` instead of DSL text; normalize
    both to one shape so `pf/assert.py` scores them identically."""
    calls: list[dict[str, Any]] = []
    for tc in tool_calls:
        fn = tc.get("function", tc) if isinstance(tc, dict) else {}
        name = fn.get("name")
        if not name:
            continue
        args = fn.get("arguments")
        if not isinstance(args, str):
            args = json.dumps(args if args is not None else {})
        calls.append({"name": name, "arguments": args})
    return json.dumps(calls)
