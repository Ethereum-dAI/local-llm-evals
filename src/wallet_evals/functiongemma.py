"""Pure translation logic for the FunctionGemma-270m promptfoo provider.

FunctionGemma is a local GGUF model served in-process via llama-cpp-python. It
emits tool calls in the Gemma DSL (FUNCTIONGEMMA dialect) as plain text rather
than OpenAI `tool_calls`. This module turns promptfoo's prompt into chat
messages and the model's raw text into something `pf/assert.py` already scores —
so the scorer, tools.json, and dataset stay unchanged.
"""
from __future__ import annotations

import json
from typing import Any

from wallet_evals.gemma_dsl import FUNCTIONGEMMA, parse_gemma_tool_calls


def decode_prompt(prompt: str) -> list[dict[str, str]]:
    """promptfoo hands the rendered prompt as a JSON conversation (our
    prompt.py:render returns a message list) or a plain string. Normalize to a
    message list and remap `system` -> `developer`, the role FunctionGemma
    requires to activate function calling."""
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
            m["role"] = "developer"
    return messages


def raw_output_to_scoreable(raw: str) -> str:
    """Convert the model's raw text into a value `pf/assert.py` understands.

    If the text contains FunctionGemma tool calls, return them as an OpenAI-shaped
    JSON string (name + JSON-string arguments) so assert.py's native path scores
    them. Otherwise return the text verbatim so refusals / clarifying questions
    (expected_calls == []) score and their prose surfaces in the reason."""
    parsed = parse_gemma_tool_calls(raw, FUNCTIONGEMMA)
    if not parsed:
        return raw
    calls: list[dict[str, Any]] = [
        {"name": name, "arguments": json.dumps(fields)} for name, fields in parsed
    ]
    return json.dumps(calls)
