"""promptfoo Python assertion.

Reuses the deterministic binary scorer: rebuild the gold `Case` from the test's
metadata, normalize the model's tool calls (whatever shape promptfoo hands us)
into a ParsedTurn, and score 1/0 with `score_case`.

Referenced from promptfooconfig.yaml as:
    assert:
      - type: python
        value: file://pf/assert.py
"""
from __future__ import annotations

import json
from typing import Any

from wallet_evals.parsing import parse_turn
from wallet_evals.promptfoo import case_from_metadata
from wallet_evals.scorer import score_case


def _normalize_calls(obj: Any) -> list[dict]:
    """Coerce an OpenAI-style tool-call list/object into parse_turn's shape."""
    items = obj if isinstance(obj, list) else [obj]
    calls: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        fn = item.get("function", item)  # OpenAI shape: {"function": {name, arguments}}
        name = fn.get("name")
        if not name:
            continue
        arguments = fn.get("arguments")
        calls.append({"name": name, "arguments": arguments if arguments is not None else "{}"})
    return calls


def _model_turn(output: Any, context: dict):
    """Build a ParsedTurn from promptfoo output / providerResponse.

    promptfoo may surface tool calls as the `output` (a list of tool-call objects
    or a JSON string of them) or as text content when the model declines to call
    a tool. Be liberal in what we accept.
    """
    candidate = output

    # providerResponse sometimes carries the structured tool calls explicitly.
    provider_response = context.get("providerResponse") or {}
    for key in ("tool_calls", "toolCalls"):
        if isinstance(provider_response.get(key), list):
            candidate = provider_response[key]
            break

    if isinstance(candidate, str):
        stripped = candidate.strip()
        if stripped[:1] in ("[", "{"):
            try:
                candidate = json.loads(stripped)
            except ValueError:
                return parse_turn(content=candidate, native_tool_calls=None, raw_text=candidate)
        else:
            # Plain prose (e.g. a clarifying question) -> no tool call.
            return parse_turn(content=candidate, native_tool_calls=None, raw_text=candidate)

    native = _normalize_calls(candidate)
    if native:
        return parse_turn(content=None, native_tool_calls=native, raw_text="")
    text = output if isinstance(output, str) else ""
    return parse_turn(content=text or None, native_tool_calls=None, raw_text=text)


def get_assert(output: Any, context: dict) -> dict:
    metadata = context["test"]["metadata"]
    case = case_from_metadata(metadata)
    turn = _model_turn(output, context)
    score = score_case(case, turn)
    return {
        "pass": bool(score),
        "score": float(score),
        "reason": (
            f"{'match' if score else 'mismatch'}: model made {len(turn.tool_calls)} call(s), "
            f"expected {len(case.expected_calls)}"
        ),
    }
