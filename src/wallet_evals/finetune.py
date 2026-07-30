"""Encode eval-style cases into FunctionGemma fine-tuning examples.

The eval harness stores gold as a promptfoo `metadata.expected_calls` list (key
`tool`, flat envelope fields). FunctionGemma is trained on chat conversations
whose assistant turn emits the tool call in its plain-text DSL:

    <start_function_call>call:NAME{key:<escape>value<escape>,...,args:[...]}<end_function_call>

This module is the encoder that turns the former into the latter — the mirror of
`wallet_evals.gemma_dsl.parse_gemma_tool_calls` (which only decodes). A training
example is a dict `{id, category, protocol, messages, tools, expected_calls}`:

- `messages` is the exact conversation the provider feeds at inference — our
  `pf.prompt.render` output with `system` roles remapped to `developer` (the role
  FunctionGemma needs to activate function calling, see functiongemma.decode_prompt)
  — plus a final assistant turn holding the target.
- The assistant target is the DSL call for cases with a gold call, a clarifying
  question for ablation (missing-field) cases, or a safety warning for refusals.
- `expected_calls` is carried through so the target can be validated offline: it
  must decode+score back to 1.0 through the unchanged scorer (see
  tests/test_finetune_integrity.py).
"""
from __future__ import annotations

import json
from typing import Any

from wallet_evals.gemma_dsl import FUNCTIONGEMMA, Dialect
from wallet_evals.generation import CLARIFICATIONS

# Canonical field order per tool (readability only; the scorer is order-free).
_EXECUTE_ORDER = ["chainId", "to", "value", "function", "args"]
_SWAP_ORDER = ["chainId", "currencyIn", "currencyOut", "amountIn",
               "amountOutMinimum", "recipient"]

# Assistant text for complete-but-dangerous requests (gold = no tool call). Keyed
# by the generator's `safety-refusal-<kind>` category. These must never contain a
# DSL opener, so the decoder finds no call and the case scores as a refusal.
REFUSAL_MESSAGES = {
    "burn-send": (
        "I won't do that. That address is a burn address — funds sent there are "
        "permanently unrecoverable. Tell me a valid recipient and I'll send it."
    ),
    "zero-send": (
        "I won't send to the zero address — those funds would be destroyed. "
        "Please give me a real recipient address or ENS name."
    ),
    "approve-unknown-spender": (
        "I won't approve token spending for an unknown address. Granting an "
        "allowance to an unrecognised spender risks draining your tokens."
    ),
    "unverified-token-swap": (
        "I won't swap into a token given only as a raw contract address I can't "
        "verify. Give me a known token symbol and I'll proceed."
    ),
}
_REFUSAL_FALLBACK = (
    "I can't do that safely. Please double-check the request and provide valid, "
    "recognised details."
)


def to_developer_roles(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy `messages`, remapping `system` -> `developer` (FunctionGemma's role
    for function-calling). Mirrors functiongemma.decode_prompt so the training
    conversation matches what the provider feeds at inference."""
    out: list[dict[str, Any]] = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "system":
            m["role"] = "developer"
        out.append(m)
    return out


def encode_gemma_call(call: dict[str, Any], dialect: Dialect = FUNCTIONGEMMA) -> str:
    """Serialize one gold call (a `metadata.expected_calls` entry) to the DSL.

    Scalars are wrapped in the dialect quote; `args` is emitted as a bare JSON
    array (the form the decoder's bracket-matcher expects); None-valued fields
    (e.g. `function` for a native transfer) are omitted."""
    call = dict(call)
    name = call.pop("tool", None) or call.pop("name")
    order = _SWAP_ORDER if name == "swap" else _EXECUTE_ORDER
    keys = [k for k in order if k in call] + [k for k in call if k not in order]
    parts: list[str] = []
    for key in keys:
        value = call[key]
        if value is None:
            continue
        if key == "args":
            parts.append(f"args:{json.dumps(value, separators=(',', ':'))}")
        else:
            parts.append(f"{key}:{dialect.quote}{value}{dialect.quote}")
    body = ",".join(parts)
    return f"{dialect.opener}call:{name}{{{body}}}{dialect.closer}"


def encode_calls(calls: list[dict[str, Any]], dialect: Dialect = FUNCTIONGEMMA) -> str:
    """Concatenate encoded calls (cases here have exactly one)."""
    return "".join(encode_gemma_call(c, dialect) for c in calls)


def refusal_message(category: str) -> str:
    """The safety-warning target for a `safety-refusal-<kind>` category."""
    kind = category.replace("safety-refusal-", "", 1)
    return REFUSAL_MESSAGES.get(kind, _REFUSAL_FALLBACK)


def assistant_target(metadata: dict[str, Any], *, reasoning_text: str | None = None,
                     dialect: Dialect = FUNCTIONGEMMA) -> str:
    """The assistant turn content for a case.

    - gold call present -> the DSL call (optionally preceded by a <think> block);
    - ablation (missing-field) case -> the canned clarifying question;
    - safety refusal -> a warning; no tool call either way.
    """
    calls = metadata.get("expected_calls") or []
    if calls:
        dsl = encode_calls(calls, dialect)
        if reasoning_text:
            return f"<think>{reasoning_text}</think>\n{dsl}"
        return dsl
    category = metadata.get("category", "")
    if category.startswith("ablation-"):
        field = category.split("-", 1)[1]
        return CLARIFICATIONS.get(field, "Could you clarify the missing detail?")
    return refusal_message(category)


def case_to_example(metadata: dict[str, Any], rendered_messages: list[dict[str, Any]],
                    tools: list[dict], *, reasoning_text: str | None = None,
                    dialect: Dialect = FUNCTIONGEMMA,
                    to_developer: bool = True) -> dict[str, Any]:
    """Build one training example from a case's metadata + rendered prompt.

    `rendered_messages` is `pf.prompt.render(context)` (system + optional protocol
    context + the user/assistant turns); we (optionally) remap roles and append
    the target.

    `to_developer` remaps `system` -> `developer` for FunctionGemma (the role it
    needs to activate function calling). Gemma-4's chat template has a native
    `<|turn>system`, so its dataset passes `to_developer=False` to keep `system`.
    """
    messages = to_developer_roles(rendered_messages) if to_developer else \
        [dict(m) for m in rendered_messages]
    messages.append({"role": "assistant",
                     "content": assistant_target(metadata, reasoning_text=reasoning_text,
                                                  dialect=dialect)})
    return {
        "id": metadata["id"],
        "category": metadata.get("category"),
        "protocol": metadata.get("protocol"),
        "messages": messages,
        "tools": tools,
        "expected_calls": metadata.get("expected_calls") or [],
    }
