"""Deterministic binary scorer (spec §4).

A case scores 1 iff the parsed call sequence matches expected_calls exactly:
same length, same order, each envelope field equal, args equal element-wise.
The only normalization is lowercasing 0x-addresses; all else is exact string
compare (the dataset is already on-chain canonical).
"""
from __future__ import annotations

import re
from typing import Any

from wallet_evals.schema import Case, ExpectedCall, ParsedToolCall, ParsedTurn

# Lowercase any 0x-prefixed string so checksummed and lowercased address/bytes
# forms compare equal. Non-0x values (decimal amounts, function signatures) are
# left untouched and compared exactly.
_ADDRESS_RE = re.compile(r"^0x", re.IGNORECASE)


def _norm_scalar(value: Any) -> Any:
    if isinstance(value, str) and _ADDRESS_RE.match(value):
        return value.lower()
    return value


def _norm(value: Any) -> Any:
    if isinstance(value, list):
        return [_norm(v) for v in value]
    return _norm_scalar(value)


def _value_or_zero(v: str | None) -> str:
    return "0" if v is None else v


def _swap_recipient(name: str, v: str | None) -> str | None:
    """A swap's recipient defaults to the user's own wallet ("<wallet>").
    Omitting it is correct, so treat a missing recipient as that default."""
    return "<wallet>" if (name == "swap" and v is None) else v


def _swap_min_out(name: str, v: str | None) -> str | None:
    """A swap's amountOutMinimum defaults to "0" when unspecified."""
    return "0" if (name == "swap" and v is None) else v


def _call_matches(expected: ExpectedCall, actual: ParsedToolCall) -> bool:
    if expected.tool != actual.name:
        return False
    if expected.chainId != actual.chainId:
        return False
    if _norm_scalar(expected.to) != _norm_scalar(actual.to):
        return False
    if _value_or_zero(expected.value) != _value_or_zero(actual.value):
        return False
    if expected.function != actual.function:
        return False
    if _norm(expected.args) != _norm(actual.args):
        return False
    # Swap intent fields (None on both sides for executeTx/readTx).
    if _norm_scalar(expected.currencyIn) != _norm_scalar(actual.currencyIn):
        return False
    if _norm_scalar(expected.currencyOut) != _norm_scalar(actual.currencyOut):
        return False
    if _norm_scalar(_swap_recipient(expected.tool, expected.recipient)) != \
            _norm_scalar(_swap_recipient(actual.name, actual.recipient)):
        return False
    if expected.amountIn != actual.amountIn:
        return False
    if _swap_min_out(expected.tool, expected.amountOutMinimum) != \
            _swap_min_out(actual.name, actual.amountOutMinimum):
        return False
    return True


def score_case(case: Case, turn: ParsedTurn) -> int:
    """Return 1 if the model's turn matches the case's expected_calls, else 0."""
    expected = case.expected_calls
    actual = turn.tool_calls
    if len(expected) != len(actual):
        return 0
    for exp, act in zip(expected, actual):
        if not _call_matches(exp, act):
            return 0
    return 1


# Field-level comparison (label, expected-value, actual-value), in the same order
# _call_matches checks them. `to`/currencies/recipient compare under address norm.
_CALL_FIELDS = (
    ("tool", lambda e: e.tool, lambda a: a.name, lambda x: x),
    ("chainId", lambda e: e.chainId, lambda a: a.chainId, lambda x: x),
    ("to", lambda e: e.to, lambda a: a.to, _norm_scalar),
    ("value", lambda e: e.value, lambda a: a.value, lambda x: _value_or_zero(x)),
    ("function", lambda e: e.function, lambda a: a.function, lambda x: x),
    ("args", lambda e: e.args, lambda a: a.args, _norm),
    ("currencyIn", lambda e: e.currencyIn, lambda a: a.currencyIn, _norm_scalar),
    ("currencyOut", lambda e: e.currencyOut, lambda a: a.currencyOut, _norm_scalar),
    ("recipient", lambda e: _swap_recipient(e.tool, e.recipient),
     lambda a: _swap_recipient(a.name, a.recipient), _norm_scalar),
    ("amountIn", lambda e: e.amountIn, lambda a: a.amountIn, lambda x: x),
    ("amountOutMinimum", lambda e: _swap_min_out(e.tool, e.amountOutMinimum),
     lambda a: _swap_min_out(a.name, a.amountOutMinimum), lambda x: x),
)


def _call_field_diffs(expected: ExpectedCall, actual: ParsedToolCall) -> list[str]:
    diffs = []
    for label, get_e, get_a, norm in _CALL_FIELDS:
        ev, av = get_e(expected), get_a(actual)
        if norm(ev) != norm(av):
            diffs.append(f"{label}: expected {ev!r} got {av!r}")
    return diffs


def explain_mismatch(case: Case, turn: ParsedTurn) -> str:
    """Human-readable reason a case passed/failed — for the promptfoo assertion."""
    expected = case.expected_calls
    actual = turn.tool_calls
    if score_case(case, turn) == 1:
        return "match"
    if len(expected) != len(actual):
        exp_tools = [c.tool for c in expected] or ["(no call)"]
        act_tools = [c.name for c in actual] or ["(no call)"]
        return f"call count: expected {len(expected)} {exp_tools}, model made {len(actual)} {act_tools}"
    parts = []
    for i, (exp, act) in enumerate(zip(expected, actual), start=1):
        diffs = _call_field_diffs(exp, act)
        if diffs:
            parts.append(f"call#{i} ({exp.tool}): " + "; ".join(diffs))
    return " | ".join(parts) if parts else "mismatch"
