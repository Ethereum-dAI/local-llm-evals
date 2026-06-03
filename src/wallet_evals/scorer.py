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
    return _norm(expected.args) == _norm(actual.args)


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
