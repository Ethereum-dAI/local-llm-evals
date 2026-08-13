"""Integrity of the app-contract dataset.

Mirrors test_generated_integrity.py, which guards the frozen base-unit dataset.
Both must pass: the scorer still has to handle executeTx gold for the protocol
datasets, and the app-contract gold has to self-score under the same scorer.
"""
from pathlib import Path
from typing import Callable

import pytest

from wallet_evals.promptfoo import load_cases
from wallet_evals.schema import Case, ExpectedCall, ParsedToolCall, ParsedTurn
from wallet_evals.scorer import score_case

TESTS = Path(__file__).resolve().parents[1] / "pf" / "tests.app-contract.yaml"


def _load():
    return load_cases(TESTS)


def test_app_contract_case_count_matches_the_frozen_dataset():
    frozen = load_cases(Path(__file__).resolve().parents[1] / "pf" / "tests.generated.yaml")
    assert len(_load()) == len(frozen) == 307


def test_app_contract_ids_unique():
    ids = [c.id for c in _load()]
    assert len(ids) == len(set(ids))


def test_app_contract_self_scores_one():
    # NOTE: this is a round-trip/schema check, NOT a fold check. as_parsed_call()
    # is a pure field-for-field copy, so every fold in the scorer sees the same
    # (tool, value) pair on both sides of the comparison — each becomes f(x) == f(x),
    # true regardless of whether the fold is correct. What this genuinely proves is
    # that every gold call in the dataset validates against ParsedToolCall's schema
    # and survives the round-trip (it would catch a builder emitting a tool name or
    # a field the schema rejects). For evidence the folds themselves score MEANING
    # rather than surface form, see test_app_contract_fold_* and the negative
    # control test_app_contract_fold_rejects_an_altered_amount below.
    for case in _load():
        turn = ParsedTurn(tool_calls=[c.as_parsed_call() for c in case.expected_calls])
        assert score_case(case, turn) == 1, f"{case.id} not self-consistent"


def _find_single_call_case(
    cases: list[Case], predicate: Callable[[ExpectedCall], bool]
) -> tuple[Case, ExpectedCall] | None:
    """First case with exactly one expected call matching predicate, or None.

    Restricted to single-call cases so the caller can build a one-element
    ParsedTurn without having to fabricate the other calls in the sequence.
    """
    for case in cases:
        if len(case.expected_calls) != 1:
            continue
        call = case.expected_calls[0]
        if predicate(call):
            return case, call
    return None


def _score_actual(case: Case, actual: ParsedToolCall) -> int:
    return score_case(case, ParsedTurn(tool_calls=[actual]))


def test_app_contract_fold_recipient_is_case_insensitive():
    # Real proof the `_recipient_text` fold folds case for ENS names, not just a
    # tautological copy: gold names an ENS recipient, the "model" emits the same
    # name in a different case, and it must still score 1.
    found = _find_single_call_case(
        _load(),
        lambda c: c.tool == "transfer" and c.to is not None
        and not c.to.startswith("0x") and "." in c.to,
    )
    if found is None:
        pytest.skip("no transfer case with an ENS-style recipient in the dataset")
    case, call = found
    actual = call.as_parsed_call().model_copy(update={"to": call.to.swapcase()})
    assert actual.to != call.to, "swapcase() should actually change an ENS name's case"
    assert _score_actual(case, actual) == 1


def test_app_contract_fold_token_symbol_is_case_insensitive():
    # Real proof `_symbol` folds case: gold names a non-ETH token, the "model"
    # emits it in the opposite case, and it must still score 1.
    found = _find_single_call_case(
        _load(),
        lambda c: c.tool == "transfer" and c.token is not None and c.token.lower() != "eth",
    )
    if found is None:
        pytest.skip("no transfer case with a non-ETH token in the dataset")
    case, call = found
    actual = call.as_parsed_call().model_copy(update={"token": call.token.swapcase()})
    assert actual.token != call.token, "swapcase() should actually change the token's case"
    assert _score_actual(case, actual) == 1


def test_app_contract_fold_amount_is_numeric_not_textual():
    # Real proof `_human_amount`/`_dec_or_raw` compares numeric value, not text:
    # gold has a fractional amount, the "model" emits the same value with an
    # extra trailing zero, and it must still score 1.
    found = _find_single_call_case(
        _load(),
        lambda c: c.tool == "transfer" and c.amount is not None and "." in c.amount,
    )
    if found is None:
        pytest.skip("no transfer case with a fractional amount in the dataset")
    case, call = found
    padded = call.amount + "0"
    actual = call.as_parsed_call().model_copy(update={"amount": padded})
    assert actual.amount != call.amount, "padding should actually change the amount text"
    assert _score_actual(case, actual) == 1


def test_app_contract_fold_missing_token_defaults_to_eth():
    # Real proof `_symbol` treats an omitted token as ETH: gold names ETH
    # explicitly, the "model" omits `token` entirely, and it must still score 1.
    found = _find_single_call_case(
        _load(),
        lambda c: c.tool == "transfer" and c.token is not None and c.token.lower() == "eth",
    )
    if found is None:
        pytest.skip("no transfer case with an explicit ETH token in the dataset")
    case, call = found
    actual = call.as_parsed_call().model_copy(update={"token": None})
    assert _score_actual(case, actual) == 1


def test_app_contract_fold_missing_amount_side_defaults_to_input():
    # Real proof `_swap_side` treats an omitted amount_side as "input": gold sets
    # it explicitly, the "model" omits it entirely, and it must still score 1.
    found = _find_single_call_case(
        _load(), lambda c: c.tool == "swap" and c.amount_side is not None
    )
    if found is None:
        pytest.skip("no swap case with an explicit amount_side in the dataset")
    case, call = found
    actual = call.as_parsed_call().model_copy(update={"amount_side": None})
    assert _score_actual(case, actual) == 1


def test_app_contract_fold_rejects_an_altered_amount():
    # NEGATIVE CONTROL — the one that proves the fold tests above can actually
    # fail. Take a real transfer amount and insert thousands separators (a change
    # a human would call cosmetic but that is NOT the same numeric value under
    # `_dec_or_raw`'s Decimal parse: a comma makes Decimal(...) raise, so the fold
    # falls back to comparing the raw, unequal strings). If this ever scores 1,
    # the fold tests above are not testing anything.
    found = _find_single_call_case(
        _load(),
        lambda c: c.tool == "transfer" and c.amount is not None
        and int(c.amount.split(".")[0]) >= 1000,
    )
    if found is None:
        pytest.skip("no transfer case with a >=4-digit amount in the dataset")
    case, call = found
    int_part, _, frac_part = call.amount.partition(".")
    grouped = f"{int(int_part):,}"
    altered = f"{grouped}.{frac_part}" if frac_part else grouped
    assert altered != call.amount, "grouping should actually change the amount text"
    actual = call.as_parsed_call().model_copy(update={"amount": altered})
    assert _score_actual(case, actual) == 0


def test_app_contract_emits_no_base_unit_gold():
    for case in _load():
        for call in case.expected_calls:
            assert call.tool in ("transfer", "swap"), f"{case.id} uses {call.tool}"
            assert call.value == "0" and call.args == []
            assert call.currencyIn is None and call.amountIn is None


def test_app_contract_has_negatives_multiturn_and_refusals():
    cases = _load()
    assert any(not c.expected_calls for c in cases)
    assert any(c.category.startswith("multiturn-") for c in cases)
    assert any(c.category.startswith("safety-refusal-") for c in cases)
