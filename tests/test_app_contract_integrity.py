"""Integrity of the app-contract dataset.

Mirrors test_generated_integrity.py, which guards the frozen base-unit dataset.
Both must pass: the scorer still has to handle executeTx gold for the protocol
datasets, and the app-contract gold has to self-score under the same scorer.
"""
from pathlib import Path

from wallet_evals.promptfoo import load_cases
from wallet_evals.schema import ParsedTurn
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
    # The load-bearing assertion: rebuild every gold call as if a model had emitted
    # it and require it to score 1. This is what proves the Task 1 builders and the
    # Task 2 scorer folds agree.
    for case in _load():
        turn = ParsedTurn(tool_calls=[c.as_parsed_call() for c in case.expected_calls])
        assert score_case(case, turn) == 1, f"{case.id} not self-consistent"


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
