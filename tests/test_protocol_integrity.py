from pathlib import Path

from wallet_evals.promptfoo import load_cases
from wallet_evals.schema import ParsedTurn
from wallet_evals.scorer import score_case

PROTOCOLS = Path(__file__).resolve().parents[1] / "pf" / "tests.protocols.yaml"


def _load():
    return load_cases(PROTOCOLS)


def test_nonempty_and_unique_ids():
    cases = _load()
    assert len(cases) > 0
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))


def test_has_add_and_remove():
    cats = {c.category for c in _load()}
    assert "safe-add-signer" in cats and "safe-remove-signer" in cats


def test_every_protocol_gold_self_scores_one():
    for case in _load():
        turn = ParsedTurn(tool_calls=[c.as_parsed_call() for c in case.expected_calls])
        assert score_case(case, turn) == 1, f"{case.id} not self-consistent"


def test_has_aave_categories():
    cats = {c.category for c in _load()}
    assert {"aave-supply", "aave-withdraw", "aave-borrow", "aave-repay"} <= cats


def test_has_all_protocols():
    protos = {c.protocol for c in _load()}
    assert {"safe", "aave", "railgun"} <= protos


def test_has_railgun_categories():
    cats = {c.category for c in _load()}
    assert {"railgun-shield", "railgun-unshield"} <= cats
    assert any(c.startswith("safety-refusal-unshield") for c in cats)


def test_railgun_cases_are_the_only_human_unit_golds():
    """`amount` is the app-mirrored human-unit field: nothing else may carry it."""
    for case in _load():
        for call in case.expected_calls:
            if call.amount is None:
                continue
            assert case.protocol == "railgun" and call.tool in ("shield", "unshield"), \
                f"{case.id} sets amount on a non-privacy call"
