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
    assert {"safe", "aave"} <= protos


def test_railgun_is_gone():
    """RAILGUN was removed from the app (local-wallet-mac#86, PR #87), so the
    protocol dataset must not resurrect shield/unshield."""
    cases = _load()
    assert not any(c.protocol == "railgun" for c in cases)
    assert not any("shield" in c.category for c in cases)


def test_no_protocol_case_carries_a_human_unit_amount():
    """The protocol datasets are base-unit executeTx only; `amount` is the
    app-mirrored human-unit field and belonged to the removed privacy tools."""
    for case in _load():
        for call in case.expected_calls:
            assert call.amount is None, \
                f"{case.id} sets a human-unit amount on a base-unit protocol call"
