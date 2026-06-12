from pathlib import Path

from wallet_evals.promptfoo import load_cases
from wallet_evals.schema import ParsedToolCall, ParsedTurn
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
        turn = ParsedTurn(tool_calls=[
            ParsedToolCall(
                name=c.tool, chainId=c.chainId, to=c.to, value=c.value,
                function=c.function, args=c.args,
                currencyIn=c.currencyIn, currencyOut=c.currencyOut,
                amountIn=c.amountIn, amountOutMinimum=c.amountOutMinimum,
                recipient=c.recipient,
            )
            for c in case.expected_calls
        ])
        assert score_case(case, turn) == 1, f"{case.id} not self-consistent"
