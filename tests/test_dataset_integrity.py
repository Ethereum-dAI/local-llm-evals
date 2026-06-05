from pathlib import Path

from wallet_evals.promptfoo import load_cases
from wallet_evals.schema import ParsedToolCall, ParsedTurn
from wallet_evals.scorer import score_case

TESTS = Path(__file__).resolve().parents[1] / "pf" / "tests.yaml"


def _load() -> list:
    return load_cases(TESTS)


def test_dataset_validates():
    assert len(_load()) > 0


def test_ids_unique():
    ids = [c.id for c in _load()]
    assert len(ids) == len(set(ids))


def test_every_case_self_scores_one():
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


def test_has_swap_cases():
    swaps = [c for c in _load() if any(call.tool == "swap" for call in c.expected_calls)]
    assert len(swaps) >= 1


def test_prompt_lists_every_lookup_token():
    """The system prompt's token table must stay in sync with datasets/lookup.json."""
    import json

    root = Path(__file__).resolve().parents[1]
    lookup = json.loads((root / "datasets" / "lookup.json").read_text())
    system = json.loads((root / "pf" / "prompt.json").read_text())[0]["content"]
    for symbol, token in lookup["tokens"].items():
        assert symbol in system, f"{symbol} missing from system prompt"
        assert f"{token['decimals']} decimals" in system, f"{symbol} decimals missing"
        if token["native"]:
            assert "0x0000000000000000000000000000000000000000" in system
        else:
            assert token["address"] in system, f"{symbol} address missing"
