from pathlib import Path

from wallet_evals.promptfoo import load_cases
from wallet_evals.schema import ParsedToolCall, ParsedTurn
from wallet_evals.scorer import score_case

GENERATED = Path(__file__).resolve().parents[1] / "pf" / "tests.generated.yaml"


def _load() -> list:
    return load_cases(GENERATED)


def test_generated_nonempty():
    assert len(_load()) > 0


def test_generated_ids_unique():
    ids = [c.id for c in _load()]
    assert len(ids) == len(set(ids))


def test_generated_self_scores_one():
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


def test_generated_has_negatives_and_multiturn():
    ids = [c.id for c in _load()]
    assert any("-neg-" in i for i in ids)
    assert any("-mt-" in i for i in ids)


def test_generated_has_safety_refusals():
    refusals = [c for c in _load() if "refusal" in c.id]
    assert refusals, "no safety-refusal cases generated"
    assert all(c.expected_calls == [] for c in refusals), "refusal gold must be no tool call"
