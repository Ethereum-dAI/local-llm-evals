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


def test_rendered_prompt_lists_every_lookup_token():
    """The rendered system message must carry the datasets/lookup.json token table."""
    import json

    from wallet_evals.prompt import build_messages

    root = Path(__file__).resolve().parents[1]
    lookup = json.loads((root / "datasets" / "lookup.json").read_text())
    messages = build_messages("Send 1 ETH to bob.eth")
    system = messages[0]["content"]
    for symbol, token in lookup["tokens"].items():
        assert symbol in system, f"{symbol} missing from system prompt"
        assert f"{token['decimals']} decimals" in system, f"{symbol} decimals missing"
        if token["native"]:
            assert "0x0000000000000000000000000000000000000000" in system
        else:
            assert token["address"] in system, f"{symbol} address missing"


def test_rendered_prompt_substitutes_user_message():
    from wallet_evals.prompt import build_messages

    messages = build_messages("Send 1 ETH to bob.eth")
    assert messages[1] == {"role": "user", "content": "Send 1 ETH to bob.eth"}
    assert "{{" not in messages[0]["content"]


def test_history_is_spliced_before_final_user_message():
    """Multi-turn: prior turns sit between the system message and the final user
    turn, which must remain last so the model answers it in context."""
    from wallet_evals.prompt import build_messages

    history = [
        {"role": "user", "content": "Send ETH to vitalik.eth"},
        {"role": "assistant", "content": "How much?"},
    ]
    messages = build_messages("0.1", history)
    assert messages[0]["role"] == "system"
    assert [m["role"] for m in messages[1:]] == ["user", "assistant", "user"]
    assert messages[1]["content"] == "Send ETH to vitalik.eth"
    assert messages[-1] == {"role": "user", "content": "0.1"}


def test_dataset_has_multi_turn_cases():
    multi = [c for c in _load() if c.query_type == "multi_turn"]
    assert len(multi) >= 1


def test_dataset_has_adversarial_cases():
    adv = [c for c in _load() if c.category == "adversarial"]
    assert len(adv) >= 3


def test_pure_adversarial_cases_expect_no_call():
    """Injection / scam / over-ask cases must have an empty gold so any fired
    call is scored as a false execution. (The confused-deputy case is the one
    adversarial exception — it carries a real gold call.)"""
    adv = [c for c in _load() if c.category == "adversarial"]
    no_call = [c for c in adv if not c.expected_calls]
    assert len(no_call) >= 3, "expected several no-call adversarial cases"
    deputy = [c for c in adv if c.expected_calls]
    for c in deputy:
        assert len(c.expected_calls) == 1, f"{c.id}: deputy case runs exactly one call"


def test_multi_turn_cases_carry_valid_history():
    """Every multi_turn test must define vars.history of {role, content} turns
    ending on an assistant turn (the final user_message follows it)."""
    import yaml

    tests = yaml.safe_load(TESTS.read_text())
    multi = [t for t in tests if t["metadata"].get("query_type") == "multi_turn"]
    assert multi, "expected at least one multi_turn case"
    for t in multi:
        history = t["vars"].get("history")
        assert history, f"{t['metadata']['id']} missing vars.history"
        for turn in history:
            assert turn["role"] in ("user", "assistant")
            assert turn["content"]
        assert history[-1]["role"] == "assistant", (
            f"{t['metadata']['id']} history should end on an assistant turn"
        )
