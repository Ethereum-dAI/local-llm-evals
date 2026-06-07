from wallet_evals.schema import Case, ExpectedCall, ParsedToolCall, ParsedTurn
from wallet_evals.scorer import score_case
from wallet_evals.parsing import parse_turn


def _swap_case(**overrides):
    base = dict(tool="swap", chainId="1",
                currencyIn="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                currencyOut="0x6B175474E89094C44Da98b954EedeAC495271d0F",
                amountIn="100000000", amountOutMinimum="0", recipient="SELF")
    base.update(overrides)
    return Case(id="s", user_message="swap", level="payload", language="english",
                category="truePositiveSwap", query_type="one_shot", protocol="uniswap",
                difficulty="medium", requires=[], expected_calls=[base], notes=None)


def test_swap_exact_match_scores_one():
    case = _swap_case()
    turn = ParsedTurn(tool_calls=[ParsedToolCall(
        name="swap", chainId="1",
        currencyIn="0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # lowercased
        currencyOut="0x6B175474E89094C44Da98b954EedeAC495271d0F",
        amountIn="100000000", amountOutMinimum="0", recipient="SELF")])
    assert score_case(case, turn) == 1


def test_swap_wrong_amount_scores_zero():
    case = _swap_case()
    turn = ParsedTurn(tool_calls=[ParsedToolCall(
        name="swap", chainId="1",
        currencyIn="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        currencyOut="0x6B175474E89094C44Da98b954EedeAC495271d0F",
        amountIn="999", amountOutMinimum="0", recipient="SELF")])
    assert score_case(case, turn) == 0


def test_swap_wrong_currency_scores_zero():
    case = _swap_case()
    turn = ParsedTurn(tool_calls=[ParsedToolCall(
        name="swap", chainId="1",
        currencyIn="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        currencyOut="0x0000000000000000000000000000000000000000",  # wrong out
        amountIn="100000000", amountOutMinimum="0", recipient="SELF")])
    assert score_case(case, turn) == 0


def test_swap_parsed_from_native_tool_call():
    native = [{"name": "swap", "arguments": '{"chainId":"1","currencyIn":"0xAAA","currencyOut":"0xBBB","amountIn":"100","amountOutMinimum":"0","recipient":"SELF"}'}]
    turn = parse_turn(content=None, native_tool_calls=native, raw_text="")
    call = turn.tool_calls[0]
    assert call.name == "swap"
    assert call.currencyIn == "0xAAA"
    assert call.currencyOut == "0xBBB"
    assert call.amountIn == "100"


def test_executeTx_unaffected_by_swap_fields():
    case = Case(id="t", user_message="x", level="payload", language="english",
                category="truePositiveTransfer", query_type="one_shot", protocol="transfer",
                difficulty="easy", requires=[],
                expected_calls=[{"tool": "executeTx", "chainId": "1", "to": "0xabc", "value": "100",
                                 "function": None, "args": []}], notes=None)
    turn = ParsedTurn(tool_calls=[ParsedToolCall(name="executeTx", chainId="1", to="0xabc", value="100", function=None, args=[])])
    assert score_case(case, turn) == 1
