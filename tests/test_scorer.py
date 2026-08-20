import json
import pytest

from wallet_evals.intents import (
    to_base_units, resolve_recipient, swap_currency,
    build_transfer_call, build_swap_call, LOOKUP, CHAIN_ID,
)
from wallet_evals.parsing import parse_turn
from wallet_evals.schema import Case, Dataset, ExpectedCall, ParsedToolCall, ParsedTurn, PreviewContext, Previewable
from wallet_evals.schema import Case, ExpectedCall, ParsedToolCall, ParsedTurn
from wallet_evals.schema import Case, ParsedToolCall, ParsedTurn
from wallet_evals.scorer import score_case

# ============================================================================
# test_scorer
# ============================================================================

def _case(expected_calls):
    return Case(
        id="t", user_message="x", level="payload", language="english",
        category="truePositiveTransfer", query_type="one_shot", protocol="transfer",
        difficulty="easy", requires=[], expected_calls=expected_calls, notes=None,
    )


def test_exact_match_scores_one():
    case = _case([{"tool": "executeTx", "chainId": "1", "to": "0xABC", "value": "0",
                   "function": "transfer(address,uint256)", "args": ["0xDEF", "100"]}])
    turn = ParsedTurn(tool_calls=[ParsedToolCall(
        name="executeTx", chainId="1", to="0xabc", value="0",
        function="transfer(address,uint256)", args=["0xdef", "100"])])
    assert score_case(case, turn) == 1


def test_address_case_is_normalized():
    case = _case([{"tool": "executeTx", "chainId": "1", "to": "0xAbCdEf0000000000000000000000000000000001",
                   "value": "0", "function": None, "args": []}])
    turn = ParsedTurn(tool_calls=[ParsedToolCall(
        name="executeTx", chainId="1", to="0xabcdef0000000000000000000000000000000001",
        value="0", function=None, args=[])])
    assert score_case(case, turn) == 1


def test_wrong_arg_scores_zero():
    case = _case([{"tool": "executeTx", "chainId": "1", "to": "0xABC", "value": "0",
                   "function": "transfer(address,uint256)", "args": ["0xDEF", "100"]}])
    turn = ParsedTurn(tool_calls=[ParsedToolCall(
        name="executeTx", chainId="1", to="0xabc", value="0",
        function="transfer(address,uint256)", args=["0xdef", "999"])])
    assert score_case(case, turn) == 0


def test_sequence_length_mismatch_scores_zero():
    case = _case([
        {"tool": "executeTx", "chainId": "1", "to": "0x1", "value": "0", "function": "approve(address,uint256)", "args": ["0xr", "100"]},
        {"tool": "executeTx", "chainId": "1", "to": "0xr", "value": "0", "function": "swap()", "args": []},
    ])
    turn = ParsedTurn(tool_calls=[ParsedToolCall(
        name="executeTx", chainId="1", to="0x1", value="0", function="approve(address,uint256)", args=["0xr", "100"])])
    assert score_case(case, turn) == 0


def test_order_matters():
    case = _case([
        {"tool": "executeTx", "chainId": "1", "to": "0xA", "value": "0", "function": "approve(address,uint256)", "args": []},
        {"tool": "executeTx", "chainId": "1", "to": "0xB", "value": "0", "function": "swap()", "args": []},
    ])
    turn = ParsedTurn(tool_calls=[
        ParsedToolCall(name="executeTx", chainId="1", to="0xB", value="0", function="swap()", args=[]),
        ParsedToolCall(name="executeTx", chainId="1", to="0xA", value="0", function="approve(address,uint256)", args=[]),
    ])
    assert score_case(case, turn) == 0


def test_empty_expected_passes_when_no_call_made():
    case = _case([])
    assert score_case(case, ParsedTurn(content="need more info", tool_calls=[])) == 1


def test_empty_expected_fails_when_call_made():
    case = _case([])
    turn = ParsedTurn(tool_calls=[ParsedToolCall(name="executeTx", chainId="1", to="0x1", args=[])])
    assert score_case(case, turn) == 0


def test_nested_tuple_args_compared_recursively():
    case = _case([{"tool": "executeTx", "chainId": "1", "to": "0xR", "value": "0",
                   "function": "exactInputSingle((address,address,uint24))",
                   "args": [["0xAAA", "0xBBB", "3000"]]}])
    turn = ParsedTurn(tool_calls=[ParsedToolCall(
        name="executeTx", chainId="1", to="0xr", value="0",
        function="exactInputSingle((address,address,uint24))", args=[["0xaaa", "0xbbb", "3000"]])])
    assert score_case(case, turn) == 1


def _swap_case():
    return _case([{"tool": "swap", "chainId": "1",
                   "currencyIn": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                   "currencyOut": "0x0000000000000000000000000000000000000000",
                   "amountIn": "100000000", "amountOutMinimum": "0",
                   "recipient": "<wallet>"}])


def test_swap_omitted_recipient_defaults_to_wallet():
    # A model that omits recipient/amountOutMinimum is accepting the documented
    # defaults (own wallet, 0) and must still score 1.
    case = _swap_case()
    turn = ParsedTurn(tool_calls=[ParsedToolCall(
        name="swap", chainId="1",
        currencyIn="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        currencyOut="0x0000000000000000000000000000000000000000",
        amountIn="100000000", amountOutMinimum=None, recipient=None)])
    assert score_case(case, turn) == 1


def test_swap_wrong_explicit_recipient_still_fails():
    case = _swap_case()
    turn = ParsedTurn(tool_calls=[ParsedToolCall(
        name="swap", chainId="1",
        currencyIn="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        currencyOut="0x0000000000000000000000000000000000000000",
        amountIn="100000000", amountOutMinimum="0",
        recipient="0x1111111111111111111111111111111111111111")])
    assert score_case(case, turn) == 0


def _transfer_case(token="ETH", to="vitalik.eth", amount="0.1"):
    return _case([{"tool": "transfer", "chainId": "1", "to": to,
                   "amount": amount, "token": token}])


def _transfer_turn(**kwargs):
    fields = {"name": "transfer", "chainId": "1", "to": "vitalik.eth",
              "amount": "0.1", "token": "ETH"}
    fields.update(kwargs)
    return ParsedTurn(tool_calls=[ParsedToolCall(**fields)])


def test_transfer_exact_match_scores_one():
    assert score_case(_transfer_case(), _transfer_turn()) == 1


def test_transfer_ens_case_is_folded():
    # mutate_case sends the surface as "ViTALik.eTh"; ENS is case-insensitive, so
    # copying the mangled form back is a formatting difference, not a capability gap.
    assert score_case(_transfer_case(), _transfer_turn(to="ViTALik.eTh")) == 1


def test_transfer_token_symbol_case_is_folded():
    assert score_case(_transfer_case(token="DAI"),
                      _transfer_turn(token="dAI")) == 1


def test_transfer_trailing_zero_amount_is_folded():
    # "0.010" and "0.01" are the same transfer and shift to identical base units.
    assert score_case(_transfer_case(amount="0.01"), _transfer_turn(amount="0.010")) == 1


def test_transfer_thousands_separator_fails():
    # NOT normalized on purpose: EtherAmountParser.units in the macOS app rejects
    # any non-digit character, so the grouped form genuinely fails in the wallet.
    case = _transfer_case(amount="123456.789012")
    assert score_case(case, _transfer_turn(amount="123,456.789012")) == 0


def test_transfer_omitted_token_defaults_to_eth():
    # The app's transfer schema marks `token` optional ("Default to ETH if the user
    # does not name a token"), so omitting it on an ETH transfer is correct.
    assert score_case(_transfer_case(token="ETH"), _transfer_turn(token=None)) == 1


def test_transfer_omitted_token_fails_for_erc20():
    assert score_case(_transfer_case(token="USDC"), _transfer_turn(token=None)) == 0


def test_transfer_base_units_amount_fails():
    # The whole point of the contract change: wei is now wrong, not required.
    assert score_case(_transfer_case(),
                      _transfer_turn(amount="100000000000000000")) == 0


def _app_swap_case():
    return _case([{"tool": "swap", "chainId": "1", "amount": "100",
                   "from_token": "USDC", "to_token": "ETH",
                   "amount_side": "input"}])


def test_app_swap_exact_match_scores_one():
    turn = ParsedTurn(tool_calls=[ParsedToolCall(
        name="swap", chainId="1", amount="100", from_token="USDC",
        to_token="ETH", amount_side="input")])
    assert score_case(_app_swap_case(), turn) == 1


def test_app_swap_omitted_amount_side_defaults_to_input():
    # The app's schema pins amount_side to the single enum value "input", so
    # omitting it is accepting the only legal value.
    turn = ParsedTurn(tool_calls=[ParsedToolCall(
        name="swap", chainId="1", amount="100", from_token="USDC",
        to_token="ETH", amount_side=None)])
    assert score_case(_app_swap_case(), turn) == 1


def test_app_swap_missing_from_token_fails():
    turn = ParsedTurn(tool_calls=[ParsedToolCall(
        name="swap", chainId="1", amount="100", from_token=None,
        to_token="ETH", amount_side="input")])
    assert score_case(_app_swap_case(), turn) == 0


def test_app_swap_reversed_direction_fails():
    turn = ParsedTurn(tool_calls=[ParsedToolCall(
        name="swap", chainId="1", amount="100", from_token="ETH",
        to_token="USDC", amount_side="input")])
    assert score_case(_app_swap_case(), turn) == 0


def test_app_swap_stray_token_is_ignored():
    # swap's schema has no `token` field (it names sides from_token/to_token
    # instead), so a stray one is noise the app would drop on decode, not a miss.
    turn = ParsedTurn(tool_calls=[ParsedToolCall(
        name="swap", chainId="1", amount="100", from_token="USDC",
        to_token="ETH", amount_side="input", token="USDC")])
    assert score_case(_app_swap_case(), turn) == 1


def test_transfer_stray_from_token_is_ignored():
    # transfer's schema has no from_token/to_token field (that's swap-only), so a
    # stray one is noise the app would drop on decode, not a miss.
    assert score_case(_transfer_case(), _transfer_turn(from_token="ETH")) == 1

# ============================================================================
# test_schema
# ============================================================================

def test_case_parses_payload_with_nested_args():
    raw = {
        "id": "uniswap-swap-001",
        "user_message": "swap 10 USDC for DAI on Uniswap",
        "level": "payload",
        "language": "english",
        "category": "truePositiveSwap",
        "query_type": "one_shot",
        "protocol": "uniswap",
        "difficulty": "medium",
        "requires": ["multi_step"],
        "expected_calls": [
            {
                "tool": "executeTx",
                "chainId": "1",
                "to": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                "value": "0",
                "function": "approve(address,uint256)",
                "args": ["0xE592427A0AEce92De3Edee1F18E0157C05861564", "10000000"],
            }
        ],
        "notes": None,
    }
    case = Case.model_validate(raw)
    assert case.id == "uniswap-swap-001"
    assert len(case.expected_calls) == 1
    assert case.expected_calls[0].function == "approve(address,uint256)"
    assert case.expected_calls[0].args == ["0xE592427A0AEce92De3Edee1F18E0157C05861564", "10000000"]


def test_case_empty_expected_calls_means_no_call():
    raw = {
        "id": "ambiguous-001",
        "user_message": "Send ETH to bob.eth",
        "level": "intent",
        "language": "english",
        "category": "ambiguous",
        "query_type": None,
        "protocol": "transfer",
        "difficulty": "easy",
        "requires": [],
        "expected_calls": [],
        "notes": None,
    }
    case = Case.model_validate(raw)
    assert case.expected_calls == []


def test_native_transfer_call_has_null_function():
    call = ExpectedCall.model_validate(
        {"tool": "executeTx", "chainId": "1", "to": "0xabc", "value": "100", "function": None, "args": []}
    )
    assert call.function is None
    assert call.args == []


def test_parsed_turn_round_trips():
    turn = ParsedTurn(
        content=None,
        tool_calls=[ParsedToolCall(name="readTx", chainId="1", to="0xabc", value=None, function="balanceOf(address)", args=["0xdef"])],
    )
    assert turn.tool_calls[0].name == "readTx"


def test_format_preview_renders_case_and_calls():
    case = Case.model_validate(
        {
            "id": "swap-001",
            "user_message": "swap 10 USDC for DAI",
            "level": "payload",
            "language": "english",
            "category": "truePositiveSwap",
            "protocol": "uniswap",
            "difficulty": "medium",
            "expected_calls": [
                {
                    "tool": "executeTx",
                    "chainId": "1",
                    "to": "0xToken",
                    "value": "0",
                    "function": "approve(address,uint256)",
                    "args": ["0xRouter", "10000000"],
                }
            ],
        }
    )
    text = case.format_preview()
    assert "[swap-001]  level=payload  protocol=uniswap" in text
    assert "user: swap 10 USDC for DAI" in text
    assert "expected call #1: executeTx -> 0xToken" in text
    assert "args=['0xRouter', '10000000']" in text


def test_format_preview_renders_no_call():
    case = Case.model_validate(
        {
            "id": "info-001",
            "user_message": "What is ETH?",
            "level": "intent",
            "language": "english",
            "category": "informational",
            "protocol": "general",
            "difficulty": "easy",
            "expected_calls": [],
        }
    )
    assert "expected: (no tool call)" in case.format_preview()


def test_dataset_format_preview_includes_source_and_cases():
    dataset = Dataset.model_validate({"cases": [{"id": "a", "user_message": "hi", "level": "intent", "language": "english", "category": "x", "protocol": "y", "difficulty": "easy"}]})
    text = dataset.format_preview(PreviewContext(source="datasets/tiny.json"))
    assert "DATASET: datasets/tiny.json  (1 cases)" in text
    assert "[a]" in text


def test_previewable_implementations():
    call = ExpectedCall.model_validate({"tool": "executeTx", "chainId": "1", "to": "0xabc", "args": []})
    case = Case.model_validate(
        {
            "id": "x",
            "user_message": "hi",
            "level": "intent",
            "language": "english",
            "category": "x",
            "protocol": "y",
            "difficulty": "easy",
            "expected_calls": [{"tool": "executeTx", "chainId": "1", "to": "0xabc", "args": []}],
        }
    )
    dataset = Dataset.model_validate({"cases": [case.model_dump()]})
    for previewable in (call, case, dataset):
        assert isinstance(previewable, Previewable)
        assert previewable.format_preview()

# ============================================================================
# test_swap
# ============================================================================
#
# Suffixed on merge, to clear a name this file shared with an
# earlier section: _swap_case -> _swap_case_swap.

def _swap_case_swap(**overrides):
    base = dict(tool="swap", chainId="1",
                currencyIn="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                currencyOut="0x6B175474E89094C44Da98b954EedeAC495271d0F",
                amountIn="100000000", amountOutMinimum="0", recipient="<wallet>")
    base.update(overrides)
    return Case(id="s", user_message="swap", level="payload", language="english",
                category="truePositiveSwap", query_type="one_shot", protocol="uniswap",
                difficulty="medium", requires=[], expected_calls=[base], notes=None)


def test_swap_exact_match_scores_one():
    case = _swap_case_swap()
    turn = ParsedTurn(tool_calls=[ParsedToolCall(
        name="swap", chainId="1",
        currencyIn="0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # lowercased
        currencyOut="0x6B175474E89094C44Da98b954EedeAC495271d0F",
        amountIn="100000000", amountOutMinimum="0", recipient="<wallet>")])
    assert score_case(case, turn) == 1


def test_swap_wrong_amount_scores_zero():
    case = _swap_case_swap()
    turn = ParsedTurn(tool_calls=[ParsedToolCall(
        name="swap", chainId="1",
        currencyIn="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        currencyOut="0x6B175474E89094C44Da98b954EedeAC495271d0F",
        amountIn="999", amountOutMinimum="0", recipient="<wallet>")])
    assert score_case(case, turn) == 0


def test_swap_wrong_currency_scores_zero():
    case = _swap_case_swap()
    turn = ParsedTurn(tool_calls=[ParsedToolCall(
        name="swap", chainId="1",
        currencyIn="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        currencyOut="0x0000000000000000000000000000000000000000",  # wrong out
        amountIn="100000000", amountOutMinimum="0", recipient="<wallet>")])
    assert score_case(case, turn) == 0


def test_swap_parsed_from_native_tool_call():
    native = [{"name": "swap", "arguments": '{"chainId":"1","currencyIn":"0xAAA","currencyOut":"0xBBB","amountIn":"100","amountOutMinimum":"0","recipient":"<wallet>"}'}]
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

# ============================================================================
# test_intents
# ============================================================================

def test_to_base_units():
    assert to_base_units("0.1", 18) == "100000000000000000"
    assert to_base_units("100", 6) == "100000000"
    assert to_base_units(5, 18) == "5000000000000000000"  # accepts int/float-ish


def test_resolve_recipient_ens_and_raw():
    assert resolve_recipient("vitalik.eth") == LOOKUP["ens"]["vitalik.eth"]
    raw = "0x1111111111111111111111111111111111111111"
    assert resolve_recipient(raw) == raw
    assert resolve_recipient("bob.eth") is None


def test_swap_currency_native_and_erc20():
    assert swap_currency("ETH") == ("0x0000000000000000000000000000000000000000", 18)
    assert swap_currency("USDC") == (LOOKUP["tokens"]["USDC"]["address"], 6)
    assert swap_currency("NOPE") is None


def test_build_transfer_call_is_human_units_and_unresolved_recipient():
    # Mirrors ToolDefinitions.transfer in the macOS app: human decimal amount,
    # token by symbol, recipient exactly as the user expressed it.
    call = build_transfer_call("0.1", "ETH", "vitalik.eth")
    assert call == {
        "tool": "transfer",
        "to": "vitalik.eth", "amount": "0.1", "token": "ETH",
    }


def test_build_transfer_call_erc20_does_not_encode_calldata():
    call = build_transfer_call("3", "USDC", "0x2222222222222222222222222222222222222222")
    assert call == {
        "tool": "transfer",
        "to": "0x2222222222222222222222222222222222222222",
        "amount": "3", "token": "USDC",
    }


def test_build_swap_call_is_symbols_and_human_units():
    call = build_swap_call("100", "USDC", "ETH")
    assert call == {
        "tool": "swap", "amount": "100",
        "from_token": "USDC", "to_token": "ETH", "amount_side": "input",
    }


def test_build_transfer_call_unknown_token_raises():
    with pytest.raises(ValueError):
        build_transfer_call("1", "NOPE", "0x1111111111111111111111111111111111111111")


def test_build_swap_call_unknown_currency_raises():
    with pytest.raises(ValueError):
        build_swap_call("1", "USDC", "NOPE")


def test_format_expected_summary_variants():
    from wallet_evals.intents import format_expected_summary
    # no call
    assert format_expected_summary([]) == "(no tool call)"
    # native ETH transfer
    s = format_expected_summary([{"tool": "executeTx", "chainId": "1",
        "to": "0xRecip", "value": "100", "function": None, "args": []}])
    assert "executeTx to 0xRecip" in s and "value=100" in s and "(native)" in s
    # ERC-20 transfer
    s = format_expected_summary([{"tool": "executeTx", "chainId": "1", "to": "0xToken",
        "value": "0", "function": "transfer(address,uint256)", "args": ["0xR", "5"]}])
    assert "transfer(address,uint256)" in s and "args=['0xR', '5']" in s
    # swap
    s = format_expected_summary([{"tool": "swap", "chainId": "1", "amount": "7",
        "from_token": "USDC", "to_token": "ETH", "amount_side": "input"}])
    assert "swap 7 USDC -> ETH" in s
    # transfer
    s = format_expected_summary([{"tool": "transfer",
        "to": "vitalik.eth", "amount": "0.1", "token": "ETH"}])
    assert s == "transfer 0.1 ETH to vitalik.eth"
    # multi-call joins with " | "
    s = format_expected_summary([
        {"tool": "executeTx", "chainId": "1", "to": "0x1", "value": "1", "function": None, "args": []},
        {"tool": "executeTx", "chainId": "1", "to": "0x2", "value": "2", "function": None, "args": []}])
    assert " | " in s
