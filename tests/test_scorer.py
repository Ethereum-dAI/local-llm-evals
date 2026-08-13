from wallet_evals.schema import Case, ParsedToolCall, ParsedTurn
from wallet_evals.scorer import score_case


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
