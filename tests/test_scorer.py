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
