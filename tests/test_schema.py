import json
from wallet_evals.schema import Case, ExpectedCall, ParsedToolCall, ParsedTurn


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
