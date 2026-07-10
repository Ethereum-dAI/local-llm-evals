"""End-to-end: a FunctionGemma raw output, run through the provider's
translation, must score through the UNCHANGED pf/assert.py scorer. This is the
claim the whole integration rests on — assert.py, tools.json, and the dataset
stay untouched.
"""
import importlib

from wallet_evals.functiongemma import raw_output_to_scoreable

get_assert = importlib.import_module("pf.assert").get_assert  # 'assert' is a keyword


_BASE = {  # fields the Case model requires but that don't affect scoring here
    "level": "payload",
    "language": "english",
    "category": "generated",
    "protocol": "uniswap",
    "difficulty": "medium",
}


def _score(raw: str, metadata: dict) -> dict:
    output = raw_output_to_scoreable(raw)
    ctx = {"test": {"metadata": {**_BASE, **metadata}}, "providerResponse": {"output": output}}
    return get_assert(output, ctx)


def test_correct_swap_call_scores_pass():
    metadata = {
        "id": "gen-swap-pos-0176",
        "expected_calls": [{
            "tool": "swap",
            "chainId": "1",
            "currencyIn": "0x0000000000000000000000000000000000000000",
            "currencyOut": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            "amountIn": "987654320000000000000000",
            "amountOutMinimum": "0",
            "recipient": "<wallet>",
        }],
    }
    raw = (
        "<start_function_call>call:swap{"
        "chainId:<escape>1<escape>,"
        "currencyIn:<escape>0x0000000000000000000000000000000000000000<escape>,"
        "currencyOut:<escape>0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2<escape>,"
        "amountIn:<escape>987654320000000000000000<escape>,"
        "amountOutMinimum:<escape>0<escape>,"
        "recipient:<escape><wallet><escape>"
        "}<end_function_call>"
    )
    assert _score(raw, metadata)["pass"] is True


def test_erc20_transfer_with_args_array_scores_pass():
    metadata = {
        "id": "fg-erc20",
        "expected_calls": [{
            "tool": "executeTx",
            "chainId": "1",
            "to": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
            "value": "0",
            "function": "transfer(address,uint256)",
            "args": ["0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045", "3000000"],
        }],
    }
    raw = (
        "<start_function_call>call:executeTx{"
        "chainId:<escape>1<escape>,"
        "to:<escape>0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48<escape>,"
        "value:<escape>0<escape>,"
        "function:<escape>transfer(address,uint256)<escape>,"
        'args:<escape>["0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045","3000000"]<escape>'
        "}<end_function_call>"
    )
    assert _score(raw, metadata)["pass"] is True


def test_prose_refusal_scores_pass_on_refusal_case():
    metadata = {"id": "gen-refusal-0001", "expected_calls": []}
    raw = "I can't send funds to a burn address — that would destroy them."
    assert _score(raw, metadata)["pass"] is True
