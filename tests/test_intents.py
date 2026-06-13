import pytest
from wallet_evals.intents import (
    to_base_units, resolve_recipient, swap_currency,
    build_transfer_call, build_swap_call, LOOKUP, CHAIN_ID,
)


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


def test_build_transfer_call_native():
    call = build_transfer_call("0.1", "ETH", "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045")
    assert call == {
        "tool": "executeTx", "chainId": CHAIN_ID,
        "to": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
        "value": "100000000000000000", "function": None, "args": [],
    }


def test_build_transfer_call_erc20():
    call = build_transfer_call("3", "USDC", "0x2222222222222222222222222222222222222222")
    assert call["to"] == LOOKUP["tokens"]["USDC"]["address"]
    assert call["value"] == "0"
    assert call["function"] == "transfer(address,uint256)"
    assert call["args"] == ["0x2222222222222222222222222222222222222222", "3000000"]


def test_build_swap_call():
    call = build_swap_call("100", "USDC", "ETH")
    assert call["tool"] == "swap"
    assert call["currencyIn"] == LOOKUP["tokens"]["USDC"]["address"]
    assert call["currencyOut"] == "0x0000000000000000000000000000000000000000"
    assert call["amountIn"] == "100000000"
    assert call["amountOutMinimum"] == "0"
    assert call["recipient"] == "<wallet>"


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
    s = format_expected_summary([{"tool": "swap", "chainId": "1", "currencyIn": "0xA",
        "currencyOut": "0xB", "amountIn": "7", "amountOutMinimum": "0", "recipient": "<wallet>"}])
    assert "swap 0xA -> 0xB" in s and "amountIn=7" in s
    # multi-call joins with " | "
    s = format_expected_summary([
        {"tool": "executeTx", "chainId": "1", "to": "0x1", "value": "1", "function": None, "args": []},
        {"tool": "executeTx", "chainId": "1", "to": "0x2", "value": "2", "function": None, "args": []}])
    assert " | " in s
