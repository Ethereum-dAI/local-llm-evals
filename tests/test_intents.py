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


def test_build_transfer_call_is_human_units_and_unresolved_recipient():
    # Mirrors ToolDefinitions.transfer in the macOS app: human decimal amount,
    # token by symbol, recipient exactly as the user expressed it.
    call = build_transfer_call("0.1", "ETH", "vitalik.eth")
    assert call == {
        "tool": "transfer", "chainId": CHAIN_ID,
        "to": "vitalik.eth", "amount": "0.1", "token": "ETH",
    }


def test_build_transfer_call_erc20_does_not_encode_calldata():
    call = build_transfer_call("3", "USDC", "0x2222222222222222222222222222222222222222")
    assert call == {
        "tool": "transfer", "chainId": CHAIN_ID,
        "to": "0x2222222222222222222222222222222222222222",
        "amount": "3", "token": "USDC",
    }


def test_build_swap_call_is_symbols_and_human_units():
    call = build_swap_call("100", "USDC", "ETH")
    assert call == {
        "tool": "swap", "chainId": CHAIN_ID, "amount": "100",
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
    s = format_expected_summary([{"tool": "transfer", "chainId": "1",
        "to": "vitalik.eth", "amount": "0.1", "token": "ETH"}])
    assert s == "transfer 0.1 ETH to vitalik.eth"
    # multi-call joins with " | "
    s = format_expected_summary([
        {"tool": "executeTx", "chainId": "1", "to": "0x1", "value": "1", "function": None, "args": []},
        {"tool": "executeTx", "chainId": "1", "to": "0x2", "value": "2", "function": None, "args": []}])
    assert " | " in s
