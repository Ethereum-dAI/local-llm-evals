"""Unit tests for the rewritten `<think>` trace builder (Step 0 of the
app-contract fine-tune regeneration).

The old `_reasoning_text` computed a base-unit derivation ("USDC has 6
decimals, so 0.000002 USDC = 2 base units...") that would directly contradict
an app-contract target, which emits the HUMAN-unit amount verbatim. These
tests assert the rewritten trace only ever talks about the human-unit call the
app contract actually takes, for both families it now covers
(transfer/swap), and that the deterministic separator-note path fires
correctly for the `separator` bucket.
"""
from __future__ import annotations

from scripts.generate_finetune_data import _reasoning_text
from wallet_evals.intents import LOOKUP

_FORBIDDEN = ("base units", "wei", "decimals")
_TOKEN_CONTRACT_ADDRESSES = [
    meta["address"] for meta in LOOKUP["tokens"].values() if meta.get("address")
]


def _assert_app_contract_shape(trace: str, amount: str) -> None:
    assert amount in trace, f"trace must carry the human-unit amount: {trace!r}"
    lowered = trace.lower()
    for phrase in _FORBIDDEN:
        assert phrase not in lowered, f"trace must not mention {phrase!r}: {trace!r}"
    for addr in _TOKEN_CONTRACT_ADDRESSES:
        assert addr not in trace, f"trace must not name a token contract address: {trace!r}"


def test_transfer_trace_is_app_contract_shaped():
    intent = {"action": "transfer", "amount": "42.5", "token": "USDC",
              "recipient": "vitalik.eth"}
    trace = _reasoning_text(intent)
    _assert_app_contract_shape(trace, "42.5")
    assert "transfer" in trace
    assert "vitalik.eth" in trace  # recipient copied verbatim, not resolved


def test_swap_trace_is_app_contract_shaped():
    intent = {"action": "swap", "amount": "7.25", "from_token": "USDC",
              "to_token": "ETH"}
    trace = _reasoning_text(intent)
    _assert_app_contract_shape(trace, "7.25")
    assert "swap" in trace
    assert 'amount_side "input"' in trace or "amount_side" in trace


def test_transfer_trace_never_mentions_erc20_contract_address():
    """The old trace's whole ERC-20 branch named `meta['address']`; the new one
    must never resurrect it for a non-native token like USDC/DAI/WETH."""
    for tok in ("USDC", "DAI", "WETH"):
        intent = {"action": "transfer", "amount": "1.23", "token": tok,
                  "recipient": "vitalik.eth"}
        trace = _reasoning_text(intent)
        assert LOOKUP["tokens"][tok]["address"] not in trace


def test_separator_note_names_the_grouping_and_states_plain_decimal():
    """When `surface_amount` differs from `amount` (the `separator` bucket),
    the trace must name the separator and state the plain decimal — the exact
    failure mode Step 2 exists to close."""
    intent = {"action": "transfer", "amount": "20000", "token": "ETH",
              "recipient": "vitalik.eth", "surface_amount": "20,000"}
    trace = _reasoning_text(intent)
    assert "20,000" in trace  # names the comma-grouped surface form
    assert "20000" in trace  # states the plain decimal
    _assert_app_contract_shape(trace, "20000")


def test_no_separator_note_when_surface_amount_matches_amount():
    """A normal (non-separator) case must not spuriously claim a separator was
    present when the surface never carried one."""
    intent = {"action": "transfer", "amount": "5", "token": "ETH",
              "recipient": "vitalik.eth"}
    trace = _reasoning_text(intent)
    assert "separator" not in trace.lower()
    assert "strip" not in trace.lower()  # the separator-stripping instruction is absent


def test_reasoning_text_deterministic():
    intent = {"action": "swap", "amount": "3.5", "from_token": "ETH", "to_token": "DAI"}
    assert _reasoning_text(intent) == _reasoning_text(dict(intent))
