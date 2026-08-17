"""The tool schema the eval offers must match the macOS app's own ToolDefinitions.

Source of truth is wallet-macos/Sources/WalletToolLayer/ToolDefinitions.swift in the
local-wallet-mac repo (`ToolDefinitions.phase1`). It cannot be imported from Python,
so the property names are asserted here and any drift shows up as a failure rather
than as a silently mis-scored run.
"""
import json
from pathlib import Path

TOOLS = json.loads((Path(__file__).resolve().parents[1] / "pf" / "tools.json").read_text())
BY_NAME = {t["function"]["name"]: t["function"] for t in TOOLS}


def _props(name):
    return set(BY_NAME[name]["parameters"]["properties"])


def test_all_four_tools_present():
    # transfer/swap mirror the app; executeTx/readTx serve the Aave and Safe
    # protocol datasets, which have no app counterpart. shield/unshield were
    # dropped with the RAILGUN feature (local-wallet-mac#86, PR #87).
    assert set(BY_NAME) == {"executeTx", "readTx", "transfer", "swap"}


def test_transfer_matches_app_properties():
    # Exact parity with ToolDefinitions.swift:5. chainId was removed on
    # 2026-08-14: the app declares no such argument and never reads one
    # (grep intent.args["chainId"] returns nothing), so requiring it trained the
    # model to emit an off-contract field — the same drift class as executeTx.
    assert _props("transfer") == {"to", "amount", "token"}
    assert BY_NAME["transfer"]["parameters"]["required"] == ["to", "amount"]


def test_swap_matches_app_properties():
    # Exact parity with ToolDefinitions.swift:15 — see the chainId note above.
    assert _props("swap") == {"from_token", "to_token", "amount", "amount_side"}
    assert BY_NAME["swap"]["parameters"]["required"] == [
        "from_token", "to_token", "amount"]


def test_app_tools_declare_no_chainid_but_protocol_tools_do():
    """The split that keeps the two contracts honest: transfer/swap mirror the
    app, which resolves the chain from activeChain.id; executeTx/readTx serve
    the Aave/Safe datasets, which are base-unit and do carry a chainId."""
    for name in ("transfer", "swap"):
        assert "chainId" not in _props(name)
    for name in ("executeTx", "readTx"):
        assert "chainId" in _props(name)
    assert BY_NAME["swap"]["parameters"]["properties"]["amount_side"]["enum"] == ["input"]


def test_no_app_tool_asks_for_base_units():
    # Every app tool is human-unit now, so "the exception to the base-unit rule"
    # framing the privacy tools used to carry is no longer true of anything.
    for name in ("transfer", "swap"):
        blob = json.dumps(BY_NAME[name]).lower()
        assert "base unit" not in blob and "base-unit" not in blob
