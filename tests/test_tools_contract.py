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


def test_all_six_tools_present():
    # transfer/swap/shield/unshield mirror the app; executeTx/readTx serve the
    # Aave and Safe protocol datasets, which have no app counterpart.
    assert set(BY_NAME) == {"executeTx", "readTx", "transfer", "swap",
                            "shield", "unshield"}


def test_transfer_matches_app_properties():
    assert _props("transfer") == {"chainId", "to", "amount", "token"}
    assert BY_NAME["transfer"]["parameters"]["required"] == ["chainId", "to", "amount"]


def test_swap_matches_app_properties():
    assert _props("swap") == {"chainId", "from_token", "to_token", "amount",
                              "amount_side"}
    assert BY_NAME["swap"]["parameters"]["required"] == [
        "chainId", "from_token", "to_token", "amount"]
    assert BY_NAME["swap"]["parameters"]["properties"]["amount_side"]["enum"] == ["input"]


def test_no_app_tool_asks_for_base_units():
    # Every app tool is human-unit now, so "the exception to the base-unit rule"
    # framing the privacy tools used to carry is no longer true of anything.
    for name in ("transfer", "swap", "shield", "unshield"):
        blob = json.dumps(BY_NAME[name]).lower()
        assert "base unit" not in blob and "base-unit" not in blob
