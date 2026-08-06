"""Shared gold-builders: structured intent -> on-chain `expected_calls`.

Single source of truth for gold computation, reused by the recognition.json
converter (scripts/convert_recognition.py) and the deterministic case generator
(src/wallet_evals/generation.py). Resolution uses datasets/lookup.json.
"""
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOOKUP = json.loads((ROOT / "datasets" / "lookup.json").read_text())
CHAIN_ID = LOOKUP["chainId"]
ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"


def to_base_units(amount: str | int | float, decimals: int) -> str:
    """Convert a human decimal amount to a base-unit integer string."""
    scaled = Decimal(str(amount)) * (Decimal(10) ** decimals)
    if scaled != scaled.to_integral_value():
        raise ValueError(f"amount {amount} has more precision than {decimals} decimals")
    return str(int(scaled))


def resolve_recipient(value: str) -> str | None:
    """ENS name -> address (via lookup), 0x-address -> itself, else None."""
    if value in LOOKUP["ens"]:
        return LOOKUP["ens"][value]
    if value.startswith("0x") and len(value) == 42:
        return value
    return None


def swap_currency(symbol: str) -> tuple[str, int] | None:
    """Return (address, decimals) for a swap currency, or None if unknown.
    Native ETH maps to the zero address (Uniswap v4 convention)."""
    token = LOOKUP["tokens"].get(symbol)
    if token is None:
        return None
    if token.get("native"):
        return ZERO_ADDRESS, token["decimals"]
    return token["address"], token["decimals"]


def build_transfer_call(amount: str, token_sym: str, recipient_addr: str) -> dict:
    """Build an executeTx gold call for a transfer of `amount` `token_sym`."""
    token = LOOKUP["tokens"].get(token_sym)
    if token is None:
        raise ValueError(f"unknown token symbol: {token_sym!r}")
    if token.get("native"):
        return {"tool": "executeTx", "chainId": CHAIN_ID, "to": recipient_addr,
                "value": to_base_units(amount, token["decimals"]),
                "function": None, "args": []}
    return {"tool": "executeTx", "chainId": CHAIN_ID, "to": token["address"], "value": "0",
            "function": "transfer(address,uint256)",
            "args": [recipient_addr, to_base_units(amount, token["decimals"])]}


def build_swap_call(amount: str, from_sym: str, to_sym: str) -> dict:
    """Build a synthetic swap gold call (exact-input)."""
    cin = swap_currency(from_sym)
    cout = swap_currency(to_sym)
    if cin is None or cout is None:
        raise ValueError(f"unknown swap currency: {from_sym!r} or {to_sym!r}")
    in_addr, in_dec = cin
    out_addr, _ = cout
    return {"tool": "swap", "chainId": CHAIN_ID,
            "currencyIn": in_addr, "currencyOut": out_addr,
            "amountIn": to_base_units(amount, in_dec),
            "amountOutMinimum": "0", "recipient": "<wallet>"}


def build_shield_call(amount: str, token_sym: str = "ETH") -> dict:
    """Gold for a RAILGUN shield (deposit into the private pool).

    Deliberately NOT base units: `shield`/`unshield` mirror the macOS app's own
    ToolDefinitions, which take a human decimal `amount` and an ETH-only `token`.
    """
    if token_sym.upper() != "ETH":
        raise ValueError(f"RAILGUN shield is ETH-only, got {token_sym!r}")
    return {"tool": "shield", "chainId": CHAIN_ID, "amount": str(amount), "token": "ETH"}


def build_unshield_call(amount: str, recipient_addr: str, token_sym: str = "ETH") -> dict:
    """Gold for a RAILGUN unshield (withdraw to `recipient_addr` as native ETH).

    The recipient must already be a 0x address: the app does not yet resolve ENS
    or contact names for unshield, so gold never carries an unresolved name.
    """
    if token_sym.upper() != "ETH":
        raise ValueError(f"RAILGUN unshield is ETH-only, got {token_sym!r}")
    if not (recipient_addr.startswith("0x") and len(recipient_addr) == 42):
        raise ValueError(f"unshield recipient must be a 0x address, got {recipient_addr!r}")
    return {"tool": "unshield", "chainId": CHAIN_ID, "amount": str(amount),
            "token": "ETH", "to": recipient_addr}


def format_expected_summary(calls: list[dict]) -> str:
    """A human-readable one-line-per-call summary of expected_calls.

    Read-only display aid: generators put this in `vars.expected_summary` so the
    promptfoo viewer shows an Expected column next to the model's actual output.
    It is never sent to the model (the prompt only renders user_message/messages/
    account_context) and never used for scoring (the scorer reads metadata).
    """
    if not calls:
        return "(no tool call)"
    lines: list[str] = []
    for c in calls:
        if c.get("tool") == "swap":
            lines.append(
                f"swap {c.get('currencyIn')} -> {c.get('currencyOut')} "
                f"amountIn={c.get('amountIn')} minOut={c.get('amountOutMinimum')} "
                f"recipient={c.get('recipient')}")
            continue
        if c.get("tool") in ("shield", "unshield"):
            line = f"{c.get('tool')} {c.get('amount')} {c.get('token')}"
            lines.append(line if c.get("to") is None else f"{line} to {c.get('to')}")
            continue
        parts = [f"{c.get('tool')} to {c.get('to')}"]
        if c.get("value") and c.get("value") != "0":
            parts.append(f"value={c.get('value')}")
        parts.append(c.get("function") or "(native)")
        if c.get("args"):
            parts.append(f"args={c.get('args')}")
        lines.append(" ".join(parts))
    return " | ".join(lines)
