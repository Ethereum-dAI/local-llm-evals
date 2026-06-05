"""Render the chat prompt: static base (pf/prompt.json) + dynamic token table.

The known-token table is loaded from datasets/lookup.json (the same single
source the dataset converter uses) so the context the model sees can never
drift from the gold the scorer expects.
"""
from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_PROMPT = _ROOT / "pf" / "prompt.json"
_LOOKUP = _ROOT / "datasets" / "lookup.json"

_ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"


def render_token_table(lookup: dict) -> str:
    """One sentence listing each known token's address and decimals."""
    entries = []
    for symbol, token in lookup["tokens"].items():
        if token.get("native"):
            entries.append(
                f"{symbol} — native, {token['decimals']} decimals,"
                f" as a swap currency use the zero address {_ZERO_ADDRESS}"
            )
        else:
            entries.append(f"{symbol} — {token['address']}, {token['decimals']} decimals")
    return (
        "Known tokens (Ethereum mainnet): " + "; ".join(entries) + "."
        " Use these exact addresses and decimals; for a token not in this list,"
        " ask rather than guess."
    )


def build_messages(user_message: str) -> list[dict]:
    """The pf/prompt.json messages with the token table appended to the system
    message and {{user_message}} substituted."""
    messages = json.loads(_PROMPT.read_text())
    lookup = json.loads(_LOOKUP.read_text())
    rendered = []
    for message in messages:
        content = message["content"].replace("{{user_message}}", user_message)
        if message["role"] == "system":
            content = f"{content} {render_token_table(lookup)}"
        rendered.append({"role": message["role"], "content": content})
    return rendered
