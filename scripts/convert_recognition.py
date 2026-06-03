"""Convert the legacy recognition.json into the unified executeTx schema.

Mechanical cases only (transfers + no-call). Swaps, "all" amounts, and any
unresolved ENS/token are routed to needs_manual for hand-authoring (later task).
Resolution uses datasets/lookup.json and happens ONLY here, at authoring time.
"""
from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOOKUP = json.loads((ROOT / "datasets" / "lookup.json").read_text())

_DIFFICULTY = "easy"


def to_base_units(amount: str, decimals: int) -> str:
    """Convert a human decimal string to a base-unit integer string."""
    scaled = Decimal(amount) * (Decimal(10) ** decimals)
    if scaled != scaled.to_integral_value():
        raise ValueError(f"amount {amount} has more precision than {decimals} decimals")
    return str(int(scaled))


def _resolve_recipient(value: str) -> str | None:
    if value in LOOKUP["ens"]:
        return LOOKUP["ens"][value]
    if value.startswith("0x") and len(value) == 42:
        return value
    return None


def _base_meta(raw: dict, protocol: str = "transfer") -> dict:
    return {
        "id": raw["id"],
        "user_message": raw["user_message"],
        "language": raw["language"],
        "category": raw["category"],
        "protocol": protocol,
        "difficulty": _DIFFICULTY,
        "notes": raw.get("notes"),
    }


def _swap_currency(symbol: str) -> tuple[str, int] | None:
    """Return (address, decimals) for a swap currency, or None if unknown.
    Native ETH maps to the zero address (Uniswap v4 convention)."""
    token = LOOKUP["tokens"].get(symbol)
    if token is None:
        return None
    if token.get("native"):
        return "0x0000000000000000000000000000000000000000", token["decimals"]
    return token["address"], token["decimals"]


def _convert_swap(raw: dict) -> tuple[dict | None, str | None]:
    args = raw["expected_args"]
    side = args.get("amount_side", {}).get("value", "input")
    if side != "input":
        return None, raw["id"]  # exact-output not supported in v1 -> manual
    amount = args.get("amount", {}).get("value")
    if amount is None or amount == "all":
        return None, raw["id"]
    in_sym = args.get("from_token", {}).get("value")
    out_sym = args.get("to_token", {}).get("value")
    cin = _swap_currency(in_sym) if in_sym else None
    cout = _swap_currency(out_sym) if out_sym else None
    if cin is None or cout is None:
        return None, raw["id"]
    in_addr, in_dec = cin
    out_addr, _ = cout
    call = {
        "tool": "swap", "chainId": LOOKUP["chainId"],
        "currencyIn": in_addr, "currencyOut": out_addr,
        "amountIn": to_base_units(amount, in_dec),
        "amountOutMinimum": "0", "recipient": "<wallet>",
    }
    case = _base_meta(raw, protocol="uniswap") | {
        "level": "payload", "query_type": "one_shot",
        "requires": ["token_address_lookup"], "expected_calls": [call],
    }
    return case, None


def convert_case(raw: dict) -> tuple[dict | None, str | None]:
    """Return (converted_case, None) or (None, case_id_needing_manual)."""
    if raw.get("expected_tool") is None:
        case = _base_meta(raw) | {"level": "intent", "query_type": None, "requires": [], "expected_calls": []}
        return case, None

    if raw["expected_tool"] == "swap":
        return _convert_swap(raw)

    if raw["expected_tool"] != "transfer":
        return None, raw["id"]  # swaps -> manual

    args = raw["expected_args"]
    amount = args["amount"]["value"]
    if amount == "all":
        return None, raw["id"]

    token_sym = args.get("token", {}).get("value", "ETH")
    token = LOOKUP["tokens"].get(token_sym)
    if token is None:
        return None, raw["id"]

    recipient = _resolve_recipient(args["to"]["value"])
    if recipient is None:
        return None, raw["id"]

    chain_id = LOOKUP["chainId"]
    if token.get("native"):
        call = {"tool": "executeTx", "chainId": chain_id, "to": recipient,
                "value": to_base_units(amount, token["decimals"]), "function": None, "args": []}
    else:
        call = {"tool": "executeTx", "chainId": chain_id, "to": token["address"], "value": "0",
                "function": "transfer(address,uint256)",
                "args": [recipient, to_base_units(amount, token["decimals"])]}

    case = _base_meta(raw) | {
        "level": "payload",
        "query_type": "one_shot",
        "requires": ["token_address_lookup"] if not token.get("native") else [],
        "expected_calls": [call],
    }
    return case, None


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT.parent / "local-wallet-mac" / "wallet-macos" / "Sources" / "wallet-eval" / "Dataset" / "recognition.json"
    )
    out = ROOT / "datasets" / "cases.json"
    legacy = json.loads(src.read_text())

    converted: list[dict] = []
    needs_manual: list[str] = []
    for raw in legacy["cases"]:
        case, manual = convert_case(raw)
        if manual is not None:
            needs_manual.append(manual)
        else:
            converted.append(case)

    payload = {"schema": "evals-local-llm/cases/v1", "cases": converted}
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Converted {len(converted)} cases -> {out}")
    print(f"Needs manual authoring ({len(needs_manual)}): {needs_manual}")


if __name__ == "__main__":
    main()
