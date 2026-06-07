"""Convert the legacy recognition.json into pf/tests.yaml.

`pf/tests.yaml` is the single source of truth for test cases: a promptfoo-native
list of tests, each `{vars: {user_message}, metadata: {gold + slices}}`. This
script generates it from the Swift app's recognition.json so the dataset stays
reproducible from the production source. Swaps become the synthetic `swap`
intent; transfers/approvals become `executeTx`; ambiguous cases become no-call.
Exact-output swaps, "all" amounts, and unresolved ENS/tokens are reported as
needing manual authoring. Resolution uses datasets/lookup.json, here only.

Multi-turn / conversational cases (metadata.query_type == "multi_turn") cannot
come from recognition.json — they are authored directly in pf/tests.yaml. To
keep regeneration idempotent, those cases are read back from the existing
tests.yaml and re-appended after the generated ones.
"""
from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
LOOKUP = json.loads((ROOT / "datasets" / "lookup.json").read_text())

def _difficulty(raw: dict, protocol: str) -> str:
    """Derived, not authored: swaps and non-English prompts are harder targets."""
    if protocol == "uniswap" or raw["language"] != "english":
        return "medium"
    return "easy"


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
        "difficulty": _difficulty(raw, protocol),
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
        "amountOutMinimum": "0", "recipient": "SELF",
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

    to_value = args["to"]["value"]
    recipient = _resolve_recipient(to_value)
    if recipient is None:
        return None, raw["id"]

    chain_id = LOOKUP["chainId"]
    if token.get("native"):
        call = {"tool": "executeTx", "chainId": chain_id, "to": recipient,
                "value": to_base_units(amount, token["decimals"]), "function": None, "args": []}
        if to_value in LOOKUP["ens"]:
            # The wallet resolves ENS post-call: the unresolved name is equally correct.
            call["to_aliases"] = [to_value]
    else:
        call = {"tool": "executeTx", "chainId": chain_id, "to": token["address"], "value": "0",
                "function": "transfer(address,uint256)",
                "args": [recipient, to_base_units(amount, token["decimals"])]}

    requires: list[str] = []
    if to_value in LOOKUP["ens"]:
        # Gold carries the resolved address, so passing demands ENS resolution —
        # flag it so the report can slice this capability out.
        requires.append("ens_resolution")
    if not token.get("native"):
        requires.append("token_address_lookup")

    case = _base_meta(raw) | {
        "level": "payload",
        "query_type": "one_shot",
        "requires": requires,
        "expected_calls": [call],
    }
    return case, None


def _to_promptfoo_test(case: dict) -> dict:
    """Map a converted case dict to a promptfoo-native test.

    The input is the prompt var; everything else (gold + slice fields) is
    metadata the python assertion reads back via case_from_metadata.
    """
    metadata = {k: v for k, v in case.items() if k != "user_message"}
    return {"vars": {"user_message": case["user_message"]}, "metadata": metadata}


def _is_manual(test: dict) -> bool:
    """True for cases authored directly in tests.yaml rather than derived from
    recognition.json: multi-turn cases (carry conversation history) and
    adversarial/safety cases (injection, scams, over-execution)."""
    metadata = test.get("metadata", {})
    return (
        metadata.get("query_type") == "multi_turn"
        or metadata.get("category") == "adversarial"
    )


def _preserved_manual_tests(out: Path) -> list[dict]:
    """Manually-authored cases carried across regens, in their existing order.

    They aren't derivable from recognition.json, so read them back verbatim and
    re-append, keeping regeneration idempotent.
    """
    if not out.exists():
        return []
    existing = yaml.safe_load(out.read_text()) or []
    return [t for t in existing if _is_manual(t)]


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT.parent / "local-wallet-mac" / "wallet-macos" / "Sources" / "wallet-eval" / "Dataset" / "recognition.json"
    )
    out = ROOT / "pf" / "tests.yaml"
    legacy = json.loads(src.read_text())

    converted: list[dict] = []
    needs_manual: list[str] = []
    for raw in legacy["cases"]:
        case, manual = convert_case(raw)
        if manual is not None:
            needs_manual.append(manual)
        else:
            converted.append(case)

    tests = [_to_promptfoo_test(c) for c in converted]
    tests.extend(_preserved_manual_tests(out))
    header = (
        "# Single source of truth for eval test cases (promptfoo-native).\n"
        "# Generated from the Swift recognition.json by scripts/convert_recognition.py.\n"
        "# Each test: vars.user_message (input) + metadata (gold expected_calls + slices).\n"
    )
    out.write_text(header + yaml.safe_dump(tests, sort_keys=False, allow_unicode=True))
    print(f"Converted {len(converted)} cases -> {out}")
    print(f"Needs manual authoring ({len(needs_manual)}): {needs_manual}")


if __name__ == "__main__":
    main()
