"""Aave v3 protocol module: fixtures -> executeTx eval cases (mainnet Pool).

Gold is a generic executeTx to the Aave v3 Pool, computed from {asset, amount}
plus fixed defaults: onBehalfOf/to = "<wallet>", referralCode = "0",
interestRateMode = "2" (variable). No on-chain state is needed.
"""
from __future__ import annotations

import random
from pathlib import Path

from wallet_evals.generation import apply_mutators
from wallet_evals.intents import format_expected_summary

NAME = "aave"
FIXTURES = Path(__file__).resolve().parents[3] / "datasets" / "protocols" / "aave.fixtures.json"

POOL = "0x87870Bca3F3fD6335C3F4ce8392D69350B4fa4E2"
WALLET = "<wallet>"

# symbol -> token address (decimals live in the prompt reference, not needed for gold)
ASSETS = {
    "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
    "DAI": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
    "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    "WBTC": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
}


def gold_call(fx: dict) -> dict:
    """Generic executeTx gold for an Aave op (to = the Pool)."""
    asset = ASSETS[fx["asset"]]
    amount = fx["amount"]
    base = {"tool": "executeTx", "chainId": fx["chainId"], "to": POOL, "value": "0"}
    op = fx["op"]
    if op == "supply":
        return {**base, "function": "supply(address,uint256,address,uint16)",
                "args": [asset, amount, WALLET, "0"]}
    if op == "withdraw":
        return {**base, "function": "withdraw(address,uint256,address)",
                "args": [asset, amount, WALLET]}
    if op == "borrow":
        return {**base, "function": "borrow(address,uint256,uint256,uint16,address)",
                "args": [asset, amount, str(fx["rate_mode"]), "0", WALLET]}
    if op == "repay":
        return {**base, "function": "repay(address,uint256,uint256,address)",
                "args": [asset, amount, str(fx["rate_mode"]), WALLET]}
    raise ValueError(f"unknown Aave op: {op!r}")


TEMPLATES = {
    "supply": [("direct", [
        "Supply {amount} {asset} to Aave v3.",
        "Deposit {amount} {asset} into Aave v3.",
        "Lend {amount} {asset} on Aave v3.",
    ]), ("narrative", [
        "I'd like to put {amount} {asset} to work earning yield on Aave v3.",
        "Let's deposit {amount} {asset} into Aave v3 so it earns interest.",
    ])],
    "withdraw": [("direct", [
        "Withdraw {amount} {asset} from my Aave v3 supply position.",
        "Redeem {amount} {asset} from Aave v3.",
        "Pull {amount} {asset} out of Aave v3.",
    ]), ("narrative", [
        "I need {amount} {asset} back from my Aave v3 deposit, please.",
        "Take {amount} {asset} out of what I lent on Aave v3.",
    ])],
    "borrow": [("direct", [
        "Borrow {amount} {asset} from Aave v3.",
        "Take out a loan of {amount} {asset} on Aave v3.",
        "Draw {amount} {asset} of debt from Aave v3.",
    ]), ("narrative", [
        "I want to borrow against my collateral — pull {amount} {asset} from Aave v3.",
        "Open a {amount} {asset} loan for me on Aave v3.",
    ])],
    "repay": [("direct", [
        "Repay {amount} {asset} on my Aave v3 loan.",
        "Pay back {amount} {asset} to Aave v3.",
        "Settle {amount} {asset} of my Aave v3 debt.",
    ]), ("narrative", [
        "Let's pay down {amount} {asset} of what I owe on Aave v3.",
        "I'd like to clear {amount} {asset} from my Aave v3 borrow.",
    ])],
}
CATEGORY = {"supply": "aave-supply", "withdraw": "aave-withdraw",
            "borrow": "aave-borrow", "repay": "aave-repay"}


def _fill(template: str, fx: dict) -> str:
    return template.format(amount=fx["amount_human"], asset=fx["asset"])


def build_cases(fixtures: list[dict], rng: random.Random, start_idx: int = 1) -> list[dict]:
    """One case per (fixture, template); gold computed; vars.protocol = 'aave'."""
    cases: list[dict] = []
    idx = start_idx
    for fx in fixtures:
        gold = gold_call(fx)
        for style, templates in TEMPLATES[fx["op"]]:
            for template in templates:
                surface, labels = apply_mutators(_fill(template, fx), rng)
                cases.append({
                    "vars": {"user_message": surface, "protocol": "aave",
                             "expected_summary": format_expected_summary([gold])},
                    "metadata": {
                        "id": f"aave-{fx['op']}-{idx:04d}",
                        "source": "generated-protocol",
                        "protocol": "aave",
                        "language": "english",
                        "category": CATEGORY[fx["op"]],
                        "difficulty": "hard",
                        "level": "payload",
                        "query_type": "one_shot",
                        "requires": ["aave_pool"],
                        "style": style,
                        "mutators": labels,
                        "expected_calls": [gold],
                        "notes": None,
                    },
                })
                idx += 1
    return cases
