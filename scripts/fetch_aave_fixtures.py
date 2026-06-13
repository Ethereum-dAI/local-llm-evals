"""Reproducibility tool: decode real Aave v3 Pool events into
datasets/protocols/aave.fixtures.json. Requires network + web3.

    uv run --with web3 python scripts/fetch_aave_fixtures.py

Not part of the test suite. Scans recent blocks via eth_getLogs on the mainnet
Aave v3 Pool, keeps known assets with clean human amounts (<=2 decimals), and
writes one fixture per (op, asset). The committed fixtures are canonical; re-run
only to refresh or extend.
"""
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from web3 import Web3

RPC = "https://ethereum-rpc.publicnode.com"
OUT = Path(__file__).resolve().parent.parent / "datasets" / "protocols" / "aave.fixtures.json"
POOL = Web3.to_checksum_address("0x87870Bca3F3fD6335C3F4ce8392D69350B4fa4E2")
BLOCK_SPAN = 3000

ASSETS = {  # address(lower) -> (symbol, decimals)
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": ("USDC", 6),
    "0xdac17f958d2ee523a2206206994597c13d831ec7": ("USDT", 6),
    "0x6b175474e89094c44da98b954eedeac495271d0f": ("DAI", 18),
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": ("WETH", 18),
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599": ("WBTC", 8),
}
ABI = [
    {"type": "event", "anonymous": False, "name": "Supply", "inputs": [
        {"name": "reserve", "type": "address", "indexed": True},
        {"name": "user", "type": "address", "indexed": False},
        {"name": "onBehalfOf", "type": "address", "indexed": True},
        {"name": "amount", "type": "uint256", "indexed": False},
        {"name": "referralCode", "type": "uint16", "indexed": True}]},
    {"type": "event", "anonymous": False, "name": "Withdraw", "inputs": [
        {"name": "reserve", "type": "address", "indexed": True},
        {"name": "user", "type": "address", "indexed": True},
        {"name": "to", "type": "address", "indexed": True},
        {"name": "amount", "type": "uint256", "indexed": False}]},
    {"type": "event", "anonymous": False, "name": "Borrow", "inputs": [
        {"name": "reserve", "type": "address", "indexed": True},
        {"name": "user", "type": "address", "indexed": False},
        {"name": "onBehalfOf", "type": "address", "indexed": True},
        {"name": "amount", "type": "uint256", "indexed": False},
        {"name": "interestRateMode", "type": "uint8", "indexed": False},
        {"name": "borrowRate", "type": "uint256", "indexed": False},
        {"name": "referralCode", "type": "uint16", "indexed": True}]},
    {"type": "event", "anonymous": False, "name": "Repay", "inputs": [
        {"name": "reserve", "type": "address", "indexed": True},
        {"name": "user", "type": "address", "indexed": True},
        {"name": "repayer", "type": "address", "indexed": True},
        {"name": "amount", "type": "uint256", "indexed": False},
        {"name": "useATokens", "type": "bool", "indexed": False}]},
]


def _clean(amount: int, dec: int) -> bool:
    h = Decimal(amount) / (Decimal(10) ** dec)
    return amount > 0 and h == h.quantize(Decimal("0.01")) and h <= Decimal("1000000")


def main() -> None:
    w3 = Web3(Web3.HTTPProvider(RPC, request_kwargs={"headers": {"User-Agent": "Mozilla/5.0"}}))
    c = w3.eth.contract(address=POOL, abi=ABI)
    latest = w3.eth.block_number
    frm = latest - BLOCK_SPAN
    out: list[dict] = []
    for op, evt in [("supply", "Supply"), ("withdraw", "Withdraw"),
                    ("borrow", "Borrow"), ("repay", "Repay")]:
        seen: set[str] = set()
        for ev in getattr(c.events, evt)().get_logs(from_block=frm, to_block=latest):
            res = ev.args["reserve"].lower()
            if res not in ASSETS:
                continue
            sym, dec = ASSETS[res]
            amount = ev.args["amount"]
            if sym in seen or not _clean(amount, dec):
                continue
            seen.add(sym)
            human = str(Decimal(amount) / (Decimal(10) ** dec))
            entry = {"op": op, "chainId": "1", "asset": sym, "amount": str(amount),
                     "amount_human": human}
            if op in ("borrow", "repay"):
                entry["rate_mode"] = int(ev.args["interestRateMode"]) if op == "borrow" else 2
            entry["tx_hash"] = ev.transactionHash.hex()
            out.append(entry)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {len(out)} fixtures -> {OUT}")


if __name__ == "__main__":
    main()
