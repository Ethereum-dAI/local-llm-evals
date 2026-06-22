"""Reproducibility tool: decode the listed Safe owner-management mainnet txs into
datasets/protocols/safe.fixtures.json. Requires network + web3.

    uv run --with web3 python scripts/fetch_safe_fixtures.py

Not part of the test suite. The committed fixtures must equal this output; re-run
only to refresh or extend the tx list. Tx hashes are sourced from
docs/superpowers/safe_add_owner_with_threshold.md and safe_remove_owner_txs.md.
"""
from __future__ import annotations

import json
from pathlib import Path

from web3 import Web3

RPC = "https://ethereum-rpc.publicnode.com"
OUT = Path(__file__).resolve().parent.parent / "datasets" / "protocols" / "safe.fixtures.json"

ADD_HASHES = [
    "0x35cbdcfe8f99fb1f42e4201ebb3b97050ff82bf0d09b85e10d727a955dc464f6",
    "0xd785009fc10ab196b976d6ae153d10a2c906d7c95a42782899c87890cba5bc11",
    "0x7d5a320fe91da31630cf46681d0936d2026d3ef07ae00ebedeeaf85928ed89dd",
    "0x9b1808cd10990ba9a2e0144f42a74f5763b0c1f071313c295f92a89eb6244004",
    "0x91283fc3ae492d06c78c626878a71d8dc0864dba03adfa7d1f1225e1ba2a1357",
]
REMOVE_HASHES = [
    "0x22dc55b121b7a576a547efefa3b62e0bf2ac3d0363f17dcf2c9fc15e3c3d3a0e",
    "0xb331c58612766b30e4770e0976325e8cd50921903dc5a93420962c8d718d8eb6",
    "0xa8267fdca74dce1eaaab443850176efb78e0ea6cb6692714184a3b0df9ccfe34",
]

SAFE_ABI = [
    {"name": "execTransaction", "type": "function", "inputs": [
        {"name": "to", "type": "address"}, {"name": "value", "type": "uint256"},
        {"name": "data", "type": "bytes"}, {"name": "operation", "type": "uint8"},
        {"name": "safeTxGas", "type": "uint256"}, {"name": "baseGas", "type": "uint256"},
        {"name": "gasPrice", "type": "uint256"}, {"name": "gasToken", "type": "address"},
        {"name": "refundReceiver", "type": "address"}, {"name": "signatures", "type": "bytes"}]},
    {"name": "addOwnerWithThreshold", "type": "function", "inputs": [
        {"name": "owner", "type": "address"}, {"name": "_threshold", "type": "uint256"}]},
    {"name": "removeOwner", "type": "function", "inputs": [
        {"name": "prevOwner", "type": "address"}, {"name": "owner", "type": "address"},
        {"name": "_threshold", "type": "uint256"}]},
    {"name": "multiSend", "type": "function", "inputs": [{"name": "transactions", "type": "bytes"}]},
]


def _parse_multisend(packed: bytes):
    ops, i = [], 0
    while i < len(packed):
        to = "0x" + packed[i + 1:i + 21].hex()
        dlen = int.from_bytes(packed[i + 53:i + 85], "big")
        data = packed[i + 85:i + 85 + dlen]
        ops.append((to, data))
        i += 85 + dlen
    return ops


def main() -> None:
    w3 = Web3(Web3.HTTPProvider(RPC, request_kwargs={"headers": {"User-Agent": "Mozilla/5.0"}}))
    c = w3.eth.contract(abi=SAFE_ABI)
    out: list[dict] = []

    def inner_ops(tx):
        fn, args = c.decode_function_input(tx["input"])
        safe = tx["to"]
        inner = args["data"]
        ifn, iargs = c.decode_function_input(inner)
        if ifn.fn_name == "multiSend":
            for _to, data in _parse_multisend(iargs["transactions"]):
                try:
                    sfn, sargs = c.decode_function_input(data)
                except Exception:
                    continue
                if sfn.fn_name in ("addOwnerWithThreshold", "removeOwner"):
                    yield safe, sfn.fn_name, sargs
        elif ifn.fn_name in ("addOwnerWithThreshold", "removeOwner"):
            yield safe, ifn.fn_name, iargs

    for h in ADD_HASHES + REMOVE_HASHES:
        tx = w3.eth.get_transaction(h)
        for safe, op, a in inner_ops(tx):
            if op == "addOwnerWithThreshold":
                params = {"owner": a["owner"], "threshold": a["_threshold"]}
            else:
                params = {"prevOwner": a["prevOwner"], "owner": a["owner"], "threshold": a["_threshold"]}
            out.append({"op": op, "chainId": "1", "safe": safe, "params": params, "tx_hash": h})

    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {len(out)} fixtures -> {OUT}")


if __name__ == "__main__":
    main()
