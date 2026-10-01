"""Would the wallet actually EXECUTE this gold call? A mirror of the app's guards.

`local-wallet-mac` turns a model's tool call into a UserOp through
`ChatDashboardView.transferRequest` / `swapRequest`, and a gold call that one of
those guards rejects scores models on emitting something the product throws on.
This module reproduces the guards for the chain the app actually runs (Ethereum
Sepolia, 11155111) so a dataset can assert every gold call is executable:

  * token: symbol or contract address in `WalletTokenRegistry` for Sepolia
    (UserOperationModels.swift). The app knows NO mainnet addresses -- which is why
    the 1000's `token_address` cases (mainnet addresses from datasets/lookup.json)
    are not executable in the app, only in wallet-eval's mainnet-ported harness.
  * amount: `EtherAmountParser.units(fromDecimalString:decimals:)` -- digits with
    at most one dot, no sign or separators, extra fraction digits only if zeros;
    "all" is rejected outright.
  * recipient: 20 hex bytes, or a dotted name that resolves. The daemon resolves on
    Sepolia and falls back to MAINNET ENS when Sepolia has no record
    (resolve_name.rs, MAINNET_FALLBACK_EXECUTION_RPC), so a name is executable iff
    it resolves on either. RESOLVABLE_ENS records the names verified on 2026-09-25
    against the same public mainnet RPC the daemon uses.
  * swap: both tokens known, different, amount_side "input".

It is a mirror kept by hand, like wallet-eval's own PortedAppEncoding.swift. The
end-to-end check is `wallet-eval userop` in local-wallet-mac.
"""
from __future__ import annotations

import re

SEPOLIA_CHAIN_ID = 11155111

#: symbol -> (decimals, contract address or None for native). WalletTokenRegistry,
#: Sepolia rows, local-wallet-mac/wallet-macos/Sources/WalletMacOSApp/UserOperationModels.swift.
SEPOLIA_TOKENS: dict[str, tuple[int, str | None]] = {
    "ETH": (18, None),
    "WETH": (18, "0xfFf9976782d46CC05630D1f6eBAb18b2324d6B14"),
    "USDC": (6, "0x1c7D4B196Cb0C7B01d743Fbc6116a902379C7238"),
    "USDT": (6, "0xaa8E23Fb1079EA71e0a56F48a2aA51851D8433D0"),
    "DAI": (18, "0x776b6FC2eD15d6bB5fC32e0c89DE68683118c62a"),
    "AAVE": (18, "0x5Bb220aFc6e2E008cB2302A83536A019ed245Aa2"),
    "UNI": (18, "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984"),
}

#: ENS names that resolve through the daemon's policy (Sepolia, then mainnet
#: fallback), with the address each resolved to on mainnet on 2026-09-25 via
#: https://ethereum-rpc.publicnode.com. Of generation.ENS_NAMES, carla, treasury,
#: grants, team-ops, ops.mydao and pay.acme did NOT resolve and are excluded.
RESOLVABLE_ENS: dict[str, str] = {
    "vitalik.eth": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
    "alice.eth": "0xcd2E72aEBe2A203b84f46DEEC948E6465dB51c75",
    "bob.eth": "0x34cd8a21E92b0Abd558Ff02D6cc7a9e12DAf0ff1",
    "erin.eth": "0xaCA0b936b0966A4f97c4E7Be7063D85F5E4fC5C2",
    "payroll.eth": "0x96619cD92Fe0c7dEaaEE1C8043a7FEA79CfA3E71",
    "devfund.eth": "0x6c3A6424EbfeE77215ec4DF1D25e88748D0c3696",
    "cold-storage.eth": "0x652F4BEA195D5e0089c02744b78F46b754Fdce78",
    "wallet2024.eth": "0x74cC457616fF7CBaFe4959BCCFAd21f5F2307664",
    "vault7.eth": "0x0D3f5a7A1Ee78E743e25C18e66942FcBcD84CcAD",
}

_HEX20 = re.compile(r"0x[0-9a-fA-F]{40}")


def _token(raw: str | None) -> tuple[int, str | None] | None:
    """`WalletTokenRegistry.token(matching:on:)`: empty means ETH; a 0x value matches
    a contract address case-insensitively; anything else matches a symbol."""
    value = (raw or "ETH").strip() or "ETH"
    if value.lower().startswith("0x"):
        for decimals, address in SEPOLIA_TOKENS.values():
            if address and address.lower() == value.lower():
                return decimals, address
        return None
    return SEPOLIA_TOKENS.get(value.upper())


def _amount_ok(raw: str | None, decimals: int) -> bool:
    """`EtherAmountParser.units(fromDecimalString:decimals:)`."""
    value = (raw or "").strip()
    if not value or value.lower() == "all":
        return False
    parts = value.split(".")
    if len(parts) > 2 or not all(p.isdigit() or p == "" for p in parts):
        return False
    frac = parts[1] if len(parts) == 2 else ""
    return set(frac[decimals:]) <= {"0"}


def why_not_executable(call: dict) -> str | None:
    """None if the wallet would build a UserOp from this gold call, else the reason."""
    tool = call.get("tool")
    if tool == "transfer":
        token = _token(call.get("token"))
        if token is None:
            return f"unsupported-token({call.get('token')})"
        if not _amount_ok(call.get("amount"), token[0]):
            return f"amount-parse-failed({call.get('amount')})"
        to = (call.get("to") or "").strip()
        if _HEX20.fullmatch(to):
            return None
        if "." not in to:
            return f"invalid-recipient({to})"
        if to.lower() not in RESOLVABLE_ENS:
            return f"ens-unresolvable({to})"
        return None
    if tool == "swap":
        if (call.get("amount_side") or "input").lower() != "input":
            return "unsupported-amount-side"
        src, dst = _token(call.get("from_token")), _token(call.get("to_token"))
        if src is None or dst is None:
            return f"unsupported-swap-token({call.get('from_token')}->{call.get('to_token')})"
        if src == dst:
            return "same-swap-token"
        if not _amount_ok(call.get("amount"), src[0]):
            return f"amount-parse-failed({call.get('amount')})"
        return None
    return f"tool-not-executable({tool})"


def case_is_executable(case: dict) -> bool:
    """A case is executable when every gold call is; empty gold (no call) always is."""
    return all(why_not_executable(c) is None for c in case["metadata"]["expected_calls"])
