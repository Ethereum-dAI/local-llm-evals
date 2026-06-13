import json
import random
from pathlib import Path

from wallet_evals.protocols import aave as aave_mod

FIXTURES = Path(__file__).resolve().parents[1] / "datasets" / "protocols" / "aave.fixtures.json"


def _fixtures():
    return json.loads(FIXTURES.read_text())


def test_fixtures_shape():
    fx = _fixtures()
    counts = {}
    for f in fx:
        counts[f["op"]] = counts.get(f["op"], 0) + 1
        assert f["chainId"] == "1"
        assert f["asset"] in aave_mod.ASSETS
        assert int(f["amount"]) > 0
        if f["op"] in ("borrow", "repay"):
            assert f["rate_mode"] == 2
    assert counts == {"supply": 5, "withdraw": 4, "borrow": 5, "repay": 4}


def test_gold_call_supply():
    fx = {"op": "supply", "chainId": "1", "asset": "USDC", "amount": "3000000"}
    assert aave_mod.gold_call(fx) == {
        "tool": "executeTx", "chainId": "1",
        "to": "0x87870Bca3F3fD6335C3F4ce8392D69350B4fa4E2", "value": "0",
        "function": "supply(address,uint256,address,uint16)",
        "args": ["0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", "3000000", "<wallet>", "0"]}


def test_gold_call_withdraw():
    fx = {"op": "withdraw", "chainId": "1", "asset": "WETH", "amount": "4000000000000000000"}
    g = aave_mod.gold_call(fx)
    assert g["function"] == "withdraw(address,uint256,address)"
    assert g["args"] == ["0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", "4000000000000000000", "<wallet>"]


def test_gold_call_borrow_uses_rate_mode():
    fx = {"op": "borrow", "chainId": "1", "asset": "DAI", "amount": "5000000000000000000000", "rate_mode": 2}
    g = aave_mod.gold_call(fx)
    assert g["function"] == "borrow(address,uint256,uint256,uint16,address)"
    assert g["args"] == ["0x6B175474E89094C44Da98b954EedeAC495271d0F",
                         "5000000000000000000000", "2", "0", "<wallet>"]


def test_gold_call_repay():
    fx = {"op": "repay", "chainId": "1", "asset": "USDT", "amount": "50000000", "rate_mode": 2}
    g = aave_mod.gold_call(fx)
    assert g["function"] == "repay(address,uint256,uint256,address)"
    assert g["args"] == ["0xdAC17F958D2ee523a2206206994597C13D831ec7", "50000000", "2", "<wallet>"]


def test_build_cases_structure_and_gold():
    fx = _fixtures()
    cases = aave_mod.build_cases(fx, random.Random(0))
    assert len(cases) >= len(fx)
    ids = [c["metadata"]["id"] for c in cases]
    assert len(ids) == len(set(ids))
    for c in cases:
        md = c["metadata"]
        assert md["protocol"] == "aave"
        assert c["vars"]["protocol"] == "aave"
        assert md["style"] in {"direct", "narrative"}
        assert "user_message" in c["vars"]
        assert "account_context" not in c["vars"]
        assert c["metadata"]["expected_calls"][0]["to"] == aave_mod.POOL
        assert "expected_summary" in c["vars"]


def test_build_cases_surface_has_asset():
    fx = [f for f in _fixtures() if f["op"] == "supply"]
    for c in aave_mod.build_cases(fx, random.Random(2)):
        um = c["vars"]["user_message"].lower()
        assert any(sym.lower() in um for sym in ("usdc", "weth", "dai", "wbtc", "usdt"))


def test_build_cases_deterministic():
    fx = _fixtures()
    assert aave_mod.build_cases(fx, random.Random(7)) == aave_mod.build_cases(fx, random.Random(7))
