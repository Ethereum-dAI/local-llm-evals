import json
import random

from pathlib import Path
from wallet_evals.promptfoo import load_cases
from wallet_evals.protocols import aave as aave_mod
from wallet_evals.protocols import safe as safe_mod
from wallet_evals.schema import ParsedTurn
from wallet_evals.scorer import score_case

# ============================================================================
# test_protocol_aave
# ============================================================================

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

# ============================================================================
# test_protocol_safe
# ============================================================================
#
# Suffixed on merge, to clear a name this file shared with an
# earlier section: FIXTURES -> FIXTURES_PROTOCOL_SAFE, test_fixtures_shape -> test_fixtures_shape_protocol_safe, test_build_cases_structure_and_gold -> test_build_cases_structure_and_gold_protocol_safe, test_build_cases_deterministic -> test_build_cases_deterministic_protocol_safe.

FIXTURES_PROTOCOL_SAFE = Path(__file__).resolve().parents[1] / "datasets" / "protocols" / "safe.fixtures.json"


def _fixtures_protocol_safe():
    return json.loads(FIXTURES_PROTOCOL_SAFE.read_text())


def test_fixtures_shape_protocol_safe():
    fx = _fixtures_protocol_safe()
    adds = [f for f in fx if f["op"] == "addOwnerWithThreshold"]
    removes = [f for f in fx if f["op"] == "removeOwner"]
    assert len(adds) == 5 and len(removes) == 5
    for f in fx:
        assert f["chainId"] == "1"
        assert f["safe"].startswith("0x") and len(f["safe"]) == 42
        assert f["params"]["threshold"] >= 1
        if f["op"] == "removeOwner":
            assert f["params"]["prevOwner"].startswith("0x")


def test_registry_lists_safe():
    from wallet_evals.protocols import PROTOCOL_MODULES
    names = {m.NAME for m in PROTOCOL_MODULES}
    assert "safe" in names
    for m in PROTOCOL_MODULES:
        assert hasattr(m, "FIXTURES") and hasattr(m, "build_cases")


def test_derive_prev_owner_head_is_sentinel():
    owners = ["0xAAd0000000000000000000000000000000000001",
              "0xBb00000000000000000000000000000000000002"]
    assert safe_mod.derive_prev_owner(owners, owners[0]) == safe_mod.SENTINEL_OWNERS
    assert safe_mod.derive_prev_owner(owners, owners[1]) == owners[0]


def test_derive_prev_owner_is_case_insensitive():
    owners = ["0xAbCd000000000000000000000000000000000001",
              "0xEf01000000000000000000000000000000000002"]
    assert safe_mod.derive_prev_owner(owners, owners[1].lower()) == owners[0]


def test_synth_owners_remove_is_self_consistent():
    for fx in (f for f in json.loads(safe_mod.FIXTURES.read_text())
               if f["op"] == "removeOwner"):
        ctx = safe_mod.account_context(fx)
        # post-removal owner count must still satisfy the threshold
        assert len(ctx["owners"]) - 1 >= fx["params"]["threshold"]
        assert safe_mod.derive_prev_owner(ctx["owners"], fx["params"]["owner"]) \
            == fx["params"]["prevOwner"]


def test_synth_owners_add_excludes_new_owner():
    fx = next(f for f in json.loads(safe_mod.FIXTURES.read_text())
              if f["op"] == "addOwnerWithThreshold")
    ctx = safe_mod.account_context(fx)
    assert fx["params"]["owner"].lower() not in {o.lower() for o in ctx["owners"]}
    assert ctx["threshold"] == max(1, fx["params"]["threshold"] - 1)


def test_gold_call_add():
    fx = {"op": "addOwnerWithThreshold", "chainId": "1", "safe": "0xSafe",
          "params": {"owner": "0xNewOwner", "threshold": 3}}
    assert safe_mod.gold_call(fx) == {
        "tool": "executeTx", "chainId": "1", "to": "0xSafe", "value": "0",
        "function": "addOwnerWithThreshold(address,uint256)",
        "args": ["0xNewOwner", "3"]}


def test_gold_call_remove():
    fx = {"op": "removeOwner", "chainId": "1", "safe": "0xSafe",
          "params": {"prevOwner": "0xPrev", "owner": "0xGone", "threshold": 2}}
    assert safe_mod.gold_call(fx) == {
        "tool": "executeTx", "chainId": "1", "to": "0xSafe", "value": "0",
        "function": "removeOwner(address,address,uint256)",
        "args": ["0xPrev", "0xGone", "2"]}


def test_build_cases_structure_and_gold_protocol_safe():
    fx = json.loads(safe_mod.FIXTURES.read_text())
    cases = safe_mod.build_cases(fx, random.Random(0))
    assert len(cases) >= len(fx)
    ids = [c["metadata"]["id"] for c in cases]
    assert len(ids) == len(set(ids))
    for c in cases:
        md = c["metadata"]
        assert md["protocol"] == "safe"
        assert md["style"] in {"direct", "narrative"}
        assert c["vars"]["account_context"]["safe"] == md["expected_calls"][0]["to"]
        assert "user_message" in c["vars"]
        fn = md["expected_calls"][0]["function"]
        assert fn in ("addOwnerWithThreshold(address,uint256)",
                      "removeOwner(address,address,uint256)")


def test_build_cases_deterministic_protocol_safe():
    fx = json.loads(safe_mod.FIXTURES.read_text())
    assert safe_mod.build_cases(fx, random.Random(7)) == safe_mod.build_cases(fx, random.Random(7))


def test_remove_surface_has_owner_address():
    fx = [f for f in json.loads(safe_mod.FIXTURES.read_text()) if f["op"] == "removeOwner"]
    cases = safe_mod.build_cases(fx, random.Random(1))
    for c in cases:
        owner = c["metadata"]["expected_calls"][0]["args"][1]
        assert owner.lower() in c["vars"]["user_message"].lower()


def test_generate_build_all_deterministic():
    from scripts.generate_protocol_cases import build_all
    assert build_all(random.Random(1)) == build_all(random.Random(1))

# ============================================================================
# test_protocol_integrity
# ============================================================================

PROTOCOLS = Path(__file__).resolve().parents[1] / "pf" / "tests.protocols.yaml"


def _load():
    return load_cases(PROTOCOLS)


def test_nonempty_and_unique_ids():
    cases = _load()
    assert len(cases) > 0
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))


def test_has_add_and_remove():
    cats = {c.category for c in _load()}
    assert "safe-add-signer" in cats and "safe-remove-signer" in cats


def test_every_protocol_gold_self_scores_one():
    for case in _load():
        turn = ParsedTurn(tool_calls=[c.as_parsed_call() for c in case.expected_calls])
        assert score_case(case, turn) == 1, f"{case.id} not self-consistent"


def test_has_aave_categories():
    cats = {c.category for c in _load()}
    assert {"aave-supply", "aave-withdraw", "aave-borrow", "aave-repay"} <= cats


def test_has_all_protocols():
    protos = {c.protocol for c in _load()}
    assert {"safe", "aave"} <= protos


def test_railgun_is_gone():
    """RAILGUN was removed from the app (local-wallet-mac#86, PR #87), so the
    protocol dataset must not resurrect shield/unshield."""
    cases = _load()
    assert not any(c.protocol == "railgun" for c in cases)
    assert not any("shield" in c.category for c in cases)


def test_no_protocol_case_carries_a_human_unit_amount():
    """The protocol datasets are base-unit executeTx only; `amount` is the
    app-mirrored human-unit field and belonged to the removed privacy tools."""
    for case in _load():
        for call in case.expected_calls:
            assert call.amount is None, \
                f"{case.id} sets a human-unit amount on a base-unit protocol call"
