import json
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "datasets" / "protocols" / "safe.fixtures.json"


def _fixtures():
    return json.loads(FIXTURES.read_text())


def test_fixtures_shape():
    fx = _fixtures()
    adds = [f for f in fx if f["op"] == "addOwnerWithThreshold"]
    removes = [f for f in fx if f["op"] == "removeOwner"]
    assert len(adds) == 5 and len(removes) == 5
    for f in fx:
        assert f["chainId"] == "1"
        assert f["safe"].startswith("0x") and len(f["safe"]) == 42
        assert f["params"]["threshold"] >= 1
        if f["op"] == "removeOwner":
            assert f["params"]["prevOwner"].startswith("0x")


import random
from wallet_evals.protocols import safe as safe_mod


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
        assert len(ctx["owners"]) >= fx["params"]["threshold"]
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


def test_build_cases_structure_and_gold():
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


def test_build_cases_deterministic():
    fx = json.loads(safe_mod.FIXTURES.read_text())
    assert safe_mod.build_cases(fx, random.Random(7)) == safe_mod.build_cases(fx, random.Random(7))


def test_remove_surface_has_owner_address():
    fx = [f for f in json.loads(safe_mod.FIXTURES.read_text()) if f["op"] == "removeOwner"]
    cases = safe_mod.build_cases(fx, random.Random(1))
    for c in cases:
        owner = c["metadata"]["expected_calls"][0]["args"][1]
        assert owner.lower() in c["vars"]["user_message"].lower()
