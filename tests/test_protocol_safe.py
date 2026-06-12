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
