"""Safe owner-management protocol module: fixtures -> executeTx eval cases.

Gold is a generic executeTx calling the Safe's OwnerManager functions. The
owners list given to the model is synthesized deterministically and is
self-consistent (prevOwner immediately precedes owner for removeOwner).
"""
from __future__ import annotations

import random
from pathlib import Path

from wallet_evals.generation import apply_mutators

NAME = "safe"
FIXTURES = Path(__file__).resolve().parents[3] / "datasets" / "protocols" / "safe.fixtures.json"

SENTINEL_OWNERS = "0x0000000000000000000000000000000000000001"

# Deterministic padding owners (real checksummed addresses from the fixtures) used
# to pad a synthesized owners list up to the threshold. Never the op's own owner.
PAD_OWNERS = [
    "0x8DbC9ba71C340f00518f29FaD98F5e9F7Fd1D5Cb",
    "0xc011F5503d270C89F358468b6Dee9b51adeb5413",
    "0xE31CB422D3c7b65ecbAE7131a4e9d0Cd369D5f86",
    "0x00C3aed1d4Da64f5A822eDA97321BBBaE6B9974D",
]


def derive_prev_owner(owners: list[str], owner: str) -> str:
    """The owner listed immediately before `owner` (sentinel if it is the head)."""
    low = [o.lower() for o in owners]
    i = low.index(owner.lower())  # ValueError if owner absent — an authoring bug
    return SENTINEL_OWNERS if i == 0 else owners[i - 1]


def synth_owners(fx: dict) -> list[str]:
    """A deterministic owners list consistent with the op.

    removeOwner: [prevOwner, owner, *pad] so prevOwner precedes owner, padded to
    reach the threshold. addOwnerWithThreshold: existing owners (excluding the new
    owner), sized to a plausible pre-add set.
    """
    p = fx["params"]
    if fx["op"] == "removeOwner":
        assert p["prevOwner"] != SENTINEL_OWNERS, (
            "prevOwner is the sentinel (owner is first); synth_owners cannot "
            "represent this — add such fixtures with an explicit owners list")
    th = p["threshold"]
    if fx["op"] == "removeOwner":
        base = [p["prevOwner"], p["owner"]]
        # Pre-removal list needs threshold+1 owners so the post-removal Safe
        # (one fewer owner) still satisfies the threshold — otherwise the
        # scenario shown to the model is an on-chain-invalid state.
        need = max(th + 1 - len(base), 0)
    else:  # addOwnerWithThreshold
        base = []
        need = max(th - 1, 1)
    used = {a.lower() for a in base} | {p.get("owner", "").lower()}
    pad = [a for a in PAD_OWNERS if a.lower() not in used]
    return base + pad[:need]


def account_context(fx: dict) -> dict:
    """The model-facing account state: Safe address, ordered owners, threshold."""
    p = fx["params"]
    threshold = p["threshold"] if fx["op"] == "removeOwner" else max(1, p["threshold"] - 1)
    return {"safe": fx["safe"], "owners": synth_owners(fx), "threshold": threshold}


def gold_call(fx: dict) -> dict:
    """Generic executeTx gold for a Safe owner op (a self-call: to = the Safe)."""
    safe = fx["safe"]
    threshold = str(fx["params"]["threshold"])
    base = {"tool": "executeTx", "chainId": fx["chainId"], "to": safe, "value": "0"}
    if fx["op"] == "addOwnerWithThreshold":
        return {**base, "function": "addOwnerWithThreshold(address,uint256)",
                "args": [fx["params"]["owner"], threshold]}
    if fx["op"] == "removeOwner":
        return {**base, "function": "removeOwner(address,address,uint256)",
                "args": [fx["params"]["prevOwner"], fx["params"]["owner"], threshold]}
    raise ValueError(f"unknown Safe op: {fx['op']!r}")


ADD_TEMPLATES = [
    "Add {owner} as a signer on my Safe and set the threshold to {threshold}.",
    "Add owner {owner} to the Safe and require {threshold} confirmations.",
    "/addowner {owner} threshold {threshold}",
]
ADD_NARRATIVE = [
    "We've got a new teammate — please add {owner} to the Safe and bump confirmations to {threshold}.",
    "{owner} is joining as a co-signer; add them and set the approval threshold to {threshold}.",
]
REMOVE_TEMPLATES = [
    "Remove signer {owner} from my Safe.",
    "Take {owner} off the Safe owners.",
    "/removeowner {owner}",
]
REMOVE_NARRATIVE = [
    "{owner} has left the team — please take them off the Safe.",
    "We're offboarding {owner}; remove them from the Safe signers.",
]

_TEMPLATES = {
    "addOwnerWithThreshold": [("direct", ADD_TEMPLATES), ("narrative", ADD_NARRATIVE)],
    "removeOwner": [("direct", REMOVE_TEMPLATES), ("narrative", REMOVE_NARRATIVE)],
}
_CATEGORY = {"addOwnerWithThreshold": "safe-add-signer", "removeOwner": "safe-remove-signer"}


def _fill(template: str, fx: dict) -> str:
    p = fx["params"]
    return template.format(owner=p["owner"], threshold=p["threshold"])


def build_cases(fixtures: list[dict], rng: random.Random, start_idx: int = 1) -> list[dict]:
    """One case per (fixture, template); gold computed; account_context attached."""
    cases: list[dict] = []
    idx = start_idx
    for fx in fixtures:
        gold = gold_call(fx)
        ctx = account_context(fx)
        for style, templates in _TEMPLATES[fx["op"]]:
            for template in templates:
                surface, labels = apply_mutators(_fill(template, fx), rng)
                short = "add" if fx["op"] == "addOwnerWithThreshold" else "remove"
                cases.append({
                    "vars": {"user_message": surface, "account_context": ctx},
                    "metadata": {
                        "id": f"safe-{short}-{idx:04d}",
                        "source": "generated-protocol",
                        "protocol": "safe",
                        "language": "english",
                        "category": _CATEGORY[fx["op"]],
                        "difficulty": "hard",
                        "level": "payload",
                        "query_type": "one_shot",
                        "requires": ["safe_owner_state"],
                        "style": style,
                        "mutators": labels,
                        "expected_calls": [gold],
                        "notes": None,
                    },
                })
                idx += 1
    return cases
