"""Concatenate the app-contract and conversation datasets into one benchmark file.

`pf/tests.app-contract.yaml` (transfer/swap, including the arithmetic slice and
the refusal banks) and `pf/tests.conversations.yaml` (2-6 round conversations)
are each generated separately from their own seeds — see
scripts/generate_cases.py and scripts/generate_conversation_cases.py. This
script concatenates them, in that order, into `pf/tests.combined.yaml`.

Aave and Safe are deliberately NOT part of this benchmark. `pf/tests.protocols.yaml`
and its generator still exist and still pass their own integrity tests, but the
wallet ships no lending or multisig tool, so those cases scored a capability the
product does not expose and are no longer part of the number anyone quotes. To
run them, point the eval at that file directly:

    EVAL_DATASET=pf/tests.protocols.yaml scripts/eval.sh -o protocols.out.json

This script does not generate cases itself — it only concatenates two already
-generated files, asserting the concatenation is lossless (no dropped or
duplicated cases). Run: uv run python scripts/build_combined_benchmark.py
"""
from __future__ import annotations

import collections
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
APP_CONTRACT = ROOT / "pf" / "tests.app-contract.yaml"
CONVERSATIONS = ROOT / "pf" / "tests.conversations.yaml"
OUT = ROOT / "pf" / "tests.combined.yaml"


def _load(path: Path) -> list[dict]:
    return yaml.safe_load(path.read_text()) or []


def build_combined(app_contract: list[dict], conversations: list[dict]) -> list[dict]:
    """Concatenate app_contract then conversations; fail loudly on any id collision.

    De-duplicating silently would hide a real authoring bug (two generators
    minting the same id) behind a benchmark that looks fine but is quietly
    missing a case, so this raises instead.
    """
    combined = app_contract + conversations
    ids = [case["metadata"]["id"] for case in combined]
    seen: set[str] = set()
    dupes: set[str] = set()
    for id_ in ids:
        (dupes if id_ in seen else seen).add(id_)
    if dupes:
        raise ValueError(
            f"duplicate case id(s) across {APP_CONTRACT.name} and {CONVERSATIONS.name}: "
            f"{sorted(dupes)}"
        )
    assert len(combined) == len(app_contract) + len(conversations), (
        "combined count must equal the sum of its two parts — "
        f"got {len(combined)}, expected {len(app_contract) + len(conversations)}"
    )
    return combined


def _rounds(case: dict) -> int:
    """How many user turns this case's prompt carries (1 for a single-turn case)."""
    messages = case["vars"].get("messages")
    if not messages:
        return 1
    return sum(1 for m in messages if m.get("role") == "user")


def print_round_census(cases: list[dict]) -> None:
    counts = collections.Counter(_rounds(c) for c in cases)
    total = len(cases)
    multi = total - counts.get(1, 0)
    print("\nrounds  cases   share")
    for rounds in sorted(counts):
        print(f"{rounds:>6}  {counts[rounds]:>5}  {counts[rounds] / total:>6.1%}")
    print(f"{'total':>6}  {total:>5}   multi-turn {multi} ({multi / total:.1%})")


def main() -> None:
    app_contract = _load(APP_CONTRACT)
    conversations = _load(CONVERSATIONS)
    combined = build_combined(app_contract, conversations)

    header = (
        "# Generated combined benchmark — DO NOT EDIT BY HAND.\n"
        "# Produced by scripts/build_combined_benchmark.py by concatenating "
        f"{APP_CONTRACT.relative_to(ROOT)} (transfer/swap + the arithmetic slice + "
        f"refusals) then {CONVERSATIONS.relative_to(ROOT)} (2-6 round conversations).\n"
        "# Aave/Safe are intentionally excluded — the wallet ships no lending or "
        "multisig tool. Run pf/tests.protocols.yaml directly if you want them.\n"
        "# Regenerate the two source files first if either is stale; this script "
        "only concatenates them.\n"
    )
    OUT.write_text(header + yaml.safe_dump(combined, sort_keys=False, allow_unicode=True))
    print(f"app-contract:  {len(app_contract)} cases")
    print(f"conversations: {len(conversations)} cases")
    print_round_census(combined)
    print(f"\nWrote {len(combined)} cases -> {OUT}")


if __name__ == "__main__":
    main()
