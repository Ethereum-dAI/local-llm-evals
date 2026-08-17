"""Concatenate the app-contract and protocol datasets into one benchmark file.

`pf/tests.app-contract.yaml` (transfer/swap, including the arithmetic slice)
and `pf/tests.protocols.yaml` (aave/safe) are each generated
separately from their own seeds/fixtures — see scripts/generate_cases.py and
scripts/generate_protocol_cases.py. Neither one alone exercises every tool the
app ships, so this script concatenates them, in that order, into
`pf/tests.combined.yaml` for a single full-coverage benchmark run.

This script does not generate cases itself — it only concatenates two already
-generated files, asserting the concatenation is lossless (no dropped or
duplicated cases). Run: uv run python scripts/build_combined_benchmark.py
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
APP_CONTRACT = ROOT / "pf" / "tests.app-contract.yaml"
PROTOCOLS = ROOT / "pf" / "tests.protocols.yaml"
OUT = ROOT / "pf" / "tests.combined.yaml"


def _load(path: Path) -> list[dict]:
    return yaml.safe_load(path.read_text()) or []


def build_combined(app_contract: list[dict], protocols: list[dict]) -> list[dict]:
    """Concatenate app_contract then protocols; fail loudly on any id collision.

    De-duplicating silently would hide a real authoring bug (two generators
    minting the same id) behind a benchmark that looks fine but is quietly
    missing a case, so this raises instead.
    """
    combined = app_contract + protocols
    ids = [case["metadata"]["id"] for case in combined]
    seen: set[str] = set()
    dupes: set[str] = set()
    for id_ in ids:
        (dupes if id_ in seen else seen).add(id_)
    if dupes:
        raise ValueError(
            f"duplicate case id(s) across {APP_CONTRACT.name} and {PROTOCOLS.name}: "
            f"{sorted(dupes)}"
        )
    assert len(combined) == len(app_contract) + len(protocols), (
        "combined count must equal the sum of its two parts — "
        f"got {len(combined)}, expected {len(app_contract) + len(protocols)}"
    )
    return combined


def main() -> None:
    app_contract = _load(APP_CONTRACT)
    protocols = _load(PROTOCOLS)
    combined = build_combined(app_contract, protocols)

    header = (
        "# Generated combined benchmark — DO NOT EDIT BY HAND.\n"
        "# Produced by scripts/build_combined_benchmark.py by concatenating "
        f"{APP_CONTRACT.relative_to(ROOT)} (transfer/swap + the arithmetic slice) "
        f"then {PROTOCOLS.relative_to(ROOT)} (aave/safe).\n"
        "# Regenerate the two source files first if either is stale; this script "
        "only concatenates them.\n"
    )
    OUT.write_text(header + yaml.safe_dump(combined, sort_keys=False, allow_unicode=True))
    print(f"app-contract: {len(app_contract)} cases")
    print(f"protocols:    {len(protocols)} cases")
    print(f"Wrote {len(combined)} cases -> {OUT}")


if __name__ == "__main__":
    main()
