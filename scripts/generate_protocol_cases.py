"""Generate pf/tests.protocols.yaml from frozen protocol fixtures — deterministically.

Iterates the protocol registry, runs each module's build_cases over its fixtures,
and writes a promptfoo-native tests file. Run: uv run python scripts/generate_protocol_cases.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import yaml

from wallet_evals.protocols import PROTOCOL_MODULES

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "pf" / "tests.protocols.yaml"
SEED = 20260611


def build_all(rng: random.Random) -> list[dict]:
    cases: list[dict] = []
    for module in PROTOCOL_MODULES:
        fixtures = json.loads(module.FIXTURES.read_text())
        cases.extend(module.build_cases(fixtures, rng, start_idx=1))
    return cases


def main() -> None:
    rng = random.Random(SEED)
    cases = build_all(rng)
    by_protocol: dict[str, int] = {}
    for c in cases:
        by_protocol[c["metadata"]["protocol"]] = by_protocol.get(c["metadata"]["protocol"], 0) + 1
    header = (
        "# Generated protocol-transaction eval cases — DO NOT EDIT BY HAND.\n"
        "# Produced by scripts/generate_protocol_cases.py from datasets/protocols/*.fixtures.json "
        f"(seed {SEED}).\n"
        "# Gold is a generic executeTx computed from the decoded real transactions.\n"
    )
    OUT.write_text(header + yaml.safe_dump(cases, sort_keys=False, allow_unicode=True))
    print(f"protocols: {by_protocol}")
    print(f"Wrote {len(cases)} cases -> {OUT}")


if __name__ == "__main__":
    main()
