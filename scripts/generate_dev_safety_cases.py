"""Generate pf/tests.dev.safety.yaml — the dev set's missing safety coverage.

A SEPARATE file rather than an append to pf/tests.dev.yaml, on purpose: the 145-case
dev numbers already recorded in results/ stay directly comparable, and a safety
regression is a different question from an accuracy regression anyway. Score both.

Run:
    uv run python scripts/generate_dev_safety_cases.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wallet_evals.dev_safety import DEV_REFUSAL_SCENARIOS  # noqa: E402
from wallet_evals.generation import build_refusal_case  # noqa: E402

#: Its own seed, so adding a case here can never reshuffle the dev set's selections.
SEED_DEV_SAFETY = 20260819
OUT = ROOT / "pf" / "tests.dev.safety.yaml"


def main() -> None:
    rng = random.Random(SEED_DEV_SAFETY)
    cases = []
    idx = 0
    for scenario in DEV_REFUSAL_SCENARIOS:
        for template in scenario["templates"]:
            idx += 1
            case = build_refusal_case(scenario, template, rng, idx)
            # Re-id into the dev namespace so these can never be confused with the
            # benchmark's `gen-refusal-*` cases in a report or a merged run.
            case["metadata"]["id"] = f"devsafety-{scenario['kind']}-{idx:04d}"
            case["metadata"]["source"] = "generated-dev-safety"
            cases.append(case)

    OUT.write_text(yaml.safe_dump(cases, sort_keys=False, width=10**6,
                                  allow_unicode=True))
    kinds = sorted({c["metadata"]["category"] for c in cases})
    print(f"wrote {len(cases)} cases -> {OUT}")
    print(f"{len(kinds)} kinds: " + ", ".join(k.replace('safety-refusal-', '')
                                              for k in kinds))


if __name__ == "__main__":
    main()
