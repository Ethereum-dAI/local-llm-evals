"""Extract the safety-refusal slice of the combined benchmark into its own file.

The refusal slice is the only place both fine-tunes lost to the untuned base, so
it gets iterated on directly: 49 cases run in ~2 minutes per model against ~25
for the full 569, which makes a train/measure loop practical.

Derived, never hand-edited — run after scripts/build_combined_benchmark.py.
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "pf" / "tests.combined.yaml"
OUT = ROOT / "pf" / "tests.refusals.yaml"

# Kinds present in the TRAINING refusal bank. The rest are the held-out
# generalization probe (see test_held_out_refusal_kinds_never_enter_training).
TRAINED_KINDS = {
    "burn-send", "zero-send", "approve-unknown-spender", "unverified-token-swap",
    "unlimited-approval", "seed-phrase-exfiltration", "private-key-exfiltration",
    "malformed-address", "wrong-chain-address", "negative-amount",
    "prompt-injection", "impersonation-scam",
}


def main() -> None:
    cases = yaml.safe_load(SRC.read_text())
    refusals = [c for c in cases
                if c["metadata"]["category"].startswith("safety-refusal-")]
    trained, held_out = [], []
    for c in refusals:
        kind = c["metadata"]["category"].replace("safety-refusal-", "")
        (trained if kind in TRAINED_KINDS else held_out).append(kind)

    header = (
        "# The safety-refusal slice of pf/tests.combined.yaml — DO NOT EDIT BY HAND.\n"
        "# Produced by scripts/build_refusal_subset.py.\n"
        f"# {len(trained)} cases on kinds the fine-tune trains on, {len(held_out)} on "
        "held-out kinds it has never seen.\n"
    )
    OUT.write_text(header + yaml.safe_dump(refusals, sort_keys=False, allow_unicode=True))
    print(f"trained-kind cases:  {len(trained)}")
    print(f"held-out-kind cases: {len(held_out)}  ({sorted(set(held_out))})")
    print(f"Wrote {len(refusals)} cases -> {OUT}")


if __name__ == "__main__":
    main()
