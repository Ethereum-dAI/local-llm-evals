"""Generate pf/tests.conversations.yaml from datasets/seeds.conversations.yaml.

The multi-round slice: 2-6 round conversations across the four mechanisms in
src/wallet_evals/conversations.py. ROUND_PLAN below fixes exactly how many cases
each (rounds, mechanism) pair contributes, so the round distribution is a
declared property of the dataset rather than a by-product of pool sizes.

This slice is a SEPARATE file with its own seeds and its own
`random.Random(SEED)` for the same reason the arithmetic slice is: appending it
must never reshuffle, renumber or resize pf/tests.app-contract.yaml, whose 307
base cases are byte-identical to the frozen base-unit dataset and carry every
published score. scripts/build_combined_benchmark.py concatenates the two.

Run: uv run python scripts/generate_conversation_cases.py
"""
from __future__ import annotations

import argparse
import collections
import itertools
import random
from pathlib import Path

import yaml

from wallet_evals.conversations import (
    ACTION_FIELDS, MECHANISM_ROUNDS, build_correction_case, build_distractor_case,
    build_progressive_case, build_switch_case,
)
from wallet_evals.generation import expand_vary

ROOT = Path(__file__).resolve().parent.parent
SEEDS = ROOT / "datasets" / "seeds.conversations.yaml"
OUT = ROOT / "pf" / "tests.conversations.yaml"
SEED = 20260817

#: How many cases each (rounds, mechanism) pair contributes. The shape is
#: deliberately decaying in the round count: the long conversations are the
#: expensive ones to run and the most likely to be dominated by a single failure
#: mode, so the mass sits at 2-4 rounds while 5- and 6-round cases still carry
#: enough weight (131 cases) to move the score if a model degrades on length.
#:
#: `progressive` stops at 4 rounds (only three fields to disclose) and
#: `distractor` starts at 3 (a 2-round distractor case has no distractor in it),
#: which is why those buckets have three and four entries rather than five.
ROUND_PLAN: dict[int, dict[str, int]] = {
    2: {"progressive": 60, "correction": 60, "switch": 60},
    3: {"progressive": 38, "correction": 38, "distractor": 37, "switch": 37},
    4: {"progressive": 28, "correction": 28, "distractor": 27, "switch": 27},
    5: {"correction": 27, "distractor": 27, "switch": 26},
    6: {"correction": 17, "distractor": 17, "switch": 17},
}

TARGET_TOTAL = sum(sum(m.values()) for m in ROUND_PLAN.values())


def _valid_intent(intent: dict) -> bool:
    """Drop swaps whose from/to token are identical (no-op swap) — same rule as
    scripts/generate_cases.py."""
    if intent["action"] == "swap":
        return intent["from_token"] != intent["to_token"]
    return True


def load_intents(seeds_path: Path, rng: random.Random) -> list[dict]:
    seeds = yaml.safe_load(seeds_path.read_text())
    intents = [i for seed in seeds for i in expand_vary(seed, rng) if _valid_intent(i)]
    if not intents:
        raise ValueError(f"{seeds_path} expanded to no valid intents")
    return intents


def _pool(mechanism: str, rounds: int, intents: list[dict],
          rng: random.Random, counter: itertools.count) -> list[dict]:
    """Every candidate case for one (rounds, mechanism) pair, before capping.

    The pool is enumerated over the structural variants that make a mechanism
    distinct — the disclosure permutation for `progressive`, which field is
    withheld for `correction`/`distractor`, which replacement intent is
    substituted for `switch` — so capping picks between genuinely different
    conversations rather than between reruns of one shape.
    """
    cases: list[dict] = []
    for intent in intents:
        fields = ACTION_FIELDS[intent["action"]]
        if mechanism == "progressive":
            for order in itertools.permutations(fields):
                cases.append(build_progressive_case(intent, order, rounds, rng,
                                                    next(counter)))
        elif mechanism == "correction":
            for withheld in fields:
                cases.append(build_correction_case(intent, withheld, rounds, rng,
                                                   next(counter)))
        elif mechanism == "distractor":
            for withheld in fields:
                cases.append(build_distractor_case(intent, withheld, rounds, rng,
                                                   next(counter)))
        elif mechanism == "switch":
            # Three replacement intents per opener, drawn from the same pool but
            # never the opener itself — an "abandon it for the identical request"
            # case would have no wrong answer to catch.
            others = [o for o in intents if o is not intent]
            for second in rng.sample(others, 3):
                cases.append(build_switch_case(intent, second, rounds, rng,
                                               next(counter)))
        else:  # pragma: no cover - guarded by ROUND_PLAN's keys
            raise ValueError(f"unknown mechanism: {mechanism!r}")
    return cases


def build_selection(intents: list[dict], rng: random.Random,
                    plan: dict[int, dict[str, int]] = ROUND_PLAN) -> list[dict]:
    """Generate, shuffle and cap each (rounds, mechanism) bucket to its target."""
    selected: list[dict] = []
    counter = itertools.count(1)
    for rounds in sorted(plan):
        for mechanism in sorted(plan[rounds]):
            target = plan[rounds][mechanism]
            if rounds not in MECHANISM_ROUNDS[mechanism]:
                raise ValueError(
                    f"ROUND_PLAN asks for {mechanism} at {rounds} rounds, which it "
                    f"cannot produce (supports {MECHANISM_ROUNDS[mechanism]})")
            pool = _pool(mechanism, rounds, intents, rng, counter)
            if len(pool) < target:
                raise ValueError(
                    f"{mechanism} @ {rounds} rounds: pool of {len(pool)} cannot fill "
                    f"a target of {target} — widen datasets/seeds.conversations.yaml")
            rng.shuffle(pool)
            kept = pool[:target]
            print(f"{rounds} rounds / {mechanism:12s}: generated {len(pool):4d}, "
                  f"kept {len(kept):3d}, dropped {len(pool) - len(kept):4d}")
            selected.extend(kept)
    return selected


def print_round_distribution(cases: list[dict], label: str = "conversation slice") -> None:
    """The (rounds x mechanism) cross-tab, plus the round totals."""
    by_round = collections.Counter(c["metadata"]["rounds"] for c in cases)
    by_pair = collections.Counter(
        (c["metadata"]["rounds"], c["metadata"]["mechanism"]) for c in cases)
    mechanisms = sorted({m for _, m in by_pair})

    print(f"\n{label}: round distribution ({len(cases)} cases)")
    header = f"{'rounds':>6}  " + "".join(f"{m:>13}" for m in mechanisms) + f"{'total':>9}"
    print(header)
    print("-" * len(header))
    for rounds in sorted(by_round):
        row = f"{rounds:>6}  " + "".join(
            f"{by_pair.get((rounds, m), 0) or '-':>13}" for m in mechanisms)
        print(f"{row}{by_round[rounds]:>9}")
    print("-" * len(header))
    totals = collections.Counter(c["metadata"]["mechanism"] for c in cases)
    print(f"{'total':>6}  " + "".join(f"{totals[m]:>13}" for m in mechanisms)
          + f"{len(cases):>9}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT,
                        help=f"destination YAML (default: {OUT.relative_to(ROOT)})")
    parser.add_argument("--seeds", type=Path, default=SEEDS,
                        help=f"seed intents (default: {SEEDS.relative_to(ROOT)})")
    args = parser.parse_args()

    rng = random.Random(SEED)
    intents = load_intents(args.seeds, rng)
    print(f"{len(intents)} seed intents from {args.seeds.name}")

    selected = build_selection(intents, rng)
    assert len(selected) == TARGET_TOTAL, \
        f"selected {len(selected)} cases, ROUND_PLAN declares {TARGET_TOTAL}"
    print_round_distribution(selected)

    header = "\n".join([
        "# Generated multi-round conversation cases — DO NOT EDIT BY HAND.",
        "# Produced by scripts/generate_conversation_cases.py from "
        f"{args.seeds.relative_to(ROOT) if not args.seeds.is_absolute() else args.seeds.name}"
        f" (seed {SEED}).",
        "# 2-6 rounds per case; a round is one user turn plus the assistant's reply, "
        "and only",
        "# the model's reply to the LAST user turn is scored. Gold is computed from "
        "the final",
        "# effective intent, so a conversation may revise a value freely.",
    ]) + "\n"
    args.out.write_text(
        header + yaml.safe_dump(selected, sort_keys=False, allow_unicode=True))
    print(f"\nWrote {len(selected)} cases -> {args.out}")


if __name__ == "__main__":
    main()
