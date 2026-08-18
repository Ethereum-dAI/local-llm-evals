"""Generate pf/tests.dev.yaml — the held-out set that GATES checkpoint selection.

Not the test set. `pf/tests.combined.yaml` is frozen and reports the final number;
this one picks which checkpoint gets there. Keeping them separate is the point:
selecting on the test set is how a model comes to look better than it is, which is
exactly the failure this dev set exists to catch.

Deliberately OUT OF DISTRIBUTION relative to training:

  * conversations run 2-6 rounds, and all the memory mechanisms are 3+ — training
    is 0% three-plus, so an in-distribution
    holdout would show eval_loss falling while 6-round accuracy halved;
  * ENS names come from a bank disjoint from both the training name and the test
    bank, so "handles ENS" cannot be satisfied by recall;
  * `exact_output` and `token_address` are included, and training has none of
    either.

Uses its OWN random.Random(SEED_DEV), so regenerating it can never perturb the
frozen benchmark's selection.

Run: uv run python scripts/generate_dev_cases.py
"""
from __future__ import annotations

import argparse
import collections
import itertools
import random
from pathlib import Path

import yaml

from wallet_evals.conversations import (
    ACTION_FIELDS, MECHANISM_ROUNDS, TOKEN_ADDRESSES, build_correction_case,
    build_distractor_case, build_exact_output_case, build_progressive_case,
    build_switch_case, build_token_address_case,
)
from wallet_evals.generation import expand_vary

ROOT = Path(__file__).resolve().parent.parent
SEEDS = ROOT / "datasets" / "seeds.dev.yaml"
OUT = ROOT / "pf" / "tests.dev.yaml"
#: Distinct from every other generator seed in the repo.
SEED_DEV = 20260818

#: ENS names for DEV gold, disjoint from generation.ENS_NAMES (the TEST bank) and
#: from vitalik.eth (the TRAIN bank). Read off the seed file so the two cannot
#: drift: the seeds decide which names appear in the OPENING turn, and this bank
#: decides which appear after a `correction` revises the recipient. Both land in
#: gold, so both have to be held out — a revision drawing on the test bank is the
#: leak test_dev_set.py caught.
DEV_ENS_NAMES: tuple[str, ...] = tuple(sorted(
    str(v)
    for seed in yaml.safe_load(SEEDS.read_text())
    for v in (seed.get("recipient", {}).get("vary", [])
              if isinstance(seed.get("recipient"), dict) else [])
    if str(v).endswith(".eth")))

#: How many cases per (rounds, mechanism). Weighted towards the failure modes the
#: v4 fine-tune actually lost — depth, corrections and interruptions — plus the two
#: contract-boundary axes it scored 0/32 and 25% on. ~150 cases keeps a per-epoch
#: scored eval cheap enough to run on every checkpoint.
ROUND_PLAN: dict[int, dict[str, int]] = {
    # 2 rounds appears ONLY for the two contract-boundary mechanisms. Their round
    # support is capped (token_address at 3, exact_output at 4), so restricting the
    # whole set to 3+ left token_address with 5 cases — far too coarse to gate a
    # capability on, at +-20% per case. These two are the axes training has zero of
    # and the v4 fine-tune scored 0/32 and 25% on, so they need enough n to move a
    # selection decision. A 2-round case is in-distribution for training's DEPTH but
    # still out-of-distribution on the axis being measured, which is the point.
    2: {"token_address": 10, "exact_output": 5},
    3: {"correction": 10, "distractor": 10, "progressive": 6, "switch": 6,
        "exact_output": 5, "token_address": 5},
    4: {"correction": 10, "distractor": 10, "progressive": 6, "switch": 6,
        "exact_output": 5},
    5: {"correction": 10, "distractor": 10, "switch": 6},
    6: {"correction": 10, "distractor": 10, "switch": 5},
}
TARGET_TOTAL = sum(sum(m.values()) for m in ROUND_PLAN.values())


def _valid_intent(intent: dict) -> bool:
    if intent["action"] == "swap":
        return intent["from_token"] != intent["to_token"]
    return True


def load_intents(path: Path, rng: random.Random) -> list[dict]:
    seeds = yaml.safe_load(path.read_text())
    intents = [i for s in seeds for i in expand_vary(s, rng) if _valid_intent(i)]
    if not intents:
        raise ValueError(f"{path} expanded to no valid intents")
    return intents


def _pool(mechanism: str, rounds: int, intents: list[dict], rng: random.Random,
          counter: itertools.count) -> list[dict]:
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
                                                   next(counter),
                                                   ens_bank=DEV_ENS_NAMES))
        elif mechanism == "distractor":
            for withheld in fields:
                cases.append(build_distractor_case(intent, withheld, rounds, rng,
                                                   next(counter)))
        elif mechanism == "switch":
            others = [o for o in intents if o is not intent]
            for second in rng.sample(others, min(3, len(others))):
                cases.append(build_switch_case(intent, second, rounds, rng,
                                               next(counter)))
        elif mechanism == "exact_output":
            if intent["action"] == "swap":
                cases.append(build_exact_output_case(intent, rounds, rng,
                                                     next(counter)))
        elif mechanism == "token_address":
            for field in fields:
                if field != "amount" and intent[field] in TOKEN_ADDRESSES:
                    cases.append(build_token_address_case(intent, field, rounds,
                                                          rng, next(counter)))
        else:  # pragma: no cover
            raise ValueError(f"unknown mechanism: {mechanism!r}")
    return cases


def build_selection(intents: list[dict], rng: random.Random,
                    plan: dict[int, dict[str, int]] = ROUND_PLAN) -> list[dict]:
    selected: list[dict] = []
    counter = itertools.count(1)
    for rounds in sorted(plan):
        for mechanism in sorted(plan[rounds]):
            target = plan[rounds][mechanism]
            if rounds not in MECHANISM_ROUNDS[mechanism]:
                raise ValueError(
                    f"{mechanism} cannot produce {rounds} rounds "
                    f"(supports {MECHANISM_ROUNDS[mechanism]})")
            pool = _pool(mechanism, rounds, intents, rng, counter)
            if len(pool) < target:
                raise ValueError(
                    f"{mechanism} @ {rounds}r: pool of {len(pool)} < target {target} "
                    f"— widen datasets/seeds.dev.yaml")
            rng.shuffle(pool)
            selected.extend(pool[:target])
    # Ids collide with the benchmark's (both generators count from 1), so re-stamp
    # with a `dev-` prefix. Without it a merged view would silently overwrite cases.
    for case in selected:
        md = case["metadata"]
        md["id"] = f"dev-{md['id']}"
    return selected


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--seeds", type=Path, default=SEEDS)
    args = ap.parse_args()

    rng = random.Random(SEED_DEV)
    intents = load_intents(args.seeds, rng)
    print(f"{len(intents)} dev seed intents from {args.seeds.name}")
    selected = build_selection(intents, rng)
    assert len(selected) == TARGET_TOTAL, f"{len(selected)} != {TARGET_TOTAL}"

    by_round = collections.Counter(c["metadata"]["rounds"] for c in selected)
    by_mech = collections.Counter(c["metadata"]["mechanism"] for c in selected)
    print(f"\ndev set: {len(selected)} cases")
    print("  rounds:    " + "  ".join(f"{k}:{v}" for k, v in sorted(by_round.items())))
    print("  mechanism: " + "  ".join(f"{k}:{v}" for k, v in sorted(by_mech.items())))
    nocall = sum(1 for c in selected if not c["metadata"]["expected_calls"])
    print(f"  no-call:   {nocall} ({nocall/len(selected):.1%})")

    header = "\n".join([
        "# DEV set — gates checkpoint selection. NOT the test set, NOT training.",
        "# Generated by scripts/generate_dev_cases.py from datasets/seeds.dev.yaml "
        f"(seed {SEED_DEV}).",
        "# Disjoint from pf/tests.combined.yaml (frozen test) and from the training",
        "# rows, by surface AND by case id — asserted in tests/test_dev_set.py.",
        "# Conversations run 2-6 rounds: training is 0% three-plus, so an in-distribution",
        "# holdout cannot see the depth collapse this set exists to detect.",
    ]) + "\n"
    args.out.write_text(header + yaml.safe_dump(selected, sort_keys=False,
                                                allow_unicode=True))
    print(f"\nWrote {len(selected)} cases -> {args.out}")


if __name__ == "__main__":
    main()
