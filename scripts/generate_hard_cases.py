"""Generate pf/tests.hard.yaml from datasets/seeds.hard.yaml.

The hard slice: six mechanisms in src/wallet_evals/hard_cases.py, each aimed at
an axis where the 1000-case benchmark had stopped separating base Gemma-4 E4B,
gpt-5 and the v5 fine-tune. PLAN below fixes how many cases each
(mechanism, rounds-or-kind) bucket contributes, so the composition is declared
rather than a by-product of pool sizes.

Like every other slice this has its own seed file and its own
`random.Random(SEED)`, so it can never reshuffle the frozen files it sits beside.
scripts/build_benchmark.py combines it with a stratified subset of the 1000.

Run: uv run python scripts/generate_hard_cases.py
"""
from __future__ import annotations

import argparse
import collections
import itertools
import random
from pathlib import Path

import yaml

from wallet_evals.conversations import ACTION_FIELDS
from wallet_evals.generation import expand_vary
from wallet_evals.hard_cases import (
    HARD_ENS, LANGUAGES, MECHANISM_ROUNDS, REFUSAL_KINDS, REVISION_KINDS,
    build_embedded_refusal_case, build_injection_distractor_case,
    build_stacked_case, build_surface_case, build_surface_revision_case,
    build_unresolvable_amount_case, build_unresolvable_recipient_case,
    comma_decimal,
)

ROOT = Path(__file__).resolve().parent.parent
SEEDS = ROOT / "datasets" / "seeds.hard.yaml"
OUT = ROOT / "pf" / "tests.hard.yaml"
SEED = 20260924

#: (mechanism, bucket) -> cases. The bucket is a round count for the
#: conversational mechanisms and a kind for `surface` / `embedded_refusal`.
#:
#: 187 cases, 70 of them no-call (37%). That share is deliberately far above the
#: 1000's 12%: the no-call cases here are where the models were measured to
#: diverge (base asks when it should act; gpt-5 over-calls on refusals; v5 trades
#: hesitation for wrong arguments), so a model that never acts is NOT rewarded --
#: the 117 call cases are all ones a silent model fails.
PLAN: dict[tuple[str, int | str], int] = {
    **{("stacked", r): 15 for r in (7, 8, 9)},
    **{("injection_distractor", r): 10 for r in (3, 4, 5)},
    ("unresolvable_recipient", 1): 8, ("unresolvable_recipient", 2): 6,
    ("unresolvable_recipient", 3): 6,
    ("unresolvable_amount", 1): 8, ("unresolvable_amount", 2): 6,
    ("unresolvable_amount", 3): 6,
    # The English surface forms arrive as a round-2 correction (the single-turn
    # version scored 10/10 for every configuration); the four languages stay
    # single-turn as a regression guard for decimal-comma handling.
    **{("surface", k): 10 for k in REVISION_KINDS},
    **{("surface", lang): 3 for lang in LANGUAGES},
    **{("embedded_refusal", k): 5 for k in REFUSAL_KINDS},
}
#: Rounds for embedded_refusal cycle through its supported counts per kind.
TARGET_TOTAL = sum(PLAN.values())


def load_intents(seeds_path: Path, rng: random.Random) -> list[dict]:
    """Expand the seeds; "random_hard_ens" draws from HARD_ENS (resolvable names only),
    which expand_vary passes through untouched for this function to fill in."""
    seeds = yaml.safe_load(seeds_path.read_text())
    intents = [i for seed in seeds for i in expand_vary(seed, rng)
               if i["action"] != "swap" or i["from_token"] != i["to_token"]]
    for intent in intents:
        if intent.get("recipient") == "random_hard_ens":
            intent["recipient"] = rng.choice(HARD_ENS)
    return intents


def _pool(mechanism: str, bucket, intents: list[dict], rng: random.Random,
          counter: itertools.count) -> list[dict]:
    transfers = [i for i in intents if i["action"] == "transfer"]
    cases: list[dict] = []
    if mechanism == "stacked":
        for intent in intents:
            for withheld in ACTION_FIELDS[intent["action"]]:
                cases.append(build_stacked_case(intent, withheld, bucket, rng,
                                                next(counter)))
                first = rng.choice([o for o in intents if o is not intent
                                    and o["amount"] != intent["amount"]])
                cases.append(build_stacked_case(intent, withheld, bucket, rng,
                                                next(counter), first=first))
    elif mechanism == "injection_distractor":
        for intent in intents:
            for withheld in ACTION_FIELDS[intent["action"]]:
                cases.append(build_injection_distractor_case(
                    intent, withheld, bucket, rng, next(counter)))
    elif mechanism == "unresolvable_recipient":
        for intent in transfers:
            cases.append(build_unresolvable_recipient_case(intent, bucket, rng,
                                                           next(counter)))
    elif mechanism == "unresolvable_amount":
        for intent in intents:
            cases.append(build_unresolvable_amount_case(intent, bucket, rng,
                                                        next(counter)))
    elif mechanism == "surface":
        if bucket == "k_revision":  # magnitude suffixes read naturally on stablecoins
            eligible = [i for i in intents
                        if i.get("token", i.get("from_token")) in ("USDC", "DAI")]
        elif bucket in LANGUAGES:
            eligible = [i for i in intents if comma_decimal(i["amount"]) is not None]
        else:
            eligible = intents
        build = (build_surface_revision_case if bucket in REVISION_KINDS
                 else build_surface_case)
        for intent in eligible:
            cases.append(build(intent, bucket, rng, next(counter)))
    elif mechanism == "embedded_refusal":
        action = "swap" if bucket == "unverified-token" else "transfer"
        rounds_cycle = itertools.cycle(MECHANISM_ROUNDS["embedded_refusal"])
        for intent in (i for i in intents if i["action"] == action):
            cases.append(build_embedded_refusal_case(intent, bucket, next(rounds_cycle),
                                                     rng, next(counter)))
    else:  # pragma: no cover - guarded by PLAN's keys
        raise ValueError(f"unknown mechanism: {mechanism!r}")
    return cases


def build_selection(intents: list[dict], rng: random.Random,
                    plan: dict = PLAN) -> list[dict]:
    selected: list[dict] = []
    counter = itertools.count(1)
    for (mechanism, bucket), target in plan.items():
        pool = _pool(mechanism, bucket, intents, rng, counter)
        if len(pool) < target:
            raise ValueError(f"{mechanism}/{bucket}: pool of {len(pool)} cannot fill "
                             f"{target} -- widen datasets/seeds.hard.yaml")
        rng.shuffle(pool)
        selected.extend(pool[:target])
    return selected


def print_census(cases: list[dict]) -> None:
    by = collections.Counter(
        (c["metadata"]["mechanism"], c["metadata"]["kind"] or f"{c['metadata']['rounds']}r")
        for c in cases)
    for (mech, bucket), n in sorted(by.items()):
        print(f"  {mech:24s} {bucket:20s} {n:3d}")
    no_call = sum(1 for c in cases if not c["metadata"]["expected_calls"])
    print(f"  total {len(cases)}; no-call {no_call} ({no_call / len(cases):.0%})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    rng = random.Random(SEED)
    selected = build_selection(load_intents(SEEDS, rng), rng)
    assert len(selected) == TARGET_TOTAL, (len(selected), TARGET_TOTAL)
    print_census(selected)
    header = (
        "# Generated hard-slice cases -- DO NOT EDIT BY HAND.\n"
        f"# Produced by scripts/generate_hard_cases.py from datasets/seeds.hard.yaml "
        f"(seed {SEED}).\n"
        "# Six mechanisms; see src/wallet_evals/hard_cases.py. Only the model's reply "
        "to the LAST user turn is scored.\n")
    args.out.write_text(header + yaml.safe_dump(selected, sort_keys=False,
                                                allow_unicode=True))
    print(f"Wrote {len(selected)} cases -> {args.out}")


if __name__ == "__main__":
    main()
