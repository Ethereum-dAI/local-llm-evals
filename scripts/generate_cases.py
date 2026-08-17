"""Generate pf/tests.app-contract.yaml from datasets/seeds.yaml — deterministically.

For each seed: expand `vary` into concrete intents, then for each intent emit
positive cases (one per surface template, with seeded mutation), plus one
single-turn negative and one scripted multi-turn case per `ablate` field. Cases
are seeded-shuffled and capped to MAX_PER_ACTION per action; drops are logged.

Run: uv run python scripts/generate_cases.py
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml

from wallet_evals.generation import (
    TRANSFER_TEMPLATES, SWAP_TEMPLATES,
    TRANSFER_NARRATIVE_TEMPLATES, SWAP_NARRATIVE_TEMPLATES, REFUSAL_SCENARIOS,
    EXTRA_REFUSAL_SCENARIOS,
    expand_vary, build_positive_case, build_negative_case, build_multiturn_case,
    build_refusal_case,
)

ROOT = Path(__file__).resolve().parent.parent
SEEDS = ROOT / "datasets" / "seeds.yaml"
OUT = ROOT / "pf" / "tests.app-contract.yaml"
# The base-unit dataset every published score was measured against. The builders
# that produced it were replaced by the app-contract ones, so it can no longer be
# regenerated — it is a frozen artifact and this script must never write it.
FROZEN = ROOT / "pf" / "tests.generated.yaml"
SEED = 20260608
MAX_PER_ACTION = 150

# The labelled arithmetic slice (--extra-seeds): a SEPARATE seed file, a
# SEPARATE random.Random stream, and a SEPARATE (smaller) per-action cap, so
# appending it can never reshuffle, renumber, or resize the main 307-case
# selection above. See datasets/seeds.arithmetic.yaml for the failure modes it
# targets.
ARITHMETIC_SEEDS = ROOT / "datasets" / "seeds.arithmetic.yaml"
SEED_ARITHMETIC = 20260812
MAX_PER_ACTION_ARITHMETIC = 40

# EXTRA_REFUSAL_SCENARIOS is appended the same way, and for a sharper reason:
# build_refusal_case draws from the RNG it is given, and main() shuffles the
# "refusal" bucket BEFORE "swap"/"transfer" (alphabetical order). Growing the
# frozen 7-template REFUSAL_SCENARIOS in place therefore changes how many draws
# `rng` has consumed by the time swap/transfer are shuffled, silently
# reselecting the main 307 cases and destroying byte-identity with the frozen
# base-unit dataset — the property that makes the base-unit vs app-contract
# comparison honest. Measured: doing so moved the first swap case from
# gen-swap-pos-0176 to gen-swap-pos-0457.
SEED_REFUSAL = 20260814

_TEMPLATES = {"transfer": TRANSFER_TEMPLATES, "swap": SWAP_TEMPLATES}
_NARRATIVE_TEMPLATES = {
    "transfer": TRANSFER_NARRATIVE_TEMPLATES,
    "swap": SWAP_NARRATIVE_TEMPLATES,
}


def _valid_intent(intent: dict) -> bool:
    """Drop swaps whose from/to token are identical (no-op swap)."""
    if intent["action"] == "swap":
        return intent["from_token"] != intent["to_token"]
    return True


def build_all(seeds: list[dict], rng: random.Random,
             include_refusals: bool = True) -> dict[str, list[dict]]:
    """`include_refusals=False` is for a second, extra-seeds call (the
    arithmetic slice): REFUSAL_SCENARIOS is a hardcoded bank, not derived from
    `seeds`, so calling this twice with it left on would re-mint a second copy
    of the same "gen-refusal-####" ids, colliding with the main selection's."""
    by_action: dict[str, list[dict]] = {}
    counters: dict[str, int] = {}

    def next_idx(action: str) -> int:
        counters[action] = counters.get(action, 0) + 1
        return counters[action]

    for seed in seeds:
        for intent in expand_vary(seed, rng):
            if not _valid_intent(intent):
                continue
            action = intent["action"]
            bucket = by_action.setdefault(action, [])
            # Direct-style positives + (direct) negatives + direct multi-turn.
            for template in _TEMPLATES[action]:
                bucket.append(build_positive_case(intent, template, rng, next_idx(action)))
            for field in intent.get("ablate", []):
                bucket.append(build_negative_case(intent, field, rng, next_idx(action)))
                bucket.append(build_multiturn_case(intent, field, rng, next_idx(action)))
            # Narrative-style positives + narrative multi-turn (verbose/indirect).
            for template in _NARRATIVE_TEMPLATES[action]:
                bucket.append(build_positive_case(intent, template, rng, next_idx(action),
                                                  style="narrative"))
            for field in intent.get("ablate", []):
                bucket.append(build_multiturn_case(intent, field, rng, next_idx(action),
                                                   style="narrative"))

    if include_refusals:
        # Complete-but-dangerous requests (safety policy → no tool call expected).
        refusals = by_action.setdefault("refusal", [])
        for scenario in REFUSAL_SCENARIOS:
            for template in scenario["templates"]:
                refusals.append(build_refusal_case(scenario, template, rng, next_idx("refusal")))
    return by_action


def _relabel_arithmetic(case: dict) -> None:
    """Smallest possible override of `_base_metadata`'s id/category derivation
    so the arithmetic slice reads distinctly in a per-category report: ids get
    an `arith-` prefix (replacing `gen-`), categories get an `arithmetic-`
    prefix (replacing generation's own `generated-` prefix where present, e.g.
    "generated-transfer-pos" -> "arithmetic-transfer-pos"; categories that
    already override the default, like "ablation-amount" or
    "multiturn-recipient", just get the prefix prepended)."""
    md = case["metadata"]
    md["id"] = md["id"].replace("gen-", "arith-", 1)
    category = md["category"]
    if category.startswith("generated-"):
        category = category[len("generated-"):]
    md["category"] = f"arithmetic-{category}"


def build_extra_selection(extra_seeds: list[dict], rng: random.Random,
                          max_per_action: int = MAX_PER_ACTION_ARITHMETIC) -> list[dict]:
    """The arithmetic slice's own generate+shuffle+cap+relabel, mirroring the
    main selection loop in `main()` but against a separate seed list/RNG/cap so
    it never touches the main selection."""
    by_action = build_all(extra_seeds, rng, include_refusals=False)
    selected: list[dict] = []
    for action in sorted(by_action):
        cases = by_action[action]
        rng.shuffle(cases)
        kept = cases[:max_per_action]
        dropped = len(cases) - len(kept)
        print(f"[arithmetic] {action}: generated {len(cases)}, kept {len(kept)}, "
             f"dropped {dropped}")
        for case in kept:
            _relabel_arithmetic(case)
        selected.extend(kept)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT,
                        help=f"destination YAML (default: {OUT.relative_to(ROOT)})")
    parser.add_argument("--extra-seeds", type=Path, default=None,
                        help="optional additional seed file (e.g. "
                             f"{ARITHMETIC_SEEDS.relative_to(ROOT)}); built with its own "
                             "random.Random(SEED_ARITHMETIC) and appended AFTER the main "
                             "selection, so the main selection's ids/order are untouched "
                             "(default: none)")
    args = parser.parse_args()
    out = args.out.resolve()
    if out == FROZEN.resolve():
        parser.error(f"{FROZEN.relative_to(ROOT)} is the frozen base-unit dataset "
                     "and is no longer generated; pick another --out")

    seeds = yaml.safe_load(SEEDS.read_text())
    rng = random.Random(SEED)
    by_action = build_all(seeds, rng)

    selected: list[dict] = []
    for action in sorted(by_action):
        cases = by_action[action]
        rng.shuffle(cases)
        kept = cases[:MAX_PER_ACTION]
        dropped = len(cases) - len(kept)
        print(f"{action}: generated {len(cases)}, kept {len(kept)}, dropped {dropped}")
        selected.extend(kept)

    header_lines = [
        "# Generated eval cases — DO NOT EDIT BY HAND.",
        "# Produced by scripts/generate_cases.py from datasets/seeds.yaml (seed "
        f"{SEED}).",
        "# Gold is computed from each seed intent; surfaces carry deterministic noise.",
    ]

    if args.extra_seeds is not None:
        extra_seeds = yaml.safe_load(args.extra_seeds.read_text())
        extra_rng = random.Random(SEED_ARITHMETIC)
        extra_selected = build_extra_selection(extra_seeds, extra_rng)
        print(f"arithmetic slice: {len(extra_selected)} cases appended "
             f"(from {args.extra_seeds})")
        selected.extend(extra_selected)
        header_lines.append(
            "# The arithmetic-* cases (arith- ids) below are appended from "
            f"{args.extra_seeds.relative_to(ROOT) if args.extra_seeds.is_absolute() else args.extra_seeds} "
            f"(seed {SEED_ARITHMETIC}) via a SEPARATE random.Random stream, so the "
            "cases above are unaffected."
        )

    extra_refusals: list[dict] = []
    extra_rng_refusal = random.Random(SEED_REFUSAL)
    idx = 0
    for scenario in EXTRA_REFUSAL_SCENARIOS:
        for template in scenario["templates"]:
            idx += 1
            case = build_refusal_case(scenario, template, extra_rng_refusal, idx)
            case["metadata"]["id"] = case["metadata"]["id"].replace("gen-", "xref-", 1)
            extra_refusals.append(case)
    if extra_refusals:
        print(f"extra refusals: {len(extra_refusals)} cases appended")
        selected.extend(extra_refusals)
        header_lines.append(
            "# The xref- cases below come from EXTRA_REFUSAL_SCENARIOS via a SEPARATE "
            f"random.Random({SEED_REFUSAL}) stream, so the selections above are "
            "unaffected."
        )

    header = "\n".join(header_lines) + "\n"
    out.write_text(header + yaml.safe_dump(selected, sort_keys=False, allow_unicode=True))
    print(f"Wrote {len(selected)} cases -> {out}")


if __name__ == "__main__":
    main()
