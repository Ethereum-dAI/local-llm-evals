"""Generate pf/tests.generated.yaml from datasets/seeds.yaml — deterministically.

For each seed: expand `vary` into concrete intents, then for each intent emit
positive cases (one per surface template, with seeded mutation), plus one
single-turn negative and one scripted multi-turn case per `ablate` field. Cases
are seeded-shuffled and capped to MAX_PER_ACTION per action; drops are logged.

    uv run python scripts/generate_cases.py          # the public dev set (307 cases)

The same code path builds a **private holdout** from a different seeds file, so
the two sets can never drift in generation logic — only in their inputs:

    uv run python scripts/generate_cases.py \\
        --seeds holdout/v1/holdout_seeds.yaml \\
        --out   holdout/v1/tests.holdout.yaml \\
        --seed  20260807 --max-per-action 70

`holdout/` is gitignored on purpose. The amount literals in a holdout seeds file
ARE the secret — publishing them would defeat the split, since a leaked amount is
all a model needs to memorise instead of computing. See tests/test_holdout_integrity.py.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml

from wallet_evals.generation import (
    TRANSFER_TEMPLATES, SWAP_TEMPLATES,
    TRANSFER_NARRATIVE_TEMPLATES, SWAP_NARRATIVE_TEMPLATES, REFUSAL_SCENARIOS,
    expand_vary, build_positive_case, build_negative_case, build_multiturn_case,
    build_refusal_case,
)

ROOT = Path(__file__).resolve().parent.parent
SEEDS = ROOT / "datasets" / "seeds.yaml"
OUT = ROOT / "pf" / "tests.generated.yaml"
SEED = 20260608
MAX_PER_ACTION = 150

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


def build_all(seeds: list[dict], rng: random.Random) -> dict[str, list[dict]]:
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

    # Complete-but-dangerous requests (safety policy → no tool call expected).
    refusals = by_action.setdefault("refusal", [])
    for scenario in REFUSAL_SCENARIOS:
        for template in scenario["templates"]:
            refusals.append(build_refusal_case(scenario, template, rng, next_idx("refusal")))
    return by_action


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=Path, default=SEEDS,
                    help="intent seeds YAML (default: datasets/seeds.yaml)")
    ap.add_argument("--out", type=Path, default=OUT,
                    help="destination YAML (default: pf/tests.generated.yaml)")
    ap.add_argument("--seed", type=int, default=SEED,
                    help=f"RNG seed for surfaces and selection (default: {SEED})")
    ap.add_argument("--max-per-action", type=int, default=MAX_PER_ACTION,
                    help=f"cap per action after shuffling (default: {MAX_PER_ACTION})")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    seeds = yaml.safe_load(args.seeds.read_text())
    rng = random.Random(args.seed)
    by_action = build_all(seeds, rng)

    selected: list[dict] = []
    for action in sorted(by_action):
        cases = by_action[action]
        rng.shuffle(cases)
        kept = cases[:args.max_per_action]
        dropped = len(cases) - len(kept)
        print(f"{action}: generated {len(cases)}, kept {len(kept)}, dropped {dropped}")
        selected.extend(kept)

    # Path is written relative to the repo when possible, so the header stays
    # identical whether the script is invoked from the repo root or elsewhere.
    try:
        seeds_label = args.seeds.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        seeds_label = args.seeds.as_posix()
    header = (
        "# Generated eval cases — DO NOT EDIT BY HAND.\n"
        f"# Produced by scripts/generate_cases.py from {seeds_label} (seed "
        f"{args.seed}).\n"
        "# Gold is computed from each seed intent; surfaces carry deterministic noise.\n"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(header + yaml.safe_dump(selected, sort_keys=False, allow_unicode=True))
    print(f"Wrote {len(selected)} cases -> {args.out}")


if __name__ == "__main__":
    main()
