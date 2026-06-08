"""Generate pf/tests.generated.yaml from datasets/seeds.yaml — deterministically.

For each seed: expand `vary` into concrete intents, then for each intent emit
positive cases (one per surface template, with seeded mutation), plus one
single-turn negative and one scripted multi-turn case per `ablate` field. Cases
are seeded-shuffled and capped to MAX_PER_ACTION per action; drops are logged.

Run: uv run python scripts/generate_cases.py
"""
from __future__ import annotations

import random
from pathlib import Path

import yaml

from wallet_evals.generation import (
    TRANSFER_TEMPLATES, SWAP_TEMPLATES,
    expand_vary, build_positive_case, build_negative_case, build_multiturn_case,
)

ROOT = Path(__file__).resolve().parent.parent
SEEDS = ROOT / "datasets" / "seeds.yaml"
OUT = ROOT / "pf" / "tests.generated.yaml"
SEED = 20260608
MAX_PER_ACTION = 100

_TEMPLATES = {"transfer": TRANSFER_TEMPLATES, "swap": SWAP_TEMPLATES}


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
            for template in _TEMPLATES[action]:
                bucket.append(build_positive_case(intent, template, rng, next_idx(action)))
            for field in intent.get("ablate", []):
                bucket.append(build_negative_case(intent, field, rng, next_idx(action)))
                bucket.append(build_multiturn_case(intent, field, rng, next_idx(action)))
    return by_action


def main() -> None:
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

    header = (
        "# Generated eval cases — DO NOT EDIT BY HAND.\n"
        "# Produced by scripts/generate_cases.py from datasets/seeds.yaml (seed "
        f"{SEED}).\n"
        "# Gold is computed from each seed intent; surfaces carry deterministic noise.\n"
    )
    OUT.write_text(header + yaml.safe_dump(selected, sort_keys=False, allow_unicode=True))
    print(f"Wrote {len(selected)} cases -> {OUT}")


if __name__ == "__main__":
    main()
