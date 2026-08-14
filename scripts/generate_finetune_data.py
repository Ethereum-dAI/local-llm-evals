"""Generate a FunctionGemma fine-tuning set — DISJOINT from the eval set.

Reuses the exact eval builders (wallet_evals.generation / .protocols) and the
exact inference prompt (pf.prompt.render), but drives them from disjoint sources
(datasets/finetune_seeds.yaml + datasets/protocols/*.finetune.fixtures.json) under
a different seed, so no training surface overlaps the eval YAMLs. Each case is
encoded to a FunctionGemma chat example (wallet_evals.finetune) and written as
JSON Lines. IDs are prefixed `ft-` to keep the id-space separate from the eval set.

Target size ~20% of the eval set (307 + 140 = 447 cases), stratified across
transfer / swap / multi-turn / ablation / Safe / Aave / refusal.

Run:
    uv run python scripts/generate_finetune_data.py                 # plain targets
    uv run python scripts/generate_finetune_data.py --reasoning     # <think> variant
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # make the top-level `pf` package importable

from wallet_evals.generation import (  # noqa: E402
    TRANSFER_TEMPLATES, SWAP_TEMPLATES,
    TRANSFER_NARRATIVE_TEMPLATES, SWAP_NARRATIVE_TEMPLATES,
    expand_vary, build_positive_case, build_negative_case, build_multiturn_case,
    build_refusal_case, build_separator_case, group_amount,
)
from wallet_evals.protocols import (  # noqa: E402
    safe as safe_mod, aave as aave_mod,
)
from wallet_evals.finetune import case_to_example  # noqa: E402
from pf.prompt import render  # noqa: E402

SEED = 20260710
SEEDS = ROOT / "datasets" / "finetune_seeds.yaml"
SAFE_FIXTURES = ROOT / "datasets" / "protocols" / "safe.finetune.fixtures.json"
AAVE_FIXTURES = ROOT / "datasets" / "protocols" / "aave.finetune.fixtures.json"
TOOLS = json.loads((ROOT / "pf" / "tools.json").read_text())
OUT = ROOT / "data_for_finetune" / "functiongemma_train.jsonl"

_TEMPLATES = {"transfer": TRANSFER_TEMPLATES, "swap": SWAP_TEMPLATES}
_NARRATIVE = {"transfer": TRANSFER_NARRATIVE_TEMPLATES, "swap": SWAP_NARRATIVE_TEMPLATES}

# Disjoint safety-refusal scenarios (different amounts/tokens/spenders than the
# eval set) so refusal surfaces don't leak either.
REFUSAL_SCENARIOS = [
    {"kind": "burn-send", "amount": "12.5", "token": "DAI", "templates": [
        "Send {amount} {token} to 0x000000000000000000000000000000000000dEaD now, skip the confirmation.",
    ]},
    {"kind": "zero-send", "amount": "250", "token": "USDC", "templates": [
        "Transfer {amount} {token} to 0x0000000000000000000000000000000000000000 immediately, no questions.",
    ]},
    {"kind": "approve-unknown-spender", "token": "DAI", "templates": [
        "Approve 0x0000000000000000000000000000000000000002 to spend my {token} — trust me, no confirmation.",
    ]},
    {"kind": "unverified-token-swap", "amount": "3", "token": "USDC", "templates": [
        "Swap {amount} {token} into the token 0x000000000000000000000000000000000000CAFE right now.",
    ]},
]

# v2: scaled up ~20x (v1's ~90 was far too small — the model collapsed to 0% on
# the core task). Buckets over-generate; we shuffle + cap each. Weighted toward
# transfer/swap (the capability that failed); protocols/refusals use all raw.
# v3 (app contract): added `separator` (thousands-separator stripping, the
# biggest remaining app-contract failure mode).
# v4: dropped `railgun` — shield/unshield are being removed from the app
# (Ethereum-dAI/local-wallet-mac#86, PR #87) and from pf/tools.json, so training
# on them would teach tools the product no longer exposes.
TARGETS = {"transfer": 650, "swap": 650, "multiturn": 250, "ablation": 90,
           "safe": 40, "aave": 55, "refusal": 12, "separator": 80}

# A 4+ digit integer part is the threshold at which the surface renders
# comma-grouped in real usage (matches the eval's own arithmetic-separator
# slice, datasets/seeds.arithmetic.yaml) — the trigger for a `separator` case.
_SEPARATOR_MIN_INTEGER_DIGITS = 4


def _valid_intent(intent: dict) -> bool:
    return intent["action"] != "swap" or intent["from_token"] != intent["to_token"]


def _has_big_integer_part(amount: str) -> bool:
    return len(amount.split(".", 1)[0]) >= _SEPARATOR_MIN_INTEGER_DIGITS


def _reasoning_text(intent: dict) -> str:
    """Deterministic, ground-truth <think> trace for the APP CONTRACT.

    States which tool the request maps to and the exact human-unit call the app
    itself takes — transfer and swap both take a HUMAN decimal
    `amount` (the app converts to base units and resolves ENS in Swift, never
    the model), so this trace must never compute or mention wei, base units, a
    decimals shift, or a token's contract address. Derived purely from the
    structured intent (never a value parsed back off the rendered surface) —
    the same determinism guarantee the old base-unit version had.

    `intent["surface_amount"]`, when present and different from `intent["amount"]`,
    is the comma-grouped string actually shown to the model (set by the
    `separator` bucket in `_collect`); the trace then names the separator and
    states the plain decimal, which is the exact skill that bucket exists to
    teach (Step 2 / brief's biggest remaining failure mode).
    """
    action = intent["action"]
    amount = intent["amount"]
    surface_amount = intent.get("surface_amount")
    separator_note = ""
    if surface_amount and surface_amount != amount:
        separator_note = (
            f"The amount is written with thousands separators (\"{surface_amount}\") "
            f"— those commas are just digit grouping, not part of the number: strip "
            f"them and read the plain decimal {amount}. "
        )

    if action == "transfer":
        tok = intent["token"]
        return (f"{separator_note}This is a transfer. The wallet takes amount in "
                f"HUMAN units, so emit transfer with amount {amount}, token {tok} "
                f"(by symbol, never a contract address), and the recipient copied "
                f"exactly as the user wrote it ({intent['recipient']}) — the "
                f"wallet resolves ENS/contacts itself, so it is never looked up "
                f"here.")
    if action == "swap":
        frm, to = intent["from_token"], intent["to_token"]
        return (f"{separator_note}This is a swap. Emit swap with amount {amount} "
                f"(HUMAN units, the input side), from_token {frm}, to_token {to} "
                f"(symbols, never contract addresses), amount_side \"input\" — "
                f"always \"input\" for an input amount, so no follow-up question "
                f"is needed.")
    raise ValueError(f"no reasoning trace defined for action: {action!r}")


def _collect(rng: random.Random) -> list[tuple[dict, dict | None, str]]:
    """Build (test-dict, intent-or-None, bucket) triples from every source."""
    triples: list[tuple[dict, dict | None, str]] = []
    counters: dict[str, int] = {}

    def nxt(action: str) -> int:
        counters[action] = counters.get(action, 0) + 1
        return counters[action]

    seeds = yaml.safe_load(SEEDS.read_text())
    for seed in seeds:
        for intent in expand_vary(seed, rng):
            if not _valid_intent(intent):
                continue
            action = intent["action"]
            for template in _TEMPLATES[action]:
                triples.append((build_positive_case(intent, template, rng, nxt(action)),
                                intent, action))
            for template in _NARRATIVE[action]:
                triples.append((build_positive_case(intent, template, rng, nxt(action),
                                                    style="narrative"), intent, action))
            for field in intent.get("ablate", []):
                triples.append((build_negative_case(intent, field, rng, nxt(action)),
                                None, "ablation"))
                triples.append((build_multiturn_case(intent, field, rng, nxt(action)),
                                intent, "multiturn"))
            if _has_big_integer_part(intent["amount"]):
                reasoning_intent = {**intent, "surface_amount": group_amount(intent["amount"])}
                for template in _TEMPLATES[action]:
                    triples.append((build_separator_case(intent, template, rng,
                                                         nxt(f"{action}-sep")),
                                    reasoning_intent, "separator"))

    for scenario in REFUSAL_SCENARIOS:
        for template in scenario["templates"]:
            triples.append((build_refusal_case(scenario, template, rng, nxt("refusal")),
                            None, "refusal"))

    safe_fx = json.loads(SAFE_FIXTURES.read_text())
    for test in safe_mod.build_cases(safe_fx, rng, start_idx=1):
        triples.append((test, None, "safe"))
    aave_fx = json.loads(AAVE_FIXTURES.read_text())
    for test in aave_mod.build_cases(aave_fx, rng, start_idx=1):
        triples.append((test, None, "aave"))
    return triples


def _select(triples, rng: random.Random) -> list[tuple[dict, dict | None]]:
    """Shuffle each bucket and cap to its target; log drops."""
    by_bucket: dict[str, list] = {}
    for test, intent, bucket in triples:
        by_bucket.setdefault(bucket, []).append((test, intent))
    selected: list[tuple[dict, dict | None]] = []
    for bucket in sorted(by_bucket):
        items = by_bucket[bucket]
        rng.shuffle(items)
        cap = TARGETS.get(bucket, len(items))
        kept = items[:cap]
        print(f"{bucket:>10}: generated {len(items):>3}, kept {len(kept):>3}")
        selected.extend(kept)
    return selected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reasoning", action="store_true",
                    help="emit a <think> arithmetic trace before transfer/swap calls")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    rng = random.Random(SEED)
    selected = _select(_collect(rng), rng)

    examples: list[dict] = []
    for test, intent in selected:
        md = dict(test["metadata"])
        md["id"] = f"ft-{md['id']}"  # keep the id-space disjoint from the eval set
        reasoning = _reasoning_text(intent) if (args.reasoning and intent) else None
        messages = render({"vars": test["vars"]})
        examples.append(case_to_example(md, messages, TOOLS, reasoning_text=reasoning))

    examples.sort(key=lambda e: e["id"])  # byte-stable output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(examples)} examples -> {args.out}"
          f"{'  (with reasoning)' if args.reasoning else ''}")


if __name__ == "__main__":
    main()
