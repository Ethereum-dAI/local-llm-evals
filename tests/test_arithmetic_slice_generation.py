"""TDD coverage for the arithmetic slice's generator additions.

`scripts/generate_cases.py` grows a `--extra-seeds` path so the arithmetic
slice can be generated from datasets/seeds.arithmetic.yaml with its OWN RNG
stream (SEED_ARITHMETIC), appended AFTER the main 307-case selection, without
disturbing it. These tests cover the pure pieces of that path in isolation,
before the file-level Step 2 diff check (which requires a real regeneration
run and is done by hand per the brief, not as a pytest).
"""
from __future__ import annotations

import random

import yaml

from scripts.generate_cases import (
    ARITHMETIC_SEEDS,
    MAX_PER_ACTION_ARITHMETIC,
    SEED_ARITHMETIC,
    build_all,
    build_extra_selection,
    _relabel_arithmetic,
)


def test_build_all_can_omit_the_refusal_bucket():
    # Refusal scenarios are hardcoded, not seed-derived: calling build_all a
    # second time for extra seeds must not silently re-mint duplicate
    # "gen-refusal-####" ids that collide with the main selection's.
    seeds = [{"action": "transfer", "amount": "1.5", "token": "ETH",
              "recipient": "vitalik.eth", "ablate": ["amount"]}]
    with_refusals = build_all(seeds, random.Random(1))
    without_refusals = build_all(seeds, random.Random(1), include_refusals=False)
    assert "refusal" in with_refusals and with_refusals["refusal"]
    assert "refusal" not in without_refusals


def test_relabel_arithmetic_rewrites_positive_case_id_and_category():
    case = {"metadata": {"id": "gen-transfer-pos-0001",
                         "category": "generated-transfer-pos"}}
    _relabel_arithmetic(case)
    assert case["metadata"]["id"] == "arith-transfer-pos-0001"
    assert case["metadata"]["category"] == "arithmetic-transfer-pos"


def test_relabel_arithmetic_rewrites_ablation_and_multiturn_categories():
    neg = {"metadata": {"id": "gen-swap-neg-0002", "category": "ablation-amount"}}
    _relabel_arithmetic(neg)
    assert neg["metadata"]["id"] == "arith-swap-neg-0002"
    assert neg["metadata"]["category"] == "arithmetic-ablation-amount"

    mt = {"metadata": {"id": "gen-transfer-mt-0003", "category": "multiturn-recipient"}}
    _relabel_arithmetic(mt)
    assert mt["metadata"]["id"] == "arith-transfer-mt-0003"
    assert mt["metadata"]["category"] == "arithmetic-multiturn-recipient"


def test_build_extra_selection_labels_every_case():
    seeds = yaml.safe_load(ARITHMETIC_SEEDS.read_text())
    selected = build_extra_selection(seeds, random.Random(SEED_ARITHMETIC))
    assert selected  # non-empty
    for case in selected:
        md = case["metadata"]
        assert md["id"].startswith("arith-"), md["id"]
        assert md["category"].startswith("arithmetic-"), md["category"]
        assert "gen-" not in md["id"]


def test_build_extra_selection_deterministic():
    seeds = yaml.safe_load(ARITHMETIC_SEEDS.read_text())
    a = build_extra_selection(seeds, random.Random(SEED_ARITHMETIC))
    b = build_extra_selection(seeds, random.Random(SEED_ARITHMETIC))
    assert a == b


def test_build_extra_selection_respects_its_own_cap():
    seeds = yaml.safe_load(ARITHMETIC_SEEDS.read_text())
    selected = build_extra_selection(seeds, random.Random(SEED_ARITHMETIC))
    by_action: dict[str, int] = {}
    for case in selected:
        action = case["metadata"]["id"].split("-")[1]
        by_action[action] = by_action.get(action, 0) + 1
    for action, count in by_action.items():
        assert count <= MAX_PER_ACTION_ARITHMETIC, (action, count)


def test_arithmetic_seed_amounts_are_disjoint_from_both_existing_seed_files():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    main_seeds = yaml.safe_load((root / "datasets" / "seeds.yaml").read_text())
    ft_seeds = yaml.safe_load((root / "datasets" / "finetune_seeds.yaml").read_text())
    arithmetic_seeds = yaml.safe_load(ARITHMETIC_SEEDS.read_text())

    def amounts(doc):
        out = set()
        for s in doc:
            spec = s["amount"]
            vals = spec["vary"] if isinstance(spec, dict) and "vary" in spec else [spec]
            out.update(vals)
        return out

    existing = amounts(main_seeds) | amounts(ft_seeds)
    new = amounts(arithmetic_seeds)
    assert existing & new == set(), f"amount literal(s) not disjoint: {existing & new}"
    assert new  # sanity: the new file actually defines some amounts
