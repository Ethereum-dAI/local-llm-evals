"""The dev set must be disjoint from the test set AND from training.

The three-way split is the whole safeguard:

    TRAIN  data_for_finetune/*.jsonl      what the model learns from
    DEV    pf/tests.dev.yaml              what picks the checkpoint
    TEST   pf/tests.combined.yaml         the number we report (FROZEN)

Selecting a checkpoint on cases that appear in TEST is selecting on the test set.
That is not a hypothetical: this whole line of work started from a fine-tune that
scored 96.5% on a benchmark matching its training distribution and 69.3% on cases
it had not seen. Doing the same thing one level up — tuning on the reported set —
would reproduce that error while looking like progress.

So this is checked by SURFACE and by CASE ID, and on the gold as well, because a
rename or a re-seed must not be able to slip an overlap through.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "pf" / "tests.dev.yaml"
TEST = ROOT / "pf" / "tests.combined.yaml"
TRAIN_GLOB = "data_for_finetune/*.jsonl"


def _cases(path: Path) -> list[dict]:
    return yaml.safe_load(path.read_text())


def _surface(vars_: dict) -> str:
    msgs = vars_.get("messages")
    if msgs:
        return "\n".join(m["content"] for m in msgs if m.get("role") == "user")
    return vars_.get("user_message", "")


def test_dev_set_exists_and_has_expected_size():
    cases = _cases(DEV)
    assert len(cases) == 145, f"dev set is {len(cases)} cases; update this constant"


def test_dev_ids_do_not_collide_with_the_test_set():
    dev = {c["metadata"]["id"] for c in _cases(DEV)}
    test = {c["metadata"]["id"] for c in _cases(TEST)}
    assert not (dev & test), f"id collision with the frozen test set: {sorted(dev & test)[:5]}"


def test_dev_surfaces_do_not_appear_in_the_test_set():
    """The load-bearing one. Two generators sharing template banks can emit the
    same conversation from different seeds, and an id prefix would not catch it."""
    dev = {_surface(c["vars"]) for c in _cases(DEV)}
    test = {_surface(c["vars"]) for c in _cases(TEST)}
    overlap = dev & test
    assert not overlap, (
        f"{len(overlap)} dev conversation(s) also appear in the frozen test set — "
        f"gating on these would be selecting on the test set. First: "
        f"{sorted(overlap)[0][:120]!r}")


def test_dev_surfaces_do_not_appear_in_training():
    train_files = sorted(ROOT.glob(TRAIN_GLOB))
    if not train_files:
        pytest.skip("no training JSONLs present (they are gitignored)")
    dev = {_surface(c["vars"]) for c in _cases(DEV)}
    for path in train_files:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            ex = json.loads(line)
            users = "\n".join(m["content"] for m in ex["messages"]
                              if m["role"] == "user")
            assert users not in dev, f"{path.name}:{ex.get('id')} leaks into the dev set"


def test_dev_amounts_are_disjoint_from_every_other_bank():
    """A dev amount that a model trained on is not held out in any useful sense."""
    def amounts(rel: str) -> set[str]:
        found: set[str] = set()
        for seed in yaml.safe_load((ROOT / rel).read_text()) or []:
            spec = seed.get("amount")
            if isinstance(spec, dict):
                found |= {str(v) for v in spec["vary"]}
            elif spec is not None:
                found.add(str(spec))
        return found

    others: set[str] = set()
    for rel in ("datasets/seeds.yaml", "datasets/seeds.arithmetic.yaml",
                "datasets/seeds.conversations.yaml", "datasets/finetune_seeds.yaml"):
        others |= amounts(rel)
    conv = (ROOT / "src" / "wallet_evals" / "conversations.py").read_text()
    others |= set(re.findall(r'"([\d.]+)"', conv.split("ALT_AMOUNTS")[1].split(")")[0]))

    dev = amounts("datasets/seeds.dev.yaml")
    assert dev, "dev seeds declare no amounts"
    assert not (dev & others), f"dev reuses amounts: {sorted(dev & others)}"


def test_dev_ens_bank_is_disjoint_from_the_test_and_train_banks():
    """`vitalik.eth` is the training bank; generation.ENS_NAMES is the test bank.
    A dev set drawing on either cannot distinguish recall from generalisation."""
    gen = (ROOT / "src" / "wallet_evals" / "generation.py").read_text()
    test_bank = set(re.findall(r'"([a-z0-9.\-]+\.eth)"',
                              gen.split("ENS_NAMES")[1].split(")")[0]))
    dev_names = {to for c in _cases(DEV)
                 for call in (c["metadata"].get("expected_calls") or [])
                 for to in [str(call.get("to", ""))] if to.endswith(".eth")}
    assert dev_names, "dev set exercises no ENS recipients"
    assert not (dev_names & test_bank), \
        f"dev reuses TEST ENS names: {sorted(dev_names & test_bank)}"
    assert "vitalik.eth" not in dev_names, "dev reuses the TRAINING ENS name"


def test_dev_set_is_out_of_distribution_for_depth_and_the_new_axes():
    """What makes this set able to see the failure an in-distribution split cannot:
    training is 85.9% one-turn / 14.1% two-turn / 0% three-plus, and has zero
    exact-output and zero token-as-address rows."""
    cases = _cases(DEV)
    deep = [c for c in cases if c["metadata"]["rounds"] >= 3]
    assert len(deep) / len(cases) > 0.85, \
        f"only {len(deep)}/{len(cases)} cases are 3+ rounds"
    mechs = {c["metadata"]["mechanism"] for c in cases}
    for required in ("exact_output", "token_address"):
        n = sum(1 for c in cases if c["metadata"]["mechanism"] == required)
        assert n >= 15, f"{required} has only {n} dev cases — too coarse to gate on"
    assert {"correction", "distractor"} <= mechs


def test_dev_gold_self_scores_under_the_real_scorer():
    """Same invariant the other datasets carry: if gold cannot score itself, the
    checkpoint selector is measuring the scorer, not the model."""
    from wallet_evals.promptfoo import load_cases
    from wallet_evals.schema import ParsedTurn
    from wallet_evals.scorer import score_case
    for case in load_cases(DEV):
        turn = ParsedTurn(tool_calls=[c.as_parsed_call() for c in case.expected_calls])
        assert score_case(case, turn) == 1, f"{case.id} does not self-score"
