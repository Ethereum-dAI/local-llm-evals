"""Integrity + containment tests for the private versioned holdout.

Two jobs, and the first one runs even on a machine that has no holdout:

1. **Containment.** Nothing under `holdout/` may ever be tracked by git. The
   amount literals in a holdout seeds file are the secret — every base-unit gold
   answer is amount x token decimals, so a leaked amount is a memorisable answer
   and the slice stops measuring arithmetic. These tests fail loudly rather than
   letting a `git add -A` quietly publish the yardstick.

2. **Disjointness + self-consistency**, for each `holdout/v*/` present. Skipped
   when the holdout isn't checked out (it lives in the private
   Ethereum-dAI/local-llm-evals-holdout repo), so the public suite stays green.

Also codifies the dev-vs-train invariant that was previously only true by
construction and never asserted.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from wallet_evals.promptfoo import load_cases
from wallet_evals.schema import ParsedTurn
from wallet_evals.scorer import score_case

ROOT = Path(__file__).resolve().parents[1]
DEV_SEEDS = ROOT / "datasets" / "seeds.yaml"
TRAIN_SEEDS = ROOT / "datasets" / "finetune_seeds.yaml"
HOLDOUT_DIR = ROOT / "holdout"

# Value-bearing gold keys. `amount` is the RAILGUN human-decimal exception; the
# rest are base-unit integers.
_AMOUNT_KEYS = ("value", "amountIn", "amount")


def _amount_literals(seeds_path: Path) -> set[str]:
    """Every `amount` literal a seeds file can produce, vary-sets expanded."""
    out: set[str] = set()
    for seed in yaml.safe_load(seeds_path.read_text()):
        spec = seed.get("amount")
        if isinstance(spec, dict) and "vary" in spec:
            out |= {str(v) for v in spec["vary"]}
        elif spec is not None:
            out.add(str(spec))
    return out


def _gold_amounts(cases_path: Path) -> set[str]:
    """Every non-zero amount appearing in computed gold across a cases file."""
    out: set[str] = set()
    for case in yaml.safe_load(cases_path.read_text()):
        for call in case["metadata"].get("expected_calls") or []:
            for key in _AMOUNT_KEYS:
                raw = call.get(key)
                if raw not in (None, "0", 0):
                    out.add(str(raw))
    return out


def _versions() -> list[Path]:
    """Holdout version dirs that actually carry generated cases, oldest first."""
    if not HOLDOUT_DIR.is_dir():
        return []
    return sorted(p.parent for p in HOLDOUT_DIR.glob("v*/tests.holdout.yaml"))


_VERSIONS = _versions()
_needs_holdout = pytest.mark.skipif(
    not _VERSIONS, reason="no holdout/v*/ checked out (it lives in the private repo)"
)


# --------------------------------------------------------------------------- #
# 1. containment — always runs                                                #
# --------------------------------------------------------------------------- #

def test_holdout_is_gitignored():
    """`holdout/` must be ignored, whether or not it exists locally."""
    proc = subprocess.run(["git", "check-ignore", "-v", "holdout/v1/tests.holdout.yaml"],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, (
        "holdout/ is NOT gitignored — a holdout file could be committed to the "
        "public repo, which publishes the gold amounts"
    )


def test_no_holdout_file_is_tracked():
    """The real guard: git must not be tracking anything under holdout/."""
    proc = subprocess.run(["git", "ls-files", "--", "holdout"],
                          cwd=ROOT, capture_output=True, text=True, check=True)
    tracked = [line for line in proc.stdout.splitlines() if line.strip()]
    assert not tracked, (
        "these holdout files are tracked by git and would publish the private "
        f"eval amounts: {tracked}"
    )


def test_dev_and_train_amount_literals_are_disjoint():
    """The pre-existing invariant, now asserted rather than merely intended."""
    overlap = _amount_literals(DEV_SEEDS) & _amount_literals(TRAIN_SEEDS)
    assert not overlap, f"training data shares amounts with the dev eval set: {sorted(overlap)}"


# --------------------------------------------------------------------------- #
# 2. per-version disjointness + self-consistency                              #
# --------------------------------------------------------------------------- #

@_needs_holdout
@pytest.mark.parametrize("version", _VERSIONS, ids=lambda p: p.name)
def test_holdout_amounts_disjoint_from_dev_and_train(version: Path):
    """A holdout amount that also appears in the dev or training set is a leak:
    whoever trained on that set has already seen the answer."""
    holdout = _amount_literals(version / "holdout_seeds.yaml")
    assert holdout, f"{version.name} seeds declare no amounts"
    for label, path in (("dev", DEV_SEEDS), ("train", TRAIN_SEEDS)):
        overlap = holdout & _amount_literals(path)
        assert not overlap, f"{version.name} shares amounts with {label}: {sorted(overlap)}"


@_needs_holdout
@pytest.mark.parametrize("version", _VERSIONS, ids=lambda p: p.name)
def test_holdout_amounts_disjoint_from_other_versions(version: Path):
    """Rotation only means something if v2 doesn't recycle v1's arithmetic."""
    mine = _amount_literals(version / "holdout_seeds.yaml")
    for other in _VERSIONS:
        if other == version:
            continue
        overlap = mine & _amount_literals(other / "holdout_seeds.yaml")
        assert not overlap, (
            f"{version.name} recycles amounts from {other.name}: {sorted(overlap)} — "
            "rotate the literals, don't just change --seed"
        )


@_needs_holdout
@pytest.mark.parametrize("version", _VERSIONS, ids=lambda p: p.name)
def test_holdout_gold_values_disjoint_from_dev_gold(version: Path):
    """Stronger than comparing literals: compare the computed base-unit answers.
    Different literals could still collide after the decimal shift (e.g. 0.1 ETH
    and 100000000000000000 wei), and it is the computed value a model memorises."""
    dev = _gold_amounts(ROOT / "pf" / "tests.generated.yaml")
    overlap = _gold_amounts(version / "tests.holdout.yaml") & dev
    assert not overlap, (
        f"{version.name} gold reuses base-unit values from the public dev set: "
        f"{sorted(overlap)[:8]}"
    )


@_needs_holdout
@pytest.mark.parametrize("version", _VERSIONS, ids=lambda p: p.name)
def test_holdout_self_scores_one(version: Path):
    """Same bar as every other dataset here: gold must score 1 through the real
    scorer, so a holdout failure is always the model's, never the fixture's."""
    for case in load_cases(version / "tests.holdout.yaml"):
        turn = ParsedTurn(tool_calls=[c.as_parsed_call() for c in case.expected_calls])
        assert score_case(case, turn) == 1, f"{version.name}/{case.id} not self-consistent"


@_needs_holdout
@pytest.mark.parametrize("version", _VERSIONS, ids=lambda p: p.name)
def test_holdout_shape_matches_the_dev_set(version: Path):
    """A holdout that dropped a case family would silently stop measuring it."""
    cases = load_cases(version / "tests.holdout.yaml")
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids)), f"{version.name} has duplicate case ids"
    assert any("-neg-" in i for i in ids), f"{version.name} has no ablation negatives"
    assert any("-mt-" in i for i in ids), f"{version.name} has no multi-turn cases"
    refusals = [c for c in cases if "refusal" in c.id]
    assert refusals, f"{version.name} has no safety-refusal cases"
    assert all(c.expected_calls == [] for c in refusals), \
        f"{version.name} refusal gold must be no tool call"
