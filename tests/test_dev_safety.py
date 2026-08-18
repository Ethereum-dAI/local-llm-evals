"""Guards for pf/tests.dev.safety.yaml — the dev set's safety coverage.

This slice exists so a prompt or recipe change can be judged on safety as well as
accuracy. It is only worth anything if it is genuinely held out and genuinely
scoreable, which is what these check.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from wallet_evals.dev_safety import DEV_REFUSAL_SCENARIOS  # noqa: E402
from wallet_evals.generation import EXTRA_REFUSAL_SCENARIOS  # noqa: E402

DEV_SAFETY = ROOT / "pf" / "tests.dev.safety.yaml"
TRAIN = ROOT / "data_for_finetune" / "gemma4_train.jsonl"


def _cases() -> list[dict]:
    return yaml.safe_load(DEV_SAFETY.read_text())


def _surface(case: dict) -> str:
    v = case["vars"]
    if v.get("messages"):
        return "\n".join(m["content"] for m in v["messages"] if m["role"] == "user")
    return v.get("user_message", "")


def test_every_case_expects_no_call():
    """A safety case whose gold carried a call would invert the whole slice."""
    for case in _cases():
        assert case["metadata"]["expected_calls"] == [], case["metadata"]["id"]


def test_covers_every_rule_kind_in_the_prompt():
    """One kind per SAFETY rule. A rule with no case is an unmeasured rule."""
    theirs = {s["kind"] for s in EXTRA_REFUSAL_SCENARIOS}
    ours = {s["kind"] for s in DEV_REFUSAL_SCENARIOS}
    assert ours == theirs, f"missing {sorted(theirs - ours)}, extra {sorted(ours - theirs)}"


def test_surfaces_are_disjoint_from_every_held_out_set():
    """New surfaces are the point: same rule, unseen phrasing. If a surface matched
    the benchmark's, this would be measuring recall of a sentence."""
    mine = {_surface(c) for c in _cases()}
    for path in sorted((ROOT / "pf").glob("tests.*.yaml")):
        if path.name == DEV_SAFETY.name:
            continue
        theirs = {_surface(c) for c in yaml.safe_load(path.read_text()) or []}
        overlap = mine & theirs
        assert not overlap, f"{path.name} shares surfaces: {sorted(overlap)[:3]}"


def test_surfaces_never_appear_in_training():
    if not TRAIN.exists():
        pytest.skip(f"{TRAIN} not generated")
    rows = [json.loads(l) for l in TRAIN.read_text().splitlines() if l.strip()]
    trained = {"\n".join(m["content"] for m in r["messages"] if m["role"] == "user")
               for r in rows}
    for case in _cases():
        assert _surface(case) not in trained, f"{case['metadata']['id']} is trained on"


def test_templates_are_not_reused_from_the_benchmark_bank():
    """Checks the SOURCE templates too, not just the rendered surfaces — a mutator
    could otherwise mask a copied template."""
    theirs = {t for s in EXTRA_REFUSAL_SCENARIOS for t in s["templates"]}
    mine = {t for s in DEV_REFUSAL_SCENARIOS for t in s["templates"]}
    assert not mine & theirs, f"reused templates: {sorted(mine & theirs)[:3]}"


def test_ids_are_namespaced():
    """So a merged report can never confuse these with the benchmark's refusals."""
    for case in _cases():
        assert case["metadata"]["id"].startswith("devsafety-"), case["metadata"]["id"]
