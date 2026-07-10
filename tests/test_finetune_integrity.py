"""Integrity of the FunctionGemma fine-tuning set (data_for_finetune/*.jsonl).

The whole value of the harness is that gold is trustworthy. A training target is
only correct if, decoded back through the UNCHANGED provider translation + scorer,
it scores 1.0 — i.e. the model that learns to emit it would pass the eval. We also
assert the set is disjoint from the eval set (no leakage) and structurally sane.

If the JSONL is absent (not generated yet), these tests skip.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from wallet_evals.finetune import encode_gemma_call
from wallet_evals.functiongemma import raw_output_to_scoreable
from wallet_evals.parsing import parse_turn
from wallet_evals.schema import Case
from wallet_evals.scorer import score_case

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data_for_finetune" / "functiongemma_train.jsonl"
GENERATED = ROOT / "pf" / "tests.generated.yaml"
PROTOCOLS = ROOT / "pf" / "tests.protocols.yaml"


def _load_examples() -> list[dict]:
    if not TRAIN.exists():
        pytest.skip(f"{TRAIN} not generated yet (run scripts/generate_finetune_data.py)")
    return [json.loads(line) for line in TRAIN.read_text().splitlines() if line.strip()]


def _case(ex: dict) -> Case:
    """Rebuild a scoring Case from an example (only expected_calls/tool matter)."""
    return Case(
        id=ex["id"], user_message="", level="payload", language="english",
        category=ex.get("category") or "generated",
        protocol=ex.get("protocol") or "transfer",
        difficulty="easy", expected_calls=ex.get("expected_calls") or [],
    )


def _assistant_content(ex: dict) -> str:
    assert ex["messages"][-1]["role"] == "assistant", f"{ex['id']}: no assistant turn"
    return ex["messages"][-1]["content"]


def test_nonempty():
    assert len(_load_examples()) > 0


def test_ids_unique_and_prefixed():
    ids = [ex["id"] for ex in _load_examples()]
    assert len(ids) == len(set(ids)), "duplicate ids"
    assert all(i.startswith("ft-") for i in ids), "ids must be ft- prefixed"


def test_every_target_self_scores_one():
    """The crux: each assistant target, run through the same translation the
    provider applies at inference, must score 1.0 against its own gold."""
    for ex in _load_examples():
        content = _assistant_content(ex)
        scoreable = raw_output_to_scoreable(content)
        # Mirror pf/assert.py's two paths: DSL calls -> JSON list; prose -> as-is.
        if scoreable.strip().startswith("["):
            calls = json.loads(scoreable)
            turn = parse_turn(content=None, native_tool_calls=calls, raw_text="")
        else:
            turn = parse_turn(content=scoreable, native_tool_calls=None, raw_text=scoreable)
        assert score_case(_case(ex), turn) == 1, f"{ex['id']} target does not self-score"


def test_targets_match_expected_call_count():
    """A gold-call case must emit a DSL call; a no-call case must emit prose only."""
    for ex in _load_examples():
        content = _assistant_content(ex)
        has_call = "<start_function_call>" in content
        assert has_call == bool(ex.get("expected_calls")), \
            f"{ex['id']}: call presence disagrees with expected_calls"


def test_encoder_roundtrips_each_call():
    """encode_gemma_call is the exact inverse the decoder+scorer accept."""
    for ex in _load_examples():
        for call in ex.get("expected_calls") or []:
            dsl = encode_gemma_call(call)
            scoreable = raw_output_to_scoreable(dsl)
            turn = parse_turn(content=None,
                              native_tool_calls=json.loads(scoreable), raw_text="")
            case = Case(id=ex["id"], user_message="", level="payload",
                        language="english", category="c", protocol="transfer",
                        difficulty="easy", expected_calls=[call])
            assert score_case(case, turn) == 1, f"{ex['id']} call did not round-trip"


def test_tools_present():
    tools = json.loads((ROOT / "pf" / "tools.json").read_text())
    for ex in _load_examples():
        assert ex.get("tools") == tools, f"{ex['id']}: tools must equal tools.json"


def test_roles_use_developer_not_system():
    for ex in _load_examples():
        roles = {m["role"] for m in ex["messages"]}
        assert "system" not in roles, f"{ex['id']}: system role must be remapped to developer"
        assert "developer" in roles, f"{ex['id']}: missing developer turn"


def _all_user_turns_from_vars(vars_: dict) -> str:
    """The full user input: every user turn joined (multi-turn cases share tiny
    completion fragments like 'to vitalik.eth' but never the whole conversation)."""
    if "messages" in vars_:
        users = [m["content"] for m in vars_["messages"] if m.get("role") == "user"]
        return "\n".join(users)
    return vars_.get("user_message", "")


def _eval_surfaces() -> set[str]:
    import yaml
    surfaces: set[str] = set()
    for path in (GENERATED, PROTOCOLS):
        for test in yaml.safe_load(path.read_text()) or []:
            surfaces.add(_all_user_turns_from_vars(test["vars"]))
    return surfaces


def _train_surface(ex: dict) -> str:
    users = [m["content"] for m in ex["messages"] if m["role"] == "user"]
    return "\n".join(users)


def test_disjoint_from_eval_set():
    """No training conversation may appear in the eval set — the anti-leakage
    guarantee. Compares the full user-turn sequence, not a single fragment."""
    eval_surfaces = _eval_surfaces()
    for ex in _load_examples():
        assert _train_surface(ex) not in eval_surfaces, \
            f"{ex['id']} conversation leaks into the eval set"
