"""Integrity of the Qwen3 fine-tuning set (data_for_finetune/qwen_train.jsonl).

Same contract as tests/test_gemma4_finetune_integrity.py, one dialect over: the
targets are Hermes-style `<tool_call>{...}</tool_call>` JSON, decoded by
`json_tool_calls.parse_json_tool_calls` (the same parser the Qwen provider uses at
inference), and every one must self-score 1.0 through the UNCHANGED scorer.

Also asserts the three training sets stay row-identical in CONTENT: the Qwen set
must carry exactly the ids and gold of the Gemma-4 set, differing only in
encoding. That is the whole basis for comparing the two fine-tunes — if the rows
drift, the comparison measures data, not model.

If the JSONL is absent (not generated yet), these tests skip.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from wallet_evals.finetune import HERMES, encode_hermes_calls
from wallet_evals.functiongemma import json_output_to_scoreable
from wallet_evals.parsing import parse_turn
from wallet_evals.schema import Case
from wallet_evals.scorer import score_case

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data_for_finetune" / "qwen_train.jsonl"
GEMMA4_TRAIN = ROOT / "data_for_finetune" / "gemma4_train.jsonl"
GENERATED = ROOT / "pf" / "tests.generated.yaml"


def _load(path: Path = TRAIN) -> list[dict]:
    if not path.exists():
        pytest.skip(f"{path} not generated yet "
                    f"(run scripts/generate_qwen_finetune_data.py)")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _case(ex: dict) -> Case:
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
    assert len(_load()) > 0


def test_ids_unique_and_prefixed():
    ids = [ex["id"] for ex in _load()]
    assert len(ids) == len(set(ids)), "duplicate ids"
    assert all(i.startswith("ft-") for i in ids), "ids must be ft- prefixed"


def test_every_target_self_scores_one():
    """The crux: each assistant target, run through the same translation the Qwen
    provider applies at inference, must score 1.0 against its own gold."""
    for ex in _load():
        scoreable = json_output_to_scoreable(_assistant_content(ex))
        if scoreable.strip().startswith("["):
            turn = parse_turn(content=None, native_tool_calls=json.loads(scoreable),
                              raw_text="")
        else:
            turn = parse_turn(content=scoreable, native_tool_calls=None,
                              raw_text=scoreable)
        assert score_case(_case(ex), turn) == 1, f"{ex['id']} target does not self-score"


def test_targets_match_expected_call_count():
    for ex in _load():
        has_call = HERMES.opener in _assistant_content(ex)
        assert has_call == bool(ex.get("expected_calls")), \
            f"{ex['id']}: call presence disagrees with expected_calls"


def test_encoder_roundtrips_each_call():
    """encode_hermes_calls is the exact inverse of the parser the provider uses."""
    for ex in _load():
        for call in ex.get("expected_calls") or []:
            scoreable = json_output_to_scoreable(encode_hermes_calls([call]))
            turn = parse_turn(content=None, native_tool_calls=json.loads(scoreable),
                              raw_text="")
            case = Case(id=ex["id"], user_message="", level="payload",
                        language="english", category="c", protocol="transfer",
                        difficulty="easy", expected_calls=[call])
            assert score_case(case, turn) == 1, f"{ex['id']} call did not round-trip"


def test_tools_present():
    tools = json.loads((ROOT / "pf" / "tools.json").read_text())
    for ex in _load():
        assert ex.get("tools") == tools, f"{ex['id']}: tools must equal tools.json"


def test_roles_keep_system_not_developer():
    for ex in _load():
        roles = {m["role"] for m in ex["messages"]}
        assert "developer" not in roles, f"{ex['id']}: Qwen must not use developer role"
        assert "system" in roles, f"{ex['id']}: missing system turn"


def test_reasoning_only_on_call_targets():
    for ex in _load():
        if "<think>" in _assistant_content(ex):
            assert ex.get("expected_calls"), f"{ex['id']}: <think> on a no-call target"


def test_same_rows_as_the_gemma4_set():
    """Identical content, different encoding — the premise of the comparison."""
    qwen = {ex["id"]: ex for ex in _load()}
    gemma = {ex["id"]: ex for ex in _load(GEMMA4_TRAIN)}
    assert set(qwen) == set(gemma), "training row ids differ between the two sets"
    for i, ex in qwen.items():
        assert ex["expected_calls"] == gemma[i]["expected_calls"], f"{i}: gold differs"
        assert ex["messages"][:-1] == gemma[i]["messages"][:-1], f"{i}: prompt differs"


def test_disjoint_from_the_eval_set():
    """No training row may reuse an eval surface — the anti-leakage guarantee."""
    import yaml

    def surfaces(vars_: dict) -> str:
        if "messages" in vars_:
            return "\n".join(m["content"] for m in vars_["messages"]
                             if m.get("role") == "user")
        return vars_.get("user_message", "")

    eval_surfaces = {surfaces(t["vars"]).strip()
                     for t in yaml.safe_load(GENERATED.read_text())}
    for ex in _load():
        users = "\n".join(m["content"] for m in ex["messages"] if m["role"] == "user")
        assert users.strip() not in eval_surfaces, f"{ex['id']} leaks an eval surface"
