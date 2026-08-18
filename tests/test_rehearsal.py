"""Guards for the rehearsal rows (wallet_evals/rehearsal.py).

These rows exist to make "answer in prose" reachable again in a training mix that
is otherwise ~97% call-emitting. Every measured regression in the fine-tune was a
case of calling when it should not have, so the failure mode of this fix is the
mirror image: too much prose, and the model stops calling when it should. Both
directions are pinned here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_finetune_data as fg  # noqa: E402
from wallet_evals.finetune import assistant_target  # noqa: E402
from wallet_evals.rehearsal import (  # noqa: E402
    REHEARSAL_TURNS, build_rehearsal_case,
)

TRAIN = ROOT / "data_for_finetune" / "gemma4_train.jsonl"

#: Every dialect's call opener. A rehearsal target containing ANY of these teaches
#: the opposite of its lesson, and would also self-score 0 against its empty gold.
CALL_OPENERS = ("<|tool_call>", "<tool_call>", "functools[", "```json")


def _examples() -> list[dict]:
    if not TRAIN.exists():
        pytest.skip(f"{TRAIN} not generated")
    return [json.loads(line) for line in TRAIN.read_text().splitlines() if line.strip()]


def test_turns_are_unique_and_nonempty():
    users = [u for u, _ in REHEARSAL_TURNS]
    assert len(users) == len(set(users)), "duplicate rehearsal prompts"
    for user, reply in REHEARSAL_TURNS:
        assert user.strip() and reply.strip()


def test_no_rehearsal_target_contains_a_call_opener():
    for user, reply in REHEARSAL_TURNS:
        for opener in CALL_OPENERS:
            assert opener not in reply, f"{user!r} target contains {opener!r}"


def test_targets_route_through_assistant_target():
    """`target_text` must actually be honoured — otherwise these rows silently
    train the generic refusal message instead of their own reply."""
    import random
    for i, turn in enumerate(REHEARSAL_TURNS, start=1):
        case = build_rehearsal_case(turn, random.Random(i), i)
        assert assistant_target(case["metadata"]) == turn[1]


def test_target_text_never_overrides_a_real_call():
    """The override is only safe while gold is empty. If a row ever carried both,
    training and scoring would disagree about the same example."""
    md = {"category": "rehearsal", "target_text": "prose",
          "expected_calls": [{"tool": "transfer", "to": "vitalik.eth",
                              "amount": "1", "token": "ETH"}]}
    assert assistant_target(md) != "prose"


def test_rehearsal_rows_are_in_the_training_set():
    rows = [ex for ex in _examples() if ex.get("category") == "rehearsal"]
    assert len(rows) == len(REHEARSAL_TURNS), \
        f"{len(rows)} rehearsal rows, expected {len(REHEARSAL_TURNS)}"
    for ex in rows:
        assert not ex.get("expected_calls")
        target = ex["messages"][-1]["content"]
        for opener in CALL_OPENERS:
            assert opener not in target, f"{ex['id']}: target emits a call"


def test_no_call_rows_stay_a_small_minority():
    """The counterweight must not become the weight.

    Rehearsal + refusals + ablations + exact_output are all no-call rows. If they
    grow past a third of the mix, the cure starts causing the disease it treats —
    a wallet model that answers in prose when the user asked it to send money is a
    worse product than one that occasionally calls too eagerly.
    """
    examples = _examples()
    no_call = [ex for ex in examples if not ex.get("expected_calls")]
    share = len(no_call) / len(examples)
    assert share <= 0.33, \
        f"{len(no_call)}/{len(examples)} ({share:.0%}) of training rows make no call"
    assert share >= 0.05, \
        f"only {share:.0%} no-call rows — the prose path is under-trained"


def test_rehearsal_can_be_switched_off_for_reproducibility():
    assert hasattr(fg, "INCLUDE_REHEARSAL_ROWS")
    import random
    triples = fg._collect(random.Random(fg.SEED), include_rehearsal=False)
    assert not [t for t in triples if t[2] == "rehearsal"]
