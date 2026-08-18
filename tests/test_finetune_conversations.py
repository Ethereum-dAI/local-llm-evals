"""Guards for the multi-round TRAINING rows added to close the depth collapse.

The measurements these protect (results/dev-epochs.e3-lr2e4.md, and the 1000-case
benchmark before it):

  * training was 85.9% single-turn and 0% three-plus rounds, while the benchmark
    runs to six — the fine-tune held 95.8% at one round and fell to 49.0% at six
    against a base model that is flat across depth;
  * `exact_output` scored 0/15 at EVERY epoch because no training row taught it;
  * every ENS recipient in training was the single name `vitalik.eth`, and novel
    ENS names drew FALSE REFUSALS ("That isn't a valid Ethereum address").

The point of this file is that none of those can silently come back, and that the
fix cannot be achieved by leaking held-out data.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_finetune_data as fg  # noqa: E402
from wallet_evals.conversations import MECHANISM_ABBREV  # noqa: E402
from wallet_evals.generation import ENS_NAMES as TEST_ENS_NAMES  # noqa: E402

TRAIN = ROOT / "data_for_finetune" / "gemma4_train.jsonl"
DEV = ROOT / "pf" / "tests.dev.yaml"
#: EVERY held-out dataset, not just the two the older integrity file knew about.
#: `tests.combined.yaml` is the frozen 1000-case benchmark and `tests.dev.yaml` is
#: the checkpoint selector — a leak into either one invalidates a reported number or
#: every selection decision respectively, which is worse than a leak into a set
#: nobody reports.
HELD_OUT_SETS = tuple(sorted(
    p for p in (ROOT / "pf").glob("tests.*.yaml")))


def _examples() -> list[dict]:
    if not TRAIN.exists():
        pytest.skip(f"{TRAIN} not generated "
                    f"(run scripts/generate_gemma4_finetune_data.py)")
    return [json.loads(line) for line in TRAIN.read_text().splitlines() if line.strip()]


def _ens_values(call: dict) -> list[str]:
    """Every `.eth` value in one gold call.

    App-contract gold is FLAT — `{"tool": "transfer", "to": ..., "amount": ...}` —
    with no `arguments` nesting, and the recipient key differs by tool (`to` for
    transfer, `recipient` for swap). Scanning the values rather than naming keys
    means a new tool cannot quietly escape these checks.
    """
    return [v for k, v in call.items()
            if k != "tool" and isinstance(v, str) and v.endswith(".eth")]


def _user_turns(messages: list[dict]) -> str:
    return "\n".join(m["content"] for m in messages if m["role"] == "user")


def _conversation_rows(examples: list[dict]) -> list[dict]:
    return [ex for ex in examples
            if (ex.get("category") or "").startswith("conversation-")]


def _dev_ens_names() -> set[str]:
    """The DEV ENS bank, read off the dev cases rather than re-declared here."""
    names: set[str] = set()
    for case in yaml.safe_load(DEV.read_text()) or []:
        for message in case["vars"].get("messages") or []:
            names.update(w.strip(".,!?\"'`") for w in message["content"].split()
                         if w.strip(".,!?\"'`").endswith(".eth"))
        for call in case["metadata"].get("expected_calls") or []:
            names.update(_ens_values(call))
    return names


# --------------------------------------------------------------------------
# leakage
# --------------------------------------------------------------------------
def test_train_ens_bank_is_disjoint_from_test_and_dev():
    """The whole point of a separate bank. If these overlap, "handles novel ENS
    names" is satisfiable by recall and the dev set stops measuring it."""
    train = set(fg.TRAIN_ENS_NAMES)
    assert not train & set(TEST_ENS_NAMES), \
        f"train ENS bank overlaps the TEST bank: {sorted(train & set(TEST_ENS_NAMES))}"
    dev = _dev_ens_names()
    assert not train & dev, \
        f"train ENS bank overlaps the DEV bank: {sorted(train & dev)}"


def test_no_held_out_ens_name_appears_anywhere_in_training():
    """Stronger than the bank check: catches a held-out name arriving by any route,
    including a builder that draws its own recipient from a default argument."""
    forbidden = set(TEST_ENS_NAMES) | _dev_ens_names()
    offenders: list[str] = []
    for ex in _examples():
        blob = _user_turns(ex["messages"]) + json.dumps(ex.get("expected_calls") or [])
        for name in forbidden:
            if name in blob:
                offenders.append(f"{ex['id']}: {name}")
    assert not offenders, f"held-out ENS names in training: {offenders[:10]}"


@pytest.mark.parametrize("dataset", HELD_OUT_SETS, ids=lambda p: p.name)
def test_no_training_conversation_leaks_into_any_held_out_set(dataset: Path):
    surfaces = set()
    for case in yaml.safe_load(dataset.read_text()) or []:
        messages = case["vars"].get("messages")
        if messages:
            surfaces.add(_user_turns(messages))
        elif case["vars"].get("user_message"):
            surfaces.add(case["vars"]["user_message"])
    for ex in _examples():
        assert _user_turns(ex["messages"]) not in surfaces, \
            f"{ex['id']} leaks into {dataset.name}"


# --------------------------------------------------------------------------
# the holes are actually filled
# --------------------------------------------------------------------------
def test_training_covers_three_to_six_rounds():
    """0% three-plus rounds was the distribution mismatch behind the depth
    collapse. Counted from the id (conv-<mech>-<N>r-<idx>) so it reflects what the
    builders emitted rather than a field we could get wrong twice."""
    rounds: dict[int, int] = {}
    for ex in _conversation_rows(_examples()):
        part = [p for p in ex["id"].split("-") if p.endswith("r") and p[:-1].isdigit()]
        assert part, f"{ex['id']}: no round marker in id"
        rounds[int(part[0][:-1])] = rounds.get(int(part[0][:-1]), 0) + 1
    assert set(rounds) >= {3, 4}, f"missing depth in training: {sorted(rounds)}"
    assert sum(rounds.values()) >= 300, f"only {sum(rounds.values())} multi-round rows"
    assert max(rounds) >= 5, f"deepest trained conversation is {max(rounds)} rounds"


def test_exact_output_rows_exist_and_teach_no_call():
    """The 0/15-at-every-epoch mechanism. Gold must be empty AND the target must
    carry no DSL opener, or the row would teach the opposite of the lesson."""
    rows = [ex for ex in _examples()
            if (ex.get("category") or "").startswith("conversation-exact_output-")]
    assert len(rows) >= 50, f"only {len(rows)} exact_output rows"
    for ex in rows:
        assert not ex.get("expected_calls"), f"{ex['id']}: gold should be no call"
        target = ex["messages"][-1]["content"]
        assert "<|tool_call>" not in target, f"{ex['id']}: target emits a call"
        assert "exact output" in target.lower(), \
            f"{ex['id']}: target does not explain the output-side limit"


def test_recipient_diversity_no_single_name_dominates():
    """`vitalik.eth` stays in the mix on purpose, but it must not BE the mix — that
    is what taught the model to refuse every other .eth name."""
    counts: dict[str, int] = {}
    for ex in _examples():
        for call in ex.get("expected_calls") or []:
            for value in _ens_values(call):
                counts[value] = counts.get(value, 0) + 1
    assert counts, "no ENS recipients in training gold at all"
    total = sum(counts.values())
    top, top_n = max(counts.items(), key=lambda kv: kv[1])
    assert top_n / total <= 0.55, \
        f"{top} is {top_n}/{total} ({top_n / total:.0%}) of ENS recipients"
    assert len(counts) >= 8, f"only {len(counts)} distinct ENS recipients: {counts}"


# --------------------------------------------------------------------------
# the control
# --------------------------------------------------------------------------
def test_held_out_mechanisms_never_enter_training():
    """`switch` is the transfer control: it survived over-training untouched
    (23/23 at every epoch), so if trained mechanisms improve while `switch` holds,
    the gain generalized. Training on it destroys the only such control we have."""
    assert fg.HELD_OUT_MECHANISMS, "no mechanism held out — nothing measures transfer"
    abbrevs = {m: MECHANISM_ABBREV[m] for m in fg.HELD_OUT_MECHANISMS}
    leaked = []
    for ex in _examples():
        category = ex.get("category") or ""
        for mechanism, abbrev in abbrevs.items():
            if category.startswith(f"conversation-{mechanism}-") \
                    or ex["id"].startswith(f"ft-conv-{abbrev}-"):
                leaked.append(ex["id"])
    assert not leaked, f"held-out mechanisms leaked into training: {leaked[:10]}"


def test_held_out_mechanism_is_still_measured_by_the_dev_set():
    """A holdout nobody scores is just missing data. The dev set must contain the
    mechanism we refuse to train, or the control is inert."""
    dev_mechanisms = {c["metadata"].get("mechanism")
                      for c in yaml.safe_load(DEV.read_text()) or []}
    for mechanism in fg.HELD_OUT_MECHANISMS:
        assert mechanism in dev_mechanisms, \
            f"{mechanism} is held out of training but absent from the dev set"
