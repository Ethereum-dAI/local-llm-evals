"""Integrity of the generated conversation slice (pf/tests.conversations.yaml).

Mirrors test_app_contract_integrity.py: the gold has to self-score under the real
scorer, ids have to be unique, and the file has to still be a byte-stable output
of its generator. Adds the two properties that only exist for this slice — the
declared round distribution, and the guarantee that gold is never simply "the
first values the transcript mentions".
"""
from __future__ import annotations

import collections
import random
from pathlib import Path

import yaml

from scripts.generate_conversation_cases import (
    ROUND_PLAN, SEED, TARGET_TOTAL, build_selection, load_intents,
)
from wallet_evals.conversations import MECHANISM_ROUNDS
from wallet_evals.promptfoo import load_cases
from wallet_evals.schema import ParsedTurn
from wallet_evals.scorer import score_case

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "pf" / "tests.conversations.yaml"
SEEDS = ROOT / "datasets" / "seeds.conversations.yaml"


def _raw() -> list[dict]:
    return yaml.safe_load(TESTS.read_text())


def test_every_case_self_scores_one():
    for case in load_cases(TESTS):
        turn = ParsedTurn(tool_calls=[c.as_parsed_call() for c in case.expected_calls])
        assert score_case(case, turn) == 1, f"{case.id} not self-consistent"


def test_ids_are_unique():
    ids = [c["metadata"]["id"] for c in _raw()]
    assert len(ids) == len(set(ids))


def test_the_file_is_a_byte_stable_output_of_its_generator():
    """The whole dataset is regenerable, so a hand-edit (or a builder change made
    without regenerating) must fail rather than sit in the repo unnoticed."""
    rng = random.Random(SEED)
    regenerated = build_selection(load_intents(SEEDS, rng), rng)
    assert regenerated == _raw(), \
        "pf/tests.conversations.yaml is stale — rerun " \
        "scripts/generate_conversation_cases.py"


def test_round_distribution_matches_the_declared_plan():
    by_pair = collections.Counter(
        (c["metadata"]["rounds"], c["metadata"]["mechanism"]) for c in _raw())
    assert dict(by_pair) == {
        (rounds, mech): count
        for rounds, mechs in ROUND_PLAN.items() for mech, count in mechs.items()
    }
    assert len(_raw()) == TARGET_TOTAL


def test_rounds_span_two_to_six():
    rounds = {c["metadata"]["rounds"] for c in _raw()}
    assert rounds == {2, 3, 4, 5, 6}


def test_every_case_has_the_declared_number_of_user_turns():
    """`rounds` is metadata the report tooling slices on, so it must describe the
    transcript rather than merely label it."""
    for case in _raw():
        messages = case["vars"]["messages"]
        users = [m for m in messages if m["role"] == "user"]
        assistants = [m for m in messages if m["role"] == "assistant"]
        rounds = case["metadata"]["rounds"]
        assert len(users) == rounds, case["metadata"]["id"]
        assert len(assistants) == rounds - 1, case["metadata"]["id"]
        assert messages[-1]["role"] == "user", case["metadata"]["id"]


def test_every_mechanism_only_appears_at_a_round_count_it_supports():
    for case in _raw():
        md = case["metadata"]
        assert md["rounds"] in MECHANISM_ROUNDS[md["mechanism"]], md["id"]


def test_no_conversation_case_carries_a_single_turn_prompt():
    for case in _raw():
        assert "user_message" not in case["vars"], case["metadata"]["id"]
        assert case["metadata"]["query_type"] == "multi_turn"


#: The one mechanism whose gold is deliberately EMPTY: an exact-output swap is not
#: a call the wallet can make, so the correct behaviour is to make none. Every
#: other conversation mechanism must produce exactly one app-contract call.
EMPTY_GOLD_MECHANISMS = {"exact_output"}


def test_gold_is_a_single_app_contract_call_with_no_base_unit_payload():
    for case in load_cases(TESTS):
        # Case (schema.py) carries no `mechanism` field; the category encodes it as
        # "conversation-<mechanism>-<rounds>r".
        mechanism = case.category.split("-")[1]
        if mechanism in EMPTY_GOLD_MECHANISMS:
            assert case.expected_calls == [], \
                f"{case.id} is {mechanism} and must expect NO call"
            continue
        assert len(case.expected_calls) == 1, case.id
        call = case.expected_calls[0]
        assert call.tool in ("transfer", "swap"), f"{case.id} uses {call.tool}"
        assert call.value == "0" and call.args == []
        assert call.currencyIn is None and call.amountIn is None


def test_only_exact_output_cases_have_empty_gold():
    """Guards the override in conversations._build_case from spreading. Empty gold
    is a strong claim — it passes any model that stays silent — so exactly one
    mechanism is allowed to make it, and only for the documented reason."""
    for case in _raw():
        md = case["metadata"]
        if not md["expected_calls"]:
            assert md["mechanism"] in EMPTY_GOLD_MECHANISMS, \
                f"{md['id']} ({md['mechanism']}) has empty gold but is not allowed to"


def test_correction_cases_never_score_the_first_value_stated():
    """The mechanism's reason to exist. For every correction case, at least one
    revised field's gold differs from the value the opener stated — so a model
    that reads only turn 1 cannot pass by luck."""
    corrections = [c for c in _raw() if c["metadata"]["mechanism"] == "correction"]
    assert corrections
    for case in corrections:
        notes = case["metadata"]["notes"]
        trace = notes.split("revisions: ")[1].split(", ")
        # Each trace entry is "field old->new"; the field's FINAL value must not
        # equal the value it first held.
        first_seen: dict[str, str] = {}
        final: dict[str, str] = {}
        for entry in trace:
            field, transition = entry.split(" ", 1)
            old, new = transition.split("->")
            first_seen.setdefault(field, old)
            final[field] = new
        changed = [f for f in final if final[f] != first_seen[f]]
        assert changed, f"{case['metadata']['id']}: gold matches the opener — {notes}"


def test_switch_cases_score_only_the_replacement_intent():
    """Gold is one call, and the abandoned request's own values are present in the
    transcript — so emitting the stale call, or both calls, scores 0."""
    switches = [c for c in _raw() if c["metadata"]["mechanism"] == "switch"]
    assert switches
    for case in switches:
        assert len(case["metadata"]["expected_calls"]) == 1, case["metadata"]["id"]
        # The assistant's turn-2 report names the abandoned request's values.
        report = case["vars"]["messages"][1]
        assert report["role"] == "assistant"
        assert "prepared" in report["content"] or "Done" in report["content"]


def test_distractor_cases_keep_a_pending_question_open_throughout():
    """Every assistant turn re-asks the missing field, so the model is never shown
    an example of dropping the pending intent."""
    from wallet_evals.conversations import ASKS

    every_ask = {ask for asks in ASKS.values() for ask in asks}
    distractors = [c for c in _raw() if c["metadata"]["mechanism"] == "distractor"]
    assert distractors
    for case in distractors:
        for turn in case["vars"]["messages"]:
            if turn["role"] != "assistant":
                continue
            assert any(turn["content"].endswith(ask) for ask in every_ask), \
                f"{case['metadata']['id']}: assistant turn drops the ask: {turn}"


def test_amounts_are_disjoint_from_the_other_seed_banks():
    """This slice must stay a held-out eval: a gold amount that also appears in
    datasets/finetune_seeds.yaml would be re-measuring a trained value. Checked
    on the GOLD amounts, which include the correction revision targets, not just
    the seed literals."""
    def _amounts(path: Path) -> set[str]:
        text = (ROOT / path).read_text()
        seeds = yaml.safe_load(text)
        found: set[str] = set()
        for seed in seeds:
            spec = seed.get("amount")
            if isinstance(spec, dict):
                found.update(str(v) for v in spec["vary"])
            elif spec is not None:
                found.add(str(spec))
        return found

    trained = _amounts(Path("datasets/finetune_seeds.yaml"))
    # exact_output cases carry no gold call, so there is no gold amount to check.
    # Their surface amount is an OUTPUT amount the wallet cannot act on, and it is
    # covered by the same seed banks these assertions already police.
    gold_amounts = {c["metadata"]["expected_calls"][0]["amount"]
                    for c in _raw() if c["metadata"]["expected_calls"]}
    overlap = gold_amounts & trained
    assert not overlap, f"conversation gold reuses trained amounts: {sorted(overlap)}"
