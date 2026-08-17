"""Unit coverage for the multi-round conversation builders.

These assert the PROPERTIES that make a mechanism worth having — that a
correction case's gold really is the last value stated, that a distractor case's
gold really is unchanged, that a switch case's gold really is the second intent
only — rather than snapshotting surfaces. A surface snapshot would pass even if a
builder scored the wrong turn.
"""
from __future__ import annotations

import random

import pytest

from wallet_evals.conversations import (
    ACTION_FIELDS, ANSWERS, ASKS, DISTRACTORS, MECHANISM_ROUNDS, PREFIX_TEMPLATES,
    REVISIONS, build_correction_case, build_distractor_case, build_progressive_case,
    build_switch_case,
)

TRANSFER = {"action": "transfer", "amount": "6.02", "token": "USDC",
            "recipient": "vitalik.eth"}
SWAP = {"action": "swap", "amount": "0.004", "from_token": "ETH", "to_token": "DAI"}


def _rng() -> random.Random:
    return random.Random(7)


def _turns(case: dict) -> list[dict]:
    return case["vars"]["messages"]


def _user_turns(case: dict) -> list[str]:
    return [m["content"] for m in _turns(case) if m["role"] == "user"]


def _assistant_turns(case: dict) -> list[str]:
    return [m["content"] for m in _turns(case) if m["role"] == "assistant"]


def _gold(case: dict) -> dict:
    calls = case["metadata"]["expected_calls"]
    assert len(calls) == 1, f"expected exactly one gold call, got {calls}"
    return calls[0]


#: An intent field's name in the gold call. Only the recipient differs: the app's
#: transfer schema calls it `to`.
_GOLD_KEY = {"recipient": "to"}


def _gold_value(case: dict, field: str) -> str:
    return _gold(case)[_GOLD_KEY.get(field, field)]


# --------------------------------------------------------------------------
# shape — the invariant every mechanism shares
# --------------------------------------------------------------------------
def _all_cases(rng: random.Random) -> list[tuple[str, int, dict]]:
    cases = []
    for rounds in MECHANISM_ROUNDS["progressive"]:
        cases.append(("progressive", rounds, build_progressive_case(
            TRANSFER, ("amount", "token", "recipient"), rounds, rng, 1)))
    for rounds in MECHANISM_ROUNDS["correction"]:
        cases.append(("correction", rounds,
                      build_correction_case(TRANSFER, "recipient", rounds, rng, 1)))
    for rounds in MECHANISM_ROUNDS["distractor"]:
        cases.append(("distractor", rounds,
                      build_distractor_case(TRANSFER, "recipient", rounds, rng, 1)))
    for rounds in MECHANISM_ROUNDS["switch"]:
        cases.append(("switch", rounds,
                      build_switch_case(TRANSFER, SWAP, rounds, rng, 1)))
    return cases


def test_every_mechanism_produces_the_declared_number_of_rounds():
    for mechanism, rounds, case in _all_cases(_rng()):
        assert len(_user_turns(case)) == rounds, f"{mechanism} @ {rounds}"
        assert len(_assistant_turns(case)) == rounds - 1, f"{mechanism} @ {rounds}"
        assert case["metadata"]["rounds"] == rounds
        assert case["metadata"]["category"].endswith(f"-{rounds}r")


def test_the_scored_turn_is_always_the_last_one_and_is_a_user_turn():
    # The model is invoked on the transcript as-is, so a trailing assistant turn
    # would mean scoring a reply to the wallet's own message.
    for mechanism, rounds, case in _all_cases(_rng()):
        assert _turns(case)[-1]["role"] == "user", f"{mechanism} @ {rounds}"


def test_every_case_is_multi_turn_and_carries_exactly_one_gold_call():
    for mechanism, rounds, case in _all_cases(_rng()):
        assert case["metadata"]["query_type"] == "multi_turn"
        assert case["metadata"]["level"] == "payload"
        _gold(case)  # raises unless exactly one


def test_no_case_uses_the_vars_user_message_single_turn_path():
    for _, _, case in _all_cases(_rng()):
        assert "user_message" not in case["vars"]
        assert case["vars"]["messages"]


# --------------------------------------------------------------------------
# progressive disclosure
# --------------------------------------------------------------------------
def test_progressive_opener_omits_every_field_it_will_later_disclose():
    order = ("amount", "token", "recipient")
    case = build_progressive_case(TRANSFER, order, 4, _rng(), 1)
    opener = _user_turns(case)[0]
    for value in ("6.02", "USDC", "vitalik.eth"):
        assert value.lower() not in opener.lower(), \
            f"4-round opener leaked {value!r}: {opener!r}"


def test_progressive_gold_is_the_unchanged_intent():
    for rounds in MECHANISM_ROUNDS["progressive"]:
        case = build_progressive_case(SWAP, ("amount", "from_token", "to_token"),
                                      rounds, _rng(), 1)
        gold = _gold(case)
        assert (gold["amount"], gold["from_token"], gold["to_token"]) == \
            (SWAP["amount"], SWAP["from_token"], SWAP["to_token"])


def test_progressive_withholds_the_tail_of_the_disclosure_order():
    # A 2-round case withholds one field, a 4-round case all three — so the
    # round count is what controls difficulty, not the seed.
    for rounds, expected_withheld in ((2, 1), (3, 2), (4, 3)):
        case = build_progressive_case(TRANSFER, ("amount", "token", "recipient"),
                                      rounds, _rng(), 1)
        # One clarification per withheld field.
        assert len(_assistant_turns(case)) == expected_withheld


# --------------------------------------------------------------------------
# correction — the load-bearing mechanism
# --------------------------------------------------------------------------
def test_correction_gold_uses_the_revised_value_not_the_original():
    """The whole point: at least one field's gold value differs from the seed
    intent, and the ORIGINAL value is present in the transcript as a decoy."""
    rng = _rng()
    for rounds in MECHANISM_ROUNDS["correction"]:
        case = build_correction_case(TRANSFER, "recipient", rounds, rng, 1)
        gold = _gold(case)
        revised = [f for f in ("amount", "token") if gold[f] != TRANSFER[f]]
        assert revised, f"{rounds}-round correction case revised nothing: {gold}"
        # mutate_case scrambles letter case, so compare case-insensitively.
        transcript = " ".join(_user_turns(case)).lower()
        for field in revised:
            assert TRANSFER[field].lower() in transcript, \
                f"stale {field} value {TRANSFER[field]!r} missing from the transcript"


def test_correction_never_revises_a_field_back_to_its_original_value():
    """The defect this guards: with a 4-symbol token bank, USDC -> ETH -> USDC is
    a plausible two-revision sequence whose gold is identical to the seed intent.
    A model that ignored every correction would score 1 on it, so the case would
    measure nothing. Swept over many seeds because it is a probabilistic hazard,
    not a deterministic one."""
    rng = random.Random(101)
    for rounds in MECHANISM_ROUNDS["correction"]:
        for withheld in ACTION_FIELDS["transfer"]:
            stated = [f for f in ACTION_FIELDS["transfer"] if f != withheld]
            for _ in range(40):
                case = build_correction_case(TRANSFER, withheld, rounds, rng, 1)
                notes = case["metadata"]["notes"]
                touched = {entry.split()[0] for entry in
                           notes.split("revisions: ")[1].split(", ")}
                for field in touched & set(stated):
                    assert _gold_value(case, field) != TRANSFER[field], (
                        f"{field} was revised but ended back at its original "
                        f"value {TRANSFER[field]!r} — {notes}")


def test_correction_makes_one_revision_per_round_after_the_first():
    for rounds in MECHANISM_ROUNDS["correction"]:
        case = build_correction_case(SWAP, "amount", rounds, _rng(), 1)
        assert f"{rounds - 1} revisions" in case["metadata"]["notes"]


def test_correction_never_revises_the_withheld_field():
    # The withheld field is answered exactly once, in the final turn. Revising it
    # too would make the assistant's standing clarifying question incoherent.
    for withheld in ACTION_FIELDS["transfer"]:
        case = build_correction_case(TRANSFER, withheld, 6, _rng(), 1)
        notes = case["metadata"]["notes"]
        assert f"withheld={withheld}" in notes
        assert _gold_value(case, withheld) == TRANSFER[withheld]
        assert f"{withheld} " not in notes.split("revisions: ")[1]


def test_correction_never_produces_a_self_swap():
    # A revision that set from_token == to_token would be an unanswerable intent.
    rng = random.Random(3)
    for rounds in MECHANISM_ROUNDS["correction"]:
        for withheld in ACTION_FIELDS["swap"]:
            for _ in range(20):
                gold = _gold(build_correction_case(SWAP, withheld, rounds, rng, 1))
                assert gold["from_token"] != gold["to_token"], gold


def test_correction_final_turn_answers_the_withheld_field_and_revises():
    case = build_correction_case(TRANSFER, "recipient", 2, _rng(), 1)
    final = _user_turns(case)[-1]
    assert "vitalik.eth" in final.lower(), final
    # The revision phrasings all name both values; one of them must be here.
    assert any(word in final.lower() for word in ("actually", "wait", "correction",
                                                  "sorry", "change", "instead", "not")), \
        final


# --------------------------------------------------------------------------
# distractor
# --------------------------------------------------------------------------
def test_distractor_gold_is_unchanged_by_the_interruptions():
    for rounds in MECHANISM_ROUNDS["distractor"]:
        gold = _gold(build_distractor_case(TRANSFER, "amount", rounds, _rng(), 1))
        assert (gold["amount"], gold["token"], gold["to"]) == \
            (TRANSFER["amount"], TRANSFER["token"], TRANSFER["recipient"])


def test_distractor_inserts_rounds_minus_two_interruptions():
    for rounds in MECHANISM_ROUNDS["distractor"]:
        case = build_distractor_case(TRANSFER, "amount", rounds, _rng(), 1)
        questions = {q for q, _ in DISTRACTORS}
        # Mutators rewrite the surfaces, so match on the assistant's canned
        # answers instead — they are never mutated.
        answers = {a for _, a in DISTRACTORS}
        hits = sum(1 for turn in _assistant_turns(case)
                   if any(turn.startswith(a) for a in answers))
        assert hits == rounds - 2, f"{rounds} rounds: {hits} distractor replies"
        assert questions  # the bank is non-empty


def test_distractor_never_reuses_the_same_interruption_within_one_case():
    # Repeating one question would weaken the test into "can it ignore the same
    # sentence four times" rather than "does the intent survive varied noise".
    case = build_distractor_case(TRANSFER, "amount", 6, _rng(), 1)
    answers = [a for _, a in DISTRACTORS]
    used = [turn for turn in _assistant_turns(case)
            if any(turn.startswith(a) for a in answers)]
    assert len(used) == len(set(used))


def test_distractor_assistant_always_re_asks_the_missing_field():
    case = build_distractor_case(SWAP, "to_token", 6, _rng(), 1)
    asks = ASKS[("swap", "to_token")]
    for turn in _assistant_turns(case):
        assert any(turn.endswith(ask) for ask in asks), turn


# --------------------------------------------------------------------------
# intent switch
# --------------------------------------------------------------------------
def test_switch_gold_is_the_second_intent_alone():
    for rounds in MECHANISM_ROUNDS["switch"]:
        case = build_switch_case(TRANSFER, SWAP, rounds, _rng(), 1)
        gold = _gold(case)
        assert gold["tool"] == "swap"
        assert (gold["amount"], gold["from_token"], gold["to_token"]) == \
            (SWAP["amount"], SWAP["from_token"], SWAP["to_token"])
        assert len(case["metadata"]["expected_calls"]) == 1, \
            "gold must not carry the abandoned call as well"


def test_switch_metadata_describes_the_second_intent_not_the_first():
    # A report grouped by protocol must attribute the case to the action it
    # actually scores, or a swap failure would be counted against transfers.
    case = build_switch_case(TRANSFER, SWAP, 4, _rng(), 1)
    assert case["metadata"]["protocol"] == "uniswap"
    case = build_switch_case(SWAP, TRANSFER, 4, _rng(), 1)
    assert case["metadata"]["protocol"] == "transfer"


def test_switch_transcript_contains_the_abandoned_request_as_a_decoy():
    case = build_switch_case(TRANSFER, SWAP, 2, _rng(), 1)
    transcript = " ".join(_user_turns(case) + _assistant_turns(case)).lower()
    assert "6.02" in transcript and "vitalik.eth" in transcript


def test_switch_cancellation_is_never_corrupted_by_a_mutator():
    """The cancellation verb is the ONLY signal to abandon the first request, so
    a typo'd "Forget that one." -> "Forgte that one." makes the case unanswerable
    rather than harder — the same reason generation._PROTECTED_WORDS shields the
    RAILGUN privacy verbs. Observed from the typo mutator before the fix; swept
    over many seeds because it only fired on some."""
    from wallet_evals.conversations import CANCELS

    rng = random.Random(23)
    for rounds in MECHANISM_ROUNDS["switch"]:
        for _ in range(40):
            case = build_switch_case(TRANSFER, SWAP, rounds, rng, 1)
            turns = _user_turns(case)
            # Either a dedicated cancel round (4+ rounds) or the final turn's
            # lead-in carries one of the CANCELS verbatim.
            assert any(any(c in turn for c in CANCELS) for turn in turns) \
                or turns[-1].startswith(("Instead, ", "What I actually want",
                                         "Let's do this one instead")), \
                f"{rounds} rounds: no intact cancellation in {turns}"


# --------------------------------------------------------------------------
# determinism + bank completeness
# --------------------------------------------------------------------------
def test_builders_are_deterministic_for_a_fixed_seed():
    for build in (
        lambda rng: build_progressive_case(TRANSFER, ("amount", "token", "recipient"),
                                           4, rng, 1),
        lambda rng: build_correction_case(TRANSFER, "recipient", 6, rng, 1),
        lambda rng: build_distractor_case(TRANSFER, "amount", 6, rng, 1),
        lambda rng: build_switch_case(TRANSFER, SWAP, 6, rng, 1),
    ):
        assert build(random.Random(11)) == build(random.Random(11))


@pytest.mark.parametrize("action", sorted(ACTION_FIELDS))
def test_every_proper_subset_of_fields_has_an_opener(action):
    """Progressive disclosure opens on any subset of stated fields, so a missing
    template is a KeyError at generation time — assert the bank is complete."""
    fields = ACTION_FIELDS[action]
    for size in range(len(fields)):
        for subset in __import__("itertools").combinations(sorted(fields), size):
            assert (action, subset) in PREFIX_TEMPLATES, (action, subset)


@pytest.mark.parametrize("action", sorted(ACTION_FIELDS))
def test_every_field_has_an_ask_an_answer_and_a_revision(action):
    for field in ACTION_FIELDS[action]:
        assert ASKS[(action, field)]
        assert ANSWERS[(action, field)]
        assert REVISIONS[(action, field)]


def test_distractor_bank_is_large_enough_for_the_longest_conversation():
    # A 6-round distractor case samples 4 without replacement; a 6-round switch
    # case samples 3 plus a cancel round.
    assert len(DISTRACTORS) >= 4
