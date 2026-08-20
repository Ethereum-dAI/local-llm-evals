"""Multi-round conversation cases: generation, the training rows, and rehearsal.

Three suites merged into one file — the generator, the fine-tune rows it feeds, and
the rehearsal turns bundled with them are one pipeline, and a change to any of them
breaks the others. The section banners name the file each came from.
"""
from __future__ import annotations

import json
import pytest
import random
import sys
import yaml

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_finetune_data as fg           # noqa: E402  (needs the path above)
from wallet_evals.conversations import (      # noqa: E402
    ACTION_FIELDS, ANSWERS, ASKS, DISTRACTORS, MECHANISM_ROUNDS, PREFIX_TEMPLATES,
    REVISIONS, build_correction_case, build_distractor_case, build_progressive_case,
    build_switch_case,
)
from wallet_evals.conversations import MECHANISM_ABBREV
from wallet_evals.finetune import assistant_target
from wallet_evals.generation import ENS_NAMES as TEST_ENS_NAMES
from wallet_evals.rehearsal import (  # noqa: E402
    REHEARSAL_TURNS, build_rehearsal_case,
)

# ============================================================================
# test_conversations
# ============================================================================
#
# Unit coverage for the multi-round conversation builders.
#
# These assert the PROPERTIES that make a mechanism worth having — that a
# correction case's gold really is the last value stated, that a distractor case's
# gold really is unchanged, that a switch case's gold really is the second intent
# only — rather than snapshotting surfaces. A surface snapshot would pass even if a
# builder scored the wrong turn.

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


_GOLD_KEY = {"recipient": "to"}


def _gold_value(case: dict, field: str) -> str:
    return _gold(case)[_GOLD_KEY.get(field, field)]


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

# ============================================================================
# test_finetune_conversations
# ============================================================================
#
# Guards for the multi-round TRAINING rows added to close the depth collapse.
#
# The measurements these protect (results/history.md, and the 1000-case
# benchmark before it):
#
#   * training was 85.9% single-turn and 0% three-plus rounds, while the benchmark
#     runs to six — the fine-tune held 95.8% at one round and fell to 49.0% at six
#     against a base model that is flat across depth;
#   * `exact_output` scored 0/15 at EVERY epoch because no training row taught it;
#   * every ENS recipient in training was the single name `vitalik.eth`, and novel
#     ENS names drew FALSE REFUSALS ("That isn't a valid Ethereum address").
#
# The point of this file is that none of those can silently come back, and that the
# fix cannot be achieved by leaking held-out data.

sys.path.insert(0, str(ROOT / "scripts"))


TRAIN = ROOT / "data_for_finetune" / "gemma4_train.jsonl"


DEV = ROOT / "pf" / "tests.dev.yaml"


HELD_OUT_SETS = tuple(sorted(
    p for p in (ROOT / "pf").glob("tests.*.yaml")))


def _train_examples() -> list[dict]:
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


def _joined_user_turns(messages: list[dict]) -> str:
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
    for ex in _train_examples():
        blob = _joined_user_turns(ex["messages"]) + json.dumps(ex.get("expected_calls") or [])
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
            surfaces.add(_joined_user_turns(messages))
        elif case["vars"].get("user_message"):
            surfaces.add(case["vars"]["user_message"])
    for ex in _train_examples():
        assert _joined_user_turns(ex["messages"]) not in surfaces, \
            f"{ex['id']} leaks into {dataset.name}"


def test_training_covers_three_to_six_rounds():
    """0% three-plus rounds was the distribution mismatch behind the depth
    collapse. Counted from the id (conv-<mech>-<N>r-<idx>) so it reflects what the
    builders emitted rather than a field we could get wrong twice."""
    rounds: dict[int, int] = {}
    for ex in _conversation_rows(_train_examples()):
        part = [p for p in ex["id"].split("-") if p.endswith("r") and p[:-1].isdigit()]
        assert part, f"{ex['id']}: no round marker in id"
        rounds[int(part[0][:-1])] = rounds.get(int(part[0][:-1]), 0) + 1
    assert set(rounds) >= {3, 4}, f"missing depth in training: {sorted(rounds)}"
    assert sum(rounds.values()) >= 300, f"only {sum(rounds.values())} multi-round rows"
    assert max(rounds) >= 5, f"deepest trained conversation is {max(rounds)} rounds"


def test_exact_output_rows_exist_and_teach_no_call():
    """The 0/15-at-every-epoch mechanism. Gold must be empty AND the target must
    carry no DSL opener, or the row would teach the opposite of the lesson."""
    rows = [ex for ex in _train_examples()
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
    for ex in _train_examples():
        for call in ex.get("expected_calls") or []:
            for value in _ens_values(call):
                counts[value] = counts.get(value, 0) + 1
    assert counts, "no ENS recipients in training gold at all"
    total = sum(counts.values())
    top, top_n = max(counts.items(), key=lambda kv: kv[1])
    assert top_n / total <= 0.55, \
        f"{top} is {top_n}/{total} ({top_n / total:.0%}) of ENS recipients"
    assert len(counts) >= 8, f"only {len(counts)} distinct ENS recipients: {counts}"


def test_held_out_mechanisms_never_enter_training():
    """`switch` is the transfer control: it survived over-training untouched
    (23/23 at every epoch), so if trained mechanisms improve while `switch` holds,
    the gain generalized. Training on it destroys the only such control we have."""
    assert fg.HELD_OUT_MECHANISMS, "no mechanism held out — nothing measures transfer"
    abbrevs = {m: MECHANISM_ABBREV[m] for m in fg.HELD_OUT_MECHANISMS}
    leaked = []
    for ex in _train_examples():
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

# ============================================================================
# test_rehearsal
# ============================================================================
#
# Guards for the rehearsal rows (wallet_evals/rehearsal.py).
#
# These rows exist to make "answer in prose" reachable again in a training mix that
# is otherwise ~97% call-emitting. Every measured regression in the fine-tune was a
# case of calling when it should not have, so the failure mode of this fix is the
# mirror image: too much prose, and the model stops calling when it should. Both
# directions are pinned here.

sys.path.insert(0, str(ROOT / "scripts"))


CALL_OPENERS = ("<|tool_call>", "<tool_call>", "functools[", "```json")


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
    rows = [ex for ex in _train_examples() if ex.get("category") == "rehearsal"]
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
    examples = _train_examples()
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
