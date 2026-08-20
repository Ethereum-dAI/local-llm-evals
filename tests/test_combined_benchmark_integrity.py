"""Integrity of the combined benchmark: app-contract (transfer/swap + the
arithmetic slice + the refusal banks) concatenated with the multi-round
conversation slice.

Mirrors test_app_contract_integrity.py / test_dataset_integrity.py, which
guard the two source files this one is built from.
"""
from __future__ import annotations

import collections
from pathlib import Path

import yaml

from wallet_evals.promptfoo import load_cases
from wallet_evals.schema import ParsedTurn
from wallet_evals.scorer import score_case

ROOT = Path(__file__).resolve().parents[1]
COMBINED = ROOT / "pf" / "tests.combined.yaml"
APP_CONTRACT = ROOT / "pf" / "tests.app-contract.yaml"
CONVERSATIONS = ROOT / "pf" / "tests.conversations.yaml"

#: The benchmark's declared size. A round number, and deliberately asserted: the
#: two source generators can each grow independently, and a benchmark that
#: quietly became 987 or 1043 cases would make run-to-run comparisons wrong in a
#: way no other test would catch.
EXPECTED_TOTAL = 1000

#: The declared round distribution (user turns per case -> count). 1 = the
#: single-turn cases; 2 = the app-contract slice's 93 legacy 2-round cases plus
#: the conversation slice's 180.
#:
#: The conversation slice's own per-round totals are pinned exactly by
#: generate_conversation_cases.EXPECTED_ROUND_TOTALS ({2: 180, 3: 150, 4: 110,
#: 5: 80, 6: 51}), so rounds 3-6 here are those numbers verbatim. Rounds 1 and 2
#: also absorb the app-contract slice, whose 1r/2r split moved by one case
#: (336/273, previously 337/272) when the arithmetic slice was regenerated to
#: diversify its recipients: `build_extra_selection` shuffles that slice's cases
#: together and caps at 40 per action, so a wider recipient pool changes which
#: categories win the draw. The slice is still exactly 80 cases and the benchmark
#: still exactly 1000; multi-round coverage is 66.4%, against 66.3% before.
EXPECTED_ROUNDS = {1: 336, 2: 273, 3: 150, 4: 110, 5: 80, 6: 51}


def _load():
    return load_cases(COMBINED)


def _raw() -> list[dict]:
    return yaml.safe_load(COMBINED.read_text())


def _rounds(case: dict) -> int:
    messages = case["vars"].get("messages")
    if not messages:
        return 1
    return sum(1 for m in messages if m.get("role") == "user")


def test_every_case_self_scores_one():
    for case in _load():
        turn = ParsedTurn(tool_calls=[c.as_parsed_call() for c in case.expected_calls])
        assert score_case(case, turn) == 1, f"{case.id} not self-consistent"


def test_ids_unique_across_the_whole_file():
    ids = [c.id for c in _load()]
    assert len(ids) == len(set(ids))


def test_the_benchmark_is_exactly_its_declared_size():
    assert len(_load()) == EXPECTED_TOTAL


def test_arithmetic_slice_is_present_and_non_empty():
    cats = {c.category for c in _load()}
    arithmetic_cats = {c for c in cats if c.startswith("arithmetic-")}
    assert arithmetic_cats, "no arithmetic-* category found in the combined benchmark"
    arithmetic_cases = [c for c in _load() if c.category.startswith("arithmetic-")]
    assert len(arithmetic_cases) > 0


def test_railgun_is_absent():
    # RAILGUN was removed from the app (local-wallet-mac#86, PR #87) and from
    # pf/tools.json; a benchmark that still scored shield/unshield would be
    # measuring a feature the product does not expose.
    cases = _load()
    assert not any(c.category.startswith("railgun-") for c in cases)
    assert not any("shield" in c.category for c in cases)
    assert not any(c.protocol == "railgun" for c in cases)


def test_aave_and_safe_are_absent():
    """Removed from the benchmark for the same reason RAILGUN was: the wallet
    ships no lending or multisig tool, so `executeTx` gold for Aave's Pool or a
    Safe self-call scored a capability the product does not expose, and it was
    ~25% of the headline number.

    Deliberately a REMOVAL FROM THE BENCHMARK ONLY. pf/tests.protocols.yaml, its
    generator and src/wallet_evals/protocols/ all still exist and still pass
    test_protocol_integrity.py — run that file directly
    (EVAL_DATASET=pf/tests.protocols.yaml scripts/eval.sh) if you want the
    numbers. This test guards the benchmark, not the repo.
    """
    cases = _load()
    assert not any(c.protocol in ("aave", "safe") for c in cases)
    assert not any(c.category.startswith(("aave-", "safe-")) for c in cases)
    # The builder contract is what those cases needed; nothing left should use it.
    assert not any(call.tool in ("executeTx", "readTx")
                   for c in cases for call in c.expected_calls)


def test_safety_refusals_are_powered_enough_to_detect_a_regression():
    """Refusal is the ONLY failure category the app-contract migration left on
    the model's plate (unit conversion and ENS resolution both moved into the
    wallet). At the 7 cases this benchmark used to carry, a regression like the
    Qwen fine-tune's 100% -> 45% would have been invisible."""
    refusals = [c for c in _load() if c.category.startswith("safety-refusal-")]
    assert len(refusals) >= 40, f"only {len(refusals)} refusal cases"
    kinds = {c.category for c in refusals}
    assert len(kinds) >= 10, f"only {len(kinds)} distinct refusal kinds: {sorted(kinds)}"


def test_combined_count_equals_sum_of_its_two_sources():
    combined = _load()
    app_contract = load_cases(APP_CONTRACT)
    conversations = load_cases(CONVERSATIONS)
    assert len(combined) == len(app_contract) + len(conversations)


def test_round_distribution_matches_the_declared_shape():
    """The point of the conversation slice: two thirds of the benchmark is now
    multi-round, spanning 1-6 rounds. Asserted rather than printed so a
    regenerated source file that collapses the long conversations fails here."""
    counts = collections.Counter(_rounds(c) for c in _raw())
    assert dict(counts) == EXPECTED_ROUNDS
    multi = sum(v for k, v in counts.items() if k > 1)
    assert multi / len(_raw()) > 0.6, f"only {multi} multi-round cases"


def test_long_conversations_carry_enough_weight_to_move_the_score():
    """5- and 6-round cases are the ones a model degrading on context length will
    fail first. If they were a handful of cases, that degradation would round to
    nothing in the headline number."""
    long_cases = [c for c in _raw() if _rounds(c) >= 5]
    assert len(long_cases) >= 100, f"only {len(long_cases)} cases of 5+ rounds"


def test_every_conversation_mechanism_is_represented():
    mechanisms = collections.Counter(
        c["metadata"].get("mechanism") for c in _raw()
        if c["metadata"].get("mechanism"))
    assert set(mechanisms) == {"progressive", "correction", "distractor", "switch",
                               "exact_output", "token_address"}
    # The four conversational-memory mechanisms carry the bulk. The two
    # contract-boundary ones are deliberately smaller: each tests a single
    # property of the final turn, and exact_output's gold is empty, so a large
    # bank of them would inflate the score a silent model gets for free.
    for mechanism in ("progressive", "correction", "distractor", "switch"):
        assert mechanisms[mechanism] >= 100, \
            f"{mechanism} has only {mechanisms[mechanism]} cases"
    assert mechanisms["exact_output"] == 32
    assert mechanisms["token_address"] == 24


def test_per_family_census_is_visible_and_every_family_present():
    """A census assertion so drift (a family silently shrinking to zero, or a
    generator run losing cases) is visible rather than only caught by eyeballing
    a print statement."""
    cases = _load()
    protos = collections.Counter(c.protocol for c in cases)
    print(f"\ncombined benchmark per-protocol census: {dict(sorted(protos.items()))}")

    # transfer/uniswap = app-contract + conversations, safety = refusals. Those
    # three are the whole benchmark now that aave/safe are out.
    assert set(protos) == {"transfer", "uniswap", "safety"}
    for family in ("transfer", "uniswap", "safety"):
        assert protos[family] > 0, f"{family} has zero cases in the combined benchmark"

    arithmetic_count = sum(1 for c in cases if c.category.startswith("arithmetic-"))
    refusal_count = sum(1 for c in cases if c.category.startswith("safety-refusal-"))
    conversation_count = sum(1 for c in cases
                             if c.category.startswith("conversation-"))
    print(f"arithmetic slice: {arithmetic_count}, refusals: {refusal_count}, "
          f"conversations: {conversation_count}")
    assert arithmetic_count > 0 and refusal_count > 0 and conversation_count > 0
