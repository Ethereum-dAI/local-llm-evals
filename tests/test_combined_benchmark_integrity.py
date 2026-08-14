"""Integrity of the combined benchmark: app-contract (transfer/swap + the
arithmetic slice) concatenated with the protocol families (aave/safe).

Mirrors test_app_contract_integrity.py / test_protocol_integrity.py, which
guard the two source files this one is built from.
"""
from __future__ import annotations

import collections
from pathlib import Path

from wallet_evals.promptfoo import load_cases
from wallet_evals.schema import ParsedTurn
from wallet_evals.scorer import score_case

ROOT = Path(__file__).resolve().parents[1]
COMBINED = ROOT / "pf" / "tests.combined.yaml"
APP_CONTRACT = ROOT / "pf" / "tests.app-contract.yaml"
PROTOCOLS = ROOT / "pf" / "tests.protocols.yaml"


def _load():
    return load_cases(COMBINED)


def test_every_case_self_scores_one():
    for case in _load():
        turn = ParsedTurn(tool_calls=[c.as_parsed_call() for c in case.expected_calls])
        assert score_case(case, turn) == 1, f"{case.id} not self-consistent"


def test_ids_unique_across_the_whole_file():
    ids = [c.id for c in _load()]
    assert len(ids) == len(set(ids))


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


def test_safety_refusals_are_powered_enough_to_detect_a_regression():
    """Refusal is the ONLY failure category the app-contract migration left on
    the model's plate (unit conversion and ENS resolution both moved into the
    wallet). At the 7 cases this benchmark used to carry, a regression like the
    Qwen fine-tune's 100% -> 45% would have been invisible."""
    refusals = [c for c in _load() if c.category.startswith("safety-refusal-")]
    assert len(refusals) >= 40, f"only {len(refusals)} refusal cases"
    kinds = {c.category for c in refusals}
    assert len(kinds) >= 10, f"only {len(kinds)} distinct refusal kinds: {sorted(kinds)}"


def test_aave_and_safe_are_present():
    protos = {c.protocol for c in _load()}
    assert "aave" in protos
    assert "safe" in protos


def test_combined_count_equals_sum_of_its_two_sources():
    combined = _load()
    app_contract = load_cases(APP_CONTRACT)
    protocols = load_cases(PROTOCOLS)
    assert len(combined) == len(app_contract) + len(protocols)


def test_per_family_census_is_visible_and_every_family_present():
    """A census assertion so drift (a family silently shrinking to zero, or a
    generator run losing cases) is visible rather than only caught by eyeballing
    a print statement."""
    cases = _load()
    protos = collections.Counter(c.protocol for c in cases)
    print(f"\ncombined benchmark per-protocol census: {dict(sorted(protos.items()))}")

    # transfer/uniswap = app-contract (incl. arithmetic), safety = refusals;
    # aave/safe = the protocol families this benchmark adds.
    for family in ("transfer", "uniswap", "aave", "safe"):
        assert protos[family] > 0, f"{family} has zero cases in the combined benchmark"

    arithmetic_count = sum(1 for c in cases if c.category.startswith("arithmetic-"))
    refusal_count = sum(1 for c in cases if c.category.startswith("safety-refusal-"))
    aave_count = sum(1 for c in cases if c.protocol == "aave")
    safe_count = sum(1 for c in cases if c.protocol == "safe")
    print(f"arithmetic slice: {arithmetic_count}, refusals: {refusal_count}, "
         f"aave: {aave_count}, safe: {safe_count}")
    assert arithmetic_count > 0 and refusal_count > 0 and aave_count > 0 and safe_count > 0
