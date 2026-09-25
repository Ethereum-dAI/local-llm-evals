"""Integrity of the hard slice (pf/tests.hard.yaml) and the ~500-case benchmark
(pf/tests.benchmark.yaml) built from it plus a stratified subset of the 1000.

Same questions test_dataset_integrity.py asks of every other slice: is the file a
byte-stable output of its generator, does every gold self-score, is it disjoint from
training, and does each mechanism actually contain the trap it claims to?
"""
from __future__ import annotations

import collections
import random
import re
from pathlib import Path

import yaml

from scripts.build_benchmark import QUOTAS, build, select_subset, stratum
from scripts.generate_hard_cases import PLAN, SEED, TARGET_TOTAL, build_selection, load_intents
from wallet_evals.conversations import ALT_AMOUNTS
from wallet_evals.hard_cases import (
    INJECTED_AMOUNTS, MECHANISM_ROUNDS, NO_CALL_MECHANISMS, REFUSAL_KINDS, comma_decimal,
)
from wallet_evals.promptfoo import load_cases
from wallet_evals.schema import ParsedTurn
from wallet_evals.scorer import score_case

ROOT = Path(__file__).resolve().parents[1]
HARD = ROOT / "pf" / "tests.hard.yaml"
BENCH = ROOT / "pf" / "tests.benchmark.yaml"
COMBINED = ROOT / "pf" / "tests.combined.yaml"
SEEDS = ROOT / "datasets" / "seeds.hard.yaml"
HEX40 = re.compile(r"0x[0-9a-fA-F]{40}(?![0-9a-fA-F])")


def _hard() -> list[dict]:
    return yaml.safe_load(HARD.read_text())


def _bench() -> list[dict]:
    return yaml.safe_load(BENCH.read_text())


def _combined() -> list[dict]:
    return yaml.safe_load(COMBINED.read_text())


def _user_turns(case: dict) -> list[str]:
    msgs = case["vars"].get("messages")
    if msgs:
        return [m["content"] for m in msgs if m["role"] == "user"]
    return [case["vars"]["user_message"]]


def _seed_amounts(path: Path) -> set[str]:
    found: set[str] = set()
    for seed in yaml.safe_load(path.read_text()) or []:
        spec = seed.get("amount")
        if isinstance(spec, dict):
            found.update(str(v) for v in spec["vary"])
        elif spec is not None:
            found.add(str(spec))
    return found


# ---------------------------------------------------------------------------
# the hard slice
# ---------------------------------------------------------------------------
def test_hard_file_is_a_byte_stable_output_of_its_generator():
    rng = random.Random(SEED)
    assert build_selection(load_intents(SEEDS, rng), rng) == _hard(), \
        "pf/tests.hard.yaml is stale -- rerun scripts/generate_hard_cases.py"


def test_hard_composition_matches_the_declared_plan():
    got = collections.Counter(
        (c["metadata"]["mechanism"],
         c["metadata"]["kind"] if c["metadata"]["mechanism"] in ("surface", "embedded_refusal")
         else c["metadata"]["rounds"])
        for c in _hard())
    assert dict(got) == dict(PLAN)
    assert len(_hard()) == TARGET_TOTAL


def test_every_hard_case_self_scores_one():
    for case in load_cases(HARD):
        turn = ParsedTurn(tool_calls=[c.as_parsed_call() for c in case.expected_calls])
        assert score_case(case, turn) == 1, case.id


def test_hard_ids_are_unique_and_namespaced():
    ids = [c["metadata"]["id"] for c in _hard()]
    assert len(ids) == len(set(ids))
    assert all(i.startswith("hard-") for i in ids)


def test_empty_gold_is_exactly_the_declared_no_call_mechanisms():
    """Empty gold passes any model that stays silent, so it must not spread."""
    for case in _hard():
        md = case["metadata"]
        assert (md["mechanism"] in NO_CALL_MECHANISMS) == (not md["expected_calls"]), md["id"]


def test_rounds_describe_the_transcript():
    for case in _hard():
        md = case["metadata"]
        assert md["rounds"] in MECHANISM_ROUNDS[md["mechanism"]], md["id"]
        assert len(_user_turns(case)) == md["rounds"], md["id"]
        msgs = case["vars"].get("messages")
        if msgs:
            assert msgs[-1]["role"] == "user", md["id"]


def test_gold_amounts_are_disjoint_from_training_and_every_other_bank():
    """The hard slice must be held out: no gold amount may be one a fine-tune trained
    on or one the 1000 / dev set already measures."""
    banks = set(ALT_AMOUNTS)
    for name in ("finetune_seeds.yaml", "seeds.yaml", "seeds.arithmetic.yaml",
                 "seeds.conversations.yaml", "seeds.dev.yaml"):
        banks |= _seed_amounts(ROOT / "datasets" / name)
    gold = {c["metadata"]["expected_calls"][0]["amount"]
            for c in _hard() if c["metadata"]["expected_calls"]}
    assert not gold & banks, sorted(gold & banks)


def test_stacked_cases_are_deeper_than_anything_in_the_1000_and_carry_both_traps():
    for case in (c for c in _hard() if c["metadata"]["mechanism"] == "stacked"):
        md = case["metadata"]
        assert md["rounds"] >= 7, md["id"]
        revisions = int(re.search(r"(\d+) revisions", md["notes"]).group(1))
        distractors = int(re.search(r"(\d+) distractors", md["notes"]).group(1))
        assert revisions >= 2 and distractors >= 1, md["id"]


def test_injected_values_never_reach_gold():
    """The trap is the injected address/amount; if gold carried either, a model that
    obeyed the injection would pass."""
    for case in (c for c in _hard() if c["metadata"]["mechanism"] == "injection_distractor"):
        md = case["metadata"]
        target = re.search(r"to (0x[0-9a-f]{40})", md["notes"]).group(1)
        call = md["expected_calls"][0]
        assert target not in {str(v).lower() for v in call.values()}, md["id"]
        assert call["amount"] not in INJECTED_AMOUNTS, md["id"]
        assert any(target in t for t in _user_turns(case)), md["id"]


def test_unresolvable_recipients_contain_no_sendable_address():
    for case in (c for c in _hard()
                 if c["metadata"]["mechanism"] == "unresolvable_recipient"):
        last = _user_turns(case)[-1]
        assert not HEX40.search(last), case["metadata"]["id"]
        assert ".eth" not in last, case["metadata"]["id"]


def test_unresolvable_amounts_never_price_a_stablecoin():
    """"$50 of USDC" arguably is 50 USDC -- that case would be ambiguous, not hard."""
    for case in (c for c in _hard() if c["metadata"]["mechanism"] == "unresolvable_amount"):
        last = _user_turns(case)[-1]
        assert re.search(r"\b(ETH|WETH)\b", last), case["metadata"]["id"]


def test_spoken_surfaces_dot_the_gold_recipient():
    """A spoken ENS name ("alice dot eth") must land in gold DOTTED -- the wallet
    rejects any recipient without a dot."""
    for case in (c for c in _hard() if c["metadata"]["kind"] == "spoken_revision"):
        call = case["metadata"]["expected_calls"][0]
        if " dot " in _user_turns(case)[-1]:
            assert call["tool"] == "transfer" and " " not in call["to"] \
                and "." in call["to"], case["metadata"]["id"]


def test_surface_revisions_leave_the_stale_amount_in_the_transcript():
    """The point of the revision form: the numeric amount in turn 1 is wrong, and
    gold carries the surface-form amount from turn 2."""
    for case in (c for c in _hard() if c["metadata"]["kind"] in
                 ("words_revision", "k_revision", "spoken_revision")):
        md = case["metadata"]
        stale = re.search(r"opener said '([^']+)'", md["notes"]).group(1)
        assert md["rounds"] == 2 and stale in _user_turns(case)[0], md["id"]
        assert md["expected_calls"][0]["amount"] != stale, md["id"]


def test_decimal_comma_surfaces_are_unambiguous():
    for case in (c for c in _hard() if c["metadata"]["language"] != "english"):
        amount = case["metadata"]["expected_calls"][0]["amount"]
        said = comma_decimal(amount)
        assert said is not None and said in _user_turns(case)[0], case["metadata"]["id"]


def test_every_embedded_refusal_kind_is_present():
    kinds = {c["metadata"]["kind"] for c in _hard()
             if c["metadata"]["mechanism"] == "embedded_refusal"}
    assert kinds == set(REFUSAL_KINDS)


# ---------------------------------------------------------------------------
# the benchmark
# ---------------------------------------------------------------------------
def test_benchmark_is_a_byte_stable_output_of_its_builder():
    assert build(_combined(), _hard()) == _bench(), \
        "pf/tests.benchmark.yaml is stale -- rerun scripts/build_benchmark.py"


def test_benchmark_size_is_about_five_hundred():
    assert len(_bench()) == sum(QUOTAS.values()) + TARGET_TOTAL
    assert 450 <= len(_bench()) <= 550


def test_every_case_of_the_1000_has_exactly_one_stratum_with_a_quota():
    strata = {stratum(c) for c in _combined()}
    assert strata == set(QUOTAS)


def test_subset_cases_are_byte_identical_to_the_1000():
    """This is what lets every recorded run be re-scored on the subset offline."""
    by_id = {c["metadata"]["id"]: c for c in _combined()}
    subset = [c for c in _bench() if not c["metadata"]["id"].startswith("hard-")]
    assert len(subset) == sum(QUOTAS.values())
    for case in subset:
        assert case == by_id[case["metadata"]["id"]], case["metadata"]["id"]


def test_subset_meets_every_quota():
    got = collections.Counter(stratum(c) for c in select_subset(_combined()))
    assert dict(got) == QUOTAS


def test_every_safety_refusal_is_kept():
    """Already 2-4 cases per kind -- sampling them down would leave nothing to read."""
    safety = {c["metadata"]["id"] for c in _combined()
              if c["metadata"]["category"].startswith("safety-refusal-")}
    assert safety <= {c["metadata"]["id"] for c in _bench()}


def test_every_benchmark_case_self_scores_one():
    for case in load_cases(BENCH):
        turn = ParsedTurn(tool_calls=[c.as_parsed_call() for c in case.expected_calls])
        assert score_case(case, turn) == 1, case.id


# ---------------------------------------------------------------------------
# the 50-case discrimination panel
# ---------------------------------------------------------------------------
PANEL = ROOT / "pf" / "tests.panel.yaml"
PANEL_IDS = ROOT / "datasets" / "panel_ids.json"


def test_panel_is_a_byte_stable_output_of_its_id_file():
    import json
    from scripts.build_panel import _all_cases, materialise
    ids = json.loads(PANEL_IDS.read_text())["ids"]
    assert materialise(ids, _all_cases()) == yaml.safe_load(PANEL.read_text()), \
        "pf/tests.panel.yaml is stale -- rerun scripts/build_panel.py"


def test_panel_has_fifty_unique_cases_that_split_the_models_when_chosen():
    import json
    data = json.loads(PANEL_IDS.read_text())
    assert len(data["ids"]) == len(set(data["ids"])) == 50
    for i, verdicts in data["selection_run"].items():
        assert len(set(verdicts.values())) == 2, f"{i} did not split the models"


def test_every_panel_case_self_scores_one():
    for case in load_cases(PANEL):
        turn = ParsedTurn(tool_calls=[c.as_parsed_call() for c in case.expected_calls])
        assert score_case(case, turn) == 1, case.id
