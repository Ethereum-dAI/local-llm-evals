"""Dataset integrity: the dev slices, the conversation cases, and degeneracy.

Four suites merged into one file. They all answer the same question about different
slices — is this dataset actually what it claims to be, and is it disjoint from the
things it must be disjoint from (training rows, the frozen benchmark, each other)?
The frozen 1000-case set has its own file, `test_combined_benchmark_integrity.py`.

The last section is this file's ORIGINAL content, kept because it guards the legacy
`pf/tests.yaml` set that predates all of this and is still scored by `scripts/eval.sh`
with no dataset argument. Two of its names were prefixed `legacy` to clear a clash.
"""
from __future__ import annotations

import collections
import json
import pytest
import random
import re
import sys
import yaml

from pathlib import Path
from wallet_evals.promptfoo import load_cases
from wallet_evals.schema import ParsedTurn
from wallet_evals.scorer import score_case
from scripts.generate_conversation_cases import (
    ROUND_PLAN, SEED, TARGET_TOTAL, build_selection, load_intents,
)
from wallet_evals.conversations import MECHANISM_ROUNDS
from wallet_evals.dev_safety import DEV_REFUSAL_SCENARIOS
from wallet_evals.generation import EXTRA_REFUSAL_SCENARIOS
from wallet_evals.promptfoo import load_cases
from wallet_evals.schema import ParsedTurn
from wallet_evals.scorer import score_case

# ============================================================================
# test_dev_set
# ============================================================================
#
# The dev set must be disjoint from the test set AND from training.
#
# The three-way split is the whole safeguard:
#
#     TRAIN  data_for_finetune/*.jsonl      what the model learns from
#     DEV    pf/tests.dev.yaml              what picks the checkpoint
#     TEST   pf/tests.combined.yaml         the number we report (FROZEN)
#
# Selecting a checkpoint on cases that appear in TEST is selecting on the test set.
# That is not a hypothetical: this whole line of work started from a fine-tune that
# scored 96.5% on a benchmark matching its training distribution and 69.3% on cases
# it had not seen. Doing the same thing one level up — tuning on the reported set —
# would reproduce that error while looking like progress.
#
# So this is checked by SURFACE and by CASE ID, and on the gold as well, because a
# rename or a re-seed must not be able to slip an overlap through.

ROOT = Path(__file__).resolve().parents[1]


DEV = ROOT / "pf" / "tests.dev.yaml"


TEST = ROOT / "pf" / "tests.combined.yaml"


TRAIN_GLOB = "data_for_finetune/*.jsonl"


def _yaml_cases(path: Path) -> list[dict]:
    return yaml.safe_load(path.read_text())


def _surface_from_vars(vars_: dict) -> str:
    msgs = vars_.get("messages")
    if msgs:
        return "\n".join(m["content"] for m in msgs if m.get("role") == "user")
    return vars_.get("user_message", "")


def test_dev_set_exists_and_has_expected_size():
    cases = _yaml_cases(DEV)
    assert len(cases) == 145, f"dev set is {len(cases)} cases; update this constant"


def test_dev_ids_do_not_collide_with_the_test_set():
    dev = {c["metadata"]["id"] for c in _yaml_cases(DEV)}
    test = {c["metadata"]["id"] for c in _yaml_cases(TEST)}
    assert not (dev & test), f"id collision with the frozen test set: {sorted(dev & test)[:5]}"


def test_dev_surfaces_do_not_appear_in_the_test_set():
    """The load-bearing one. Two generators sharing template banks can emit the
    same conversation from different seeds, and an id prefix would not catch it."""
    dev = {_surface_from_vars(c["vars"]) for c in _yaml_cases(DEV)}
    test = {_surface_from_vars(c["vars"]) for c in _yaml_cases(TEST)}
    overlap = dev & test
    assert not overlap, (
        f"{len(overlap)} dev conversation(s) also appear in the frozen test set — "
        f"gating on these would be selecting on the test set. First: "
        f"{sorted(overlap)[0][:120]!r}")


def test_dev_surfaces_do_not_appear_in_training():
    train_files = sorted(ROOT.glob(TRAIN_GLOB))
    if not train_files:
        pytest.skip("no training JSONLs present (they are gitignored)")
    dev = {_surface_from_vars(c["vars"]) for c in _yaml_cases(DEV)}
    for path in train_files:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            ex = json.loads(line)
            users = "\n".join(m["content"] for m in ex["messages"]
                              if m["role"] == "user")
            assert users not in dev, f"{path.name}:{ex.get('id')} leaks into the dev set"


def test_dev_amounts_are_disjoint_from_every_other_bank():
    """A dev amount that a model trained on is not held out in any useful sense."""
    def amounts(rel: str) -> set[str]:
        found: set[str] = set()
        for seed in yaml.safe_load((ROOT / rel).read_text()) or []:
            spec = seed.get("amount")
            if isinstance(spec, dict):
                found |= {str(v) for v in spec["vary"]}
            elif spec is not None:
                found.add(str(spec))
        return found

    others: set[str] = set()
    for rel in ("datasets/seeds.yaml", "datasets/seeds.arithmetic.yaml",
                "datasets/seeds.conversations.yaml", "datasets/finetune_seeds.yaml"):
        others |= amounts(rel)
    conv = (ROOT / "src" / "wallet_evals" / "conversations.py").read_text()
    others |= set(re.findall(r'"([\d.]+)"', conv.split("ALT_AMOUNTS")[1].split(")")[0]))

    dev = amounts("datasets/seeds.dev.yaml")
    assert dev, "dev seeds declare no amounts"
    assert not (dev & others), f"dev reuses amounts: {sorted(dev & others)}"


def test_dev_ens_bank_is_disjoint_from_the_test_and_train_banks():
    """`vitalik.eth` is the training bank; generation.ENS_NAMES is the test bank.
    A dev set drawing on either cannot distinguish recall from generalisation."""
    gen = (ROOT / "src" / "wallet_evals" / "generation.py").read_text()
    test_bank = set(re.findall(r'"([a-z0-9.\-]+\.eth)"',
                              gen.split("ENS_NAMES")[1].split(")")[0]))
    dev_names = {to for c in _yaml_cases(DEV)
                 for call in (c["metadata"].get("expected_calls") or [])
                 for to in [str(call.get("to", ""))] if to.endswith(".eth")}
    assert dev_names, "dev set exercises no ENS recipients"
    assert not (dev_names & test_bank), \
        f"dev reuses TEST ENS names: {sorted(dev_names & test_bank)}"
    assert "vitalik.eth" not in dev_names, "dev reuses the TRAINING ENS name"


def test_dev_set_is_out_of_distribution_for_depth_and_the_new_axes():
    """What makes this set able to see the failure an in-distribution split cannot:
    training is 85.9% one-turn / 14.1% two-turn / 0% three-plus, and has zero
    exact-output and zero token-as-address rows."""
    cases = _yaml_cases(DEV)
    deep = [c for c in cases if c["metadata"]["rounds"] >= 3]
    assert len(deep) / len(cases) > 0.85, \
        f"only {len(deep)}/{len(cases)} cases are 3+ rounds"
    mechs = {c["metadata"]["mechanism"] for c in cases}
    for required in ("exact_output", "token_address"):
        n = sum(1 for c in cases if c["metadata"]["mechanism"] == required)
        assert n >= 15, f"{required} has only {n} dev cases — too coarse to gate on"
    assert {"correction", "distractor"} <= mechs


def test_dev_gold_self_scores_under_the_real_scorer():
    """Same invariant the other datasets carry: if gold cannot score itself, the
    checkpoint selector is measuring the scorer, not the model."""
    from wallet_evals.promptfoo import load_cases
    from wallet_evals.schema import ParsedTurn
    from wallet_evals.scorer import score_case
    for case in load_cases(DEV):
        turn = ParsedTurn(tool_calls=[c.as_parsed_call() for c in case.expected_calls])
        assert score_case(case, turn) == 1, f"{case.id} does not self-score"

# ============================================================================
# test_dev_safety
# ============================================================================
#
# Guards for pf/tests.dev.safety.yaml — the dev set's safety coverage.
#
# This slice exists so a prompt or recipe change can be judged on safety as well as
# accuracy. It is only worth anything if it is genuinely held out and genuinely
# scoreable, which is what these check.

sys.path.insert(0, str(ROOT / "scripts"))


DEV_SAFETY = ROOT / "pf" / "tests.dev.safety.yaml"


TRAIN = ROOT / "data_for_finetune" / "gemma4_train.jsonl"


def _dev_safety_cases() -> list[dict]:
    return yaml.safe_load(DEV_SAFETY.read_text())


def _surface_of_case(case: dict) -> str:
    v = case["vars"]
    if v.get("messages"):
        return "\n".join(m["content"] for m in v["messages"] if m["role"] == "user")
    return v.get("user_message", "")


def test_every_case_expects_no_call():
    """A safety case whose gold carried a call would invert the whole slice."""
    for case in _dev_safety_cases():
        assert case["metadata"]["expected_calls"] == [], case["metadata"]["id"]


def test_covers_every_rule_kind_in_the_prompt():
    """One kind per SAFETY rule. A rule with no case is an unmeasured rule."""
    theirs = {s["kind"] for s in EXTRA_REFUSAL_SCENARIOS}
    ours = {s["kind"] for s in DEV_REFUSAL_SCENARIOS}
    assert ours == theirs, f"missing {sorted(theirs - ours)}, extra {sorted(ours - theirs)}"


def test_surfaces_are_disjoint_from_every_held_out_set():
    """New surfaces are the point: same rule, unseen phrasing. If a surface matched
    the benchmark's, this would be measuring recall of a sentence."""
    mine = {_surface_of_case(c) for c in _dev_safety_cases()}
    for path in sorted((ROOT / "pf").glob("tests.*.yaml")):
        if path.name == DEV_SAFETY.name:
            continue
        theirs = {_surface_of_case(c) for c in yaml.safe_load(path.read_text()) or []}
        overlap = mine & theirs
        assert not overlap, f"{path.name} shares surfaces: {sorted(overlap)[:3]}"


def test_surfaces_never_appear_in_training():
    if not TRAIN.exists():
        pytest.skip(f"{TRAIN} not generated")
    rows = [json.loads(l) for l in TRAIN.read_text().splitlines() if l.strip()]
    trained = {"\n".join(m["content"] for m in r["messages"] if m["role"] == "user")
               for r in rows}
    for case in _dev_safety_cases():
        assert _surface_of_case(case) not in trained, f"{case['metadata']['id']} is trained on"


def test_templates_are_not_reused_from_the_benchmark_bank():
    """Checks the SOURCE templates too, not just the rendered surfaces — a mutator
    could otherwise mask a copied template."""
    theirs = {t for s in EXTRA_REFUSAL_SCENARIOS for t in s["templates"]}
    mine = {t for s in DEV_REFUSAL_SCENARIOS for t in s["templates"]}
    assert not mine & theirs, f"reused templates: {sorted(mine & theirs)[:3]}"


def test_ids_are_namespaced():
    """So a merged report can never confuse these with the benchmark's refusals."""
    for case in _dev_safety_cases():
        assert case["metadata"]["id"].startswith("devsafety-"), case["metadata"]["id"]

# ============================================================================
# test_conversation_integrity
# ============================================================================
#
# Integrity of the generated conversation slice (pf/tests.conversations.yaml).
#
# Mirrors test_app_contract_integrity.py: the gold has to self-score under the real
# scorer, ids have to be unique, and the file has to still be a byte-stable output
# of its generator. Adds the two properties that only exist for this slice — the
# declared round distribution, and the guarantee that gold is never simply "the
# first values the transcript mentions".

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

# ============================================================================
# test_dataset_degeneracy
# ============================================================================
#
# Guards against a scored gold field that cannot discriminate.
#
# This file exists because two such fields shipped, and neither was caught by any
# existing test — both were found by reading the dataset by hand:
#
#   * `amount_side` was `"input"` in all 436 swap golds, so a model that hardcoded
#     the string scored the field perfectly and it measured nothing.
#   * `to` was `vitalik.eth` in all 36 transfer golds of the ARITHMETIC slice, and
#     51.5% of transfer golds overall — while being the ONLY ENS name in
#     datasets/finetune_seeds.yaml, so the whole ENS axis was n=1 on a value every
#     fine-tune had memorised.
#
# The second one is why the checks below run PER SLICE as well as globally: across
# the full dataset `to` had 63 distinct values and looked healthy, which is exactly
# how the arithmetic slice's constant hid.
#
# A constant field is not automatically a bug — some are forced by the app contract
# or deliberately held fixed to isolate another variable. So constants are allowed,
# but only by name, and only with a reason recorded here. That is the point: a NEW
# constant fails loudly, and every existing one carries its justification in code
# rather than in someone's memory.

COMBINED = ROOT / "pf" / "tests.combined.yaml"


REGENERATED_SLICES = {"arithmetic", "conversation"}


ALLOWED_CONSTANTS: dict[tuple[str | None, str, str], str] = {
    (None, "swap", "amount_side"): (
        "Forced by the app contract, not a dataset choice: pf/tools.app.json pins "
        "amount_side to enum [\"input\"], and the wallet enforces it twice "
        "(ChatDashboardView.swift:3335, SlashCommandParser.swift:66). No faithful "
        "gold can carry another value, so this field cannot be made to vary. The "
        "free point it used to hand out is instead removed by the `exact_output` "
        "mechanism, whose 32 cases expect NO call at all -- so a model that "
        "hardcodes \"input\" and emits a swap now loses them. See "
        "test_exact_output_cases_remove_the_amount_side_free_point below."),
    ("arithmetic", "swap", "from_token"): (
        "Deliberate: the arithmetic slice isolates amount handling, so tokens are "
        "held fixed per group and only the amount varies "
        "(datasets/seeds.arithmetic.yaml says so explicitly). ETH's 18 decimals "
        "tolerate every amount in the slice."),
    ("arithmetic", "swap", "to_token"): "Same as from_token above.",
    ("arithmetic", "transfer", "token"): "Same as from_token above.",
}


def _combined_cases() -> list[dict]:
    return yaml.safe_load(COMBINED.read_text())


def _slice_of(case: dict) -> str:
    return case["metadata"]["category"].split("-")[0]


def _field_values() -> dict[tuple[str, str, str], collections.Counter]:
    """(slice, tool, field) -> Counter of gold values."""
    out: dict[tuple[str, str, str], collections.Counter] = collections.defaultdict(
        collections.Counter)
    for case in _combined_cases():
        sl = _slice_of(case)
        for call in case["metadata"].get("expected_calls") or []:
            tool = call["tool"]
            for field, value in call.items():
                if field == "tool":
                    continue
                out[(sl, tool, field)][str(value)] += 1
    return out


def _is_allowed(sl: str, tool: str, field: str) -> bool:
    return ((sl, tool, field) in ALLOWED_CONSTANTS
            or (None, tool, field) in ALLOWED_CONSTANTS)


def test_no_gold_field_is_constant_without_a_recorded_reason():
    offenders = []
    for (sl, tool, field), counter in sorted(_field_values().items()):
        if len(counter) == 1 and not _is_allowed(sl, tool, field):
            only = next(iter(counter))
            offenders.append(
                f"{sl}/{tool}.{field} is always {only!r} ({counter[only]} golds)")
    assert not offenders, (
        "constant gold field(s) a model could hardcode:\n  "
        + "\n  ".join(offenders)
        + "\n\nEither make the field vary, or add it to ALLOWED_CONSTANTS with the "
          "reason it cannot.")


def test_every_allowed_constant_is_still_actually_constant():
    """Keeps the allowlist honest in the other direction. Once a field starts to
    vary, its exemption is stale and must go, or it will mask a real regression
    later."""
    values = _field_values()
    for (sl, tool, field) in ALLOWED_CONSTANTS:
        keys = ([(s, tool, field) for (s, t, f) in values if (t, f) == (tool, field)
                 for s in [s]] if sl is None else [(sl, tool, field)])
        seen = [values[k] for k in keys if k in values]
        assert seen, f"ALLOWED_CONSTANTS names {sl}/{tool}.{field}, which no gold has"
        for counter in seen:
            assert len(counter) == 1, (
                f"{sl}/{tool}.{field} now has {len(counter)} distinct values, so its "
                f"ALLOWED_CONSTANTS entry is stale — delete it.")


def test_recipient_is_not_dominated_by_one_value_in_the_regenerated_slices():
    """The specific defect: one recipient standing in for the whole axis. 0.30 is
    chosen with real margin — the arithmetic slice's top recipient sits at 14% and
    the conversation slice's at 12%, against 100% and 48% before this change."""
    for (sl, tool, field), counter in sorted(_field_values().items()):
        if field != "to" or sl not in REGENERATED_SLICES:
            continue
        top, n = counter.most_common(1)[0]
        total = sum(counter.values())
        assert n / total < 0.30, (
            f"{sl}/{tool}.to is {n}/{total} ({n / total:.0%}) {top!r} — one "
            f"recipient is standing in for the whole axis")


def test_the_ens_axis_is_more_than_one_name():
    """`vitalik.eth` was the only ENS name in the eval set AND the only one in
    datasets/finetune_seeds.yaml, so "handles ENS" and "has memorised one string"
    were indistinguishable. The regenerated slices must exercise a real bank."""
    per_slice: dict[str, set[str]] = collections.defaultdict(set)
    for case in _combined_cases():
        sl = _slice_of(case)
        for call in case["metadata"].get("expected_calls") or []:
            to = call.get("to")
            if isinstance(to, str) and to.endswith(".eth"):
                per_slice[sl].add(to)

    for sl in sorted(REGENERATED_SLICES):
        assert len(per_slice[sl]) >= 8, (
            f"{sl} uses only {len(per_slice[sl])} ENS name(s): "
            f"{sorted(per_slice[sl])}")
    # And the bank must not be the trained name wearing new clothes.
    trained = {"vitalik.eth"}
    for sl in sorted(REGENERATED_SLICES):
        assert not (per_slice[sl] & trained), (
            f"{sl} reuses the ENS name every fine-tune trained on: "
            f"{sorted(per_slice[sl] & trained)}")


def test_exact_output_cases_remove_the_amount_side_free_point():
    """The compensating mechanism named in ALLOWED_CONSTANTS. Without these cases
    `amount_side` would be exempt AND unmeasured, which is the situation this whole
    file exists to prevent."""
    xout = [c for c in _combined_cases()
            if c["metadata"].get("mechanism") == "exact_output"]
    assert len(xout) == 32, f"expected 32 exact_output cases, found {len(xout)}"
    for case in xout:
        assert case["metadata"]["expected_calls"] == [], \
            f"{case['metadata']['id']} must expect no call"
        # The final user turn is the one scored; it has to actually be output-side.
        last = case["vars"]["messages"][-1]
        assert last["role"] == "user"
        text = last["content"].lower()
        assert ("exactly" in text or "precisely" in text), \
            f"{case['metadata']['id']}: final turn does not pin an exact output"


def test_token_address_cases_carry_the_address_verbatim():
    """The other new axis: a 0x contract address in a token slot, which
    pf/tools.app.json documents and WalletTokenRegistry.token(matching:) resolves.
    Gold must be the address itself, never translated to a symbol."""
    import json
    lookup = json.loads((ROOT / "datasets" / "lookup.json").read_text())
    addresses = {m["address"].lower() for m in lookup["tokens"].values()
                 if m.get("address")}

    cases = [c for c in _combined_cases()
             if c["metadata"].get("mechanism") == "token_address"]
    assert len(cases) == 24, f"expected 24 token_address cases, found {len(cases)}"
    for case in cases:
        calls = case["metadata"]["expected_calls"]
        assert len(calls) == 1, case["metadata"]["id"]
        call = calls[0]
        # Only the TOKEN slots can hold a contract address. `to` is the recipient,
        # and a random recipient is 0x-prefixed as well — collecting it here would
        # assert that a recipient address is a known token, which it never is.
        token_fields = ("token", "from_token", "to_token")
        hexed = [call[f] for f in token_fields
                 if isinstance(call.get(f), str) and call[f].lower().startswith("0x")]
        assert hexed, f"{case['metadata']['id']} has no 0x token in gold"
        for value in hexed:
            assert value.lower() in addresses, (
                f"{case['metadata']['id']} gold names {value!r}, which is not a "
                f"known token address — the wallet's registry could not resolve it")
        # And the address has to appear in the transcript, or the case is unanswerable.
        transcript = " ".join(m["content"] for m in case["vars"]["messages"]).lower()
        for value in hexed:
            assert value.lower() in transcript, (
                f"{case['metadata']['id']}: gold address {value!r} never appears in "
                f"the conversation")


# ============================================================================
# test_dataset_integrity (original content, predating the merge)
# ============================================================================
#
# The legacy `pf/tests.yaml` set: it validates, ids are unique, every gold
# self-scores to 1, and swap cases exist at all.

LEGACY_TESTS = Path(__file__).resolve().parents[1] / "pf" / "tests.yaml"


def _load() -> list:
    return load_cases(LEGACY_TESTS)


def test_dataset_validates():
    assert len(_load()) > 0


def test_ids_unique():
    ids = [c.id for c in _load()]
    assert len(ids) == len(set(ids))


def test_every_legacy_case_self_scores_one():
    for case in _load():
        turn = ParsedTurn(tool_calls=[c.as_parsed_call() for c in case.expected_calls])
        assert score_case(case, turn) == 1, f"{case.id} not self-consistent"


def test_has_swap_cases():
    swaps = [c for c in _load() if any(call.tool == "swap" for call in c.expected_calls)]
    assert len(swaps) >= 1
