from __future__ import annotations

import collections
import json
import pytest
import random
import re
import sys
import yaml

from pathlib import Path
from scripts.generate_cases import (
    ARITHMETIC_SEEDS,
    MAX_PER_ACTION_ARITHMETIC,
    SEED_ARITHMETIC,
    build_all,
    build_extra_selection,
    _relabel_arithmetic,
)
from scripts.generate_conversation_cases import (
    ROUND_PLAN, SEED, TARGET_TOTAL, build_selection, load_intents,
)
from typing import Callable
from wallet_evals.conversations import MECHANISM_ROUNDS
from wallet_evals.dev_safety import DEV_REFUSAL_SCENARIOS
from wallet_evals.generation import EXTRA_REFUSAL_SCENARIOS
from wallet_evals.promptfoo import load_cases
from wallet_evals.schema import Case, ExpectedCall, ParsedToolCall, ParsedTurn
from wallet_evals.schema import ParsedTurn
from wallet_evals.scorer import score_case

# ============================================================================
# test_dataset_integrity
# ============================================================================
#
# Dataset integrity: the dev slices, the conversation cases, and degeneracy.
#
# Four suites merged into one file. They all answer the same question about different
# slices — is this dataset actually what it claims to be, and is it disjoint from the
# things it must be disjoint from (training rows, the frozen benchmark, each other)?
# The frozen 1000-case set has its own file, `test_combined_benchmark_integrity.py`.
#
# The last section is this file's ORIGINAL content, kept because it guards the legacy
# `pf/tests.yaml` set that predates all of this and is still scored by `scripts/eval.sh`
# with no dataset argument. Two of its names were prefixed `legacy` to clear a clash.

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

# ============================================================================
# test_combined_benchmark_integrity
# ============================================================================
#
# Integrity of the combined benchmark: app-contract (transfer/swap + the
# arithmetic slice + the refusal banks) concatenated with the multi-round
# conversation slice.
#
# Mirrors test_app_contract_integrity.py / test_dataset_integrity.py, which
# guard the two source files this one is built from.
#
# Suffixed on merge, to clear a name this file shared with an
# earlier section: _load -> _load_combined_benchmark_integrity, _raw -> _raw_combined_benchmark_integrity, test_every_case_self_scores_one -> test_every_case_self_scores_one_combined_benchmark_integrity.

APP_CONTRACT = ROOT / "pf" / "tests.app-contract.yaml"


CONVERSATIONS = ROOT / "pf" / "tests.conversations.yaml"


EXPECTED_TOTAL = 1000


EXPECTED_ROUNDS = {1: 336, 2: 273, 3: 150, 4: 110, 5: 80, 6: 51}


def _load_combined_benchmark_integrity():
    return load_cases(COMBINED)


def _raw_combined_benchmark_integrity() -> list[dict]:
    return yaml.safe_load(COMBINED.read_text())


def _rounds(case: dict) -> int:
    messages = case["vars"].get("messages")
    if not messages:
        return 1
    return sum(1 for m in messages if m.get("role") == "user")


def test_every_case_self_scores_one_combined_benchmark_integrity():
    for case in _load_combined_benchmark_integrity():
        turn = ParsedTurn(tool_calls=[c.as_parsed_call() for c in case.expected_calls])
        assert score_case(case, turn) == 1, f"{case.id} not self-consistent"


def test_ids_unique_across_the_whole_file():
    ids = [c.id for c in _load_combined_benchmark_integrity()]
    assert len(ids) == len(set(ids))


def test_the_benchmark_is_exactly_its_declared_size():
    assert len(_load_combined_benchmark_integrity()) == EXPECTED_TOTAL


def test_arithmetic_slice_is_present_and_non_empty():
    cats = {c.category for c in _load_combined_benchmark_integrity()}
    arithmetic_cats = {c for c in cats if c.startswith("arithmetic-")}
    assert arithmetic_cats, "no arithmetic-* category found in the combined benchmark"
    arithmetic_cases = [c for c in _load_combined_benchmark_integrity() if c.category.startswith("arithmetic-")]
    assert len(arithmetic_cases) > 0


def test_railgun_is_absent():
    # RAILGUN was removed from the app (local-wallet-mac#86, PR #87) and from
    # pf/tools.json; a benchmark that still scored shield/unshield would be
    # measuring a feature the product does not expose.
    cases = _load_combined_benchmark_integrity()
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
    cases = _load_combined_benchmark_integrity()
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
    refusals = [c for c in _load_combined_benchmark_integrity() if c.category.startswith("safety-refusal-")]
    assert len(refusals) >= 40, f"only {len(refusals)} refusal cases"
    kinds = {c.category for c in refusals}
    assert len(kinds) >= 10, f"only {len(kinds)} distinct refusal kinds: {sorted(kinds)}"


def test_combined_count_equals_sum_of_its_two_sources():
    combined = _load_combined_benchmark_integrity()
    app_contract = load_cases(APP_CONTRACT)
    conversations = load_cases(CONVERSATIONS)
    assert len(combined) == len(app_contract) + len(conversations)


def test_round_distribution_matches_the_declared_shape():
    """The point of the conversation slice: two thirds of the benchmark is now
    multi-round, spanning 1-6 rounds. Asserted rather than printed so a
    regenerated source file that collapses the long conversations fails here."""
    counts = collections.Counter(_rounds(c) for c in _raw_combined_benchmark_integrity())
    assert dict(counts) == EXPECTED_ROUNDS
    multi = sum(v for k, v in counts.items() if k > 1)
    assert multi / len(_raw_combined_benchmark_integrity()) > 0.6, f"only {multi} multi-round cases"


def test_long_conversations_carry_enough_weight_to_move_the_score():
    """5- and 6-round cases are the ones a model degrading on context length will
    fail first. If they were a handful of cases, that degradation would round to
    nothing in the headline number."""
    long_cases = [c for c in _raw_combined_benchmark_integrity() if _rounds(c) >= 5]
    assert len(long_cases) >= 100, f"only {len(long_cases)} cases of 5+ rounds"


def test_every_conversation_mechanism_is_represented():
    mechanisms = collections.Counter(
        c["metadata"].get("mechanism") for c in _raw_combined_benchmark_integrity()
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
    cases = _load_combined_benchmark_integrity()
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

# ============================================================================
# test_app_contract_integrity
# ============================================================================
#
# Integrity of the app-contract dataset.
#
# Mirrors test_generated_integrity.py, which guards the frozen base-unit dataset.
# Both must pass: the scorer still has to handle executeTx gold for the protocol
# datasets, and the app-contract gold has to self-score under the same scorer.
#
# Suffixed on merge, to clear a name this file shared with an
# earlier section: TESTS -> TESTS_APP_CONTRACT_INTEGRITY, _load -> _load_app_contract_integrity.

TESTS_APP_CONTRACT_INTEGRITY = Path(__file__).resolve().parents[1] / "pf" / "tests.app-contract.yaml"


def _load_app_contract_integrity():
    return load_cases(TESTS_APP_CONTRACT_INTEGRITY)


def test_app_contract_first_307_match_the_frozen_dataset_count_and_the_arithmetic_slice_is_appended():
    # pf/tests.app-contract.yaml now holds the original 307 transfer/swap cases
    # PLUS an appended arithmetic slice (scripts/generate_cases.py's
    # --extra-seeds path, from datasets/seeds.arithmetic.yaml, its own RNG
    # stream). The frozen count is still exactly 307; the first 307 cases here
    # are byte-identical to it (proven by scripts/generate_cases.py's own
    # SEPARATE RNG for the extra seeds, plus the diff check in the brief); the
    # rest is the arithmetic slice, labelled distinctly so it's never mistaken
    # for part of the original 307.
    frozen = load_cases(Path(__file__).resolve().parents[1] / "pf" / "tests.generated.yaml")
    assert len(frozen) == 307

    cases = _load_app_contract_integrity()
    # xref- cases are the EXTRA_REFUSAL_SCENARIOS bank, appended from its own
    # RNG stream just like the arithmetic slice.
    base = [c for c in cases
            if not c.category.startswith("arithmetic-") and not c.id.startswith("xref-")]
    arithmetic = [c for c in cases if c.category.startswith("arithmetic-")]
    extra_refusals = [c for c in cases if c.id.startswith("xref-")]
    assert len(base) == 307
    assert len(arithmetic) > 0
    assert len(extra_refusals) > 0
    assert len(cases) == 307 + len(arithmetic) + len(extra_refusals)

    # The load-bearing property, asserted rather than asserted-in-a-comment: the
    # 307 carry the SAME prompts as the frozen base-unit dataset, only different
    # gold. Without it, "base 9.8% -> 87.9% on the same intents" is not a claim
    # about the contract change, and the whole comparison collapses.
    assert {c.id for c in base} == {c.id for c in frozen}

    # Compare the raw `vars` blocks, not the parsed Case: multi-turn prompts
    # live in vars["messages"], which the Case model does not surface.
    root = Path(__file__).resolve().parents[1]
    import yaml
    def _vars_by_id(path):
        return {t["metadata"]["id"]: t["vars"]
                for t in yaml.safe_load((root / path).read_text())}
    frozen_vars = _vars_by_id("pf/tests.generated.yaml")
    current_vars = _vars_by_id("pf/tests.app-contract.yaml")
    for case_id in frozen_vars:
        fv, cv = frozen_vars[case_id], current_vars[case_id]
        for key in ("user_message", "messages"):
            assert fv.get(key) == cv.get(key), \
                f"{case_id}: {key} drifted from the frozen base-unit dataset"


def test_app_contract_ids_unique():
    ids = [c.id for c in _load_app_contract_integrity()]
    assert len(ids) == len(set(ids))


def test_app_contract_self_scores_one():
    # NOTE: this is a round-trip/schema check, NOT a fold check. as_parsed_call()
    # is a pure field-for-field copy, so every fold in the scorer sees the same
    # (tool, value) pair on both sides of the comparison — each becomes f(x) == f(x),
    # true regardless of whether the fold is correct. What this genuinely proves is
    # that every gold call in the dataset validates against ParsedToolCall's schema
    # and survives the round-trip (it would catch a builder emitting a tool name or
    # a field the schema rejects). For evidence the folds themselves score MEANING
    # rather than surface form, see test_app_contract_fold_* and the negative
    # control test_app_contract_fold_rejects_an_altered_amount below.
    for case in _load_app_contract_integrity():
        turn = ParsedTurn(tool_calls=[c.as_parsed_call() for c in case.expected_calls])
        assert score_case(case, turn) == 1, f"{case.id} not self-consistent"


def _find_single_call_case(
    cases: list[Case], predicate: Callable[[ExpectedCall], bool]
) -> tuple[Case, ExpectedCall] | None:
    """First case with exactly one expected call matching predicate, or None.

    Restricted to single-call cases so the caller can build a one-element
    ParsedTurn without having to fabricate the other calls in the sequence.
    """
    for case in cases:
        if len(case.expected_calls) != 1:
            continue
        call = case.expected_calls[0]
        if predicate(call):
            return case, call
    return None


def _score_actual(case: Case, actual: ParsedToolCall) -> int:
    return score_case(case, ParsedTurn(tool_calls=[actual]))


def test_app_contract_fold_recipient_is_case_insensitive():
    # Real proof the `_recipient_text` fold folds case for ENS names, not just a
    # tautological copy: gold names an ENS recipient, the "model" emits the same
    # name in a different case, and it must still score 1.
    found = _find_single_call_case(
        _load_app_contract_integrity(),
        lambda c: c.tool == "transfer" and c.to is not None
        and not c.to.startswith("0x") and "." in c.to,
    )
    if found is None:
        pytest.skip("no transfer case with an ENS-style recipient in the dataset")
    case, call = found
    actual = call.as_parsed_call().model_copy(update={"to": call.to.swapcase()})
    assert actual.to != call.to, "swapcase() should actually change an ENS name's case"
    assert _score_actual(case, actual) == 1


def test_app_contract_fold_token_symbol_is_case_insensitive():
    # Real proof `_symbol` folds case: gold names a non-ETH token, the "model"
    # emits it in the opposite case, and it must still score 1.
    found = _find_single_call_case(
        _load_app_contract_integrity(),
        lambda c: c.tool == "transfer" and c.token is not None and c.token.lower() != "eth",
    )
    if found is None:
        pytest.skip("no transfer case with a non-ETH token in the dataset")
    case, call = found
    actual = call.as_parsed_call().model_copy(update={"token": call.token.swapcase()})
    assert actual.token != call.token, "swapcase() should actually change the token's case"
    assert _score_actual(case, actual) == 1


def test_app_contract_fold_amount_is_numeric_not_textual():
    # Real proof `_human_amount`/`_dec_or_raw` compares numeric value, not text:
    # gold has a fractional amount, the "model" emits the same value with an
    # extra trailing zero, and it must still score 1.
    found = _find_single_call_case(
        _load_app_contract_integrity(),
        lambda c: c.tool == "transfer" and c.amount is not None and "." in c.amount,
    )
    if found is None:
        pytest.skip("no transfer case with a fractional amount in the dataset")
    case, call = found
    padded = call.amount + "0"
    actual = call.as_parsed_call().model_copy(update={"amount": padded})
    assert actual.amount != call.amount, "padding should actually change the amount text"
    assert _score_actual(case, actual) == 1


def test_app_contract_fold_missing_token_defaults_to_eth():
    # Real proof `_symbol` treats an omitted token as ETH: gold names ETH
    # explicitly, the "model" omits `token` entirely, and it must still score 1.
    found = _find_single_call_case(
        _load_app_contract_integrity(),
        lambda c: c.tool == "transfer" and c.token is not None and c.token.lower() == "eth",
    )
    if found is None:
        pytest.skip("no transfer case with an explicit ETH token in the dataset")
    case, call = found
    actual = call.as_parsed_call().model_copy(update={"token": None})
    assert _score_actual(case, actual) == 1


def test_app_contract_fold_missing_amount_side_defaults_to_input():
    # Real proof `_swap_side` treats an omitted amount_side as "input": gold sets
    # it explicitly, the "model" omits it entirely, and it must still score 1.
    found = _find_single_call_case(
        _load_app_contract_integrity(), lambda c: c.tool == "swap" and c.amount_side is not None
    )
    if found is None:
        pytest.skip("no swap case with an explicit amount_side in the dataset")
    case, call = found
    actual = call.as_parsed_call().model_copy(update={"amount_side": None})
    assert _score_actual(case, actual) == 1


def test_app_contract_fold_rejects_an_altered_amount():
    # NEGATIVE CONTROL — the one that proves the fold tests above can actually
    # fail. Take a real transfer amount and insert thousands separators (a change
    # a human would call cosmetic but that is NOT the same numeric value under
    # `_dec_or_raw`'s Decimal parse: a comma makes Decimal(...) raise, so the fold
    # falls back to comparing the raw, unequal strings). If this ever scores 1,
    # the fold tests above are not testing anything.
    found = _find_single_call_case(
        _load_app_contract_integrity(),
        lambda c: c.tool == "transfer" and c.amount is not None
        and int(c.amount.split(".")[0]) >= 1000,
    )
    if found is None:
        pytest.skip("no transfer case with a >=4-digit amount in the dataset")
    case, call = found
    int_part, _, frac_part = call.amount.partition(".")
    grouped = f"{int(int_part):,}"
    altered = f"{grouped}.{frac_part}" if frac_part else grouped
    assert altered != call.amount, "grouping should actually change the amount text"
    actual = call.as_parsed_call().model_copy(update={"amount": altered})
    assert _score_actual(case, actual) == 0


def test_app_contract_emits_no_base_unit_gold():
    for case in _load_app_contract_integrity():
        for call in case.expected_calls:
            assert call.tool in ("transfer", "swap"), f"{case.id} uses {call.tool}"
            assert call.value == "0" and call.args == []
            assert call.currencyIn is None and call.amountIn is None


def test_app_contract_has_negatives_multiturn_and_refusals():
    cases = _load_app_contract_integrity()
    assert any(not c.expected_calls for c in cases)
    assert any(c.category.startswith("multiturn-") for c in cases)
    assert any(c.category.startswith("safety-refusal-") for c in cases)

# ============================================================================
# test_generated_integrity
# ============================================================================
#
# Suffixed on merge, to clear a name this file shared with an
# earlier section: _load -> _load_generated_integrity.

GENERATED = Path(__file__).resolve().parents[1] / "pf" / "tests.generated.yaml"


def _load_generated_integrity() -> list:
    return load_cases(GENERATED)


def test_generated_nonempty():
    assert len(_load_generated_integrity()) > 0


def test_generated_ids_unique():
    ids = [c.id for c in _load_generated_integrity()]
    assert len(ids) == len(set(ids))


def test_generated_self_scores_one():
    for case in _load_generated_integrity():
        turn = ParsedTurn(tool_calls=[c.as_parsed_call() for c in case.expected_calls])
        assert score_case(case, turn) == 1, f"{case.id} not self-consistent"


def test_generated_has_negatives_and_multiturn():
    ids = [c.id for c in _load_generated_integrity()]
    assert any("-neg-" in i for i in ids)
    assert any("-mt-" in i for i in ids)


def test_generated_has_safety_refusals():
    refusals = [c for c in _load_generated_integrity() if "refusal" in c.id]
    assert refusals, "no safety-refusal cases generated"
    assert all(c.expected_calls == [] for c in refusals), "refusal gold must be no tool call"

# ============================================================================
# test_arithmetic_slice_generation
# ============================================================================
#
# TDD coverage for the arithmetic slice's generator additions.
#
# `scripts/generate_cases.py` grows a `--extra-seeds` path so the arithmetic
# slice can be generated from datasets/seeds.arithmetic.yaml with its OWN RNG
# stream (SEED_ARITHMETIC), appended AFTER the main 307-case selection, without
# disturbing it. These tests cover the pure pieces of that path in isolation,
# before the file-level Step 2 diff check (which requires a real regeneration
# run and is done by hand per the brief, not as a pytest).

def test_build_all_can_omit_the_refusal_bucket():
    # Refusal scenarios are hardcoded, not seed-derived: calling build_all a
    # second time for extra seeds must not silently re-mint duplicate
    # "gen-refusal-####" ids that collide with the main selection's.
    seeds = [{"action": "transfer", "amount": "1.5", "token": "ETH",
              "recipient": "vitalik.eth", "ablate": ["amount"]}]
    with_refusals = build_all(seeds, random.Random(1))
    without_refusals = build_all(seeds, random.Random(1), include_refusals=False)
    assert "refusal" in with_refusals and with_refusals["refusal"]
    assert "refusal" not in without_refusals


def test_relabel_arithmetic_rewrites_positive_case_id_and_category():
    case = {"metadata": {"id": "gen-transfer-pos-0001",
                         "category": "generated-transfer-pos"}}
    _relabel_arithmetic(case)
    assert case["metadata"]["id"] == "arith-transfer-pos-0001"
    assert case["metadata"]["category"] == "arithmetic-transfer-pos"


def test_relabel_arithmetic_rewrites_ablation_and_multiturn_categories():
    neg = {"metadata": {"id": "gen-swap-neg-0002", "category": "ablation-amount"}}
    _relabel_arithmetic(neg)
    assert neg["metadata"]["id"] == "arith-swap-neg-0002"
    assert neg["metadata"]["category"] == "arithmetic-ablation-amount"

    mt = {"metadata": {"id": "gen-transfer-mt-0003", "category": "multiturn-recipient"}}
    _relabel_arithmetic(mt)
    assert mt["metadata"]["id"] == "arith-transfer-mt-0003"
    assert mt["metadata"]["category"] == "arithmetic-multiturn-recipient"


def test_build_extra_selection_labels_every_case():
    seeds = yaml.safe_load(ARITHMETIC_SEEDS.read_text())
    selected = build_extra_selection(seeds, random.Random(SEED_ARITHMETIC))
    assert selected  # non-empty
    for case in selected:
        md = case["metadata"]
        assert md["id"].startswith("arith-"), md["id"]
        assert md["category"].startswith("arithmetic-"), md["category"]
        assert "gen-" not in md["id"]


def test_build_extra_selection_deterministic():
    seeds = yaml.safe_load(ARITHMETIC_SEEDS.read_text())
    a = build_extra_selection(seeds, random.Random(SEED_ARITHMETIC))
    b = build_extra_selection(seeds, random.Random(SEED_ARITHMETIC))
    assert a == b


def test_build_extra_selection_respects_its_own_cap():
    seeds = yaml.safe_load(ARITHMETIC_SEEDS.read_text())
    selected = build_extra_selection(seeds, random.Random(SEED_ARITHMETIC))
    by_action: dict[str, int] = {}
    for case in selected:
        action = case["metadata"]["id"].split("-")[1]
        by_action[action] = by_action.get(action, 0) + 1
    for action, count in by_action.items():
        assert count <= MAX_PER_ACTION_ARITHMETIC, (action, count)


def test_arithmetic_seed_amounts_are_disjoint_from_both_existing_seed_files():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    main_seeds = yaml.safe_load((root / "datasets" / "seeds.yaml").read_text())
    ft_seeds = yaml.safe_load((root / "datasets" / "finetune_seeds.yaml").read_text())
    arithmetic_seeds = yaml.safe_load(ARITHMETIC_SEEDS.read_text())

    def amounts(doc):
        out = set()
        for s in doc:
            spec = s["amount"]
            vals = spec["vary"] if isinstance(spec, dict) and "vary" in spec else [spec]
            out.update(vals)
        return out

    existing = amounts(main_seeds) | amounts(ft_seeds)
    new = amounts(arithmetic_seeds)
    assert existing & new == set(), f"amount literal(s) not disjoint: {existing & new}"
    assert new  # sanity: the new file actually defines some amounts

# ============================================================================
# test_tools_contract
# ============================================================================
#
# The tool schema the eval offers must match the macOS app's own ToolDefinitions.
#
# Source of truth is wallet-macos/Sources/WalletToolLayer/ToolDefinitions.swift in the
# local-wallet-mac repo (`ToolDefinitions.phase1`). It cannot be imported from Python,
# so the property names are asserted here and any drift shows up as a failure rather
# than as a silently mis-scored run.

TOOLS = json.loads((Path(__file__).resolve().parents[1] / "pf" / "tools.json").read_text())


BY_NAME = {t["function"]["name"]: t["function"] for t in TOOLS}


def _props(name):
    return set(BY_NAME[name]["parameters"]["properties"])


def test_all_four_tools_present():
    # transfer/swap mirror the app; executeTx/readTx serve the Aave and Safe
    # protocol datasets, which have no app counterpart. shield/unshield were
    # dropped with the RAILGUN feature (local-wallet-mac#86, PR #87).
    assert set(BY_NAME) == {"executeTx", "readTx", "transfer", "swap"}


def test_transfer_matches_app_properties():
    # Exact parity with ToolDefinitions.swift:5. chainId was removed on
    # 2026-08-14: the app declares no such argument and never reads one
    # (grep intent.args["chainId"] returns nothing), so requiring it trained the
    # model to emit an off-contract field — the same drift class as executeTx.
    assert _props("transfer") == {"to", "amount", "token"}
    assert BY_NAME["transfer"]["parameters"]["required"] == ["to", "amount"]


def test_swap_matches_app_properties():
    # Exact parity with ToolDefinitions.swift:15 — see the chainId note above.
    assert _props("swap") == {"from_token", "to_token", "amount", "amount_side"}
    assert BY_NAME["swap"]["parameters"]["required"] == [
        "from_token", "to_token", "amount"]


def test_app_tools_declare_no_chainid_but_protocol_tools_do():
    """The split that keeps the two contracts honest: transfer/swap mirror the
    app, which resolves the chain from activeChain.id; executeTx/readTx serve
    the Aave/Safe datasets, which are base-unit and do carry a chainId."""
    for name in ("transfer", "swap"):
        assert "chainId" not in _props(name)
    for name in ("executeTx", "readTx"):
        assert "chainId" in _props(name)
    assert BY_NAME["swap"]["parameters"]["properties"]["amount_side"]["enum"] == ["input"]


def test_no_app_tool_asks_for_base_units():
    # Every app tool is human-unit now, so "the exception to the base-unit rule"
    # framing the privacy tools used to carry is no longer true of anything.
    for name in ("transfer", "swap"):
        blob = json.dumps(BY_NAME[name]).lower()
        assert "base unit" not in blob and "base-unit" not in blob
