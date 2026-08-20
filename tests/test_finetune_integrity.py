from __future__ import annotations

import json
import pytest
import sys

from functools import partial
from pathlib import Path
from wallet_evals.finetune import HERMES, encode_hermes_calls
from wallet_evals.finetune import encode_gemma_call
from wallet_evals.functiongemma import json_output_to_scoreable
from wallet_evals.functiongemma import raw_output_to_scoreable
from wallet_evals.functiongemma import raw_output_to_scoreable as _raw_output_to_scoreable
from wallet_evals.gemma_dsl import GEMMA4
from wallet_evals.parsing import parse_turn
from wallet_evals.schema import Case
from wallet_evals.scorer import score_case

# ============================================================================
# test_finetune_integrity
# ============================================================================
#
# Integrity of the FunctionGemma fine-tuning set (data_for_finetune/*.jsonl).
#
# The whole value of the harness is that gold is trustworthy. A training target is
# only correct if, decoded back through the UNCHANGED provider translation + scorer,
# it scores 1.0 — i.e. the model that learns to emit it would pass the eval. We also
# assert the set is disjoint from the eval set (no leakage) and structurally sane.
#
# If the JSONL is absent (not generated yet), these tests skip.

ROOT = Path(__file__).resolve().parents[1]


TRAIN = ROOT / "data_for_finetune" / "functiongemma_train.jsonl"


GENERATED = ROOT / "pf" / "tests.generated.yaml"


PROTOCOLS = ROOT / "pf" / "tests.protocols.yaml"


COMBINED = ROOT / "pf" / "tests.combined.yaml"


def _load_examples() -> list[dict]:
    if not TRAIN.exists():
        pytest.skip(f"{TRAIN} not generated yet (run scripts/generate_finetune_data.py)")
    return [json.loads(line) for line in TRAIN.read_text().splitlines() if line.strip()]


def _case(ex: dict) -> Case:
    """Rebuild a scoring Case from an example (only expected_calls/tool matter)."""
    return Case(
        id=ex["id"], user_message="", level="payload", language="english",
        category=ex.get("category") or "generated",
        protocol=ex.get("protocol") or "transfer",
        difficulty="easy", expected_calls=ex.get("expected_calls") or [],
    )


def _assistant_content(ex: dict) -> str:
    assert ex["messages"][-1]["role"] == "assistant", f"{ex['id']}: no assistant turn"
    return ex["messages"][-1]["content"]


def test_nonempty():
    assert len(_load_examples()) > 0


def test_ids_unique_and_prefixed():
    ids = [ex["id"] for ex in _load_examples()]
    assert len(ids) == len(set(ids)), "duplicate ids"
    assert all(i.startswith("ft-") for i in ids), "ids must be ft- prefixed"


def test_every_target_self_scores_one():
    """The crux: each assistant target, run through the same translation the
    provider applies at inference, must score 1.0 against its own gold."""
    for ex in _load_examples():
        content = _assistant_content(ex)
        scoreable = raw_output_to_scoreable(content)
        # Mirror pf/assert.py's two paths: DSL calls -> JSON list; prose -> as-is.
        if scoreable.strip().startswith("["):
            calls = json.loads(scoreable)
            turn = parse_turn(content=None, native_tool_calls=calls, raw_text="")
        else:
            turn = parse_turn(content=scoreable, native_tool_calls=None, raw_text=scoreable)
        assert score_case(_case(ex), turn) == 1, f"{ex['id']} target does not self-score"


def test_targets_match_expected_call_count():
    """A gold-call case must emit a DSL call; a no-call case must emit prose only."""
    for ex in _load_examples():
        content = _assistant_content(ex)
        has_call = "<start_function_call>" in content
        assert has_call == bool(ex.get("expected_calls")), \
            f"{ex['id']}: call presence disagrees with expected_calls"


def test_encoder_roundtrips_each_call():
    """encode_gemma_call is the exact inverse the decoder+scorer accept."""
    for ex in _load_examples():
        for call in ex.get("expected_calls") or []:
            dsl = encode_gemma_call(call)
            scoreable = raw_output_to_scoreable(dsl)
            turn = parse_turn(content=None,
                              native_tool_calls=json.loads(scoreable), raw_text="")
            case = Case(id=ex["id"], user_message="", level="payload",
                        language="english", category="c", protocol="transfer",
                        difficulty="easy", expected_calls=[call])
            assert score_case(case, turn) == 1, f"{ex['id']} call did not round-trip"


def test_tools_present():
    """Each row carries the tool menu its own contract offers, not one global set.

    Wallet-path rows offer exactly the app's two tools so training matches the
    bytes the app sends; Aave/Safe rows stay on the transaction-builder superset
    their `executeTx` gold is written against.
    """
    builder = json.loads((ROOT / "pf" / "tools.json").read_text())
    app = json.loads((ROOT / "pf" / "tools.app.json").read_text())
    seen = set()
    for ex in _load_examples():
        is_protocol = ex["category"].startswith(("aave-", "safe-"))
        expected = builder if is_protocol else app
        assert ex.get("tools") == expected, (
            f"{ex['id']}: expected the "
            f"{'builder' if is_protocol else 'app'} tool set"
        )
        seen.add(is_protocol)
    # The default training set is WALLET-ONLY: mixing the builder contract into
    # it is what taught v4 a second tool vocabulary opposed to the app's own.
    # The protocol rows still exist — `--protocol-only` builds them as their own
    # set — so what this asserts is the separation, not their removal.
    assert seen == {False}, (
        "the default fine-tune set must contain no Aave/Safe rows; build those "
        "with --protocol-only (see generate_finetune_data.INCLUDE_PROTOCOL_ROWS)"
    )


def test_roles_use_developer_not_system():
    for ex in _load_examples():
        roles = {m["role"] for m in ex["messages"]}
        assert "system" not in roles, f"{ex['id']}: system role must be remapped to developer"
        assert "developer" in roles, f"{ex['id']}: missing developer turn"


def _all_user_turns_from_vars(vars_: dict) -> str:
    """The full user input: every user turn joined (multi-turn cases share tiny
    completion fragments like 'to vitalik.eth' but never the whole conversation)."""
    if "messages" in vars_:
        users = [m["content"] for m in vars_["messages"] if m.get("role") == "user"]
        return "\n".join(users)
    return vars_.get("user_message", "")


def _eval_surfaces() -> set[str]:
    import yaml
    surfaces: set[str] = set()
    for path in (GENERATED, PROTOCOLS, COMBINED):
        for test in yaml.safe_load(path.read_text()) or []:
            surfaces.add(_all_user_turns_from_vars(test["vars"]))
    return surfaces


def _train_surface(ex: dict) -> str:
    users = [m["content"] for m in ex["messages"] if m["role"] == "user"]
    return "\n".join(users)


def test_disjoint_from_eval_set():
    """No training conversation may appear in the eval set — the anti-leakage
    guarantee. Compares the full user-turn sequence, not a single fragment."""
    eval_surfaces = _eval_surfaces()
    for ex in _load_examples():
        assert _train_surface(ex) not in eval_surfaces, \
            f"{ex['id']} conversation leaks into the eval set"


def test_protocol_rows_are_separated_not_deleted():
    """`--protocol-only` still builds the Aave/Safe set, on the builder contract.

    The wallet mix dropped these 95 rows because two tool vocabularies in one
    model is what produced v4's invented `transfer` to `avev3.eth`. Dropping the
    MIX is not the same as dropping the capability: the fixtures, builders and
    caps all remain, so the protocol model can be trained the moment the wallet
    gains lending or multisig tools. This test is what keeps that true.
    """
    import subprocess
    import tempfile

    builder = json.loads((ROOT / "pf" / "tools.json").read_text())
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "protocol.jsonl"
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "generate_finetune_data.py"),
             "--protocol-only", "--out", str(out)],
            cwd=ROOT, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr[-1500:]
        rows = [json.loads(line) for line in out.read_text().splitlines() if line]

    assert rows, "--protocol-only produced nothing"
    assert all(r["category"].startswith(("aave-", "safe-")) for r in rows)
    assert all(r.get("tools") == builder for r in rows), \
        "protocol rows must keep the executeTx builder contract"


def test_the_published_v4_mix_is_reproducible():
    """`--include-protocol-rows` rebuilds the 1863-row mix v4 was trained on.

    The published gemma-4/qwen3 v4 artifacts trained on 1768 wallet rows PLUS 95
    Aave/Safe builder rows. The default set is wallet-only now, so without this
    flag the repo could no longer rebuild what those weights came from — the
    card would describe a mix nothing in the tree produces. Reproducibility of a
    shipped artifact should not require editing a module constant.

    `--no-conversation-rows` and `--no-rehearsal-rows` are the rest of it: the
    default set gained 500 multi-round rows and 20 prose-answer rows after v4
    shipped, so reproducing v4 means switching every later addition off. All flags,
    all explicit — a shipped artifact whose training mix cannot be rebuilt is
    undocumented in practice however carefully the card is written. A new bucket
    means a new flag here, and this test failing is the reminder.
    """
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "v4mix.jsonl"
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "generate_finetune_data.py"),
             "--include-protocol-rows", "--no-conversation-rows",
             "--no-rehearsal-rows", "--out", str(out)],
            cwd=ROOT, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr[-1500:]
        rows = [json.loads(line) for line in out.read_text().splitlines() if line]

    protocol = [r for r in rows if r["category"].startswith(("aave-", "safe-"))]
    assert len(rows) == 1863, f"v4 mix was 1863 rows, got {len(rows)}"
    assert len(protocol) == 95, f"v4 mix had 95 protocol rows, got {len(protocol)}"

# ============================================================================
# test_gemma4_finetune_integrity
# ============================================================================
#
# Integrity of the Gemma-4 fine-tuning set (data_for_finetune/gemma4_train.jsonl).
#
# The whole value of the harness is that gold is trustworthy. A training target is
# only correct if, decoded back through the UNCHANGED provider translation + scorer,
# it scores 1.0 — i.e. the model that learns to emit it would pass the eval. We also
# assert the set is disjoint from the eval set (no leakage) and structurally sane.
#
# Mirrors tests/test_finetune_integrity.py but for the GEMMA4 dialect: targets use
# the `<|tool_call>…<tool_call|>` shape, roles keep `system` (Gemma-4's native
# `<|turn>system`), and gold-call targets may carry a leading <think> trace.
#
# If the JSONL is absent (not generated yet), these tests skip.
#
# Suffixed on merge, to clear a name this file shared with an
# earlier section: TRAIN -> TRAIN_GEMMA4_FINETUNE_INTEGRITY, _load_examples -> _load_examples_gemma4_finetune_integrity, test_targets_match_expected_call_count -> test_targets_match_expected_call_count_gemma4_finetune_integrity, test_encoder_roundtrips_each_call -> test_encoder_roundtrips_each_call_gemma4_finetune_integrity, test_tools_present -> test_tools_present_gemma4_finetune_integrity, _all_user_turns_from_vars -> _all_user_turns_from_vars_gemma4_finetune_integrity, _eval_surfaces -> _eval_surfaces_gemma4_finetune_integrity.

# Named, not rebound: this section decodes the GEMMA4 dialect while the
# FunctionGemma section above decodes the default one, and a module-level
# rebinding of the shared name silently retargets both.
_gemma4_scoreable = partial(_raw_output_to_scoreable, dialect=GEMMA4)


TRAIN_GEMMA4_FINETUNE_INTEGRITY = ROOT / "data_for_finetune" / "gemma4_train.jsonl"


def _load_examples_gemma4_finetune_integrity() -> list[dict]:
    if not TRAIN_GEMMA4_FINETUNE_INTEGRITY.exists():
        pytest.skip(f"{TRAIN_GEMMA4_FINETUNE_INTEGRITY} not generated yet "
                    f"(run scripts/generate_gemma4_finetune_data.py)")
    return [json.loads(line) for line in TRAIN_GEMMA4_FINETUNE_INTEGRITY.read_text().splitlines() if line.strip()]


def test_nonempty_qwen():
    assert len(_load_examples_gemma4_finetune_integrity()) > 0


def test_ids_unique_and_prefixed_qwen():
    ids = [ex["id"] for ex in _load_examples_gemma4_finetune_integrity()]
    assert len(ids) == len(set(ids)), "duplicate ids"
    assert all(i.startswith("ft-") for i in ids), "ids must be ft- prefixed"


def test_every_target_self_scores_one_qwen():
    """The crux: each assistant target, run through the same translation the
    provider applies at inference, must score 1.0 against its own gold."""
    for ex in _load_examples_gemma4_finetune_integrity():
        content = _assistant_content(ex)
        scoreable = _gemma4_scoreable(content)
        # Mirror pf/assert.py's two paths: DSL calls -> JSON list; prose -> as-is.
        if scoreable.strip().startswith("["):
            calls = json.loads(scoreable)
            turn = parse_turn(content=None, native_tool_calls=calls, raw_text="")
        else:
            turn = parse_turn(content=scoreable, native_tool_calls=None, raw_text=scoreable)
        assert score_case(_case(ex), turn) == 1, f"{ex['id']} target does not self-score"


def test_targets_match_expected_call_count_gemma4_finetune_integrity():
    """A gold-call case must emit a DSL call; a no-call case must emit prose only."""
    for ex in _load_examples_gemma4_finetune_integrity():
        content = _assistant_content(ex)
        has_call = GEMMA4.opener in content
        assert has_call == bool(ex.get("expected_calls")), \
            f"{ex['id']}: call presence disagrees with expected_calls"


def test_encoder_roundtrips_each_call_gemma4_finetune_integrity():
    """encode_gemma_call (GEMMA4) is the exact inverse the decoder+scorer accept."""
    for ex in _load_examples_gemma4_finetune_integrity():
        for call in ex.get("expected_calls") or []:
            dsl = encode_gemma_call(call, GEMMA4)
            scoreable = _gemma4_scoreable(dsl)
            turn = parse_turn(content=None,
                              native_tool_calls=json.loads(scoreable), raw_text="")
            case = Case(id=ex["id"], user_message="", level="payload",
                        language="english", category="c", protocol="transfer",
                        difficulty="easy", expected_calls=[call])
            assert score_case(case, turn) == 1, f"{ex['id']} call did not round-trip"


def test_tools_present_gemma4_finetune_integrity():
    """Each row carries the tool menu its own contract offers, not one global set.

    Wallet-path rows must offer exactly `ToolDefinitions.phase1` (transfer, swap)
    so training matches the bytes the app sends. Aave/Safe rows stay on the
    transaction-builder superset, which is the contract their `executeTx` gold is
    written against. A single shared array is what let the two drift together.
    """
    builder = json.loads((ROOT / "pf" / "tools.json").read_text())
    app = json.loads((ROOT / "pf" / "tools.app.json").read_text())
    seen = set()
    for ex in _load_examples_gemma4_finetune_integrity():
        is_protocol = ex["category"].startswith(("aave-", "safe-"))
        expected = builder if is_protocol else app
        assert ex.get("tools") == expected, (
            f"{ex['id']}: expected the "
            f"{'builder' if is_protocol else 'app'} tool set"
        )
        seen.add(is_protocol)
    # The default training set is WALLET-ONLY: mixing the builder contract into
    # it is what taught v4 a second tool vocabulary opposed to the app's own.
    # The protocol rows still exist — `--protocol-only` builds them as their own
    # set — so what this asserts is the separation, not their removal.
    assert seen == {False}, (
        "the default fine-tune set must contain no Aave/Safe rows; build those "
        "with --protocol-only (see generate_finetune_data.INCLUDE_PROTOCOL_ROWS)"
    )


def test_roles_keep_system_not_developer():
    """Gemma-4 keeps `system` (native `<|turn>system`) — unlike FunctionGemma."""
    for ex in _load_examples_gemma4_finetune_integrity():
        roles = {m["role"] for m in ex["messages"]}
        assert "developer" not in roles, f"{ex['id']}: Gemma-4 must not use developer role"
        assert "system" in roles, f"{ex['id']}: missing system turn"


def test_reasoning_only_on_call_targets():
    """A <think> trace is only ever a prefix to a real tool call (transfer/swap),
    never on a refusal/clarification (which must stay pure prose)."""
    for ex in _load_examples_gemma4_finetune_integrity():
        if "<think>" in _assistant_content(ex):
            assert ex.get("expected_calls"), \
                f"{ex['id']}: <think> on a no-call target"


def _all_user_turns_from_vars_gemma4_finetune_integrity(vars_: dict) -> str:
    if "messages" in vars_:
        users = [m["content"] for m in vars_["messages"] if m.get("role") == "user"]
        return "\n".join(users)
    return vars_.get("user_message", "")


def _eval_surfaces_gemma4_finetune_integrity() -> set[str]:
    import yaml
    surfaces: set[str] = set()
    for path in (GENERATED, PROTOCOLS):
        for test in yaml.safe_load(path.read_text()) or []:
            surfaces.add(_all_user_turns_from_vars_gemma4_finetune_integrity(test["vars"]))
    return surfaces


def test_disjoint_from_eval_set_qwen():
    """No training conversation may appear in the eval set — the anti-leakage
    guarantee. Compares the full user-turn sequence, not a single fragment."""
    eval_surfaces = _eval_surfaces_gemma4_finetune_integrity()
    for ex in _load_examples_gemma4_finetune_integrity():
        assert _train_surface(ex) not in eval_surfaces, \
            f"{ex['id']} conversation leaks into the eval set"


def test_railgun_coverage_is_zero():
    """Briefly the opposite of this test: the set gained railgun rows because
    the app shipped shield/unshield. The app is removing them
    (local-wallet-mac#86, PR #87) and pf/tools.json no longer offers them, so
    training on them would teach tools the product does not expose."""
    examples = _load_examples_gemma4_finetune_integrity()
    assert not [ex for ex in examples if ex.get("protocol") == "railgun"]
    tools_used = {c.get("tool") for ex in examples for c in ex.get("expected_calls") or []}
    assert "shield" not in tools_used and "unshield" not in tools_used


def test_no_app_contract_trace_leaks_base_units_or_a_token_contract_address():
    """Step 0's whole point: transfer and swap both take a HUMAN
    decimal amount, so their <think> traces must never compute or name wei,
    base units, or a token's contract address (Aave/Safe stay on the OLD
    executeTx/base-unit contract and are deliberately exempt)."""
    from wallet_evals.intents import LOOKUP
    token_addresses = [m["address"] for m in LOOKUP["tokens"].values() if m.get("address")]
    app_contract_protocols = {"transfer", "uniswap", "railgun"}
    offenders = []
    for ex in _load_examples_gemma4_finetune_integrity():
        if ex.get("protocol") not in app_contract_protocols:
            continue
        content = _assistant_content(ex)
        if "<think>" not in content:
            continue
        think = content.split("<think>", 1)[1].split("</think>", 1)[0]
        lowered = think.lower()
        if "base unit" in lowered or "wei" in lowered:
            offenders.append((ex["id"], "base units/wei"))
        if any(addr in think for addr in token_addresses):
            offenders.append((ex["id"], "token contract address"))
    assert not offenders, offenders


HELD_OUT_REFUSAL_KINDS = {
    "non-numeric-amount", "keystore-exfiltration", "roleplay-jailbreak",
}


def test_held_out_refusal_kinds_never_enter_training():
    """These three kinds exist ONLY in the eval, as the generalization probe.

    Training now covers all twelve other refusal kinds, so without a held-out
    set there is nothing left to measure "does the model apply a SAFETY rule it
    was never shown an example of" — the question that produced the clearest
    result of the 560-case run (base 96% vs v1 71% vs v2 67% on untrained
    kinds). If someone adds these to the training bank to raise the refusal
    score, the score goes up and the measurement quietly dies.
    """
    from scripts.generate_finetune_data import REFUSAL_SCENARIOS as TRAIN_BANK
    trained = {s["kind"] for s in TRAIN_BANK}
    leaked = trained & HELD_OUT_REFUSAL_KINDS
    assert not leaked, f"held-out refusal kinds leaked into training: {sorted(leaked)}"

    examples = _load_examples_gemma4_finetune_integrity()
    for ex in examples:
        cat = ex.get("category", "")
        kind = cat.replace("safety-refusal-", "")
        assert kind not in HELD_OUT_REFUSAL_KINDS, \
            f"{ex.get('id')}: held-out kind {kind!r} present in the training set"


def test_every_trained_safety_rule_has_examples():
    """v2's regression on kinds it HAD trained on is the reason for this test.

    Its training prompt carried a 7-rule SAFETY block while only rules (a)-(c)
    had any refusal examples; training on 1815 rows where (d)-(g) never fire
    appears to teach the model to discount them. Every kind in the bank must
    therefore actually produce rows.
    """
    from scripts.generate_finetune_data import REFUSAL_SCENARIOS as TRAIN_BANK
    kinds_in_bank = {s["kind"] for s in TRAIN_BANK}
    kinds_in_data = {
        ex["category"].replace("safety-refusal-", "")
        for ex in _load_examples_gemma4_finetune_integrity()
        if ex.get("category", "").startswith("safety-refusal-")
    }
    missing = kinds_in_bank - kinds_in_data
    assert not missing, f"refusal kinds defined but absent from the data: {sorted(missing)}"
    assert len(kinds_in_data) >= 12, f"only {len(kinds_in_data)} refusal kinds trained"

# ============================================================================
# test_qwen_finetune_integrity
# ============================================================================
#
# Integrity of the Qwen3 fine-tuning set (data_for_finetune/qwen_train.jsonl).
#
# Same contract as tests/test_gemma4_finetune_integrity.py, one dialect over: the
# targets are Hermes-style `<tool_call>{...}</tool_call>` JSON, decoded by
# `json_tool_calls.parse` (the same parser the Qwen provider uses at
# inference), and every one must self-score 1.0 through the UNCHANGED scorer.
#
# Also asserts the three training sets stay row-identical in CONTENT: the Qwen set
# must carry exactly the ids and gold of the Gemma-4 set, differing only in
# encoding. That is the whole basis for comparing the two fine-tunes — if the rows
# drift, the comparison measures data, not model.
#
# If the JSONL is absent (not generated yet), these tests skip.
#
# Suffixed on merge, to clear a name this file shared with an
# earlier section: TRAIN -> TRAIN_QWEN_FINETUNE_INTEGRITY, _case -> _case_qwen_finetune_integrity, test_nonempty_qwen -> test_nonempty_qwen_finetune_integrity, test_ids_unique_and_prefixed_qwen -> test_ids_unique_and_prefixed_qwen_finetune_integrity, test_every_target_self_scores_one_qwen -> test_every_target_self_scores_one_qwen_finetune_integrity, test_targets_match_expected_call_count -> test_targets_match_expected_call_count_qwen_finetune_integrity, test_encoder_roundtrips_each_call -> test_encoder_roundtrips_each_call_qwen_finetune_integrity, test_tools_present -> test_tools_present_qwen_finetune_integrity, test_roles_keep_system_not_developer -> test_roles_keep_system_not_developer_qwen_finetune_integrity, test_reasoning_only_on_call_targets -> test_reasoning_only_on_call_targets_qwen_finetune_integrity.

TRAIN_QWEN_FINETUNE_INTEGRITY = ROOT / "data_for_finetune" / "qwen_train.jsonl"


GEMMA4_TRAIN = ROOT / "data_for_finetune" / "gemma4_train.jsonl"


def _load(path: Path = TRAIN_QWEN_FINETUNE_INTEGRITY) -> list[dict]:
    if not path.exists():
        pytest.skip(f"{path} not generated yet "
                    f"(run scripts/generate_qwen_finetune_data.py)")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _case_qwen_finetune_integrity(ex: dict) -> Case:
    return Case(
        id=ex["id"], user_message="", level="payload", language="english",
        category=ex.get("category") or "generated",
        protocol=ex.get("protocol") or "transfer",
        difficulty="easy", expected_calls=ex.get("expected_calls") or [],
    )


def test_nonempty_qwen_finetune_integrity():
    assert len(_load()) > 0


def test_ids_unique_and_prefixed_qwen_finetune_integrity():
    ids = [ex["id"] for ex in _load()]
    assert len(ids) == len(set(ids)), "duplicate ids"
    assert all(i.startswith("ft-") for i in ids), "ids must be ft- prefixed"


def test_every_target_self_scores_one_qwen_finetune_integrity():
    """The crux: each assistant target, run through the same translation the Qwen
    provider applies at inference, must score 1.0 against its own gold."""
    for ex in _load():
        scoreable = json_output_to_scoreable(_assistant_content(ex))
        if scoreable.strip().startswith("["):
            turn = parse_turn(content=None, native_tool_calls=json.loads(scoreable),
                              raw_text="")
        else:
            turn = parse_turn(content=scoreable, native_tool_calls=None,
                              raw_text=scoreable)
        assert score_case(_case_qwen_finetune_integrity(ex), turn) == 1, f"{ex['id']} target does not self-score"


def test_targets_match_expected_call_count_qwen_finetune_integrity():
    for ex in _load():
        has_call = HERMES.opener in _assistant_content(ex)
        assert has_call == bool(ex.get("expected_calls")), \
            f"{ex['id']}: call presence disagrees with expected_calls"


def test_encoder_roundtrips_each_call_qwen_finetune_integrity():
    """encode_hermes_calls is the exact inverse of the parser the provider uses."""
    for ex in _load():
        for call in ex.get("expected_calls") or []:
            scoreable = json_output_to_scoreable(encode_hermes_calls([call]))
            turn = parse_turn(content=None, native_tool_calls=json.loads(scoreable),
                              raw_text="")
            case = Case(id=ex["id"], user_message="", level="payload",
                        language="english", category="c", protocol="transfer",
                        difficulty="easy", expected_calls=[call])
            assert score_case(case, turn) == 1, f"{ex['id']} call did not round-trip"


def test_tools_present_qwen_finetune_integrity():
    """Each row carries the tool menu its own contract offers, not one global set.

    Wallet-path rows offer exactly the app's two tools so training matches the
    bytes the app sends; Aave/Safe rows stay on the transaction-builder superset
    their `executeTx` gold is written against.
    """
    builder = json.loads((ROOT / "pf" / "tools.json").read_text())
    app = json.loads((ROOT / "pf" / "tools.app.json").read_text())
    seen = set()
    for ex in _load():
        is_protocol = ex["category"].startswith(("aave-", "safe-"))
        expected = builder if is_protocol else app
        assert ex.get("tools") == expected, (
            f"{ex['id']}: expected the "
            f"{'builder' if is_protocol else 'app'} tool set"
        )
        seen.add(is_protocol)
    # The default training set is WALLET-ONLY: mixing the builder contract into
    # it is what taught v4 a second tool vocabulary opposed to the app's own.
    # The protocol rows still exist — `--protocol-only` builds them as their own
    # set — so what this asserts is the separation, not their removal.
    assert seen == {False}, (
        "the default fine-tune set must contain no Aave/Safe rows; build those "
        "with --protocol-only (see generate_finetune_data.INCLUDE_PROTOCOL_ROWS)"
    )


def test_roles_keep_system_not_developer_qwen_finetune_integrity():
    for ex in _load():
        roles = {m["role"] for m in ex["messages"]}
        assert "developer" not in roles, f"{ex['id']}: Qwen must not use developer role"
        assert "system" in roles, f"{ex['id']}: missing system turn"


def test_reasoning_only_on_call_targets_qwen_finetune_integrity():
    for ex in _load():
        if "<think>" in _assistant_content(ex):
            assert ex.get("expected_calls"), f"{ex['id']}: <think> on a no-call target"


def test_same_rows_as_the_gemma4_set():
    """Identical content, different encoding — the premise of the comparison."""
    qwen = {ex["id"]: ex for ex in _load()}
    gemma = {ex["id"]: ex for ex in _load(GEMMA4_TRAIN)}
    assert set(qwen) == set(gemma), "training row ids differ between the two sets"
    for i, ex in qwen.items():
        assert ex["expected_calls"] == gemma[i]["expected_calls"], f"{i}: gold differs"
        assert ex["messages"][:-1] == gemma[i]["messages"][:-1], f"{i}: prompt differs"


def test_disjoint_from_the_eval_set():
    """No training row may reuse an eval surface — the anti-leakage guarantee."""
    import yaml

    def surfaces(vars_: dict) -> str:
        if "messages" in vars_:
            return "\n".join(m["content"] for m in vars_["messages"]
                             if m.get("role") == "user")
        return vars_.get("user_message", "")

    eval_surfaces = {surfaces(t["vars"]).strip()
                     for t in yaml.safe_load(GENERATED.read_text())}
    for ex in _load():
        users = "\n".join(m["content"] for m in ex["messages"] if m["role"] == "user")
        assert users.strip() not in eval_surfaces, f"{ex['id']} leaks an eval surface"
