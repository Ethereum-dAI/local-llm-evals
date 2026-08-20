"""Everything that pins the prompt the harness sends.

Four suites merged into one file because they all guard the same invariant from
different sides, and pytest runs them together anyway: `render()` must reproduce the
wallet's own `prompt-dump` byte for byte unless `$PROMPT_VARIANT` explicitly asks for
a candidate, and a candidate must never be able to arrive silently. The section
banners below name the file each came from.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pytest
import re
import yaml

from pathlib import Path
from pf.prompt import APP_SYSTEM
from pf.prompt import APP_SYSTEM, render
from pf.prompt_candidates import SAFETY_FULL

# ============================================================================
# test_prompt_variant_env
# ============================================================================
#
# `$PROMPT_VARIANT` — the only way to A/B the prompt for a HOSTED provider.
#
# promptfoo builds the prompt before it reaches the provider, so `config.prompt_variant`
# (which `pf/provider_functiongemma.py` reads) cannot work for `openrouter:…`. The env
# var closes that gap, and these tests pin the two properties that make it safe to have:
#
#   * it is OFF unless explicitly set, so every run that does not name it keeps app
#     parity — `APP_SYSTEM` byte-for-byte, which is what makes a score transfer to the
#     product;
#   * when it IS set, the resulting system turn is byte-identical to the one the local
#     fine-tune arm received, so the gpt-5 + clause cell is comparable to the v5 +
#     clause cell rather than merely similar.

V5_SAFETY_ARM_SYSTEM_CHARS = 2110


SINGLE = {"vars": {"user_message": "send 1 eth to bob.eth"}}


def test_default_is_app_parity(monkeypatch):
    monkeypatch.delenv("PROMPT_VARIANT", raising=False)
    assert render(SINGLE)[0]["content"] == APP_SYSTEM


def test_empty_string_is_also_off(monkeypatch):
    """An exported-but-blank var is the shape a shell leaves behind; it must be OFF."""
    monkeypatch.setenv("PROMPT_VARIANT", "")
    assert render(SINGLE)[0]["content"] == APP_SYSTEM


def test_safety_variant_matches_the_local_arm_byte_for_byte(monkeypatch):
    monkeypatch.setenv("PROMPT_VARIANT", "safety")
    system = render(SINGLE)[0]["content"]
    # The single joining space is part of it: `augment` uses f"{content} {extra}",
    # and that is what the fine-tune arm was scored on.
    assert system == f"{APP_SYSTEM} {SAFETY_FULL}"
    assert len(system) == V5_SAFETY_ARM_SYSTEM_CHARS


def test_unknown_env_variant_fails_loudly(monkeypatch):
    """A typo must abort the run, not silently serve the control and look like a null
    result — this repo has already been burned by an arm that quietly wasn't an arm."""
    monkeypatch.setenv("PROMPT_VARIANT", "saftey")
    with pytest.raises(SystemExit):
        render(SINGLE)


def test_only_the_system_turn_changes(monkeypatch):
    convo = [
        {"role": "user", "content": "swap some eth"},
        {"role": "assistant", "content": "How much?"},
        {"role": "user", "content": "0.5 into usdc"},
    ]
    monkeypatch.setenv("PROMPT_VARIANT", "safety")
    out = render({"vars": {"messages": convo}})
    assert [m["role"] for m in out] == ["system", "user", "assistant", "user"]
    assert out[1:] == convo


def test_does_not_mutate_the_callers_messages(monkeypatch):
    """promptfoo reuses the vars dict across providers; a mutating render would make
    arm B inherit arm A's system turn."""
    convo = [{"role": "user", "content": "send 1 eth to bob.eth"}]
    vars_ = {"messages": convo}
    monkeypatch.setenv("PROMPT_VARIANT", "safety")
    render({"vars": vars_})
    render({"vars": vars_})
    assert vars_["messages"] == [{"role": "user", "content": "send 1 eth to bob.eth"}]
    assert convo[0] == {"role": "user", "content": "send 1 eth to bob.eth"}


def test_repeated_renders_are_stable(monkeypatch):
    """Two cases in one run must get the same system turn — not a doubled clause."""
    monkeypatch.setenv("PROMPT_VARIANT", "safety")
    a = render(SINGLE)[0]["content"]
    b = render({"vars": {"user_message": "swap 1 eth to usdc"}})[0]["content"]
    assert a == b


def test_variant_works_when_loaded_by_path(monkeypatch):
    """promptfoo's real entrypoint, which the imports above do NOT exercise.

    `file://pf/prompt.py:render` is executed as a standalone module, so `pf` is not an
    importable package in that process. A package-style import of the candidates
    module raises ModuleNotFoundError there and every case ERRORS — which is how this
    was actually found, on a 6-case smoke run before the paid one.
    """
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "pf" / "prompt.py"
    spec = importlib.util.spec_from_file_location("pf_prompt_standalone", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setenv("PROMPT_VARIANT", "safety")
    system = module.render(SINGLE)[0]["content"]
    assert system == f"{APP_SYSTEM} {SAFETY_FULL}"
    assert len(system) == V5_SAFETY_ARM_SYSTEM_CHARS

# ============================================================================
# test_prompt_candidates
# ============================================================================
#
# Guards for pf/prompt_candidates.py — the A/B-only prompt variants.
#
# These encode lessons that were paid for, so a future edit cannot quietly undo them.

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    path = ROOT / "pf" / "prompt_candidates.py"
    spec = importlib.util.spec_from_file_location("pc", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ZERO_LITERAL = "0x0000000000000000000000000000000000000000"


def test_every_variant_is_registered_and_nonempty():
    m = _mod()
    assert m.PROMPT_CANDIDATES
    for name, parts in m.PROMPT_CANDIDATES.items():
        assert parts and all(p.strip() for p in parts), name


def test_augment_is_a_noop_for_the_control_arm():
    """The A arm of an A/B must go through the identical code path, or the comparison
    includes the code path as a variable."""
    m = _mod()
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "u"}]
    assert m.augment(msgs, "none") == msgs
    assert m.augment(msgs, "") == msgs


def test_augment_only_touches_the_system_turn():
    m = _mod()
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "u"}]
    out = m.augment(msgs, "safety")
    assert out[1] == msgs[1], "user turns must be untouched"
    assert out[0]["content"].startswith("S")
    assert len(out[0]["content"]) > len("S")


def test_augment_rejects_an_unknown_variant():
    """A typo'd variant name must not silently score the control prompt under a
    variant label."""
    m = _mod()
    with pytest.raises(SystemExit):
        m.augment([{"role": "system", "content": "S"}], "safetyy")


def test_augment_refuses_a_conversation_with_no_system_turn():
    m = _mod()
    with pytest.raises(SystemExit):
        m.augment([{"role": "user", "content": "u"}], "safety")


def test_safety_variants_never_colocate_swap_and_the_zero_address():
    """A PAID-FOR lesson. The sentence that once fixed base's zero-address refusals
    named `swap` and `0x0` together, and the swap-heavy fine-tune then emitted
    `swap` with `currencyIn=0x0` for plain requests. Keep them in separate sentences.
    """
    m = _mod()
    for name in ("safety", "safety-min"):
        text = " ".join(m.PROMPT_CANDIDATES[name])
        for sentence in re.split(r"(?<=[.;])\s+|\n", text):
            has_zero = ZERO_LITERAL in sentence or "zero address" in sentence.lower()
            has_swap = "swap" in sentence.lower()
            assert not (has_zero and has_swap), \
                f"{name}: sentence names both swap and the zero address: {sentence!r}"


def test_safety_variants_carry_no_mechanical_address_heuristic():
    """Also rejected before: "refuse if `to` starts with 4+ zeros" passes the cases
    and ships a false positive, because real addresses can begin with zeros. The rule
    must be stated by REASON, not by string shape."""
    m = _mod()
    banned = ("starts with", "begins with", "leading zero", "first four", "prefix")
    for name in ("safety", "safety-min"):
        text = " ".join(m.PROMPT_CANDIDATES[name]).lower()
        for phrase in banned:
            assert phrase not in text, f"{name}: mechanical heuristic {phrase!r}"


def test_safety_variants_still_permit_ordinary_transfers():
    """A safety clause that does not carve out the normal case buys refusals with
    false refusals — which the accuracy slice would catch, but stating it here makes
    the intent explicit."""
    m = _mod()
    for name in ("safety", "safety-min"):
        text = " ".join(m.PROMPT_CANDIDATES[name]).lower()
        assert "ordinary address or ens name is fine" in text, name


def test_safety_full_keeps_the_known_token_address_carve_out():
    """The app accepts a KNOWN token given as its 0x address (the token_address
    mechanism, whose gold carries it verbatim). A blanket "refuse raw addresses" rule
    would break a documented capability to win safety cases."""
    m = _mod()
    text = " ".join(m.PROMPT_CANDIDATES["safety"]).lower()
    assert "known token given as its address is fine" in text


def test_act_not_ask_keeps_its_refusal_carve_out():
    """The whole clause hinges on this sentence.

    "Always emit a call" would destroy the refusal slice — the other half of this
    effort — and would also break the conversation-exact_output cases whose gold is
    deliberately no call. The clause must scope itself to already-determined requests
    AND restate that a refusal wins.
    """
    m = _mod()
    text = m.ACT_NOT_ASK.lower()
    assert "already determines" in text, "clause must be scoped, not unconditional"
    assert "refuse it and make no tool call" in text, "missing refusal precedence"
    assert "genuinely absent" in text, "must still allow a real clarifying question"


def test_act_not_ask_addresses_typos_since_the_dataset_mutates_them():
    """`mutate_typos` is applied on purpose, so treating a misspelling as unresolvable
    converts a solvable case into a question — which is exactly what base did."""
    m = _mod()
    assert "misspelling" in m.ACT_NOT_ASK.lower()


def test_composite_variant_concatenates_its_parts_in_order():
    m = _mod()
    msgs = [{"role": "system", "content": "BASE."}, {"role": "user", "content": "hi"}]
    composed = m.augment(msgs, "safety+act")[0]["content"]
    safety_only = m.augment(msgs, "safety")[0]["content"]
    assert composed.startswith(safety_only), "composite must preserve part order"
    assert m.ACT_NOT_ASK in composed


def test_registered_composite_wins_over_the_split_spelling():
    """`safety+act` is registered explicitly so its ORDER is pinned rather than left to
    however a caller spelled it — sentence order has changed behaviour in this prompt
    before."""
    m = _mod()
    assert "safety+act" in m.PROMPT_CANDIDATES
    assert m.PROMPT_CANDIDATES["safety+act"] == [m.SAFETY_FULL, m.ACT_NOT_ASK]


def test_unknown_part_in_a_composite_is_rejected_by_name():
    m = _mod()
    msgs = [{"role": "system", "content": "BASE."}]
    try:
        m.augment(msgs, "safety+nope")
    except SystemExit as exc:
        assert "nope" in str(exc), "error must name the offending part"
    else:
        raise AssertionError("unknown composite part must raise")

# ============================================================================
# test_prompt_parity
# ============================================================================
#
# The harness must render the app's prompt BYTE-for-byte, per template family.
#
# History this guards, in order:
#
#   1. llama-cpp-python's `create_chat_completion` renders without
#      `enable_thinking`, dropping the `<|think|>` marker the app emits — 2925 chars
#      against the app's 2935. Fixed by rendering explicitly in the provider.
#   2. The provider then compared EVERY model against the Gemma dump, so a Qwen GGUF
#      failed by construction and was waved through with "expected for non-Gemma
#      templates". The wallet's own Qwen dump sat unused in pf/.
#   3. With the Qwen dump actually wired up, Qwen rendered 3309 against the app's
#      3299. Cause: Jinja's `tojson` is `htmlsafe_json_dumps`, which escapes `'` to
#      `\u0027` AFTER dumping, so `json.dumps_kwargs={"ensure_ascii": False}` cannot
#      switch it off. Both app tool descriptions contain "user's" — 2 x 5 chars = the
#      10-char gap. llama.cpp's C++ minja does no HTML escaping.
#
# Gemma never hit (3) because its template emits descriptions as raw text in the
# FunctionGemma DSL, never through `tojson`. That is luck, not design, which is why
# both families are asserted here.

def _provider():
    spec = importlib.util.spec_from_file_location(
        "prov_parity", ROOT / "pf" / "provider_functiongemma.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_prompt_reference_dumps_agree_on_the_contract():
    """The dumps may differ ONLY in `rendered`. If the system prompt or the tool
    schemas ever diverge between them, the two models are being scored on
    different contracts and no cross-model comparison means anything."""
    prov = _provider()
    dumps = {name: json.loads(path.read_text())
             for name, path in prov._REFERENCE_PATHS.items()}
    assert len(dumps) >= 2
    def h(s: str) -> str:
        return hashlib.sha256(s.encode()).hexdigest()
    system = {n: h(d["systemPrompt"]) for n, d in dumps.items()}
    tools = {n: h(d["toolsJSON"]) for n, d in dumps.items()}
    assert len(set(system.values())) == 1, f"systemPrompt differs across dumps: {system}"
    assert len(set(tools.values())) == 1, f"toolsJSON differs across dumps: {tools}"
    # And `rendered` MUST differ — same contract, different template family. If
    # these ever matched, one dump would be a stale copy of the other.
    rendered = {n: h("".join(c["rendered"] for c in d["cases"]))
                for n, d in dumps.items()}
    assert len(set(rendered.values())) == len(rendered), \
        f"two dumps carry identical rendered bytes — one is a stale copy: {rendered}"


@pytest.mark.parametrize("config,expected", [
    ({}, "gemma"),
    ({"tool_format": "gemma"}, "gemma"),
    ({"tool_format": "json"}, "none"),
    ({"tool_format": "json", "prompt_reference": "qwen"}, "qwen"),
    ({"prompt_reference": "none"}, "none"),
])
def test_reference_selection(config, expected):
    """`tool_format: json` alone cannot identify Qwen — Phi-4-mini and SmolLM3 also
    emit JSON-in-text but have no dump from the wallet. So the Qwen dump must be
    named explicitly, and the default must stay Gemma so existing configs are
    untouched."""
    name, path = _provider()._reference_for(config)
    assert name == expected
    assert (path is None) == (expected == "none")


def test_unknown_reference_is_rejected():
    with pytest.raises(ValueError, match="prompt_reference"):
        _provider()._reference_for({"prompt_reference": "llama"})


def test_tojson_does_not_html_escape_apostrophes():
    """The actual defect from (3). Jinja's stock `tojson` turns `'` into `\\u0027`;
    the wallet's C++ minja does not, so the harness has to match minja, not Jinja."""
    import jinja2

    payload = {"description": "from the user's smart account", "lt": "a<b", "amp": "x&y"}

    stock = jinja2.Environment(loader=jinja2.BaseLoader())
    stock.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
    escaped = stock.from_string("{{ v | tojson }}").render(v=payload)
    # Proves ensure_ascii=False is NOT sufficient — the escaping is post-dump.
    assert "\\u0027" in escaped, "jinja2 changed; this guard needs revisiting"

    fixed = jinja2.Environment(loader=jinja2.BaseLoader())
    fixed.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
    fixed.filters["tojson"] = lambda value, indent=None: json.dumps(
        value, ensure_ascii=False, indent=indent)
    out = fixed.from_string("{{ v | tojson }}").render(v=payload)
    assert "user's" in out and "\\u0027" not in out
    assert "a<b" in out and "\\u003c" not in out
    assert "x&y" in out and "\\u0026" not in out


_MODELS = {
    "qwen": ROOT.parent / "human-units-arithmetic" / "models"
    / "qwen3-8b-wallet-ft-appcontract-v4.Q4_K_M.gguf",
    "gemma": ROOT.parent / "human-units-arithmetic" / "models"
    / "gemma4-e4b-wallet-ft-appcontract-v4.Q4_K_M.gguf",
}


@pytest.mark.parametrize("refname", sorted(_MODELS))
def test_gguf_renders_the_apps_exact_bytes(refname):
    path = _MODELS[refname]
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    import warnings
    warnings.filterwarnings("ignore")
    from llama_cpp import Llama

    prov = _provider()
    llm = Llama(model_path=str(path), vocab_only=True, verbose=False)
    template = prov._chat_template(llm, (str(path), None, None, None, 4096),
                                   refname, prov._REFERENCE_PATHS[refname])
    reference = json.loads(prov._REFERENCE_PATHS[refname].read_text())
    tools = json.loads(reference["toolsJSON"])
    for case in reference["cases"]:
        got = prov._render(template, case["messages"], tools)
        assert got == case["rendered"], (
            f"{refname}/{case['label']}: rendered {len(got)} chars, app sends "
            f"{len(case['rendered'])} — the harness is scoring a different prompt")

# ============================================================================
# test_app_prompt_with_clause
# ============================================================================
#
# The wallet's prompt AFTER `ToolDefinitions.safetyClause` landed, and its exact delta.
#
# `pf/app_contract_reference.json` is the LIVE reference: it is what `APP_SYSTEM` is read
# from, so every recorded number on this benchmark was measured against those bytes.
# `pf/app_contract_reference.with_clause.json` is the dump taken from the wallet branch
# that adds the clause (`wallet-eval prompt-dump`), kept here so the app's new bytes are
# in this repo rather than only in the other one.
#
# **It is deliberately NOT wired into any run.** Swapping it in would make
# `PROMPT_VARIANT=none` mean "clause on", silently changing what every existing config
# measures, and it carries a third tool that `pf/tools.app.json` does not offer. The
# decision to re-baseline is a separate, explicit one; these tests pin the relationship in
# the meantime so neither file can drift unnoticed.
#
# What they establish, and why each matters:
#
#   * the clause in the app's dump is byte-identical to `SAFETY_FULL` — so the clause-on
#     numbers in results/ describe the string the app now sends, not a near-miss;
#   * it is appended as a suffix joined by ONE space — the concatenation `augment()`
#     produces, and the one the models were scored on;
#   * the ONLY other change from the live reference is the pre-existing
#     `top_up_bundler` sentence, which is exercised by 0 of the 1000 cases. That bounds
#     the re-measurement question: the prompt delta that matters is the clause, which is
#     already measured on three models.

WITH_CLAUSE = json.loads((ROOT / "pf" / "app_contract_reference.with_clause.json").read_text())


MEASURED_SYSTEM_CHARS = 2110


def test_the_clause_in_the_app_dump_is_byte_identical() -> None:
    assert SAFETY_FULL in WITH_CLAUSE["systemPrompt"]


def test_the_clause_is_a_suffix_joined_by_one_space() -> None:
    system = WITH_CLAUSE["systemPrompt"]
    assert system.endswith(SAFETY_FULL)
    assert system.endswith(f" {SAFETY_FULL}")
    # ...and not by a newline, which would be a different string from the measured one.
    assert not system.endswith(f"\n{SAFETY_FULL}")


def test_the_only_other_delta_is_the_pre_existing_top_up_sentence() -> None:
    """If this fails, the app's prompt changed in some way BEYOND the clause, and the
    recorded numbers no longer bound the difference — re-measure rather than reasoning
    about it."""
    prefix = WITH_CLAUSE["systemPrompt"][: -len(SAFETY_FULL) - 1]
    assert prefix.startswith(APP_SYSTEM)
    extra = prefix[len(APP_SYSTEM):]
    assert extra == (
        "\nFor a bundler top-up, call top_up_bundler with only the requested amount; "
        "never invent or request a destination address."
    ), extra


def test_the_top_up_tool_is_exercised_by_no_case_in_the_frozen_set() -> None:
    """Which is what makes the delta above ignorable for now. The frozen 1000 are all
    transfer/swap/refusal, so a third tool in the prompt changes the tool list the model
    sees but nothing it is asked to do."""
    assert WITH_CLAUSE["toolNames"] == ["transfer", "swap", "top_up_bundler"]
    cases = yaml.safe_load((ROOT / "pf" / "tests.combined.yaml").read_text())
    tools_wanted = {
        call["tool"]
        for c in cases
        for call in (c["metadata"].get("expected_calls") or [])
    }
    assert tools_wanted == {"transfer", "swap"}, tools_wanted


def test_the_live_reference_still_has_no_clause() -> None:
    """The guard on the footgun: if someone swaps the with-clause dump in as the live
    reference, `PROMPT_VARIANT=none` starts meaning "clause on" and every recorded
    number silently describes a prompt it was not measured under."""
    assert SAFETY_FULL not in APP_SYSTEM
    assert len(APP_SYSTEM) == 533
    assert len(f"{APP_SYSTEM} {SAFETY_FULL}") == MEASURED_SYSTEM_CHARS
