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
from pf.prompt import render, SYSTEM, APP_SYSTEM, APP_TOOLS
from pf.prompt_candidates import SAFETY_FULL

# ============================================================================
# test_prompt_contract
# ============================================================================
#
# Everything that pins the prompt the harness sends.
#
# Four suites merged into one file because they all guard the same invariant from
# different sides, and pytest runs them together anyway: `render()` must reproduce the
# wallet's own `prompt-dump` byte for byte unless `$PROMPT_VARIANT` explicitly asks for
# a candidate, and a candidate must never be able to arrive silently. The section
# banners below name the file each came from.

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

# ============================================================================
# test_prompt_render
# ============================================================================

def test_render_single_turn():
    """A wallet-path case gets the app's own system prompt, not the builder one.

    `SYSTEM` is now reserved for the Aave/Safe transaction-builder cases; every
    other case must see byte-for-byte what ToolDefinitions.systemNudge produces,
    which is what APP_SYSTEM is read from.
    """
    chat = render({"vars": {"user_message": "Send 0.1 ETH to vitalik.eth"}})
    assert chat[0] == {"role": "system", "content": APP_SYSTEM}
    assert chat[0]["content"] != SYSTEM
    assert chat[1] == {"role": "user", "content": "Send 0.1 ETH to vitalik.eth"}
    assert len(chat) == 2


def test_render_multi_turn():
    msgs = [
        {"role": "user", "content": "Send 0.1 ETH"},
        {"role": "assistant", "content": "Which address?"},
        {"role": "user", "content": "to vitalik.eth"},
    ]
    chat = render({"vars": {"messages": msgs}})
    assert chat[0]["role"] == "system"
    assert chat[1:] == msgs


def test_render_without_account_context_unchanged():
    chat = render({"vars": {"user_message": "Send 0.1 ETH to vitalik.eth"}})
    assert len(chat) == 2 and chat[0]["role"] == "system" and chat[1]["role"] == "user"


def test_render_safe_protocol_adds_reference_and_context():
    ctx = {"safe": "0xSafe", "owners": ["0xA", "0xB"], "threshold": 2}
    chat = render({"vars": {"user_message": "Remove signer 0xB from my Safe.",
                            "protocol": "safe", "account_context": ctx}})
    assert [m["role"] for m in chat] == ["system", "system", "user"]
    addendum = chat[1]["content"]
    assert "addOwnerWithThreshold(address,uint256)" in addendum
    assert "removeOwner(address,address,uint256)" in addendum
    assert "0xSafe" in addendum and "0xA, 0xB" in addendum


def test_render_aave_protocol_adds_reference():
    chat = render({"vars": {"user_message": "Supply 3 USDC to Aave v3.", "protocol": "aave"}})
    assert [m["role"] for m in chat] == ["system", "system", "user"]
    ref = chat[1]["content"]
    assert "supply(address,uint256,address,uint16)" in ref
    assert "borrow(address,uint256,uint256,uint16,address)" in ref
    assert "0x87870Bca3F3fD6335C3F4ce8392D69350B4fa4E2" in ref
    assert "<wallet>" in ref


def test_railgun_protocol_no_longer_resolves_to_a_reference():
    """RAILGUN was removed from the app (local-wallet-mac#86, PR #87). An
    unknown protocol key must fall through silently rather than inject a
    reference for tools that no longer exist."""
    chat = render({"vars": {"user_message": "Shield 0.01 ETH.", "protocol": "railgun"}})
    assert [m["role"] for m in chat] == ["system", "user"]
    assert all("unshield" not in m["content"] for m in chat)


def test_render_no_protocol_unchanged():
    chat = render({"vars": {"user_message": "Send 0.1 ETH to vitalik.eth"}})
    assert len(chat) == 2 and chat[0]["role"] == "system" and chat[1]["role"] == "user"


def test_expected_summary_var_is_not_leaked_to_model():
    # expected_summary is a viewer-only var; render must never put it in the chat.
    chat = render({"vars": {"user_message": "Send 0.1 ETH to vitalik.eth",
                            "expected_summary": "executeTx to 0xSECRETGOLD (native)"}})
    assert all("0xSECRETGOLD" not in m["content"] for m in chat)


def test_app_contract_lives_in_the_app_prompt_not_the_builder_one():
    """The human-units/verbatim-recipient rules belong to the WALLET path.

    The app states them in its TOOL SCHEMAS, not its system prompt — APP_SYSTEM
    is 533 characters and says nothing about units or resolution, because the
    app leans on the `transfer`/`swap` parameter descriptions instead. That is
    the contract the product actually ships, so it is what the wallet path must
    carry.

    They must NOT be in SYSTEM, which after the prompt split is reached only by
    Aave/Safe, whose gold is base units and resolved addresses. Asserting them
    on SYSTEM is how the contradiction got in.
    """
    app_tools = json.dumps(APP_TOOLS)
    assert "human units" in app_tools
    assert "do not attempt to resolve ENS yourself" in app_tools

    assert "HUMAN units" not in SYSTEM
    assert "Never convert to wei or base units" not in SYSTEM
    assert "NOT resolve it to an address yourself" not in SYSTEM
    assert "never by contract address" not in SYSTEM


def test_builder_system_keeps_the_contract_its_gold_is_written_against():
    """Aave/Safe gold is executeTx with base units and resolved addresses, so
    the only prompt those cases see has to ask for exactly that."""
    assert "Convert every human amount to base units" in SYSTEM
    assert "Resolve any ENS name or token symbol to its address" in SYSTEM
    assert "executeTx" in SYSTEM


def test_aave_reference_still_carries_the_base_unit_rule():
    # The protocol datasets keep executeTx + base units, so the rule must survive
    # in the reference block that renders only for them.
    chat = render({"vars": {"user_message": "Supply 3 USDC to Aave v3.",
                            "protocol": "aave"}})
    assert "base units" in chat[1]["content"]
