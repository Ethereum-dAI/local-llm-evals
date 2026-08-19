"""`$PROMPT_VARIANT` — the only way to A/B the prompt for a HOSTED provider.

promptfoo builds the prompt before it reaches the provider, so `config.prompt_variant`
(which `pf/provider_functiongemma.py` reads) cannot work for `openrouter:…`. The env
var closes that gap, and these tests pin the two properties that make it safe to have:

  * it is OFF unless explicitly set, so every run that does not name it keeps app
    parity — `APP_SYSTEM` byte-for-byte, which is what makes a score transfer to the
    product;
  * when it IS set, the resulting system turn is byte-identical to the one the local
    fine-tune arm received, so the gpt-5 + clause cell is comparable to the v5 +
    clause cell rather than merely similar.
"""
from __future__ import annotations

import pytest

from pf.prompt import APP_SYSTEM, render
from pf.prompt_candidates import SAFETY_FULL

#: The v5 + SAFETY_FULL arm's system turn, measured before that run and recorded in
#: results/v5-safety-full.1000.md. A change here means the clause or the app prompt
#: moved, and the cross-model comparison is no longer like-for-like.
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


def test_unknown_variant_fails_loudly(monkeypatch):
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
