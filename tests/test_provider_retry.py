"""`retry_on_no_call`: one extra turn when the model answered without calling.

Aimed at a MEASURED bucket: 40 of base's 74 non-safety failures on the frozen set are
"expected a call, made NONE", and about 60% of them had already named the right tool and
every gold argument value in their reasoning trace before asking a clarifying question
anyway. A prompt clause did not move it (results/act-ab.base-e4b.md), so the remaining
lever is mechanical.

The danger is one-directional and is what these tests pin down: refusal cases legitimately
produce no call, so this fires on them too, and anything that makes the model less hesitant
can convert a correct refusal into a call. ACT_NOT_ASK already lost two refusals that way.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _provider():
    path = Path(__file__).resolve().parents[1] / "pf" / "provider_functiongemma.py"
    spec = importlib.util.spec_from_file_location("pf_provider_retry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_has_call_reads_the_translators_own_contract():
    """`raw_output_to_scoreable` returns an OpenAI-shaped LIST when a call was parsed and
    the prose verbatim otherwise, so the retry trigger must key off that, not re-parse."""
    m = _provider()
    assert m._has_call(json.dumps([{"name": "transfer", "arguments": "{}"}]))
    assert m._has_call(json.dumps([{"name": "swap", "arguments": "{}"},
                                   {"name": "transfer", "arguments": "{}"}]))
    assert not m._has_call("What token do you mean?")
    assert not m._has_call("[]")
    assert not m._has_call("")


def test_has_call_rejects_a_list_that_is_not_tool_calls():
    """Prose that happens to be valid JSON must not read as a call, or the retry would
    silently never fire for it."""
    m = _provider()
    assert not m._has_call(json.dumps(["transfer", "swap"]))
    assert not m._has_call(json.dumps([{"tool": "transfer"}]))
    assert not m._has_call(json.dumps({"name": "transfer"}))


def test_retry_nudge_restates_the_refusal_option():
    """Without this sentence the nudge is a pure bias toward acting, which is exactly how
    a refusal case gets converted into a transaction on the second attempt."""
    m = _provider()
    text = m.RETRY_NUDGE.lower()
    assert "refused" in text and "no tool call" in text
    assert "do not ask" in text


def test_retry_is_off_unless_a_config_asks_for_it():
    """It changes app behaviour, so it must never be the default — a run with it on is not
    app parity."""
    src = (Path(__file__).resolve().parents[1] / "pf"
           / "provider_functiongemma.py").read_text()
    assert 'config.get("retry_on_no_call")' in src, "must be opt-in via config"
    assert 'config.get("retry_on_no_call", True)' not in src, "must not default to on"


def test_retry_result_is_kept_only_when_it_produced_a_call():
    """A second refusal must not overwrite the first answer: replacing it would hide a
    refusal behind a second helping of prose, and would also mask the failure shape the
    analysis depends on."""
    src = (Path(__file__).resolve().parents[1] / "pf"
           / "provider_functiongemma.py").read_text()
    assert "if _has_call(output2):" in src


def test_exactly_one_retry_no_loop():
    """A loop would keep pushing until it got a call, which scores well by destroying the
    refusals — the opposite of what this is for."""
    src = (Path(__file__).resolve().parents[1] / "pf"
           / "provider_functiongemma.py").read_text()
    body = src[src.index("if config.get(\"retry_on_no_call\")"):]
    body = body[:body.index("result: dict[str, Any]")]
    assert "while" not in body, "retry must not loop"
    assert body.count("_has_call(output2)") == 1
