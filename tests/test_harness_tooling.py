"""The harness's own tooling: provider retry, report splitting, config pairing.

Three small suites merged into one file. None is about the models or the data; each
guards a piece of measurement machinery that has already produced a wrong number
once, which is why they exist at all.
"""
from __future__ import annotations

import importlib.util
import json
import pytest
import yaml

from pathlib import Path

# ============================================================================
# test_provider_retry
# ============================================================================
#
# `retry_on_no_call`: one extra turn when the model answered without calling.
#
# Aimed at a MEASURED bucket: 40 of base's 74 non-safety failures on the frozen set are
# "expected a call, made NONE", and about 60% of them had already named the right tool and
# every gold argument value in their reasoning trace before asking a clarifying question
# anyway. A prompt clause did not move it (results/history.md), so the remaining
# lever is mechanical.
#
# The danger is one-directional and is what these tests pin down: refusal cases legitimately
# produce no call, so this fires on them too, and anything that makes the model less hesitant
# can convert a correct refusal into a call. ACT_NOT_ASK already lost two refusals that way.

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

# ============================================================================
# test_report_provider_split
# ============================================================================
#
# A multi-provider export must become one run PER ARM, not one arm silently.
#
# `Run.cases` is keyed by case id so a re-run chunk replaces rather than double-counts.
# That is right for the chunked single-model runs it was written for, and wrong for every
# A/B config in this repo: those put 2-3 arms in ONE export, so each arm overwrote the
# previous one and the report showed the LAST arm's results as the whole run — at a case
# count of exactly 1000, so the truncation guard stayed silent too.

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location(
        "report_1000", ROOT / "scripts" / "report_1000.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _row(case_id: str, provider: str, passed: bool) -> dict:
    return {
        "provider": {"id": f"file://x:{provider}", "label": provider},
        "success": passed,
        "failureReason": 0,
        "response": {"output": "[]"},
        "gradingResult": {"reason": ""},
        "testCase": {"vars": {}, "metadata": {
            "id": case_id, "category": "generated-transfer-pos",
            "expected_calls": [{"tool": "transfer"}]}},
    }


def _export(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "e.out.json"
    p.write_text(json.dumps({"results": {"results": rows}}))
    return p


def test_two_arms_in_one_export_become_two_runs(tmp_path):
    m = _mod()
    # Same two case ids under both arms: arm A gets both right, arm B gets both wrong.
    rows = [_row("c1", "armA", True), _row("c2", "armA", True),
            _row("c1", "armB", False), _row("c2", "armB", False)]
    runs = m.split_providers("ignored", [_export(tmp_path, rows)])
    assert [r.label for r in runs] == ["armA", "armB"]
    assert [r.n for r in runs] == [2, 2], "each arm must keep all of its own cases"
    passed = {r.label: sum(c["passed"] for c in r.cases.values()) for r in runs}
    assert passed == {"armA": 2, "armB": 0}, \
        "arms must not overwrite each other — this is the bug the split exists for"


def test_single_provider_export_keeps_the_callers_label(tmp_path):
    """Existing invocations pass one model per export and name it themselves; the
    split must not rename those or every recorded table's labels change."""
    m = _mod()
    rows = [_row("c1", "only", True), _row("c2", "only", False)]
    runs = m.split_providers("my-label", [_export(tmp_path, rows)])
    assert len(runs) == 1
    assert runs[0].label == "my-label"
    assert runs[0].n == 2


def test_provider_falls_back_to_id_when_unlabelled(tmp_path):
    m = _mod()
    rows = [_row("c1", "a", True), _row("c2", "b", True)]
    for r in rows:
        del r["provider"]["label"]
    runs = m.split_providers("x", [_export(tmp_path, rows)])
    assert sorted(r.label for r in runs) == ["file://x:a", "file://x:b"]

# ============================================================================
# test_eval_config_pairs
# ============================================================================
#
# A base-vs-fine-tune config must differ ONLY in the weights.
#
# Every head-to-head headline rests on this: if the two arms of a pair differ in
# sampling, context size, tool format or prompt reference, the delta stops being
# "what the fine-tune did" and becomes "what the setup did". That is the exact error
# CLAUDE.md warns about for the Modal runs ("only the device differs, so this is not
# a full-precision re-run wearing an on-device label").
#
# Asserted here rather than left to review because the two blocks are inlined —
# promptfoo validates top-level keys, so a YAML anchor risks failing a run hours in.
#
# The v1-v4 pair configs this originally guarded have been deleted along with the rest
# of the superseded one-off configs; `results/history.md` is their record. It now
# guards the surviving pair, which is the one the shipped headline came from.

PAIRS = {
    "promptfooconfig.v5-vs-base.remote.yaml": ("gemma4-e4b-base",
                                               "gemma4-e4b-ft-v5"),
}


#: Keys that legitimately differ: they name WHICH weights to serve, nothing else.
#: `remote_url` belongs here for a rented-GPU pair — each arm points at the pod
#: holding its own GGUF, and `repo_id`/`filename` only supply the shared template.
MODEL_SOURCE_KEYS = {"model_path", "repo_id", "filename", "revision", "remote_url"}


def _config(name: str) -> dict:
    path = ROOT / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    return yaml.safe_load(path.read_text())


@pytest.mark.parametrize("name", sorted(PAIRS))
def test_pair_differs_only_in_the_model_source(name):
    cfg = _config(name)
    labels = PAIRS[name]
    by_label = {p["label"]: p["config"] for p in cfg["providers"]}
    for label in labels:
        assert label in by_label, f"{name} has no provider labelled {label!r}"
    a, b = (by_label[l] for l in labels)
    differing = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    stray = differing - MODEL_SOURCE_KEYS
    assert not stray, (
        f"{name}: {labels[0]} and {labels[1]} differ in {sorted(stray)} — a "
        f"base-vs-fine-tune delta must come from the weights alone")
    assert differing, f"{name}: the two providers are identical; one loads the wrong model"


@pytest.mark.parametrize("name", sorted(PAIRS))
def test_pair_has_no_unknown_top_level_keys(name):
    """promptfoo rejects unrecognised top-level keys. A leftover YAML anchor block
    would fail the run rather than the suite, which is the expensive way to find out."""
    allowed = {"description", "prompts", "providers", "defaultTest", "tests",
               "env", "outputPath", "sharing", "defaultTestOptions", "scenarios",
               "derivedMetrics", "extensions", "commandLineOptions", "redteam",
               "evaluateOptions", "tags", "metadata"}
    stray = set(_config(name)) - allowed
    assert not stray, f"{name} has unknown top-level key(s): {sorted(stray)}"


@pytest.mark.parametrize("name", sorted(PAIRS))
def test_pair_scores_the_full_benchmark_by_default(name):
    cfg = _config(name)
    assert "tests.combined.yaml" in cfg["tests"], \
        f"{name} does not default to the 1000-case benchmark"
    assert cfg["defaultTest"]["assert"][0]["value"] == "file://pf/assert.py"


def test_a_remote_pair_names_a_distinct_pod_per_arm():
    """The failure this catches: both arms pointing at the SAME `remote_url` env var,
    which serves one model under two labels and reports a 0-point delta as a result."""
    cfg = _config("promptfooconfig.v5-vs-base.remote.yaml")
    urls = [p["config"].get("remote_url") for p in cfg["providers"]]
    assert all(urls), "every remote arm needs a remote_url"
    assert len(set(urls)) == len(urls), f"arms share a pod: {urls}"
