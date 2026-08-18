"""A base-vs-fine-tune config must differ ONLY in the weights.

The four-model comparison rests on this: if the base and fine-tune rows of a pair
differ in sampling, context size, tool format or prompt reference, the delta stops
being "what the fine-tune did" and becomes "what the setup did". That is the exact
error CLAUDE.md warns about for the Modal runs ("only the device differs, so this
is not a full-precision re-run wearing an on-device label").

Asserted here rather than left to review because the two blocks are inlined —
promptfoo validates top-level keys, so a YAML anchor risks failing a run hours in.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

#: config -> the two provider labels that must match except for the model source.
PAIRS = {
    "promptfooconfig.qwen-v4-vs-base.yaml": ("qwen3-8b-base", "qwen3-8b-ft-v4"),
    "promptfooconfig.v4-vs-base.yaml": ("gemma4-e4b-base",
                                        "gemma4-e4b-ft-v4-appprompt"),
}

#: Keys that legitimately differ: they name WHICH weights to load, nothing else.
MODEL_SOURCE_KEYS = {"model_path", "repo_id", "filename", "revision"}


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


def test_qwen_pair_asserts_prompt_parity_against_the_qwen_dump():
    """The specific regression: `prompt_reference` defaults to the GEMMA dump, so a
    Qwen provider that omits it is silently unverified against the app."""
    cfg = _config("promptfooconfig.qwen-v4-vs-base.yaml")
    for provider in cfg["providers"]:
        c = provider["config"]
        assert c.get("prompt_reference") == "qwen", (
            f"{provider['label']} must set prompt_reference: qwen, or its prompt is "
            f"compared against the Gemma dump and waved through")
        assert c.get("tool_format") == "json", provider["label"]
