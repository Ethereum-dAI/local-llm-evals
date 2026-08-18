"""The harness must render the app's prompt BYTE-for-byte, per template family.

History this guards, in order:

  1. llama-cpp-python's `create_chat_completion` renders without
     `enable_thinking`, dropping the `<|think|>` marker the app emits — 2925 chars
     against the app's 2935. Fixed by rendering explicitly in the provider.
  2. The provider then compared EVERY model against the Gemma dump, so a Qwen GGUF
     failed by construction and was waved through with "expected for non-Gemma
     templates". The wallet's own Qwen dump sat unused in pf/.
  3. With the Qwen dump actually wired up, Qwen rendered 3309 against the app's
     3299. Cause: Jinja's `tojson` is `htmlsafe_json_dumps`, which escapes `'` to
     `\\u0027` AFTER dumping, so `json.dumps_kwargs={"ensure_ascii": False}` cannot
     switch it off. Both app tool descriptions contain "user's" — 2 x 5 chars = the
     10-char gap. llama.cpp's C++ minja does no HTML escaping.

Gemma never hit (3) because its template emits descriptions as raw text in the
FunctionGemma DSL, never through `tojson`. That is luck, not design, which is why
both families are asserted here.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


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


#: (reference name, local GGUF) pairs to assert byte-parity for. Skipped when the
#: model is absent — these are 5 GB files, so the suite stays offline-runnable.
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
