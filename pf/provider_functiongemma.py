"""promptfoo custom Python provider for local Gemma-family GGUF models.

Serves both FunctionGemma-270m and the shipped Gemma-4 E4B from one code path;
`config.dialect` + `config.system_role` select the model's conventions.

Referenced from promptfooconfig.yaml as:
    providers:
      # FunctionGemma-270m (defaults: dialect=functiongemma, system_role=developer)
      - id: file://pf/provider_functiongemma.py:call_api
        config:
          repo_id: unsloth/functiongemma-270m-it-GGUF
          filename: "*Q8_0.gguf"      # or set model_path to a local .gguf
          n_ctx: 4096
          temperature: 0.2
          max_tokens: 1024
      # Shipped Gemma-4 E4B (the exact SHA-pinned Q4_K_M the wallet installs; the
      # file was deleted from `main`, so pin the revision that still carries it).
      - id: file://pf/provider_functiongemma.py:call_api
        config:
          repo_id: ggml-org/gemma-4-E4B-it-GGUF
          filename: gemma-4-E4B-it-Q4_K_M.gguf
          revision: 1762c8e8713f
          dialect: gemma4
          system_role: system

The model is a local GGUF served in-process by llama-cpp-python. promptfoo runs
this file under $PROMPTFOO_PYTHON (the uv venv, set by scripts/eval.sh), so
`wallet_evals` and `llama_cpp` import from the same interpreter as the scorer.
The persistent worker loads the model ONCE and reuses it across all cases.

Gemma models emit tool calls as plain-text DSL, not OpenAI `tool_calls`, so we
parse them here (wallet_evals.functiongemma) into a shape pf/assert.py already
scores. If a GGUF chat template instead surfaces structured `tool_calls`, we
normalize those to the same shape. The scorer, tools.json, and dataset stay
unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wallet_evals.functiongemma import (
    decode_prompt,
    json_output_to_scoreable,
    raw_output_to_scoreable,
)
from wallet_evals.gemma_dsl import DIALECTS
from wallet_evals.llama_serving import sampling_kwargs

_TOOLS_PATH = Path(__file__).with_name("tools.json")
_REFERENCE_PATH = Path(__file__).with_name("app_contract_reference.json")
#: Rendered-prompt cache keyed by model identity, alongside `_llms`.
_templates: dict[tuple, Any] = {}


def _chat_template(llm, key: tuple):
    """The GGUF's own chat template, compiled, plus a one-time parity assertion.

    Why render here instead of calling `create_chat_completion`: llama-cpp-python
    renders the template WITHOUT `enable_thinking`, which drops the `<|think|>`
    marker the wallet app emits — 2925 chars against the app's 2935 for the same
    conversation. Everything downstream (the fine-tune, the wallet funnel) is now
    aligned on the app's bytes, so a harness that is 10 chars off is measuring a
    different prompt again, which is the exact failure this whole change exists
    to remove. Rendering explicitly also makes the divergence assertable, so it
    cannot drift back silently.
    """
    if key in _templates:
        return _templates[key]

    import jinja2

    source = (llm.metadata or {}).get("tokenizer.chat_template")
    if not source:
        raise RuntimeError("GGUF carries no tokenizer.chat_template; cannot render "
                           "the app's prompt for this model")
    env = jinja2.Environment(loader=jinja2.BaseLoader(),
                             trim_blocks=True, lstrip_blocks=True)
    env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
    template = env.from_string(source)

    # Assert against the app's own dump before scoring a single case. Only
    # meaningful for models whose template is the app's (Gemma-4); a Qwen GGUF
    # legitimately renders differently, so a mismatch there is reported, not
    # fatal.
    reference = json.loads(_REFERENCE_PATH.read_text())
    ref_tools = json.loads(reference["toolsJSON"])
    for case in reference["cases"]:
        got = _render(template, case["messages"], ref_tools)
        if got != case["rendered"]:
            print(f"[provider] NOTE prompt differs from the wallet app for "
                  f"{case['label']}: {len(got)} vs {len(case['rendered'])} chars "
                  f"(expected for non-Gemma templates)", flush=True)
            break
    else:
        print("[provider] prompt parity OK against the wallet app", flush=True)

    # promptfoo runs providers in a persistent worker and swallows their stdout,
    # so the line above is invisible in practice. Drop a sentinel next to the
    # results as well, or the parity guarantee is unverifiable after the fact —
    # which is how the original drift went unnoticed for weeks.
    try:
        verdicts = {case["label"]: _render(template, case["messages"], ref_tools)
                    == case["rendered"] for case in reference["cases"]}
        Path("/tmp/pf_prompt_parity.json").write_text(json.dumps(
            {"model": key[0] or key[1], "parity": verdicts}, indent=2))
    except OSError:
        pass  # diagnostics only; never fail a run over the sentinel

    _templates[key] = template
    return template


def _render(template, messages: list[dict], tools: list[dict]) -> str:
    """Render one conversation the way the app does.

    `enable_thinking=True` mirrors `SamplerOptions.enableThinking`. The leading
    `<bos>` is stripped because `create_completion` tokenizes with `add_bos`,
    and llama.cpp likewise adds BOS as a token rather than as text.
    """
    text = template.render(messages=messages, tools=tools,
                           add_generation_prompt=True, enable_thinking=True)
    return text[len("<bos>"):] if text.startswith("<bos>") else text
# Cache keyed by model identity, NOT a single global: a base-vs-fine-tuned config
# has two providers from this same file, and if promptfoo serves them from one
# worker a single global would make the second silently reuse the first's weights.
_llms: dict[tuple, Any] = {}


def _load_model(config: dict[str, Any]):
    """Lazily construct (and cache) the Llama model, keyed by its identity."""
    n_ctx = int(config.get("n_ctx", 4096))
    model_path = config.get("model_path")
    revision = config.get("revision")
    key = (model_path, config.get("repo_id"), config.get("filename"), revision,
           n_ctx)
    if key in _llms:
        return _llms[key]

    from llama_cpp import Llama  # heavy, optional dep — import only when serving

    # Default 0 = CPU only, which is what every earlier local run used. -1 offloads
    # every layer to Metal: same weights, same quantization, same sampling — only
    # the backend doing the arithmetic changes — but ~an order of magnitude less
    # wall clock, which is the difference between a 1-hour and a 10-hour run.
    n_gpu_layers = int(config.get("n_gpu_layers", 0))

    if model_path:
        llm = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
                    verbose=False)
    elif revision:
        # Pinned revision: llama-cpp's from_pretrained globs the repo's `main`
        # branch, but the wallet's Q4_K_M was deleted from main (it lives only at
        # this pinned commit). Resolve the exact file ourselves so the benchmark
        # loads the byte-identical GGUF the wallet ships.
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=config["repo_id"],
                               filename=config["filename"], revision=revision)
        llm = Llama(model_path=path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
                    verbose=False)
    else:
        llm = Llama.from_pretrained(
            repo_id=config["repo_id"],
            filename=config.get("filename", "*.gguf"),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
    _llms[key] = llm
    return llm


_APP_TOOLS_PATH = Path(__file__).with_name("tools.app.json")


def _load_tools(config: dict[str, Any], vars_: dict[str, Any]) -> list[dict]:
    """The tool menu for this case — the app's two, or the builder's four.

    Mirrors `pf.prompt.tools_for`, duplicated here rather than imported because
    promptfoo loads this module by path and `pf/` is not reliably importable.
    Kept honest by both reading the same two files.

    An explicit `tools_path` in the provider config still overrides everything.
    """
    override = config.get("tools_path")
    if override:
        return json.loads(Path(override).read_text())
    # Aave/Safe are deliberately still on the transaction-builder contract; every
    # other case must see exactly what the wallet app offers.
    if (vars_ or {}).get("protocol") in ("aave", "safe"):
        return json.loads(_TOOLS_PATH.read_text())
    return json.loads(_APP_TOOLS_PATH.read_text())


def call_api(prompt: str, options: dict, context: dict) -> dict:
    config = (options or {}).get("config", {}) or {}
    # `tool_format` selects how the model's OUTPUT is read: the Gemma DSL (default,
    # unchanged for the Gemma-family providers) or JSON-in-text, which is what the
    # Qwen3 fine-tune emits.
    tool_format = config.get("tool_format", "gemma")
    dialect = DIALECTS[config.get("dialect", "functiongemma")]
    system_role = config.get("system_role", "developer")
    try:
        llm = _load_model(config)
        messages = decode_prompt(prompt, system_role=system_role)
        tools = _load_tools(config, (context or {}).get("vars", {}))
        # top_p/top_k/min_p are passed only when the config names them, so the
        # Gemma-family providers keep llama-cpp's defaults untouched while a
        # model whose card prescribes sampling (Qwen3) can be run the way its
        # authors specify.
        sampling = sampling_kwargs(config)
        # Render the app's exact prompt ourselves rather than letting
        # create_chat_completion do it — see _chat_template for why.
        key = (config.get("model_path"), config.get("repo_id"),
               config.get("filename"), config.get("revision"),
               int(config.get("n_ctx", 4096)))
        rendered = _render(_chat_template(llm, key), messages, tools)
        # Explicit turn-end stops on top of the model's EOS token. The GGUF
        # declares one eos id (106) and llama-cpp stops on it, but a raw
        # `create_completion` has none of the chat wrapper's turn awareness, so a
        # model that emits the turn marker as ordinary text would run to
        # max_tokens on every case — minutes per case across a 569-case run.
        # Harmless when EOS already fires, and both markers sit after any tool
        # call, so nothing scoreable is truncated.
        stops = list(config.get("stop") or ["<turn|>", "<end_of_turn>"])
        resp = llm.create_completion(
            rendered,
            temperature=float(config.get("temperature", 0.2)),
            max_tokens=int(config.get("max_tokens", 1024)),
            stop=stops,
            **sampling,
        )
    except Exception as e:  # surface as a case error, not a crashed run
        return {"output": "", "error": f"{type(e).__name__}: {e}"}

    text = resp["choices"][0].get("text") or ""
    # No native `tool_calls` on the raw-completion path: the model's turn is text
    # and is parsed exactly as the app parses it.
    if tool_format == "json":
        output = json_output_to_scoreable(text)
    else:
        output = raw_output_to_scoreable(text, dialect)
    result: dict[str, Any] = {"output": output}
    if isinstance(resp.get("usage"), dict):
        u = resp["usage"]
        result["tokenUsage"] = {
            "prompt": u.get("prompt_tokens"),
            "completion": u.get("completion_tokens"),
            "total": u.get("total_tokens"),
        }
    return result
