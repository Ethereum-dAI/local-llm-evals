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
import os
import re
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

#: The wallet's own rendered-prompt dumps, one per chat template family, emitted by
#: `wallet-eval prompt-dump`. `systemPrompt` and `toolsJSON` are byte-identical
#: across them (asserted by test_prompt_reference_dumps_agree_on_the_contract) —
#: only `rendered` differs, because each family serializes the same tools its own
#: way (Gemma's FunctionGemma DSL vs Qwen's Hermes JSON).
_REFERENCE_PATHS = {
    "gemma": Path(__file__).with_name("app_contract_reference.json"),
    "qwen": Path(__file__).with_name("app_contract_reference.qwen.json"),
}
_REFERENCE_PATH = _REFERENCE_PATHS["gemma"]  # back-compat for importers


def _reference_for(config: dict) -> tuple[str, Path | None]:
    """Which app dump this model's rendered prompt must match.

    Explicit rather than inferred, because `tool_format: json` alone cannot tell
    Qwen from Phi-4-mini or SmolLM3: all three emit JSON-in-text, but only Qwen has
    a dump from the wallet to compare against. Guessing would either skip Qwen's
    assertion (the bug this replaces — a Qwen GGUF was compared to the GEMMA dump,
    failed, and was waved through with a NOTE) or invent a parity claim for a model
    the app does not ship.

    `prompt_reference: none` states that no dump applies, which is the honest
    answer for Phi/SmolLM3 and keeps "unchecked" distinguishable from "checked and
    matching".
    """
    name = config.get("prompt_reference")
    if name is None:
        # Gemma-family providers keep their existing behaviour untouched.
        name = "gemma" if config.get("tool_format", "gemma") == "gemma" else "none"
    if name == "none":
        return "none", None
    if name not in _REFERENCE_PATHS:
        raise ValueError(
            f"prompt_reference={name!r} is not one of "
            f"{sorted(_REFERENCE_PATHS) + ['none']}")
    return name, _REFERENCE_PATHS[name]
#: Rendered-prompt cache keyed by model identity, alongside `_llms`.
_templates: dict[tuple, Any] = {}


def _chat_template(llm, key: tuple, reference_name: str = "gemma",
                   reference_path: Path | None = None):
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
    # Jinja's built-in `tojson` is `htmlsafe_json_dumps`, which escapes <, >, & and
    # ' to \uXXXX AFTER dumping — so `json.dumps_kwargs` cannot switch it off. The
    # wallet renders through llama.cpp's C++ minja, which does no HTML escaping, so
    # any apostrophe in a tool description diverged: the app sends
    # "from the user's smart account", Jinja sent "from the user\u0027s smart
    # account". Both app tool descriptions contain "user's", which is exactly the
    # 10-character gap (2 x 5) between our 3309 and the app's 3299 for Qwen.
    #
    # Gemma was unaffected only by luck — its template serializes tools into the
    # FunctionGemma DSL and emits descriptions as raw text, never through `tojson`
    # — which is why this hid until a Hermes/JSON template was checked against its
    # own dump. Overriding the filter fixes Qwen and leaves Gemma byte-identical
    # (asserted for both in tests/test_prompt_parity.py).
    env.filters["tojson"] = lambda value, indent=None: json.dumps(
        value, ensure_ascii=False, indent=indent)
    template = env.from_string(source)

    # Assert against the app's own dump for THIS template family before scoring a
    # single case. Previously only the Gemma dump existed here, so a Qwen GGUF was
    # compared against Gemma's rendered bytes, failed by construction, and was
    # waved through with "expected for non-Gemma templates" — leaving Qwen's prompt
    # unverified while the wallet's own Qwen dump sat unused in this directory.
    reference = None
    if reference_path is not None:
        reference = json.loads(reference_path.read_text())
        ref_tools = json.loads(reference["toolsJSON"])
        mismatches = []
        for case in reference["cases"]:
            got = _render(template, case["messages"], ref_tools)
            if got != case["rendered"]:
                mismatches.append(
                    f"{case['label']}: {len(got)} vs {len(case['rendered'])} chars")
        if mismatches:
            print(f"[provider] WARNING prompt DIFFERS from the wallet app "
                  f"({reference_name} dump {reference_path.name}): "
                  f"{'; '.join(mismatches)} — scores from this run do not "
                  f"transfer to the product", flush=True)
        else:
            print(f"[provider] prompt parity OK against the wallet app "
                  f"({reference_name} dump)", flush=True)
    else:
        print("[provider] NOTE no wallet prompt dump applies to this model "
              "(prompt_reference: none) — prompt is UNVERIFIED", flush=True)

    # promptfoo runs providers in a persistent worker and swallows their stdout,
    # so the line above is invisible in practice. Drop a sentinel next to the
    # results as well, or the parity guarantee is unverifiable after the fact —
    # which is how the original drift went unnoticed for weeks.
    # The filename carries the model, not just the contents: a single eval runs
    # several providers through this same code path, and a config-wide sentinel
    # would leave only the last one's verdict on disk — the two-config-dir
    # pattern for concurrent runs clobbers across evals too. Per-model files
    # make every provider's verdict survive.
    try:
        if reference is not None:
            ref_tools = json.loads(reference["toolsJSON"])
            verdicts = {case["label"]: _render(template, case["messages"], ref_tools)
                        == case["rendered"] for case in reference["cases"]}
        else:
            verdicts = None  # not false: nothing was compared
        model = key[0] or key[1] or "unknown"
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(model)).strip("_")[:120]
        Path(f"/tmp/pf_prompt_parity.{slug}.json").write_text(json.dumps(
            {"model": model, "reference": reference_name, "parity": verdicts},
            indent=2))
    except OSError:
        pass  # diagnostics only; never fail a run over the sentinel

    _templates[key] = template
    return template


def _render(template, messages: list[dict], tools: list[dict],
            enable_thinking: bool = True) -> str:
    """Render one conversation the way the app does.

    `enable_thinking=True` is the DEFAULT because it mirrors the app's
    `SamplerOptions.enableThinking`, and that parity is load-bearing — it is what makes
    the harness's bytes match the wallet's 2935 (see `_chat_template`). Do not change
    the default to win an eval.

    It is a knob because thinking is implicated in base's single largest failure bucket:
    40 of its 74 non-safety failures on the frozen set are the model reasoning correctly
    inside `<|channel>thought` and then asking a clarifying question instead of emitting
    the call. A prompt clause telling it to act did nothing (verified in the prompt, 9/9
    failures still ended in a question), so the next question is whether the reasoning
    pass itself is what talks it out of acting. `enable_thinking=False` is A/B ONLY until
    the app changes the same setting.

    The leading `<bos>` is stripped because `create_completion` tokenizes with `add_bos`,
    and llama.cpp likewise adds BOS as a token rather than as text.
    """
    text = template.render(messages=messages, tools=tools,
                           add_generation_prompt=True,
                           enable_thinking=enable_thinking)
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
           n_ctx, bool(config.get("remote_url")))
    if key in _llms:
        return _llms[key]

    from llama_cpp import Llama  # heavy, optional dep — import only when serving

    # Default 0 = CPU only, which is what every earlier local run used. -1 offloads
    # every layer to Metal: same weights, same quantization, same sampling — only
    # the backend doing the arithmetic changes — but ~an order of magnitude less
    # wall clock, which is the difference between a 1-hour and a 10-hour run.
    n_gpu_layers = int(config.get("n_gpu_layers", 0))

    # With `remote_url`, generation happens on a rented GPU and the ONLY thing this
    # local model is for is `metadata["tokenizer.chat_template"]`. vocab_only skips
    # the 5 GB of weights entirely — seconds to load, no VRAM, and it keeps the
    # prompt bytes provably identical to a local run because the template comes from
    # the same GGUF file.
    vocab_only = bool(config.get("remote_url"))
    extra = {"vocab_only": True} if vocab_only else {}

    if model_path:
        llm = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
                    verbose=False, **extra)
    elif revision:
        # Pinned revision: llama-cpp's from_pretrained globs the repo's `main`
        # branch, but the wallet's Q4_K_M was deleted from main (it lives only at
        # this pinned commit). Resolve the exact file ourselves so the benchmark
        # loads the byte-identical GGUF the wallet ships.
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=config["repo_id"],
                               filename=config["filename"], revision=revision)
        llm = Llama(model_path=path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
                    verbose=False, **extra)
    else:
        llm = Llama.from_pretrained(
            repo_id=config["repo_id"],
            filename=config.get("filename", "*.gguf"),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
            **extra,
        )
    _llms[key] = llm
    return llm


def _clear_kv(llm) -> None:
    """Actually free the KV cache between cases.

    `Llama.reset()` alone is NOT enough. Read its source: it sets `n_tokens = 0`
    and only calls `llama_memory_clear` when the model `_is_recurrent` or
    `_is_hybrid`. Gemma-4 is a plain transformer, so the llama.cpp-side cache is
    never freed and cells accumulate across `create_completion` calls until no KV
    slot can be allocated — at which point `llama_decode` returns -3 and, because
    the Llama object is cached across the whole run (`_llms`), EVERY later case
    fails the same way.

    That is not hypothetical: a 1000-case run of gemma4-e4b-base scored 156 cases
    cleanly, failed on case 157, and then failed all 844 remaining cases with
    `RuntimeError: llama_decode returned -3` — an export that still looked
    complete (1000 rows) and reported a plausible 14.7%. The context was never
    the problem: the longest prompt in that dataset is 1133 tokens against
    n_ctx 4096.

    Clearing per case also removes cross-case state as a variable, which is worth
    the re-evaluation of the shared prompt prefix: a benchmark case's result must
    not depend on which case ran before it.
    """
    llm.reset()
    ctx = getattr(llm, "_ctx", None)
    clear = getattr(ctx, "kv_cache_clear", None)
    if clear is not None:
        clear()


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


def _resolve_remote_url(config: dict) -> str | None:
    """`remote_url`, with an explicit `env:NAME` form.

    The pod URL is only known once a GPU has been rented, so it cannot be a literal in
    a committed config. `env:NAME` reads it here instead of relying on promptfoo to
    interpolate `{{ env.NAME }}` inside a PROVIDER config block (it does so for
    prompts and vars; provider config is a different code path and this is not worth
    discovering at the cost of a rented GPU-hour). An unresolved value raises rather
    than silently falling back to local generation, which would report laptop numbers
    under a remote label.
    """
    raw = config.get("remote_url")
    if not raw:
        return None
    raw = str(raw).strip()
    if raw.startswith("env:"):
        name = raw[4:]
        value = os.environ.get(name)
        if not value:
            raise RuntimeError(f"remote_url is 'env:{name}' but ${name} is unset — "
                               f"run `scripts/runpod_serve_gguf.py up` and export it")
        return value.strip().rstrip("/")
    if "{{" in raw:
        raise RuntimeError(f"remote_url was not interpolated ({raw!r}) — use the "
                           f"explicit 'env:NAME' form instead")
    return raw.rstrip("/")


def _remote_completion(base_url: str, rendered: str, kwargs: dict,
                       timeout_s: float = 600.0) -> dict:
    """POST a fully-rendered prompt to a llama.cpp server's /completion endpoint.

    Returns the same shape `Llama.create_completion` does, so the caller's output
    translation is untouched.

    `/completion` (raw prompt) NOT `/v1/chat/completions`: the whole point of
    rendering locally is that the server must not re-apply a chat template. Sending
    messages would let the remote template the conversation its own way and silently
    change the prompt — the divergence this provider exists to prevent.
    """
    import json as _json
    import urllib.error
    import urllib.request

    payload = {
        "prompt": rendered,
        "temperature": kwargs.get("temperature", 0.2),
        "n_predict": kwargs.get("max_tokens", 1024),
        "stop": kwargs.get("stop") or [],
        # Deterministic across arms: llama-server otherwise seeds randomly per
        # request, which would add sampling noise to an A/B whose whole question is a
        # 2-case delta.
        "seed": int(kwargs.get("seed", 0)),
        "cache_prompt": False,
    }
    for src, dst in (("top_p", "top_p"), ("top_k", "top_k"), ("min_p", "min_p")):
        if src in kwargs:
            payload[dst] = kwargs[src]

    req = urllib.request.Request(
        base_url.rstrip("/") + "/completion",
        data=_json.dumps(payload).encode(),
        # The User-Agent is REQUIRED, not cosmetic. RunPod fronts pod ports with
        # Cloudflare, which rejects urllib's default `Python-urllib/3.x` signature with
        # HTTP 403 and body `error code: 1010` — every case errors while the identical
        # request via curl returns 200, so this reads as "the pod is broken" rather than
        # as a blocked client. Any ordinary UA string is accepted.
        headers={"Content-Type": "application/json",
                 "User-Agent": "wallet-evals/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as fh:
        body = _json.loads(fh.read().decode())
    usage = {
        "prompt_tokens": body.get("tokens_evaluated"),
        "completion_tokens": body.get("tokens_predicted"),
    }
    usage["total_tokens"] = (usage["prompt_tokens"] or 0) + (usage["completion_tokens"] or 0)
    return {"choices": [{"text": body.get("content", "")}], "usage": usage}


def _has_call(scoreable: str) -> bool:
    """Did the translated output contain a tool call?

    `raw_output_to_scoreable` / `json_output_to_scoreable` return an OpenAI-shaped JSON
    LIST when a call was parsed and the prose verbatim otherwise, so this is a check on
    the translator's own contract rather than a second parse of the model's text.
    """
    try:
        parsed = json.loads(scoreable)
    except Exception:
        return False
    return (isinstance(parsed, list) and bool(parsed)
            and all(isinstance(c, dict) and "name" in c for c in parsed))


#: The nudge used by `retry_on_no_call`. Phrased as a user turn because that is what an
#: app can actually do — it cannot edit the model's own turn — and it deliberately repeats
#: the refusal escape hatch, because the whole risk of this mechanism is converting a
#: correct refusal into a call on the second attempt.
RETRY_NUDGE = (
    "Do not ask me anything. If you can act, emit the tool call now. "
    "If this request must be refused, say so and make no tool call."
)


def _augment_fn():
    """Load prompt_candidates.augment from the file next to this one.

    promptfoo loads this provider by PATH (`file://pf/provider_functiongemma.py`),
    so the top-level `pf` package is not necessarily importable and
    `from pf.prompt_candidates import ...` can fail depending on how the run was
    launched. Resolving the sibling file directly works under every entrypoint.
    """
    import importlib.util
    path = Path(__file__).with_name("prompt_candidates.py")
    spec = importlib.util.spec_from_file_location("pf_prompt_candidates", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.augment


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
        # A/B ONLY, and off unless a config names it. `prompt_variant` appends
        # candidate sentences to the system turn so a wallet-prompt change can be
        # measured before it is made. It reuses pf/prompt_candidates.py rather than
        # restating the sentences, so the local arm and the Modal arm can never drift
        # into testing different text. A run with this set is NOT app parity.
        variant = config.get("prompt_variant", "none")
        if variant and variant != "none":
            messages = _augment_fn()(messages, variant)
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
        reference_name, reference_path = _reference_for(config)
        # Default True = app parity; see _render.
        rendered = _render(
            _chat_template(llm, key, reference_name, reference_path),
            messages, tools,
            enable_thinking=bool(config.get("enable_thinking", True)))
        # Explicit turn-end stops on top of the model's EOS token. The GGUF
        # declares one eos id (106) and llama-cpp stops on it, but a raw
        # `create_completion` has none of the chat wrapper's turn awareness, so a
        # model that emits the turn marker as ordinary text would run to
        # max_tokens on every case — minutes per case across a 569-case run.
        # Harmless when EOS already fires, and both markers sit after any tool
        # call, so nothing scoreable is truncated.
        stops = list(config.get("stop") or ["<turn|>", "<end_of_turn>"])
        completion_kwargs = dict(
            temperature=float(config.get("temperature", 0.2)),
            max_tokens=int(config.get("max_tokens", 1024)),
            stop=stops,
            **sampling,
        )
        # Start every case from an empty KV cache (see _clear_kv), and if a decode
        # still fails, clear and retry ONCE before giving up. Both halves matter:
        # the pre-clear stops one bad case from poisoning the rest of the run, and
        # the retry keeps a transient allocation failure from costing a case.
        remote_url = _resolve_remote_url(config)
        if remote_url:
            # Same rendered bytes, same sampling, same stops — only the arithmetic
            # moves. Everything downstream (output translation, scoring) stays here,
            # so a scorer or parser change never needs the remote redeployed.
            resp = _remote_completion(remote_url, rendered, completion_kwargs,
                                      timeout_s=float(config.get("remote_timeout", 600)))
        else:
            try:
                _clear_kv(llm)
                resp = llm.create_completion(rendered, **completion_kwargs)
            except Exception:
                _clear_kv(llm)
                resp = llm.create_completion(rendered, **completion_kwargs)
    except Exception as e:  # surface as a case error, not a crashed run
        return {"output": "", "error": f"{type(e).__name__}: {e}"}

    text = resp["choices"][0].get("text") or ""
    # No native `tool_calls` on the raw-completion path: the model's turn is text
    # and is parsed exactly as the app parses it.
    def _translate(raw: str) -> str:
        return (json_output_to_scoreable(raw) if tool_format == "json"
                else raw_output_to_scoreable(raw, dialect))

    output = _translate(text)
    retried = False
    # ONE extra turn when the model answered without calling anything. This targets the
    # largest measured failure bucket — 40 of base's 74 non-safety failures are it
    # reasoning correctly and then asking a clarifying question — which a prompt clause
    # provably did NOT fix (results/act-ab.base-e4b.md).
    #
    # It is an app-level mechanism, not a prompt tweak, and it is DANGEROUS in a specific
    # way: refusal cases legitimately produce no call, so this fires on them too and a
    # second attempt could talk the model into acting. That is why RETRY_NUDGE restates
    # the refusal option, and why any run using this must report the safety slice
    # alongside the accuracy slice. Exactly one retry: a loop would keep pushing until it
    # got a call, which would score well by destroying the refusals.
    if config.get("retry_on_no_call") and not _has_call(output):
        try:
            followup = list(messages) + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": RETRY_NUDGE},
            ]
            rendered2 = _render(
                _chat_template(llm, key, reference_name, reference_path),
                followup, tools,
                enable_thinking=bool(config.get("enable_thinking", True)))
            resp2 = (_remote_completion(remote_url, rendered2, completion_kwargs,
                                        timeout_s=float(config.get("remote_timeout", 600)))
                     if remote_url else llm.create_completion(rendered2, **completion_kwargs))
            text2 = resp2["choices"][0].get("text") or ""
            output2 = _translate(text2)
            # Keep the retry ONLY if it produced a call. If the model declined again, the
            # first answer is the honest one and replacing it would hide a refusal behind
            # a second helping of prose.
            if _has_call(output2):
                output, resp, retried = output2, resp2, True
        except Exception:
            pass  # a failed retry must not lose the first, valid answer
    result: dict[str, Any] = {"output": output}
    if retried:
        result["metadata"] = {"retried_on_no_call": True}
    if isinstance(resp.get("usage"), dict):
        u = resp["usage"]
        result["tokenUsage"] = {
            "prompt": u.get("prompt_tokens"),
            "completion": u.get("completion_tokens"),
            "total": u.get("total_tokens"),
        }
    return result
