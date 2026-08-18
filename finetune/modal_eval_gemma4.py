"""Evaluate the fine-tuned Gemma-4 LoRA ADAPTER on the eval set, scored by the
repo's own deterministic scorer — bypassing the GGUF merge/quantize path.

This is the fast checkpoint (and the fallback if the Q4_K_M GGUF ever comes out
corrupt): load base+adapter on a GPU, render every case in pf/tests.generated.yaml
with the real prompt (keeping the `system` role — no developer remap for Gemma-4),
greedy-generate, translate the raw output the same way pf/provider_functiongemma.py
does (GEMMA4 dialect, <think> stripped), and score with wallet_evals.scorer.

Returns per-category pass rates (captured locally via the entrypoint).

Run:  uv run --with modal modal run finetune/modal_eval_gemma4.py
"""
from __future__ import annotations

from pathlib import Path

import modal

OUTPUTS_DIR = "/outputs"
#: 4096, not 2048. The dev set runs to 6 rounds and the encode below passes
#: truncation=True — at 2048 a long conversation would be silently cut and score as
#: a model failure. The longest prompt in the 1000-case benchmark measures 1133
#: tokens, so 4096 is ~3.5x headroom, and _assert_no_truncation makes a future
#: overrun fail loudly instead of quietly.
MAX_SEQ_LEN = 4096


def _adapters(tag: str, all_checkpoints: bool) -> list[str]:
    """Which adapter(s) to score.

    With `all_checkpoints`, every per-epoch `checkpoint-N` in the run directory is
    returned, oldest first. That is the point of Phase 0c: `eval_loss` selects one
    checkpoint and dev-set ACCURACY may select another, and the disagreement is the
    finding — a same-distribution holdout (85.9% single-turn) cannot see a
    multi-round collapse, so it can happily prefer an over-trained epoch.
    """
    import os
    import re
    run_dir = f"{OUTPUTS_DIR}/run-{tag}" if tag else OUTPUTS_DIR
    stable = f"{OUTPUTS_DIR}/adapter-{tag}" if tag else f"{OUTPUTS_DIR}/adapter"
    out: list[str] = []
    if all_checkpoints and os.path.isdir(run_dir):
        cks = [d for d in os.listdir(run_dir) if re.fullmatch(r"checkpoint-\d+", d)]
        out += [f"{run_dir}/{d}" for d in sorted(cks, key=lambda d: int(d.split("-")[1]))]
    if os.path.isfile(f"{stable}/adapter_config.json"):
        out.append(stable)
    if not out:
        out.append(_resolve_adapter())
    return out


def _resolve_adapter() -> str:
    """Prefer the stable /outputs/adapter path; fall back to the newest
    checkpoint-N the trainer wrote (the probe-loop bug can skip the final save)."""
    import os
    import re
    stable = f"{OUTPUTS_DIR}/adapter"
    if os.path.isfile(f"{stable}/adapter_config.json"):
        return stable
    cks = [d for d in os.listdir(OUTPUTS_DIR) if re.fullmatch(r"checkpoint-\d+", d)]
    if not cks:
        raise SystemExit(f"no adapter or checkpoint-* under {OUTPUTS_DIR}")
    newest = max(cks, key=lambda d: int(d.split("-")[1]))
    return f"{OUTPUTS_DIR}/{newest}"
_REPO = Path(__file__).resolve().parent.parent

hf_cache = modal.Volume.from_name("gemma4-hf-cache", create_if_missing=True)
outputs = modal.Volume.from_name("gemma4-ft-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("unsloth", "huggingface_hub", "pyyaml", "pydantic")
    .env({"HF_HOME": "/root/.cache/huggingface"})
    .add_local_dir(str(_REPO / "src"), "/repo/src")
    .add_local_dir(str(_REPO / "pf"), "/repo/pf")
    .add_local_dir(str(_REPO / "datasets"), "/repo/datasets")
)

app = modal.App("gemma4-eval")


@app.function(image=image, gpu="A100", timeout=7200,
              volumes={"/root/.cache/huggingface": hf_cache, "/outputs": outputs})
def evaluate(dataset: str = "pf/tests.generated.yaml", tag: str = "",
             all_checkpoints: bool = False) -> dict:
    import json
    import sys
    from collections import defaultdict

    sys.path.insert(0, "/repo/src")
    sys.path.insert(0, "/repo")

    import yaml
    from unsloth import FastModel

    from wallet_evals.functiongemma import raw_output_to_scoreable
    from wallet_evals.gemma_dsl import GEMMA4
    from wallet_evals.parsing import parse_turn
    from wallet_evals.promptfoo import case_from_metadata
    from wallet_evals.scorer import score_case
    import importlib
    render = importlib.import_module("pf.prompt").render

    # TWO tool sets, chosen PER CASE exactly as pf/provider_functiongemma.py does.
    # This script previously handed pf/tools.json (the transaction-builder contract:
    # executeTx, base units) to every case — including app-contract cases whose gold
    # is transfer/swap. A model offered the wrong tools cannot pass, so the old
    # numbers understated any app-contract slice.
    builder_tools = json.loads(Path("/repo/pf/tools.json").read_text())
    app_tools = json.loads(Path("/repo/pf/tools.app.json").read_text())

    def tools_for(md: dict) -> list:
        return builder_tools if md.get("protocol") in ("aave", "safe") else app_tools

    tests = yaml.safe_load(Path(f"/repo/{dataset}").read_text())
    print(f"[eval] dataset = {dataset} ({len(tests)} cases)", flush=True)

    adapters = _adapters(tag, all_checkpoints)
    print(f"[eval] scoring {len(adapters)} adapter(s): {adapters}", flush=True)
    per_adapter: dict[str, dict] = {}

    for adapter in adapters:
        print(f"\n[eval] ===== {adapter} =====", flush=True)
        per_adapter[adapter] = _score_one(adapter, tests, tools_for)

    best = max(per_adapter.items(), key=lambda kv: kv[1]["overall_pct"])
    print("\n[eval] ===== dev-set accuracy by checkpoint =====", flush=True)
    for name, sm in per_adapter.items():
        mark = "  <== best by ACCURACY" if name == best[0] else ""
        print(f"  {name:<48} {sm['overall_pct']:>5.1f}% "
              f"({sm['overall_pass']}/{sm['overall_total']}){mark}", flush=True)
    summary = {"dataset": dataset, "tag": tag,
               "best_adapter": best[0], "by_adapter": per_adapter}
    print("[eval] SUMMARY:", json.dumps(summary, indent=2)[:4000], flush=True)
    return summary


def _score_one(adapter: str, tests: list, tools_for) -> dict:
    """Generate and score one adapter over `tests`.

    Split out so the checkpoint loop reloads weights cleanly rather than trying to
    hot-swap adapters. Imports live in here rather than being passed in from the
    caller: they are function-local because Modal's LOCAL client does not have
    unsloth or the repo on its path, and that reason applies to this function as
    much as to `evaluate` — threading a dozen modules through the signature would
    make it look as though they were injectable, which they are not."""
    import json
    import sys
    from collections import defaultdict

    sys.path.insert(0, "/repo/src")
    sys.path.insert(0, "/repo")

    import importlib
    from unsloth import FastModel

    from wallet_evals.functiongemma import raw_output_to_scoreable
    from wallet_evals.gemma_dsl import GEMMA4
    from wallet_evals.parsing import parse_turn
    from wallet_evals.promptfoo import case_from_metadata
    from wallet_evals.scorer import score_case
    render = importlib.import_module("pf.prompt").render

    model, tokenizer = FastModel.from_pretrained(
        model_name=adapter, max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False, full_finetuning=False,
    )
    FastModel.for_inference(model)

    # Gemma-4 E4B is multimodal: FastModel returns a PROCESSOR, whose __call__
    # expects text=/images= (a positional batch misroutes to images -> text=None).
    # Use the underlying text tokenizer for batch encode/decode. Keep the `system`
    # role (native <|turn>system) — no remap.
    tk = getattr(tokenizer, "tokenizer", tokenizer)
    if tk.pad_token is None:
        tk.pad_token = tk.eos_token
    tk.padding_side = "left"
    texts = [tokenizer.apply_chat_template(render({"vars": t["vars"]}),
                                           tools=tools_for(t["metadata"]),
                                           add_generation_prompt=True,
                                           tokenize=False) for t in tests]

    BATCH = 16
    outs: list[str] = []
    for start in range(0, len(texts), BATCH):
        chunk = texts[start:start + BATCH]
        enc = tk(chunk, return_tensors="pt", padding=True, truncation=True,
                 max_length=MAX_SEQ_LEN, add_special_tokens=False).to(model.device)
        # Truncation is silent and looks exactly like a model failure. A 6-round
        # dev case that got cut would be scored as a wrong answer.
        if enc["input_ids"].shape[1] >= MAX_SEQ_LEN:
            raise SystemExit(
                f"prompt hit MAX_SEQ_LEN={MAX_SEQ_LEN} at batch {start} — raise it; "
                f"truncated prompts score as model errors")
        gen = model.generate(**enc, max_new_tokens=256, do_sample=False,
                             pad_token_id=tk.pad_token_id)
        inlen = enc["input_ids"].shape[1]
        for j in range(len(chunk)):
            outs.append(tk.decode(gen[j][inlen:], skip_special_tokens=False))
        print(f"[eval] {start + len(chunk)}/{len(texts)}", flush=True)

    agg = defaultdict(lambda: [0, 0])
    by_rounds = defaultdict(lambda: [0, 0])
    by_mech = defaultdict(lambda: [0, 0])
    total = [0, 0]
    for t, text in zip(tests, outs):
        md = t["metadata"]
        scoreable = raw_output_to_scoreable(text, GEMMA4)
        if scoreable.strip().startswith("["):
            turn = parse_turn(content=None, native_tool_calls=json.loads(scoreable),
                              raw_text="")
        else:
            turn = parse_turn(content=scoreable, native_tool_calls=None,
                              raw_text=scoreable)
        s = score_case(case_from_metadata(md), turn)
        cat = md.get("category", "?")
        agg[cat][0] += s
        agg[cat][1] += 1
        r = md.get("rounds") or 1
        by_rounds[r][0] += s
        by_rounds[r][1] += 1
        mech = md.get("mechanism")
        if mech:
            by_mech[mech][0] += s
            by_mech[mech][1] += 1
        total[0] += s
        total[1] += 1

    summary = {
        "overall_pass": total[0], "overall_total": total[1],
        "overall_pct": round(100 * total[0] / total[1], 1),
        "by_rounds": {str(k): v for k, v in sorted(by_rounds.items())},
        "by_mechanism": {k: v for k, v in sorted(by_mech.items())},
        "by_category": {c: [v[0], v[1]] for c, v in sorted(agg.items())},
    }
    # Depth and mechanism are what the dev set exists to expose — print them per
    # checkpoint so an over-trained epoch is visible without digging into JSON.
    print(f"[eval] {summary['overall_pct']}% "
          f"({summary['overall_pass']}/{summary['overall_total']})", flush=True)
    print("[eval]   by rounds:    "
          + "  ".join(f"{k}r={v[0]}/{v[1]}" for k, v in sorted(by_rounds.items())),
          flush=True)
    if by_mech:
        print("[eval]   by mechanism: "
              + "  ".join(f"{k}={v[0]}/{v[1]}" for k, v in sorted(by_mech.items())),
              flush=True)
    return summary


@app.local_entrypoint()
def main(dataset: str = "pf/tests.dev.yaml", tag: str = "",
         all_checkpoints: bool = True) -> None:
    """Score the DEV set on every per-epoch checkpoint (Phase 0c).

        modal run finetune/modal_eval_gemma4.py --tag e3-lr2e4
        modal run finetune/modal_eval_gemma4.py \
            --dataset pf/tests.generated.yaml --all-checkpoints False

    Defaults to pf/tests.dev.yaml because that is the selector of record. It is
    disjoint from pf/tests.combined.yaml by construction, so scoring it never
    touches the reported number.
    """
    # spawn + --detach so a dropped client connection can't cancel the job. The
    # summary is printed to the logs ("[eval] SUMMARY: {...}"); read it with
    # `modal app logs <app-id>` if the client disconnects before it returns.
    call = evaluate.spawn(dataset=dataset, tag=tag,
                          all_checkpoints=all_checkpoints)
    print(f"SPAWNED evaluate call_id={call.object_id} — poll logs for '[eval] SUMMARY'.")
