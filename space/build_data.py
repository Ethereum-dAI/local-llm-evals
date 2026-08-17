"""Freeze the Space's static data from the eval harness.

Produces two files the Space reads at import time, so the app never needs the
promptfoo run artefacts (which are gitignored and large):

  data/eval_cases.json  — a stratified subset of pf/tests.generated.yaml, each
                          carrying its gold `expected_calls`, so the Space can
                          score a live model run with the real scorer.
  data/benchmark.json   — the frozen scoreboard (overall + per-category) read
                          out of the *.out.json promptfoo runs in the repo root.

Run from the repo root:  uv run python space/build_data.py
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "data"

# How many cases of each category to expose in the Space's "Scored cases" tab.
# Weighted toward the two core categories, but every category is represented so
# the ablation / refusal behaviour is visible too.
QUOTA = {
    "generated-transfer-pos": 10,
    "generated-swap-pos": 10,
    "multiturn-to_token": 3,
    "multiturn-recipient": 3,
    "multiturn-amount": 3,
    "multiturn-token": 2,
    "ablation-to_token": 2,
    "ablation-amount": 2,
    "ablation-recipient": 2,
    "ablation-token": 2,
    "safety-refusal-burn-send": 2,
    "safety-refusal-zero-send": 1,
    "safety-refusal-unverified-token-swap": 2,
    "safety-refusal-approve-unknown-spender": 2,
}

# (file, provider label) -> display name. Each entry is a real promptfoo run;
# nothing here is re-measured or estimated. Kept in step with
# `build_static.py:MODELS` on purpose — the playground's scoreboard and the
# report's strips describing different model sets is worse than either being
# incomplete. Same one-vintage rule: all of these are the 2026-08-10/11 relaunch,
# the first runs made with the 5-tool `pf/tools.json`.
SCOREBOARD = [
    ("relaunch/gemma4-base.final.json", "gemma4-e4b-base", "Gemma-4 E4B (base, wallet-shipped)"),
    ("relaunch/gemma4-ft.final.json", "gemma4-e4b-ft", "Gemma-4 E4B wallet-ft"),
    ("relaunch/qwen3-8b.out.json", "qwen3-8b", "Qwen3-8B (base)"),
    ("relaunch/qwen3-ft.out.json", "qwen3-8b-ft", "Qwen3-8B wallet-ft"),
    ("relaunch/gpt5.final.json", "openrouter:openai/gpt-5", "gpt-5 (anchor)"),
]


def _user_message(vars_: dict) -> str:
    if "user_message" in vars_:
        return vars_["user_message"]
    turns = [m["content"] for m in vars_.get("messages", []) if m.get("role") == "user"]
    return turns[-1] if turns else ""


def build_cases() -> list[dict]:
    tests = yaml.safe_load((ROOT / "pf" / "tests.generated.yaml").read_text())
    taken: collections.Counter[str] = collections.Counter()
    cases = []
    for test in tests:
        md = test["metadata"]
        cat = md["category"]
        if taken[cat] >= QUOTA.get(cat, 0):
            continue
        taken[cat] += 1
        cases.append({
            "id": md["id"],
            "category": cat,
            "label": f"[{cat}] {_user_message(test['vars'])[:70]}",
            "vars": test["vars"],
            "metadata": md,
        })
    cases.sort(key=lambda c: (c["category"], c["id"]))
    return cases


def _run_scores(path: Path, label: str) -> tuple[dict, dict] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    rows = [
        r for r in data.get("results", {}).get("results", [])
        if (r.get("provider", {}).get("label") or r.get("provider", {}).get("id")) == label
    ]
    if not rows:
        return None
    per_cat: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        cat = r["testCase"].get("metadata", {}).get("category", "?")
        bucket = per_cat[cat]
        bucket[1] += 1
        bucket[0] += 1 if r.get("success") else 0
    overall = {
        "passed": sum(1 for r in rows if r.get("success")),
        "total": len(rows),
    }
    return overall, {k: {"passed": v[0], "total": v[1]} for k, v in sorted(per_cat.items())}


def build_benchmark() -> dict:
    models = []
    for filename, label, display in SCOREBOARD:
        scored = _run_scores(ROOT / filename, label)
        if scored is None:
            print(f"  skip (no rows): {filename} :: {label}")
            continue
        overall, per_cat = scored
        models.append({
            "display": display,
            "provider_label": label,
            "source_run": filename,
            "overall": overall,
            "per_category": per_cat,
        })
    models.sort(key=lambda m: -m["overall"]["passed"] / max(m["overall"]["total"], 1))
    return {"dataset": "pf/tests.generated.yaml", "models": models}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cases = build_cases()
    (OUT / "eval_cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    print(f"eval_cases.json: {len(cases)} cases")

    bench = build_benchmark()
    (OUT / "benchmark.json").write_text(json.dumps(bench, indent=2) + "\n")
    for m in bench["models"]:
        o = m["overall"]
        print(f"  {100 * o['passed'] / o['total']:5.1f}%  {o['passed']:3d}/{o['total']:3d}  {m['display']}")


if __name__ == "__main__":
    main()
