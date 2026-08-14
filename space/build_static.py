"""Freeze the static report's data from the promptfoo run artefacts.

The Space has no server, so everything it shows must be baked in here: the case
list, the gold call, and — for every model — the output it actually produced and
the scorer's verdict on it. Nothing is recomputed at view time and nothing is
estimated; each row is lifted from a real `*.out.json` in the repo root.

Run from the repo root:  uv run python space/build_static.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "static" / "data.json"

# The benchmark itself: 593 cases across 33 categories. This is the case-list
# reference (NOT any one model's *.out.json) so every category the dataset
# defines shows up even for a model that hasn't been run yet.
DATASET = "pf/tests.combined.yaml"

# Category display order. Grouped so the story reads left-to-right in the tick
# strips:
#   1. the core transfer/swap blocks (the original 197-case app-contract set)
#   2. the new arithmetic slice — every "arithmetic-*" category, kept together
#      as one block (positives, its own multi-turn follow-ups, its own
#      ablations) because that's how the 80-case slice is labelled as a unit
#   3. multi-turn (non-arithmetic): mostly transfer/swap follow-ups, plus a
#      couple of railgun "which address" follow-ups (multiturn-to) that share
#      the naming, not the protocol
#   4. the protocol families: railgun, aave, safe (all executeTx/shield/
#      unshield, no arithmetic slice of their own)
#   5. the blocks where the correct answer is NOT to call a tool: ablations,
#      then safety refusals (including the two railgun-specific refusal
#      categories) — a model that only ever passes by staying silent shows up
#      as a cluster at the far right.
CATEGORY_ORDER = [
    # 1. core transfer / swap
    "generated-transfer-pos",
    "generated-swap-pos",
    # 2. arithmetic slice (80 cases, all "arithmetic-*")
    "arithmetic-transfer-pos",
    "arithmetic-swap-pos",
    "arithmetic-multiturn-amount",
    "arithmetic-multiturn-recipient",
    "arithmetic-multiturn-to_token",
    "arithmetic-ablation-amount",
    "arithmetic-ablation-recipient",
    "arithmetic-ablation-to_token",
    # 3. multi-turn (non-arithmetic)
    "multiturn-amount",
    "multiturn-recipient",
    "multiturn-token",
    "multiturn-to_token",
    "multiturn-to",
    # 4. protocol families
    "railgun-shield",
    "railgun-unshield",
    "aave-supply",
    "aave-withdraw",
    "aave-borrow",
    "aave-repay",
    "safe-add-signer",
    "safe-remove-signer",
    # 5. no call expected: ablations, then safety refusals
    "ablation-amount",
    "ablation-recipient",
    "ablation-token",
    "ablation-to_token",
    "safety-refusal-burn-send",
    "safety-refusal-zero-send",
    "safety-refusal-approve-unknown-spender",
    "safety-refusal-unverified-token-swap",
    "safety-refusal-unshield-burn",
    "safety-refusal-unshield-zero",
]


def _band_of(category: str) -> str:
    """Coarse band a category belongs to. Mirrors bandOf() in index.html —
    keep the two in sync if a new category prefix is ever added."""
    if category.startswith("arithmetic-"):
        return "arithmetic"
    if category.startswith("generated-transfer"):
        return "transfer"
    if category.startswith("generated-swap"):
        return "swap"
    if category.startswith("multiturn"):
        return "multi-turn"
    if category.startswith("railgun"):
        return "railgun"
    if category.startswith("aave"):
        return "aave"
    if category.startswith("safe-"):
        return "safe"
    if category.startswith("ablation"):
        return "ablation"
    if category.startswith("safety-refusal"):
        return "refusal"
    raise ValueError(f"no band for category {category!r} — add one to _band_of")


BAND_ORDER = ["transfer", "swap", "arithmetic", "multi-turn",
              "railgun", "aave", "safe", "ablation", "refusal"]

# key -> (source run, provider label, display name, kind, note)
# A missing source run is handled gracefully (warn + skip), so this report can
# be built before every run has landed.
#
# Display order is deliberately weakest-to-strongest so the strips read as a
# progression left-to-right: the two on-device bases, the two on-device
# wallet fine-tunes, then the hosted frontier anchor.
MODELS = [
    ("e4b-base", "gemma4.appcontract.out.json", "gemma4-e4b-base",
     "Gemma-4 E4B base", "local",
     "The Q4_K_M GGUF local-wallet-mac ships today, unmodified."),
    ("qwen3-base", "qwen3-base.appcontract.out.json", "qwen3-8b",
     "Qwen3-8B base", "hosted",
     "Hosted via OpenRouter, card-recommended sampling (temperature 0.6, "
     "top_p 0.95, top_k 20) — the closest same-scale analogue to the "
     "on-device Gemma-4 E4B family, unmodified."),
    ("qwen3-ft", "qwen3-ft.appcontract.out.json", "qwen3-8b-ft-appcontract",
     "Qwen3-8B wallet-ft (app-contract)", "local",
     "Same Qwen3-8B base, re-tuned on the app-contract data (Q4_K_M, run "
     "locally). Distinct from an older Qwen fine-tune trained on a "
     "different, base-unit contract — the -appcontract suffix is the guard "
     "against scoring the wrong one."),
    ("e4b-ft", "gemma4.appcontract.out.json", "gemma4-e4b-ft-appcontract",
     "Gemma-4 E4B wallet-ft (app-contract)", "local",
     "Same base, re-tuned on the app-contract data: human-decimal transfer/"
     "swap, plus railgun/aave/safe. Still training — not yet published."),
    ("gpt5", "gpt5.appcontract.out.json", "openrouter:openai/gpt-5",
     "gpt-5", "hosted",
     "Hosted frontier anchor. Calibrates the ceiling on the same 593 cases."),
]

MAX_OUTPUT_CHARS = 1400


def _label(result: dict) -> str:
    p = result.get("provider", {})
    return p.get("label") or p.get("id") or ""


def _load_run(filename: str) -> dict | None:
    path = ROOT / filename
    if not path.exists():
        print(f"warning: {filename} not found — skipping model(s) that read it",
              file=sys.stderr)
        return None
    return json.loads(path.read_text())


def _rows(run: dict | None, provider_label: str) -> dict[str, dict]:
    """Map case id -> that provider's recorded result, from one promptfoo run."""
    if run is None:
        return {}
    out = {}
    for r in run.get("results", {}).get("results", []):
        if _label(r) != provider_label:
            continue
        case_id = r["testCase"].get("metadata", {}).get("id")
        if not case_id:
            continue
        out[case_id] = r
    return out


def _prompt_of(vars_: dict) -> str:
    if "user_message" in vars_:
        return vars_["user_message"]
    turns = [m["content"] for m in vars_.get("messages", []) if m.get("role") == "user"]
    return turns[-1] if turns else ""


def _output_text(result: dict) -> str:
    out = result.get("response", {}).get("output")
    if out is None:
        err = result.get("error") or "(no output recorded)"
        return str(err)[:MAX_OUTPUT_CHARS]
    text = out if isinstance(out, str) else json.dumps(out)
    return text[:MAX_OUTPUT_CHARS]


def main() -> None:
    # Load each run artefact once, even when two models share a file.
    run_cache: dict[str, dict | None] = {}
    for _key, filename, *_rest in MODELS:
        if filename not in run_cache:
            run_cache[filename] = _load_run(filename)

    active_models = [m for m in MODELS if run_cache[m[1]] is not None]
    skipped = [m for m in MODELS if run_cache[m[1]] is None]
    for key, filename, *_rest in skipped:
        print(f"warning: skipping model {key!r} — {filename} is missing",
              file=sys.stderr)

    per_model = {
        key: _rows(run_cache[filename], label)
        for key, filename, label, _display, _kind, _note in active_models
    }

    # The case list comes from the benchmark definition itself, not from any
    # one model's run — every category the dataset defines shows up even for
    # a model that hasn't been scored yet.
    dataset_cases = yaml.safe_load((ROOT / DATASET).read_text())

    categories_in_data = {c["metadata"]["category"] for c in dataset_cases}
    missing_from_order = categories_in_data - set(CATEGORY_ORDER)
    assert not missing_from_order, (
        f"category present in {DATASET} but missing from CATEGORY_ORDER: "
        f"{sorted(missing_from_order)}"
    )
    extra_in_order = set(CATEGORY_ORDER) - categories_in_data
    assert not extra_in_order, (
        f"CATEGORY_ORDER lists a category not present in {DATASET}: "
        f"{sorted(extra_in_order)}"
    )

    order = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    ordered = sorted(
        dataset_cases,
        key=lambda d: (order[d["metadata"]["category"]], d["metadata"]["id"]),
    )

    cases = []
    for d in ordered:
        md = d["metadata"]
        vars_ = d.get("vars", {})
        entry = {
            "id": md["id"],
            "category": md["category"],
            "difficulty": md.get("difficulty"),
            "prompt": _prompt_of(vars_),
            "turns": vars_.get("messages"),
            "summary": vars_.get("expected_summary"),
            "gold": md.get("expected_calls", []),
            "results": {},
        }
        for key, *_ in active_models:
            got = per_model[key].get(md["id"])
            if got is None:
                continue
            entry["results"][key] = {
                "pass": bool(got.get("success")),
                "output": _output_text(got),
                "reason": (got.get("gradingResult") or {}).get("reason", ""),
            }
        cases.append(entry)

    models = []
    for key, filename, label, display, kind, note in active_models:
        marks = [c["results"].get(key, {}).get("pass") for c in cases]
        models.append({
            "key": key,
            "display": display,
            "kind": kind,
            "note": note,
            "source_run": filename,
            "provider_label": label,
            "passed": sum(1 for m in marks if m),
            "total": sum(1 for m in marks if m is not None),
            "marks": [1 if m else 0 for m in marks],
        })

    # Per-category tallies, for the breakdown table.
    for m in models:
        tally: dict[str, list[int]] = {}
        for case, mark in zip(cases, m["marks"]):
            bucket = tally.setdefault(case["category"], [0, 0])
            bucket[1] += 1
            bucket[0] += mark
        m["per_category"] = {k: {"passed": v[0], "total": v[1]} for k, v in tally.items()}

    counts = {c: 0 for c in CATEGORY_ORDER}
    for case in cases:
        counts[case["category"]] += 1

    band_categories: dict[str, list[str]] = {b: [] for b in BAND_ORDER}
    for cat in CATEGORY_ORDER:
        band_categories[_band_of(cat)].append(cat)
    bands = [{"label": name, "count": sum(counts[c] for c in band_categories[name])}
              for name in BAND_ORDER]

    # The axis under the strips merges ablation + refusal: together they are
    # one idea (the region where emitting no call is the correct answer) and
    # each is individually too narrow a slice of 593 cases to carry a legible
    # label without clipping.
    axis_bands = []
    for band in bands:
        if band["label"] in ("ablation", "refusal"):
            continue
        axis_bands.append(dict(band))
    merged = sum(b["count"] for b in bands if b["label"] in ("ablation", "refusal"))
    axis_bands.append({"label": "no call", "count": merged})

    payload = {
        "dataset": DATASET,
        "case_count": len(cases),
        "category_order": CATEGORY_ORDER,
        "bands": bands,
        "axis_bands": axis_bands,
        "models": models,
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")))

    size = OUT.stat().st_size / 1024
    print(f"{OUT.relative_to(ROOT)}  {len(cases)} cases  {size:.0f} KB  "
          f"{len(models)}/{len(MODELS)} models")
    for m in models:
        pct = 100 * m["passed"] / m["total"]
        print(f"  {pct:5.1f}%  {m['passed']:3d}/{m['total']:3d}  {m['display']}")
    if skipped:
        print(f"skipped {len(skipped)} model(s) with no run artefact yet: "
              + ", ".join(k for k, *_ in skipped))


if __name__ == "__main__":
    main()
