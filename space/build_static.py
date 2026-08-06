"""Freeze the static report's data from the promptfoo run artefacts.

The Space has no server, so everything it shows must be baked in here: the case
list, the gold call, and — for every model — the output it actually produced and
the scorer's verdict on it. Nothing is recomputed at view time and nothing is
estimated; each row is lifted from a real `*.out.json` in the repo root.

Run from the repo root:  uv run python space/build_static.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "static" / "data.json"

# Case display order. Grouped so the story is legible left-to-right in the tick
# strips: the two big "must emit a call" blocks, then multi-turn, then the two
# blocks where the correct answer is NOT to call a tool. A model that only ever
# passes by staying silent shows up as a cluster at the far right.
CATEGORY_ORDER = [
    "generated-transfer-pos",
    "generated-swap-pos",
    "multiturn-amount",
    "multiturn-recipient",
    "multiturn-token",
    "multiturn-to_token",
    "ablation-amount",
    "ablation-recipient",
    "ablation-token",
    "ablation-to_token",
    "safety-refusal-burn-send",
    "safety-refusal-zero-send",
    "safety-refusal-approve-unknown-spender",
    "safety-refusal-unverified-token-swap",
]

# Coarser bands, for the axis under the strips.
BANDS = [
    ("transfer", ["generated-transfer-pos"]),
    ("swap", ["generated-swap-pos"]),
    ("multi-turn", ["multiturn-amount", "multiturn-recipient",
                    "multiturn-token", "multiturn-to_token"]),
    ("ablation", ["ablation-amount", "ablation-recipient",
                  "ablation-token", "ablation-to_token"]),
    ("refusal", ["safety-refusal-burn-send", "safety-refusal-zero-send",
                 "safety-refusal-approve-unknown-spender",
                 "safety-refusal-unverified-token-swap"]),
]

# key -> (source run, provider label, display name, kind, note)
# Every source run below covers all 307 cases, so the strips are comparable.
MODELS = [
    ("fg270m-ft", "functiongemma.ft.out.json", "functiongemma-ft",
     "FunctionGemma-270M wallet-ft", "local",
     "The fine-tune. LoRA on 1739 examples, exported Q8_0."),
    ("fg270m", "functiongemma-all.out.json", "functiongemma-270m-it",
     "FunctionGemma-270M base", "local",
     "What it started from, untuned."),
    ("e4b", "gemma4.ft.out.json", "gemma4-e4b-base",
     "Gemma-4 E4B base", "local",
     "The GGUF the wallet actually ships today."),
    ("e4b-ft", "gemma4ft.fresh.out.json", "gemma4-e4b-ft",
     "Gemma-4 E4B wallet-ft", "local",
     "Same data, same recipe, bigger model."),
    ("4o-mini", "functiongemma-all.out.json", "openrouter:openai/gpt-4o-mini",
     "gpt-4o-mini", "hosted", "Hosted reference point."),
    ("gemma26b", "functiongemma-all.out.json", "openrouter:google/gemma-4-26b-a4b-it",
     "Gemma-4 26B-A4B", "hosted", "Hosted reference point."),
    ("gpt5", "functiongemma-all.out.json", "openrouter:openai/gpt-5",
     "gpt-5", "hosted", "The capable anchor. Calibrates the ceiling."),
]

MAX_OUTPUT_CHARS = 1400


def _label(result: dict) -> str:
    p = result.get("provider", {})
    return p.get("label") or p.get("id") or ""


def _rows(filename: str, provider_label: str) -> dict[str, dict]:
    """Map case id -> that provider's recorded result, from one promptfoo run."""
    data = json.loads((ROOT / filename).read_text())
    out = {}
    for r in data.get("results", {}).get("results", []):
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
    per_model = {
        key: _rows(filename, label)
        for key, filename, label, _display, _kind, _note in MODELS
    }

    # The case list comes from the run with full coverage; every model's run is
    # keyed by the same generated case ids.
    reference = json.loads((ROOT / "functiongemma-all.out.json").read_text())
    seen: dict[str, dict] = {}
    for r in reference["results"]["results"]:
        md = r["testCase"].get("metadata", {})
        if md.get("id") and md["id"] not in seen:
            seen[md["id"]] = r

    order = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    ordered = sorted(seen.values(),
                     key=lambda r: (order.get(r["testCase"]["metadata"]["category"], 99),
                                    r["testCase"]["metadata"]["id"]))

    cases = []
    for r in ordered:
        md = r["testCase"]["metadata"]
        vars_ = r["testCase"].get("vars", {})
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
        for key, *_ in MODELS:
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
    for key, filename, label, display, kind, note in MODELS:
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
    bands = [{"label": name, "count": sum(counts[c] for c in cats)} for name, cats in BANDS]

    # The axis under the strips merges ablation + refusal: at 7/307 the refusal
    # band is too narrow to carry a legible label, and the two together are one
    # idea anyway — the region where emitting no call is the correct answer.
    axis_bands = []
    for band in bands:
        if band["label"] in ("ablation", "refusal"):
            continue
        axis_bands.append(dict(band))
    merged = sum(b["count"] for b in bands if b["label"] in ("ablation", "refusal"))
    # Kept short deliberately: at 35/307 this band is ~11% of the strip width,
    # which is not enough room for a longer label without clipping.
    axis_bands.append({"label": "no call", "count": merged})

    payload = {
        "dataset": "pf/tests.generated.yaml",
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
    print(f"{OUT.relative_to(ROOT)}  {len(cases)} cases  {size:.0f} KB")
    for m in models:
        pct = 100 * m["passed"] / m["total"]
        print(f"  {pct:5.1f}%  {m['passed']:3d}/{m['total']:3d}  {m['display']}")


if __name__ == "__main__":
    main()
