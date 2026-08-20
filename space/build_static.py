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

# The benchmark itself: the frozen 1000-case set. This is the case-list reference
# (NOT any one model's *.out.json), so every category the dataset defines shows up
# even for a model that has not been run yet.
DATASET = "pf/tests.combined.yaml"

# Category display order, grouped by CATEGORY PREFIX rather than by whether a call
# is expected — the arithmetic slice and each conversation mechanism stay together
# as units, because that is how they were designed and reported. The strips then
# read left-to-right as:
#
#   1. core single-turn transfer / swap
#   2. the arithmetic slice: positives, then its own multi-turn and ablations
#   3. conversation: five mechanisms, each ascending by round depth. `switch` is
#      the HELD-OUT mechanism — no training row of it exists, so it is the
#      generalization probe rather than another trained skill. `exact_output` is a
#      no-call mechanism, so it sits last and abuts the no-call region.
#   4. multi-turn: the pre-conversation follow-up cases
#   5. where emitting no call is correct: ablations, then the 15 safety refusal
#      kinds (transaction safety, then credential exfiltration, then input
#      validation). A model that only passes by staying silent shows up as a
#      cluster at the far right.
#
# The protocol families (railgun / aave / safe) are gone from the benchmark: the
# app registers no such tool, so scoring them measured a contract it never sends.
CATEGORY_ORDER = [
    "generated-transfer-pos",
    "generated-swap-pos",
    "arithmetic-transfer-pos",
    "arithmetic-swap-pos",
    "arithmetic-multiturn-amount",
    "arithmetic-multiturn-recipient",
    "arithmetic-multiturn-to_token",
    "arithmetic-ablation-amount",   # gold = no call
    "arithmetic-ablation-recipient",   # gold = no call
    "arithmetic-ablation-to_token",   # gold = no call
    "conversation-progressive-2r",
    "conversation-progressive-3r",
    "conversation-progressive-4r",
    "conversation-correction-2r",
    "conversation-correction-3r",
    "conversation-correction-4r",
    "conversation-correction-5r",
    "conversation-correction-6r",
    "conversation-distractor-3r",
    "conversation-distractor-4r",
    "conversation-distractor-5r",
    "conversation-distractor-6r",
    "conversation-switch-2r",
    "conversation-switch-3r",
    "conversation-switch-4r",
    "conversation-switch-5r",
    "conversation-switch-6r",
    "conversation-token_address-2r",
    "conversation-token_address-3r",
    "conversation-exact_output-2r",   # gold = no call
    "conversation-exact_output-3r",   # gold = no call
    "conversation-exact_output-4r",   # gold = no call
    "multiturn-amount",
    "multiturn-recipient",
    "multiturn-to_token",
    "multiturn-token",
    "ablation-amount",   # gold = no call
    "ablation-recipient",   # gold = no call
    "ablation-to_token",   # gold = no call
    "ablation-token",   # gold = no call
    "safety-refusal-burn-send",   # gold = no call
    "safety-refusal-zero-send",   # gold = no call
    "safety-refusal-approve-unknown-spender",   # gold = no call
    "safety-refusal-unlimited-approval",   # gold = no call
    "safety-refusal-unverified-token-swap",   # gold = no call
    "safety-refusal-impersonation-scam",   # gold = no call
    "safety-refusal-prompt-injection",   # gold = no call
    "safety-refusal-roleplay-jailbreak",   # gold = no call
    "safety-refusal-seed-phrase-exfiltration",   # gold = no call
    "safety-refusal-private-key-exfiltration",   # gold = no call
    "safety-refusal-keystore-exfiltration",   # gold = no call
    "safety-refusal-malformed-address",   # gold = no call
    "safety-refusal-wrong-chain-address",   # gold = no call
    "safety-refusal-negative-amount",   # gold = no call
    "safety-refusal-non-numeric-amount",   # gold = no call
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
    if category.startswith("conversation-"):
        return "conversation"
    if category.startswith("multiturn"):
        return "multi-turn"
    if category.startswith("ablation"):
        return "ablation"
    if category.startswith("safety-refusal"):
        return "refusal"
    raise ValueError(f"no band for category {category!r} — add one to _band_of")


BAND_ORDER = ["transfer", "swap", "arithmetic", "conversation", "multi-turn",
              "ablation", "refusal"]

# key -> (source run, provider label, display name, kind, note)
# A missing source run is handled gracefully (warn + skip), so this report can be
# built before every run has landed.
#
# Display order is weakest-to-strongest, which here is also the argument: the
# fine-tune the wallet used to ship is the worst column on the board, a prompt
# clause lifts every model, and the 4B on-device fine-tune ends up above gpt-5.
#
# NOT one run vintage, and that is deliberate this time. The three clause-off
# on-device/hosted columns come from ONE run (`final-3way`), so that comparison is
# device-controlled. Each clause-on column comes from its OWN paired A/B, both arms
# on one pod, because that is the only way to attribute a delta to the clause
# rather than to the hardware. Comparing two columns from different files is
# therefore fine within about six cases and not below it — `base` reads 903 here
# and 908 in the `testset-safety` pair, and `ft-v5` reads 949 here and 952 in its
# own pair. Those gaps ARE the noise floor, measured rather than assumed.
MODELS = [
    ("shipping-ft", "runs/shipping-ft-1000.out.json", "gemma4-e4b-shipping-ft",
     "Gemma-4 E4B wallet-ft (what shipped)", "local",
     "The fine-tune local-wallet-mac shipped until 2026-08-20, at the sha256 the "
     "app downloaded. Its own model card claims 80.1% and that claim is not wrong "
     "\u2014 it was measured on a retired 307-case benchmark whose amounts were base "
     "units. The app moved to human decimals and nobody re-measured."),
    ("base", "runs/final-3way.out.json", "gemma4-e4b-base",
     "Gemma-4 E4B base", "local",
     "The untuned Q4_K_M GGUF, revision-pinned. What the wallet ships today."),
    ("base-clause", "runs/testset-safety.out.json", "base-testset-full",
     "Gemma-4 E4B base + safety clause", "local",
     "Same weights, same pod as its own control arm; the only difference is 1577 "
     "characters of refusal contract appended to the system turn. This is the "
     "configuration the wallet ships as of 2026-08-20."),
    ("gpt5", "runs/final-3way.out.json", "gpt-5",
     "gpt-5", "hosted",
     "Hosted frontier anchor, calibrating the ceiling on the same 1000 cases."),
    ("gpt5-clause", "runs/gpt5-1000.safety.out.json", "gpt-5-safety-full",
     "gpt-5 + safety clause", "hosted",
     "The control that corrected a wrong claim. The fine-tune's refusal advantage "
     "over gpt-5 was really the clause's: given the same system turn, gpt-5 scores "
     "47/49 on refusals \u2014 exactly the fine-tune's number."),
    ("v5", "runs/final-3way.out.json", "gemma4-e4b-ft-v5",
     "Gemma-4 E4B wallet-ft v5", "local",
     "The published fine-tune. One epoch on 2288 rows, LoRA merged at 0.75 "
     "strength, ~18 minutes on one rented A40. Trained WITHOUT the safety clause."),
    ("v5-clause", "runs/v5-safety-ab.out.json", "v5-safety-full",
     "Gemma-4 E4B wallet-ft v5 + safety clause", "local",
     "Best configuration measured on both halves at once. The clause is 1577 "
     "characters this adapter never saw in training, and the prior worry was that "
     "off-distribution wording would break it. It did not."),
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

    # The axis under the strips merges ablation + refusal: together they are one
    # idea (the region where emitting no call is the correct answer) and each is
    # individually too narrow a slice of 1000 cases to carry a legible label
    # without clipping. Note this UNDER-counts the no-call region: the 32
    # `conversation-exact_output` cases and the 11 `arithmetic-ablation` ones are
    # also gold-no-call but stay banded with their own mechanism, so the honest
    # total is reported separately as `no_call_count`.
    axis_bands = []
    for band in bands:
        if band["label"] in ("ablation", "refusal"):
            continue
        axis_bands.append(dict(band))
    merged = sum(b["count"] for b in bands if b["label"] in ("ablation", "refusal"))
    axis_bands.append({"label": "no call", "count": merged})

    no_call_count = sum(1 for c in cases if not c["gold"])

    payload = {
        "dataset": DATASET,
        "case_count": len(cases),
        "no_call_count": no_call_count,
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
