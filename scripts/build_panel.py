"""The 50-case discrimination panel: pf/tests.panel.yaml.

A small, cheap set whose every case is already known to split the models. Two modes:

  uv run python scripts/build_panel.py --base runs/pool-base.out.json \
      runs/hard2-base.out.json --v5 ... --gemini ...
      Choose the cases from recorded runs over the candidate pool and write
      datasets/panel_ids.json (ids + every model's recorded verdict). Several
      exports per model are merged by case id, later files winning -- so a
      re-run of one part of the pool replaces just those verdicts.

  uv run python scripts/build_panel.py
      Materialise pf/tests.panel.yaml from datasets/panel_ids.json, copying each
      case byte-for-byte out of pf/tests.hard.yaml or pf/tests.combined.yaml.
      This is the byte-stable step the tests check; it needs no run outputs.

How cases are chosen, and what that costs:

  The candidate pool was the 187 hard cases plus the 160 cases of the 1000 where
  base, v5 and gpt-5 disagreed. base Gemma-4 E4B, the v5 fine-tune and Gemini 3.1
  Pro (the frontier stand-in: gpt-5 is blocked on this OpenRouter key and the OpenAI
  key has no credits) each ran the whole pool once. A case is a CANDIDATE only if the
  three disagree. Candidates are sampled IN PROPORTION to how often each pass/fail
  pattern occurs, and round-robin across mechanisms inside a pattern, so the panel
  does not overweight any one model's wins or any one mechanism.

  This is selection on the outcome, on purpose: the panel is built to separate these
  three models, so its separation is partly built in, and single-run verdicts include
  noise flips. `stability` in panel_ids.json records a second, independent run of
  the chosen 50 -- read that, not the selection run, for how well it separates.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

import yaml

from wallet_evals.wallet_executable import case_is_executable

ROOT = Path(__file__).resolve().parent.parent
IDS = ROOT / "datasets" / "panel_ids.json"
OUT = ROOT / "pf" / "tests.panel.yaml"
SOURCES = (ROOT / "pf" / "tests.hard.yaml", ROOT / "pf" / "tests.combined.yaml")
MODELS = ("base", "v5", "gemini")
SIZE = 50
SEED = 20260926


def _group(md: dict) -> str:
    """Coarse mechanism for round-robin diversity."""
    if md.get("mechanism"):
        return md["mechanism"]
    return md["category"].split("-")[0]


def _load_verdicts(paths: dict[str, list[Path]]) -> dict[str, dict[str, bool]]:
    out: dict[str, dict[str, bool]] = {}
    for model in MODELS:
        out[model] = {}
        for path in paths[model]:
            rows = json.loads(path.read_text())["results"]["results"]
            errors = [r for r in rows if r.get("failureReason") == 2]
            assert not errors, f"{path}: {len(errors)} provider errors -- rerun it"
            out[model].update({r["testCase"]["metadata"]["id"]: bool(r["success"])
                               for r in rows})
    return out


def select(verdicts: dict[str, dict[str, bool]], cases: dict[str, dict],
           size: int = SIZE, seed: int = SEED) -> list[str]:
    """Proportional-by-pattern, round-robin-by-mechanism sample of split cases."""
    # Only cases present in the current sources (a regenerated case id with stale
    # verdicts would otherwise sneak in) and whose gold the wallet would execute.
    ids = sorted(i for i in set.intersection(*(set(v) for v in verdicts.values()))
                 if i in cases and case_is_executable(cases[i]))
    pattern = {i: "".join("1" if verdicts[m][i] else "0" for m in MODELS) for i in ids}
    split = [i for i in ids if pattern[i] not in ("000", "111")]
    by_pat: dict[str, list[str]] = collections.defaultdict(list)
    for i in split:
        by_pat[pattern[i]].append(i)
    # Largest-remainder apportionment of `size` across patterns.
    raw = {p: size * len(v) / len(split) for p, v in by_pat.items()}
    quota = {p: int(r) for p, r in raw.items()}
    for p in sorted(raw, key=lambda p: (raw[p] - quota[p], p), reverse=True)[
            : size - sum(quota.values())]:
        quota[p] += 1
    rng = random.Random(seed)
    chosen: list[str] = []
    for p in sorted(by_pat):
        groups: dict[str, list[str]] = collections.defaultdict(list)
        for i in by_pat[p]:
            groups[_group(cases[i]["metadata"])].append(i)
        for g in groups.values():
            rng.shuffle(g)
        order = sorted(groups)
        rng.shuffle(order)
        picked: list[str] = []
        while len(picked) < quota[p]:
            for g in order:
                if groups[g] and len(picked) < quota[p]:
                    picked.append(groups[g].pop())
        chosen += picked
    return sorted(chosen)


def _all_cases() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for src in SOURCES:
        for case in yaml.safe_load(src.read_text()):
            out.setdefault(case["metadata"]["id"], case)
    return out


def materialise(ids: list[str], cases: dict[str, dict]) -> list[dict]:
    missing = [i for i in ids if i not in cases]
    assert not missing, f"panel ids missing from the sources: {missing}"
    return [cases[i] for i in ids]


#: Human-readable category per hard mechanism/kind, and how far the frozen 1000
#: already covers it. A panel case drawn from the 1000 IS in the 1000.
HARD_CATEGORIES: dict[str, tuple[str, str]] = {
    "stacked": ("Stacked corrections + distractors, 7-9 rounds",
                "Partly: the 1000 has corrections and distractors separately, up to 6 rounds"),
    "injection_distractor": ("Pasted injection mid-conversation",
                             "No: the 1000's interruptions never try to redirect the request"),
    "unresolvable_recipient": ("Unresolvable recipient -> no call",
                               "No: no case gives a truncated or remembered recipient"),
    "unresolvable_amount": ("Fiat or fraction amount -> no call",
                            "Partly: exact-output swaps are the 1000's only no-call amount case"),
    "surface": ("Hard number form, given as a correction",
                "Partly: numeric corrections and comma-grouped amounts, never words, 2.4k or speech"),
    "surface-lang": ("Other language with a decimal comma",
                     "No: the 1000 is English-only"),
    "embedded_refusal": ("Dangerous answer under pressure -> no call",
                         "Partly: the same refusal kinds, but asked directly in one turn"),
}


def _category(md: dict) -> tuple[str, str, str]:
    """(category, mechanism, coverage in the 1000)."""
    cat = md["category"]
    if md["id"].startswith("hard-"):
        mech = md["mechanism"]
        key = "surface-lang" if mech == "surface" and md.get("language") != "english" else mech
        label, coverage = HARD_CATEGORIES[key]
        return label, mech + (f"/{md['kind']}" if md.get("kind") else ""), coverage
    covered = "Yes: this exact case is in the 1000"
    if cat.startswith("safety-refusal-"):
        return f"Safety refusal: {cat.removeprefix('safety-refusal-')}", "safety", covered
    if cat.startswith("conversation-"):
        mech = md["mechanism"]
        return (f"{mech.replace('_', ' ').capitalize()} conversation, {md['rounds']} rounds",
                mech, covered)
    if cat.startswith(("multiturn-", "arithmetic-multiturn-")):
        return f"2-round clarification ({cat.rsplit('-', 1)[-1]})", "multiturn", covered
    if cat.startswith(("ablation-", "arithmetic-ablation-")):
        return f"Missing field -> ask ({cat.rsplit('-', 1)[-1]})", "ablation", covered
    if cat.startswith("arithmetic-"):
        return "Amount handling (arithmetic slice)", "arithmetic", covered
    return f"Single-turn {cat.split('-')[1]}", "single-turn", covered


def write_csv(path: Path, panel: list[dict], data: dict) -> None:
    """One row per panel case: category, coverage in the 1000, gold, verdicts."""
    import csv
    sel, stab = data["selection_run"], data.get("stability_run", {})
    with path.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["id", "category", "mechanism", "source", "in_1000", "coverage_in_1000",
                    "rounds", "expected", "gold", "last_user_turn",
                    *(f"{m}_selection" for m in MODELS), *(f"{m}_rerun" for m in MODELS),
                    "gold_executes_in_wallet"])
        for case in panel:
            md = case["metadata"]
            label, mech, coverage = _category(md)
            msgs = case["vars"].get("messages")
            last = msgs[-1]["content"] if msgs else case["vars"]["user_message"]
            hard = md["id"].startswith("hard-")
            w.writerow([md["id"], label, mech,
                        "hard slice (new)" if hard else "frozen 1000",
                        "no" if hard else "yes", coverage, sum(1 for m in msgs if m["role"] == "user") if msgs else 1,
                        "call" if md["expected_calls"] else "no call",
                        case["vars"]["expected_summary"], last,
                        *("pass" if sel[md["id"]][m] else "fail" for m in MODELS),
                        *(("pass" if stab[md["id"]][m] else "fail") if stab else ""
                          for m in MODELS),
                        "yes" if md["expected_calls"] else "n/a (no call)"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for model in MODELS:
        parser.add_argument(f"--{model}", nargs="+", type=Path, default=[],
                            help=f"recorded pool run(s) for {model}")
    parser.add_argument("--csv", type=Path, default=ROOT / "results" / "panel50.csv",
                        help="where to write the per-case CSV")
    args = parser.parse_args()
    cases = _all_cases()
    if args.base or args.v5 or args.gemini:
        verdicts = _load_verdicts({m: getattr(args, m) for m in MODELS})
        ids = select(verdicts, cases)
        IDS.write_text(json.dumps({
            "models": {"base": "Gemma-4 E4B Q4_K_M (local)",
                       "v5": "gemma-4-E4B-wallet-ft-v5 a075 Q4_K_M (local)",
                       "gemini": "google/gemini-3.1-pro-preview via OpenRouter"},
            "selection_run": {i: {m: verdicts[m][i] for m in MODELS} for i in ids},
            "ids": ids,
        }, indent=1) + "\n")
        print(f"selected {len(ids)} -> {IDS.relative_to(ROOT)}")
    ids = json.loads(IDS.read_text())["ids"]
    panel = materialise(ids, cases)
    header = ("# Generated 50-case discrimination panel -- DO NOT EDIT BY HAND.\n"
              "# Produced by scripts/build_panel.py from datasets/panel_ids.json; every "
              "case is copied\n# byte-for-byte from pf/tests.hard.yaml or "
              "pf/tests.combined.yaml.\n")
    OUT.write_text(header + yaml.safe_dump(panel, sort_keys=False, allow_unicode=True))
    write_csv(args.csv, panel, json.loads(IDS.read_text()))
    no_call = sum(1 for c in panel if not c["metadata"]["expected_calls"])
    groups = collections.Counter(_group(c["metadata"]) for c in panel)
    print(f"Wrote {len(panel)} cases -> {OUT.relative_to(ROOT)}; no-call {no_call}")
    print("mechanisms:", dict(groups.most_common()))


if __name__ == "__main__":
    main()
