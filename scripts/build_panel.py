"""The 50-case discrimination panel: pf/tests.panel.yaml.

A small, cheap set whose every case is already known to split the models. Two modes:

  uv run python scripts/build_panel.py --select runs/pool-base.out.json \
      runs/pool-v5.out.json runs/pool-gemini.out.json
      Choose the cases from recorded runs over the candidate pool and write
      datasets/panel_ids.json (ids + every model's recorded verdict).

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


def _load_verdicts(paths: list[Path]) -> dict[str, dict[str, bool]]:
    out: dict[str, dict[str, bool]] = {}
    for model, path in zip(MODELS, paths):
        rows = json.loads(path.read_text())["results"]["results"]
        out[model] = {r["testCase"]["metadata"]["id"]: bool(r["success"]) for r in rows}
    return out


def select(verdicts: dict[str, dict[str, bool]], cases: dict[str, dict],
           size: int = SIZE, seed: int = SEED) -> list[str]:
    """Proportional-by-pattern, round-robin-by-mechanism sample of split cases."""
    ids = sorted(set.intersection(*(set(v) for v in verdicts.values())))
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--select", nargs=3, type=Path, metavar=("BASE", "V5", "GEMINI"),
                        help="recorded pool runs to choose the panel from")
    args = parser.parse_args()
    cases = _all_cases()
    if args.select:
        verdicts = _load_verdicts(args.select)
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
    no_call = sum(1 for c in panel if not c["metadata"]["expected_calls"])
    groups = collections.Counter(_group(c["metadata"]) for c in panel)
    print(f"Wrote {len(panel)} cases -> {OUT.relative_to(ROOT)}; no-call {no_call}")
    print("mechanisms:", dict(groups.most_common()))


if __name__ == "__main__":
    main()
