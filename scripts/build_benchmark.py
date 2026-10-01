"""Build pf/tests.benchmark.yaml: a ~500-case benchmark that still discriminates.

Two parts, concatenated in this order:

1. A STRATIFIED SUBSET of pf/tests.combined.yaml (the frozen 1000). Every case
   is byte-identical to its copy in the 1000, so every recorded run can be
   re-scored on it offline, per case id, with no new inference.
2. pf/tests.hard.yaml, the six hard mechanisms (scripts/generate_hard_cases.py).

How the subset is chosen, and what that does and does not bias:

  * QUOTAS fixes how many cases each stratum keeps. The strata where the seven
    recorded configurations were at or near ceiling (progressive disclosure,
    short `switch`, `exact_output`, single-turn swaps, ablation) keep a small
    regression-guard sample. The strata where they diverged (long `distractor`,
    `correction`, long `switch`, `token_address`, single-turn transfers, every
    safety refusal) keep most or all of their cases.
  * Only cases whose gold the WALLET would execute are eligible
    (wallet_executable.case_is_executable): the frozen 1000 carries 104 that it
    would not (unresolvable ENS, mainnet token addresses, unparseable amounts).
  * INSIDE a stratum, cases are drawn by a seeded RNG over the sorted ids, never
    by how any model scored on them. So the quotas used the recorded results at
    STRATUM level; no individual case was kept or dropped for its verdict.
    That is a milder selection effect than case-level filtering, but it is not
    zero: separation measured on this subset with the same seven configurations
    is partly built in. The hard slice has no such bias -- it was written before
    any model saw it -- and is the part to read for an unbiased separation.

Run: uv run python scripts/build_benchmark.py [--project]
`--project` re-scores the subset from space/static/data.json (the recorded
per-case verdicts of every configuration on the 1000) and prints the result.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

import yaml

from wallet_evals.wallet_executable import case_is_executable

ROOT = Path(__file__).resolve().parent.parent
COMBINED = ROOT / "pf" / "tests.combined.yaml"
HARD = ROOT / "pf" / "tests.hard.yaml"
OUT = ROOT / "pf" / "tests.benchmark.yaml"
DATA = ROOT / "space" / "static" / "data.json"
SEED = 20260925

#: stratum -> cases kept from the 1000. Comments give the stratum's size in the
#: 1000 and why it keeps what it keeps.
QUOTAS: dict[str, int] = {
    "single-transfer": 45,      # 101; mutated one-shot transfers still split models
    "single-swap": 15,          # 96; 95-99% for every current model
    "arithmetic-single": 20,    # 51; the slice all three models tie on (93.8%)
    "legacy-2round": 15,        # 93; the old fixed 2-round shape, mostly saturated
    "ablation-token": 4,        # 4; base 0/4, gpt-5 0/4, v5 4/4 -- all kept
    "ablation-other": 8,        # 35; 100% for every model since base
    "safety": 49,               # 49; all kept -- already 2-4 cases per kind
    "progressive": 12,          # 111; 94-100% everywhere
    "switch-short": 12,         # 106 (2-4 rounds); 96-100% everywhere
    "switch-long": 20,          # 43 (5-6 rounds); where v5's held-out regression is
    "correction-short": 30,     # 110 (2-4 rounds)
    "correction-long": 20,      # 44 (5-6 rounds)
    "distractor-short": 6,      # 32 (3 rounds)
    "distractor-long": 57,      # 69 (4-6 rounds); base 52-72%, the widest spread.
                                # 59 are executable; takes token_address's 12
    "exact_output": 8,          # 32; 97-100% for every model but ft-v4
    "token_address": 0,         # 24, NONE executable: every one names a MAINNET token
                                # address (datasets/lookup.json) and the app's
                                # registry is Sepolia-only -- see wallet_executable
}


def stratum(case: dict) -> str:
    """Map a case of the 1000 to its QUOTAS key. Total over the 1000 (tested)."""
    md = case["metadata"]
    cat = md["category"]
    if cat == "generated-transfer-pos":
        return "single-transfer"
    if cat == "generated-swap-pos":
        return "single-swap"
    if cat in ("arithmetic-transfer-pos", "arithmetic-swap-pos"):
        return "arithmetic-single"
    if cat.startswith(("multiturn-", "arithmetic-multiturn-")):
        return "legacy-2round"
    if cat in ("ablation-token", "arithmetic-ablation-token"):
        return "ablation-token"
    if cat.startswith(("ablation-", "arithmetic-ablation-")):
        return "ablation-other"
    if cat.startswith("safety-refusal-"):
        return "safety"
    mech = md.get("mechanism")
    if mech in ("progressive", "exact_output", "token_address"):
        return mech
    if mech in ("switch", "correction"):
        return f"{mech}-{'long' if md['rounds'] >= 5 else 'short'}"
    if mech == "distractor":
        return f"distractor-{'long' if md['rounds'] >= 4 else 'short'}"
    raise ValueError(f"{md['id']}: no stratum for category {cat!r}")


def select_subset(combined: list[dict], quotas: dict[str, int] = QUOTAS,
                  seed: int = SEED) -> list[dict]:
    """Seeded per-stratum sample, returned in the 1000's own order."""
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for case in combined:
        groups[stratum(case)].append(case)
    unknown = set(groups) - set(quotas)
    assert not unknown, f"strata without a quota: {sorted(unknown)}"
    rng = random.Random(seed)
    keep: set[str] = set()
    for name in sorted(quotas):
        # Only cases whose gold the wallet would execute: 104 of the 1000 name an
        # ENS the daemon cannot resolve, a mainnet token address, or an amount its
        # parser rejects (wallet_executable.why_not_executable).
        members = sorted((c for c in groups[name] if case_is_executable(c)),
                         key=lambda c: c["metadata"]["id"])
        if len(members) < quotas[name]:
            raise ValueError(f"{name}: {len(members)} cases < quota {quotas[name]}")
        keep.update(c["metadata"]["id"] for c in rng.sample(members, quotas[name]))
    return [c for c in combined if c["metadata"]["id"] in keep]


def build(combined: list[dict], hard: list[dict]) -> list[dict]:
    subset = select_subset(combined)
    out = subset + hard
    ids = [c["metadata"]["id"] for c in out]
    assert len(ids) == len(set(ids)), "id collision between the subset and the hard slice"
    return out


def _rounds(case: dict) -> int:
    msgs = case["vars"].get("messages")
    return sum(1 for m in msgs if m["role"] == "user") if msgs else 1


def print_census(cases: list[dict]) -> None:
    rounds = collections.Counter(_rounds(c) for c in cases)
    no_call = sum(1 for c in cases if not c["metadata"]["expected_calls"])
    print(f"{len(cases)} cases; no-call {no_call} ({no_call / len(cases):.1%})")
    print("rounds: " + ", ".join(f"{r}:{n}" for r, n in sorted(rounds.items())))


def project(subset: list[dict]) -> None:
    """Re-score the subset from the recorded per-case verdicts on the 1000."""
    data = json.loads(DATA.read_text())
    keys = [m["key"] for m in data["models"]]
    by_id = {c["id"]: c for c in data["cases"]}
    kept = {c["metadata"]["id"] for c in subset}
    no_call = {c["metadata"]["id"] for c in subset if not c["metadata"]["expected_calls"]}
    print(f"\nprojected on the {len(kept)}-case subset (recorded verdicts, no inference):")
    print(f"{'config':14s} {'full 1000':>10s} {'subset':>8s} {'wants call':>11s} "
          f"{'no call':>8s}")
    passed: dict[str, set[str]] = {}
    for k in keys:
        full = sum(c["results"][k]["pass"] for c in data["cases"])
        passed[k] = {i for i in kept if by_id[i]["results"][k]["pass"]}
        call = kept - no_call
        print(f"{k:14s} {full / 10:9.1f}% {100 * len(passed[k]) / len(kept):7.1f}% "
              f"{100 * len(passed[k] & call) / len(call):10.1f}% "
              f"{100 * len(passed[k] & no_call) / max(1, len(no_call)):7.1f}%")
    print("\npairwise on the subset (only-A / only-B / net, sigma ~ net/sqrt(flips)):")
    for a, b in (("v5", "base"), ("v5", "gpt5"), ("gpt5", "base"),
                 ("v5-clause", "gpt5-clause"), ("v5-clause", "base-clause")):
        oa, ob = len(passed[a] - passed[b]), len(passed[b] - passed[a])
        flips = oa + ob
        sigma = (oa - ob) / flips ** 0.5 if flips else 0.0
        print(f"  {a:>10s} vs {b:<12s} +{oa:<3d} -{ob:<3d} net {oa - ob:+d} "
              f"({sigma:+.1f} sigma)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", action="store_true",
                        help="re-score the subset from space/static/data.json")
    args = parser.parse_args()
    combined = yaml.safe_load(COMBINED.read_text())
    hard = yaml.safe_load(HARD.read_text())
    cases = build(combined, hard)
    subset = cases[: len(cases) - len(hard)]
    header = (
        "# Generated benchmark -- DO NOT EDIT BY HAND.\n"
        "# Produced by scripts/build_benchmark.py: a seeded, stratified subset of "
        f"{COMBINED.relative_to(ROOT)} ({len(subset)} cases, byte-identical to the 1000)\n"
        f"# followed by {HARD.relative_to(ROOT)} ({len(hard)} cases). Regenerate both "
        "sources first if either is stale.\n")
    OUT.write_text(header + yaml.safe_dump(cases, sort_keys=False, allow_unicode=True))
    print(f"subset {len(subset)} + hard {len(hard)}")
    print_census(cases)
    print(f"Wrote {len(cases)} cases -> {OUT}")
    if args.project:
        project(subset)


if __name__ == "__main__":
    main()
