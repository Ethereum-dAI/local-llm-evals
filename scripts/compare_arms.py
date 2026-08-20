#!/usr/bin/env python3
"""Per-case A/B between two single-provider exports.

    uv run python scripts/compare_arms.py runs/gpt5-1000.out.json \
        runs/gpt5-1000.safety.out.json --label-a gpt5 --label-b gpt5+clause

Reproduces the tables in results/gpt5-safety-clause.1000.md. For two arms inside ONE
export (the usual local A/B shape) use scripts/report_1000.py, which splits providers;
this is for arms that had to run as separate processes, which is the only way to A/B a
prompt against a HOSTED provider (see $PROMPT_VARIANT in CLAUDE.md).

Joins on metadata.id — never on row order, since promptfoo's export order is not the
dataset's and a positional join would silently misalign the arms.

Reports the SD of a difference as sqrt(flips), the noise floor this repo uses, so a
net move that is not resolved cannot be read as a result.
"""
import argparse, collections, json, sys
from pathlib import Path

def _is_scorer_reason(e):
    return e is None or str(e).startswith(("call count:", "call#"))


def load(p):
    d = json.load(open(p))
    out = {}
    for r in d["results"]["results"]:
        md = r["testCase"]["metadata"]
        out[md["id"]] = {"ok": bool(r["success"]), "cat": md["category"],
                         "gold": md.get("expected_calls"),
                         # `error` carries the SCORER's verdict for a normal failing
                         # case ("call count: expected 1 ['swap'], model made 0"), so
                         # counting it as a provider error flags every failure. Only a
                         # message that is not a scorer verdict is a real transport
                         # failure — the thing that must be zero for a run to count.
                         "err": None if _is_scorer_reason(r.get("error")) else r.get("error")}
    return out

def slice_of(ids, arm):
    return sum(arm[i]["ok"] for i in ids), len(ids)

ap = argparse.ArgumentParser()
ap.add_argument("control"); ap.add_argument("treatment")
ap.add_argument("--label-a", default="control"); ap.add_argument("--label-b", default="treatment")
a = ap.parse_args()

A, B = load(a.control), load(a.treatment)
print(f"{a.label_a}: {len(A)} cases   {a.label_b}: {len(B)} cases")
for nm, arm in ((a.label_a, A), (a.label_b, B)):
    errs = [i for i, v in arm.items() if v["err"]]
    if len(arm) != 1000: print(f"  !! {nm} has {len(arm)} cases, NOT 1000")
    if errs: print(f"  !! {nm} has {len(errs)} provider errors")

ids = sorted(set(A) & set(B))
if len(ids) != 1000: print(f"  !! only {len(ids)} shared ids")

safety = [i for i in ids if A[i]["cat"].startswith("safety-refusal")]
nocall = [i for i in ids if not A[i]["gold"]]
call   = [i for i in ids if A[i]["gold"]]

print(f"\n{'slice':<22}{a.label_a:>20}{a.label_b:>20}")
for nm, s in (("overall", ids), ("wants a call", call), ("wants NO call", nocall), ("safety-refusal", safety)):
    pa, na = slice_of(s, A); pb, nb = slice_of(s, B)
    print(f"{nm:<22}{pa:>6}/{na:<5}{100*pa/na:>6.1f}%{pb:>6}/{nb:<5}{100*pb/nb:>6.1f}%")

fixed  = [i for i in ids if not A[i]["ok"] and B[i]["ok"]]
broken = [i for i in ids if A[i]["ok"] and not B[i]["ok"]]
print(f"\nflips: {len(fixed)} fixed, {len(broken)} broken, net {len(fixed)-len(broken):+d}"
      f"  (SD of a difference ~ sqrt({len(fixed)+len(broken)}) = {(len(fixed)+len(broken))**.5:.1f} cases)")

print("\nrefusal kind                          ctl    trt")
kinds = collections.defaultdict(list)
for i in safety: kinds[A[i]["cat"]].append(i)
for k in sorted(kinds):
    pa, n = slice_of(kinds[k], A); pb, _ = slice_of(kinds[k], B)
    mark = "  <<" if pb > pa else ("  >> REGRESSED" if pb < pa else "")
    print(f"{k.replace('safety-refusal-',''):<38}{pa}/{n}   {pb}/{n}{mark}")

print("\nbiggest movers by category (non-safety)")
mv = collections.defaultdict(lambda: [0, 0, 0])
for i in ids:
    if A[i]["cat"].startswith("safety-refusal"): continue
    c = A[i]["cat"]; mv[c][2] += 1
    if not A[i]["ok"] and B[i]["ok"]: mv[c][0] += 1
    if A[i]["ok"] and not B[i]["ok"]: mv[c][1] += 1
for c, (f, b, n) in sorted(mv.items(), key=lambda kv: -(abs(kv[1][0]-kv[1][1]))):
    if f or b:
        print(f"{c:<34} n={n:<4} fixed {f:<3} broke {b:<3} net {f-b:+d}")
