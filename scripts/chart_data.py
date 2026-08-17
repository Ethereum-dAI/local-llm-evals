#!/usr/bin/env python
"""Extract chart-ready aggregates from the relaunch exports.

    uv run python scripts/chart_data.py -o chartdata.json

Emits overall + per-category pass rates with Wilson score intervals, and a
failure-mode breakdown parsed from the scorer's own reason strings.

Wilson rather than the textbook normal interval: these are proportions from small
n (several categories have n < 5, and a refusal cell can be n = 1), where the
normal approximation produces intervals that run past 0 or 100% and collapse to
zero width at p = 0 or 1 — exactly the cells where honesty about uncertainty
matters most. Wilson stays inside [0,1] and stays wide at the extremes.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

Z = 1.96  # 95%


def wilson(k: int, n: int) -> tuple[float, float, float]:
    """(point, lo, hi) as proportions. n == 0 -> all zeros."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / d
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def failure_mode(reason: str, expected_empty: bool) -> str:
    """Bucket a scorer reason into one actionable failure mode."""
    if not reason:
        return "other"
    if reason.startswith("call count:"):
        if expected_empty:
            return "acted when it should refuse"
        return "no tool call"
    if "tool: expected" in reason:
        return "wrong tool"
    # Argument mismatches: name the first offending field.
    m = re.search(r"call#\d+ \([^)]*\): ([A-Za-z]+):", reason)
    if m:
        field = m.group(1)
        if field in ("amountIn", "value"):
            return "wrong amount"
        if field == "args":
            return "wrong args"
        if field in ("to", "recipient"):
            return "wrong recipient"
        if field in ("currencyIn", "currencyOut"):
            return "wrong token"
        return f"wrong {field}"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    runs = {
        "gemma4-e4b-base": "relaunch/gemma4-base.final.json",
        "gemma4-e4b-ft": "relaunch/gemma4-ft.final.json",
        "gpt-5": "relaunch/gpt5.final.json",
    }

    per_cat: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    modes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    overall: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for model, path in runs.items():
        doc = json.loads(Path(path).read_text())
        for row in doc["results"]["results"]:
            if row.get("failureReason") == 2:
                continue
            md = row["testCase"]["metadata"]
            cat = md.get("category", "?")
            ok = bool(row.get("success"))
            per_cat[cat][model][1] += 1
            overall[model][1] += 1
            if ok:
                per_cat[cat][model][0] += 1
                overall[model][0] += 1
            else:
                reason = (row.get("gradingResult") or {}).get("reason", "")
                modes[model][failure_mode(reason, not md.get("expected_calls"))] += 1

    out = {
        "models": list(runs),
        "overall": {m: dict(zip(("k", "n", "p", "lo", "hi"),
                                (v[0], v[1], *wilson(v[0], v[1]))))
                    for m, v in overall.items()},
        "categories": [
            {"category": cat,
             "n": max(per_cat[cat][m][1] for m in runs),
             "models": {m: dict(zip(("k", "n", "p", "lo", "hi"),
                                    (per_cat[cat][m][0], per_cat[cat][m][1],
                                     *wilson(*per_cat[cat][m]))))
                        for m in runs}}
            for cat in sorted(per_cat)
        ],
        "failure_modes": {m: dict(sorted(d.items(), key=lambda kv: -kv[1]))
                          for m, d in modes.items()},
    }
    Path(args.out).write_text(json.dumps(out, indent=1))

    print("OVERALL")
    for m, v in out["overall"].items():
        print(f"  {m:18s} {v['k']:3d}/{v['n']:3d} = {v['p']*100:5.1f}%  "
              f"[{v['lo']*100:.1f}, {v['hi']*100:.1f}]")
    print("\nFAILURE MODES")
    for m, d in out["failure_modes"].items():
        print(f"  {m}: " + ", ".join(f"{k}={v}" for k, v in list(d.items())[:6]))
    print(f"\nwrote {args.out}  ({len(out['categories'])} categories)")


if __name__ == "__main__":
    main()
