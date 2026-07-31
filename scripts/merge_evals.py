#!/usr/bin/env python
"""Merge several single-provider promptfoo eval exports into ONE eval.

Why: promptfoo's UI shows one eval at a time, and a provider becomes a *column*
only if it lives in the same eval. We deliberately ran gpt-5 (network-bound,
-j 8) and the local GGUF (CPU-bound, -j 1) as separate evals because sharing one
serialized run made each wait on the other. This stitches them back together so
the browser shows a real side-by-side.

Only valid when every input ran the SAME prompt over the SAME dataset — asserted
below on promptId and on the test-case id sets. Rows are keyed by test-case id
(not by position), so a provider missing a case leaves that cell blank rather
than silently shifting every row beneath it.

    uv run python scripts/merge_evals.py -o merged.out.json \
        gpt5.fresh.out.json gemma4ft.fresh.out.json
    npx promptfoo import merged.out.json --new-id

Nothing is synthesized: a cell is either a real recorded result or absent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="single-provider *.out.json exports")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--dataset", default="pf/tests.generated.yaml",
                    help="source of canonical row order")
    ap.add_argument("--description")
    args = ap.parse_args()

    # Canonical row order comes from the dataset, so rows line up across providers
    # even when one provider is missing a case.
    cases = yaml.safe_load(Path(args.dataset).read_text())
    order = {t["metadata"]["id"]: i for i, t in enumerate(cases)}

    docs = [json.loads(Path(p).read_text()) for p in args.inputs]

    # Guard: same prompt, or the comparison is meaningless.
    pids = {d["results"]["prompts"][0].get("id") for d in docs}
    if len(pids) != 1:
        raise SystemExit(f"refusing to merge: differing promptIds {pids}")

    merged_prompts, merged_results = [], []
    for idx, (path, doc) in enumerate(zip(args.inputs, docs)):
        res = doc["results"]
        if len(res["prompts"]) != 1:
            raise SystemExit(f"{path}: expected 1 provider, got {len(res['prompts'])}")
        merged_prompts.append(res["prompts"][0])
        for r in res["results"]:
            cid = r["testCase"]["metadata"]["id"]
            if cid not in order:
                raise SystemExit(f"{path}: case {cid} not in {args.dataset}")
            r = dict(r)
            r["promptIdx"] = idx          # provider -> column
            r["testIdx"] = order[cid]     # case id -> row
            merged_results.append(r)

    merged_results.sort(key=lambda r: (r["testIdx"], r["promptIdx"]))

    base = docs[0]
    stats = {"successes": 0, "failures": 0, "errors": 0,
             "tokenUsage": {"prompt": 0, "completion": 0, "total": 0, "cached": 0}}
    for d in docs:
        s = d["results"].get("stats") or {}
        for k in ("successes", "failures", "errors"):
            stats[k] += s.get(k) or 0
        for k in stats["tokenUsage"]:
            stats["tokenUsage"][k] += (s.get("tokenUsage") or {}).get(k) or 0

    # Config lists every provider so the UI labels columns correctly.
    config = dict(base.get("config") or {})
    providers = []
    for d in docs:
        p = (d.get("config") or {}).get("providers") or []
        providers.extend(p if isinstance(p, list) else [p])
    config["providers"] = providers
    if args.description:
        config["description"] = args.description

    out = {
        "evalId": None,
        "config": config,
        "results": {
            "version": base["results"].get("version"),
            "timestamp": base["results"].get("timestamp"),
            "prompts": merged_prompts,
            "results": merged_results,
            "stats": stats,
        },
    }
    Path(args.out).write_text(json.dumps(out))

    print(f"merged {len(docs)} evals -> {args.out}")
    for i, (p, path) in enumerate(zip(merged_prompts, args.inputs)):
        n = sum(1 for r in merged_results if r["promptIdx"] == i)
        ok = sum(1 for r in merged_results if r["promptIdx"] == i and r.get("success"))
        prov = p.get("provider")
        print(f"  col {i}: {prov:34s} {ok:3d}/{n:<3d} pass   ({Path(path).name})")
    print(f"  rows: {len({r['testIdx'] for r in merged_results})} of {len(cases)} dataset cases")


if __name__ == "__main__":
    main()
