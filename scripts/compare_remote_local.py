"""Compare a RunPod-served run against the local run of the same slice.

Two questions, and they are NOT the same question:

  1. Is it faster?          — wall clock and per-case latency.
  2. Is it the SAME eval?   — per-case verdicts, arm by arm.

(2) is the one that matters. Moving generation to a rented GPU is only legitimate if it
does not change what is measured, and this repo already keeps a backend control
(promptfooconfig.phi4-mini.modal.yaml) for exactly that reason. A remote run that is
fast and disagrees is worthless; report the disagreement rather than the speedup.

    uv run python scripts/compare_remote_local.py runs/remote.out.json runs/local.out.json
"""
from __future__ import annotations

import collections
import json
import statistics
import sys


def load(path: str) -> tuple[dict, dict]:
    """-> {(arm, case_id): passed}, {arm: [latency_s]}"""
    data = json.load(open(path))
    results = data["results"]["results"]
    verdicts: dict[tuple[str, str], bool] = {}
    latency: dict[str, list[float]] = collections.defaultdict(list)
    for r in results:
        arm = r.get("provider", {}).get("label") or r["provider"]["id"]
        md = r["testCase"].get("metadata") or {}
        case = md.get("id") or r["testCase"].get("description") or "?"
        verdicts[(arm, case)] = bool(r.get("success"))
        if r.get("latencyMs"):
            latency[arm].append(r["latencyMs"] / 1000)
    return verdicts, latency


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    remote_path, local_path = sys.argv[1], sys.argv[2]
    remote, r_lat = load(remote_path)
    local, l_lat = load(local_path)

    shared = sorted(set(remote) & set(local))
    if not shared:
        raise SystemExit("no (arm, case) pairs in common — different slices or labels?")
    print(f"comparable (arm, case) pairs: {len(shared)}  "
          f"[remote {len(remote)}, local {len(local)}]")

    print("\n=== SPEED ===")
    for name, lat in (("remote", r_lat), ("local ", l_lat)):
        allv = [x for v in lat.values() for x in v]
        if not allv:
            continue
        print(f"  {name}: n={len(allv):4} median={statistics.median(allv):6.2f}s "
              f"mean={statistics.fmean(allv):6.2f}s  sum={sum(allv)/60:6.1f}min")
    ra = [x for v in r_lat.values() for x in v]
    la = [x for v in l_lat.values() for x in v]
    if ra and la:
        print(f"  per-case median speedup: "
              f"{statistics.median(la)/statistics.median(ra):.2f}x")
        print("  NOTE: per-case latency understates the real gain — the remote run also "
              "runs at -j 8,\n        so wall clock improves by concurrency ON TOP of "
              "this. Compare driver elapsed times\n        for the number that matters.")

    print("\n=== AGREEMENT (does the device change the measurement?) ===")
    by_arm: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    disagreements = []
    for key in shared:
        arm, case = key
        agree = remote[key] == local[key]
        by_arm[arm][0] += 1
        by_arm[arm][1] += agree
        if not agree:
            disagreements.append((arm, case, local[key], remote[key]))
    for arm, (n, ok) in sorted(by_arm.items()):
        r_pass = sum(remote[(a, c)] for a, c in shared if a == arm)
        l_pass = sum(local[(a, c)] for a, c in shared if a == arm)
        print(f"  {arm:22} agree {ok:3}/{n:3} ({100*ok/n:5.1f}%)   "
              f"local {l_pass:3}/{n:3} vs remote {r_pass:3}/{n:3}")
    if disagreements:
        print(f"\n  {len(disagreements)} disagreement(s) — local -> remote:")
        for arm, case, lv, rv in disagreements[:25]:
            print(f"    {arm:22} {case:28} {'PASS' if lv else 'FAIL'} -> "
                  f"{'PASS' if rv else 'FAIL'}")
        print("\n  Disagreements are NOT automatically a bug: temperature is 0.2, so a "
              "case near the\n  decision boundary can flip. What would be a bug is a "
              "SYSTEMATIC gap (one arm much\n  worse remotely), which points at the "
              "template or sampling rather than the device.")
    else:
        print("\n  perfect agreement on every shared pair — same eval, different device.")


if __name__ == "__main__":
    main()
