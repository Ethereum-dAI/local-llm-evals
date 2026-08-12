"""One table for every model ever scored on the 307-case dev set.

`report_relaunch.py` answers "how did these runs do"; this answers "where does
each model stand", which needs three things that report can't assume:

  * the CALL vs NO-CALL split. 35 of the 307 cases want no tool call, so a model
    that never acts still scores ~11% without ever getting one right — a headline
    alone cannot tell that apart from real capability.
  * the fine-tune pairing, so each gain is stated against its OWN base rather
    than against whatever else happens to be in the table.
  * how each model was served (hosted / local Q4_K_M / Modal GPU) and at what
    sampling, because those differ per model card and the reader must not have to
    take a same-conditions comparison on faith.

    uv run python scripts/compare_all_models.py            # markdown
    uv run python scripts/compare_all_models.py --plain    # aligned text
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (label, export path, how it was served, sampling, base model for gain pairing)
RUNS: list[tuple[str, str, str, str, str | None]] = [
    ("gpt-5", "relaunch/gpt5.final.json", "OpenRouter", "T=0.1", None),
    ("gemma-4-E4B ft", "relaunch/gemma4-ft.local.out.json", "local Q4_K_M", "T=0.2",
     "gemma-4-E4B stock"),
    ("qwen3-8b ft", "relaunch/qwen3-ft.verified.out.json", "local Q4_K_M", "T=0.6/0.95/20",
     "qwen3-8b"),
    ("qwen3-8b", "relaunch/qwen3-8b.out.json", "OpenRouter", "T=0.6/0.95/20", None),
    ("gemma-4-E4B stock", "relaunch/gemma4-base.final.json", "local Q4_K_M", "T=0.2",
     None),
]


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    rows = json.loads(path.read_text())["results"]["results"]
    call = [r for r in rows if r["testCase"]["metadata"].get("expected_calls")]
    nocall = [r for r in rows if not r["testCase"]["metadata"].get("expected_calls")]
    # failureReason 2 = the provider blew up; such a case was never scored and must
    # not be counted as either a pass or a fail.
    errors = sum(1 for r in rows if r.get("failureReason") == 2)
    return {
        "total": len(rows), "errors": errors,
        "passed": sum(1 for r in rows if r.get("success")),
        "call_n": len(call), "call_pass": sum(1 for r in call if r.get("success")),
        "nocall_n": len(nocall),
        "nocall_pass": sum(1 for r in nocall if r.get("success")),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", action="store_true", help="aligned text, not markdown")
    args = ap.parse_args()

    stats: dict[str, dict] = {}
    rows: list[tuple] = []
    for label, rel, served, sampling, base in RUNS:
        s = load(ROOT / rel)
        if s is None:
            rows.append((label, served, sampling, None, base))
            continue
        stats[label] = s
        rows.append((label, served, sampling, s, base))

    rows.sort(key=lambda r: (r[3]["passed"] / r[3]["total"]) if r[3] else -1,
              reverse=True)

    header = ["model", "served", "sampling", "overall", "needs a call", "no call wanted"]
    body: list[list[str]] = []
    for label, served, sampling, s, base in rows:
        if s is None:
            body.append([label, served, sampling, "not run yet", "—", "—"])
            continue
        note = f"  ({s['errors']} unscored)" if s["errors"] else ""
        flag = "" if s["total"] == 307 else f"  << only {s['total']} cases!"
        body.append([
            label, served, sampling,
            f"{s['passed']}/{s['total']} = {100*s['passed']/s['total']:.1f}%{note}{flag}",
            f"{s['call_pass']}/{s['call_n']} = {100*s['call_pass']/s['call_n']:.1f}%",
            f"{s['nocall_pass']}/{s['nocall_n']} = {100*s['nocall_pass']/s['nocall_n']:.0f}%",
        ])

    if args.plain:
        widths = [max(len(h), *(len(r[i]) for r in body))
                  for i, h in enumerate(header)]
        print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
        print("  ".join("-" * w for w in widths))
        for r in body:
            print("  ".join(c.ljust(w) for c, w in zip(r, widths)))
    else:
        print("| " + " | ".join(header) + " |")
        print("|" + "|".join([" --- "] * len(header)) + "|")
        for r in body:
            print("| " + " | ".join(r) + " |")

    print("\n### Fine-tune gains (same dataset rows, same eval, each vs its OWN base)\n")
    for label, _, _, s, base in rows:
        if not base or not s or base not in stats:
            continue
        b = stats[base]
        d = 100 * s["passed"] / s["total"] - 100 * b["passed"] / b["total"]
        dc = 100 * s["call_pass"] / s["call_n"] - 100 * b["call_pass"] / b["call_n"]
        print(f"  {base} -> {label}: {100*b['passed']/b['total']:.1f}% -> "
              f"{100*s['passed']/s['total']:.1f}%  ({d:+.1f} pp overall, "
              f"{dc:+.1f} pp on cases needing a call)")

    print("\n35 of the 307 cases want NO tool call (ablations + safety refusals), so a "
          "model that\nnever acts still scores ~11% overall — always read the "
          "'needs a call' column.")


if __name__ == "__main__":
    main()
