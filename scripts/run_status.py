"""One-line-per-model progress for the four benchmark runs.

Reads promptfoo's own SQLite DBs, not just the chunk exports, so a chunk that is
half-finished still shows its case count. An export-only view jumps 0 -> 250 and
looks stalled for 30 minutes at a time.

    uv run python scripts/run_status.py            # human
    uv run python scripts/run_status.py --oneline  # compact, for a monitor event
"""
from __future__ import annotations

import argparse
import glob
import os
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ["gemma4-e4b-base", "gemma4-e4b-ft-v4-appprompt",
          "qwen3-8b-base", "qwen3-8b-ft-v4"]
TOTAL = 1000


def scored(label: str) -> tuple[int, int]:
    """(cases scored, passed) across every chunk DB for this model."""
    n = p = 0
    for db in sorted(glob.glob(str(ROOT / f".promptfoo-{label}-part*/promptfoo.db"))):
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            a, b = con.execute(
                "select count(*), sum(success) from eval_results").fetchone()
            n += a or 0
            p += b or 0
        except sqlite3.Error:
            continue  # a DB being written mid-transaction is not an error here
    return n, p


def exports(label: str) -> int:
    return len(glob.glob(str(ROOT / f"runs/{label}.part*.out.json")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oneline", action="store_true")
    args = ap.parse_args()

    lock = ROOT / "runs" / ".eval.lock"
    pid = lock.read_text().strip() if lock.exists() else ""
    alive = False
    if pid:
        try:
            os.kill(int(pid), 0)
            alive = True
        except (OSError, ValueError):
            alive = False

    rows = []
    for label in MODELS:
        n, p = scored(label)
        rows.append((label, exports(label), n, p))

    if args.oneline:
        parts = []
        for label, ex, n, p in rows:
            short = (label.replace("gemma4-e4b-", "G-").replace("qwen3-8b-", "Q-")
                     .replace("ft-v4-appprompt", "ft").replace("-appprompt", ""))
            parts.append(f"{short} {n}/{TOTAL}"
                         + (f" {p/n:.0%}" if n else "")
                         + (" DONE" if ex == 4 else ""))
        state = f"driver pid {pid} alive" if alive else "DRIVER NOT RUNNING"
        print(f"[{time.strftime('%H:%MZ', time.gmtime())}] "
              + " | ".join(parts) + f" || {state}")
        return

    print(f"{'model':<30}{'chunks':>8}{'scored':>9}{'passed':>9}{'rate':>8}")
    print("-" * 64)
    for label, ex, n, p in rows:
        rate = f"{p/n:.1%}" if n else "—"
        print(f"{label:<30}{ex:>5}/4{n:>7}/{TOTAL}{p:>9}{rate:>8}")
    print("-" * 64)
    print(f"driver: pid {pid or 'none'} {'alive' if alive else 'NOT RUNNING'}")
    sys.exit(0)


if __name__ == "__main__":
    main()
