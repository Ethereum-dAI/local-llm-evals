"""Print (and optionally export) the composition of a promptfoo tests file.

The review table for a dataset change: categories and counts, the round
distribution, and the slice fields the report tooling groups by. Reads the same
YAML promptfoo eats, so it describes what will actually be run.

    uv run python scripts/dataset_census.py                          # combined
    uv run python scripts/dataset_census.py pf/tests.conversations.yaml
    uv run python scripts/dataset_census.py --csv census.csv --cases-csv cases.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "pf" / "tests.combined.yaml"

#: Metadata fields worth a breakdown. `rounds`/`mechanism` only exist on the
#: conversation slice, so they are reported as "(n/a)" elsewhere.
SLICE_FIELDS = ("protocol", "query_type", "level", "difficulty", "style")


def load(path: Path) -> list[dict]:
    return yaml.safe_load(path.read_text()) or []


def rounds_of(case: dict) -> int:
    """User turns in this case's prompt — 1 for a single-turn case."""
    messages = case["vars"].get("messages")
    if not messages:
        return 1
    return sum(1 for m in messages if m.get("role") == "user")


def family_of(category: str) -> str:
    """The coarse family a category belongs to, for the summary table."""
    for prefix, family in (
        ("conversation-", "conversation (multi-round)"),
        ("safety-refusal-", "safety refusal"),
        ("arithmetic-", "arithmetic slice"),
        ("multiturn-", "multi-turn (2-round, legacy)"),
        ("ablation-", "ablation (single-turn, incomplete)"),
        ("generated-", "single-turn positive"),
    ):
        if category.startswith(prefix):
            return family
    return "other"


def _table(title: str, counter: collections.Counter, total: int,
           key_width: int = 44) -> None:
    print(f"\n{title}")
    print(f"{'':2}{'':<{key_width}} {'count':>6} {'share':>7}")
    for key, count in sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        print(f"{'':2}{str(key):<{key_width}} {count:>6} {count / total:>6.1%}")
    print(f"{'':2}{'TOTAL':<{key_width}} {total:>6} {1.0:>6.1%}")


def report(cases: list[dict], path: Path) -> None:
    total = len(cases)
    print(f"=== {path} — {total} cases ===")

    categories = collections.Counter(c["metadata"]["category"] for c in cases)
    families = collections.Counter(
        family_of(c["metadata"]["category"]) for c in cases)
    _table("category", categories, total)
    _table("family", families, total, key_width=30)

    rounds = collections.Counter(rounds_of(c) for c in cases)
    multi = total - rounds.get(1, 0)
    print("\nconversation rounds (1 round = one user turn + the assistant's reply;")
    print("only the model's reply to the LAST user turn is scored)")
    print(f"{'':2}{'rounds':>6} {'count':>6} {'share':>7}")
    for r in sorted(rounds):
        print(f"{'':2}{r:>6} {rounds[r]:>6} {rounds[r] / total:>6.1%}")
    print(f"{'':2}{'multi':>6} {multi:>6} {multi / total:>6.1%}  (2+ rounds)")

    mechanisms = collections.Counter(
        c["metadata"].get("mechanism") or "(single/legacy shape)" for c in cases)
    _table("conversation mechanism", mechanisms, total, key_width=30)

    for field in SLICE_FIELDS:
        counter = collections.Counter(
            str(c["metadata"].get(field)) for c in cases)
        _table(field, counter, total, key_width=30)

    calls = collections.Counter(
        (c["metadata"]["expected_calls"][0]["tool"]
         if c["metadata"]["expected_calls"] else "(no call — refusal/ablation)")
        for c in cases)
    _table("gold tool", calls, total, key_width=30)


def write_census_csv(cases: list[dict], out: Path) -> None:
    """One row per (family, category, rounds) bucket — the review table."""
    buckets = collections.Counter(
        (family_of(c["metadata"]["category"]), c["metadata"]["category"],
         rounds_of(c), c["metadata"].get("mechanism") or "")
        for c in cases)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "category", "rounds", "mechanism", "count"])
        for (family, category, rounds, mechanism), count in sorted(buckets.items()):
            w.writerow([family, category, rounds, mechanism, count])
    print(f"\nWrote census CSV ({len(buckets)} rows) -> {out}")


#: Joins the turns of a transcript when `single_line=True`. A visible glyph
#: rather than a real newline: Google Sheets imports a quoted newline correctly
#: but then renders a 6-round case as an 11-line-tall row, which makes the sheet
#: unscannable. "\n" stays the default for terminal/grep use.
TURN_SEPARATOR = "  ⏎  "


def write_cases_csv(cases: list[dict], out: Path, single_line: bool = False) -> None:
    """One row per case: the full transcript, the gold call and the slice fields.

    `single_line` keeps every cell on one line, for a spreadsheet import.
    """
    joiner = TURN_SEPARATOR if single_line else "\n"
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "family", "category", "mechanism", "rounds", "protocol",
                    "difficulty", "level", "query_type", "style", "mutators",
                    "expected_summary", "expected_calls", "notes",
                    "scored_user_turn", "transcript"])
        for c in cases:
            md, v = c["metadata"], c["vars"]
            messages = v.get("messages") or [
                {"role": "user", "content": v.get("user_message", "")}]
            transcript = joiner.join(f"{m['role']}: {m['content']}" for m in messages)
            # The turn the model is actually scored on — the one column you want
            # when eyeballing a failure, and buried at the end of a 6-round
            # transcript otherwise.
            scored = [m["content"] for m in messages if m["role"] == "user"][-1]
            w.writerow([
                md["id"], family_of(md["category"]), md["category"],
                md.get("mechanism") or "", rounds_of(c), md["protocol"],
                md["difficulty"], md["level"], md.get("query_type") or "",
                md.get("style") or "", "|".join(md.get("mutators") or []),
                v.get("expected_summary", ""),
                # JSON, not yaml.safe_dump: PyYAML wraps flow style at 80 columns,
                # which put a real newline inside this cell for the longer swap
                # calls and split 165 of the 1000 rows across two lines in the
                # exported CSV.
                json.dumps(md["expected_calls"], separators=(",", ":")),
                md.get("notes") or "", scored, transcript,
            ])
    print(f"Wrote per-case CSV ({len(cases)} rows"
          f"{', single-line cells' if single_line else ''}) -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT,
                        help=f"tests YAML (default: {DEFAULT.relative_to(ROOT)})")
    parser.add_argument("--csv", type=Path, default=None,
                        help="write the (family, category, rounds) census here")
    parser.add_argument("--cases-csv", type=Path, default=None,
                        help="write one row per case (full transcripts) here")
    parser.add_argument("--single-line", action="store_true",
                        help="keep every cell on one line (transcript turns joined "
                             "by a visible glyph) — for a spreadsheet import")
    parser.add_argument("--quiet", action="store_true",
                        help="skip the terminal report; just write the CSVs")
    args = parser.parse_args()

    cases = load(args.dataset)
    if not args.quiet:
        report(cases, args.dataset)
    if args.csv:
        write_census_csv(cases, args.csv)
    if args.cases_csv:
        write_cases_csv(cases, args.cases_csv, single_line=args.single_line)


if __name__ == "__main__":
    main()
