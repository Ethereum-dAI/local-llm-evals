"""Compare model runs over the 1000-case benchmark, sliced by the new axes.

`compare_all_models.py` answers "where does each model stand" on the 307-case dev
set. This answers the question the conversation slice was built to ask: does
accuracy hold up as a conversation gets longer, and which failure mode breaks
first. So every table here is cut by ROUNDS and by MECHANISM, not just by category.

Four things it refuses to paper over:

  * A truncated export. promptfoo writes an `-o` file even when a run dies
    mid-way (a crash under SQLITE_BUSY produced a plausible-looking 289-of-307
    export once). Any run whose case count is not the full dataset is flagged
    loudly and excluded from the headline.
  * Provider errors. `failureReason == 2` means the provider raised and the case
    was never scored; counting it as a failure would blame the model for a
    timeout. Reported separately, excluded from the denominator.
  * The call / no-call split. 88 of the 1000 cases want NO tool call (refusals
    and single-turn ablations), so a model that never acts still scores 8.8%
    without getting a single call right.
  * Per-case pairing. The interesting number is not two headlines but the
    cases where the models DISAGREE, which is where a fine-tune's effect lives.

    uv run python scripts/report_1000.py runs/base1000.out.json runs/v41000.out.json
    uv run python scripts/report_1000.py runs/*.out.json --json runs/report.json
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_SIZE = 1000

#: Case ids that are BYTE-IDENTICAL (vars + gold) to the published v4 429-case set
#: on the Hub. Written by the comparability check; absent is fine.
#:
#: This is the ONLY subset on which a per-case comparison against the published
#: numbers (base 89.7% / gemma-ft-v4 96.5% / qwen-ft-v4 94.4%) is honest. It comes
#: to 349 cases — the frozen 307 plus the 42 extra refusals. The arithmetic slice
#: is excluded because it was regenerated to fix a constant recipient, which both
#: changed 12 shared cases and reshuffled which 40-per-action were drawn.
#: Lives in datasets/ rather than runs/, which is gitignored — a force-added file
#: under an ignored directory reads as disposable, and .gitignore says exactly
#: that about runs/.
_COMPARABLE = ROOT / "datasets" / "comparable_case_ids.json"


def comparable_ids() -> set[str]:
    if not _COMPARABLE.exists():
        return set()
    return set(json.loads(_COMPARABLE.read_text()).get("identical", []))


def _rounds(md: dict, vars_: dict) -> int:
    """User turns in the prompt. Prefer the metadata the generator declared; fall
    back to counting, so app-contract cases (which carry no `rounds`) still slot in."""
    if md.get("rounds"):
        return int(md["rounds"])
    messages = (vars_ or {}).get("messages")
    if not messages:
        return 1
    return sum(1 for m in messages if m.get("role") == "user")


def family_of(category: str) -> str:
    for prefix, family in (
        ("conversation-", "conversation (multi-round)"),
        ("safety-refusal-", "safety refusal"),
        ("arithmetic-", "arithmetic slice"),
        ("multiturn-", "multi-turn (2-round, legacy)"),
        ("ablation-", "ablation (single-turn)"),
        ("generated-", "single-turn positive"),
    ):
        if category.startswith(prefix):
            return family
    return "other"


def _provider_of(row: dict) -> str:
    """The arm a scored row belongs to: promptfoo's provider label, or its id."""
    prov = row.get("provider") or {}
    return prov.get("label") or prov.get("id") or "?"


def split_providers(label: str, paths: list[Path]) -> list["Run"]:
    """One Run per provider found across `paths`.

    A single-provider export keeps the caller's label, so existing invocations read
    exactly as before. A multi-provider export becomes one Run per arm, named by the
    provider label — which is the name the config gave it, and the only name that makes
    a three-way table legible.
    """
    seen: list[str] = []
    for p in paths:
        for row in json.loads(p.read_text())["results"]["results"]:
            name = _provider_of(row)
            if name not in seen:
                seen.append(name)
    if len(seen) <= 1:
        return [Run(label, paths)]
    return [Run(name, paths, provider=name) for name in seen]


class Run:
    """One model's scored cases, keyed by case id.

    Accepts SEVERAL exports for one model, because a 1000-case local run is done
    in 250-case chunks — promptfoo writes the `-o` export only at the end, so an
    unchunked run that dies loses everything — and each chunk writes its own
    export. Keying by case id makes the merge safe: a chunk re-run that overlaps
    replaces the case rather than double-counting it.
    """

    def __init__(self, label: str, paths: list[Path] | Path,
                 provider: str | None = None):
        self.label = label
        self.paths = [paths] if isinstance(paths, Path) else list(paths)
        self.path = self.paths[0]
        self.provider = provider
        rows = []
        for p in self.paths:
            rows.extend(json.loads(p.read_text())["results"]["results"])
        # ONE PROVIDER PER RUN. `cases` is keyed by case id so that a re-run chunk
        # replaces rather than double-counts (see the docstring) — which means a
        # MULTI-provider export would have every arm overwrite the previous one and
        # report the last arm's results as if they were the whole run, at a case count
        # of exactly 1000 so the truncation guard would not fire either. Every A/B
        # config in this repo puts two or three arms in one export, so this is the
        # normal case, not an edge one. `split_providers` below is what callers use.
        if provider is not None:
            rows = [r for r in rows if _provider_of(r) == provider]
        self.cases: dict[str, dict] = {}
        for r in rows:
            tc = r["testCase"]
            md = tc["metadata"]
            self.cases[md["id"]] = {
                "id": md["id"],
                "category": md["category"],
                "family": family_of(md["category"]),
                "mechanism": md.get("mechanism") or "",
                "rounds": _rounds(md, tc.get("vars", {})),
                "wants_call": bool(md.get("expected_calls")),
                # failureReason 2 = the provider raised; never scored.
                "error": r.get("failureReason") == 2,
                "passed": bool(r.get("success")),
                "reason": (r.get("gradingResult") or {}).get("reason", ""),
                "output": r.get("response", {}).get("output"),
            }

    # -- integrity -------------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.cases)

    @property
    def truncated(self) -> bool:
        return self.n != DATASET_SIZE

    @property
    def errors(self) -> int:
        return sum(1 for c in self.cases.values() if c["error"])

    def scored(self) -> list[dict]:
        """Cases that actually got a verdict — the honest denominator."""
        return [c for c in self.cases.values() if not c["error"]]

    def rate(self, cases: list[dict] | None = None) -> tuple[int, int]:
        cases = self.scored() if cases is None else [c for c in cases if not c["error"]]
        return sum(1 for c in cases if c["passed"]), len(cases)


def _pct(passed: int, total: int) -> str:
    return f"{passed / total:6.1%}" if total else "     —"


def _cell(run: Run, cases: list[dict]) -> str:
    passed, total = run.rate(cases)
    return f"{_pct(passed, total)} ({passed}/{total})" if total else "        —"


def _grouped(run: Run, key) -> dict:
    out: dict = collections.defaultdict(list)
    for c in run.scored():
        out[key(c)].append(c)
    return out


def table(runs: list[Run], title: str, key, order=None, width: int = 30) -> list[str]:
    """One row per group, one column per run."""
    groups = {}
    for run in runs:
        for k, cases in _grouped(run, key).items():
            # None means "this row does not apply to this case" (e.g. a
            # single-turn case in a per-mechanism cut). Rendering it as a row
            # labelled "None" invites reading it as a real bucket.
            if k is None:
                continue
            groups.setdefault(k, True)
    keys = order or sorted(groups)
    lines = [f"\n{title}"]
    header = f"  {'':<{width}}" + "".join(f"{r.label:>26}" for r in runs)
    lines.append(header)
    lines.append("  " + "-" * (width + 26 * len(runs)))
    for k in keys:
        if k not in groups:
            continue
        row = f"  {str(k):<{width}}"
        for run in runs:
            cases = [c for c in run.scored() if key(c) == k]
            row += f"{_cell(run, cases):>26}"
        lines.append(row)
    return lines


def disagreement(a: Run, b: Run) -> list[str]:
    """Where the two models differ, and on what. Two headlines can be equal while
    the models pass disjoint halves of the benchmark, so this is the table that
    says whether the fine-tune moved anything."""
    shared = [i for i in a.cases if i in b.cases
              and not a.cases[i]["error"] and not b.cases[i]["error"]]
    both = a_only = b_only = neither = 0
    a_only_by: collections.Counter = collections.Counter()
    b_only_by: collections.Counter = collections.Counter()
    for i in shared:
        pa, pb = a.cases[i]["passed"], b.cases[i]["passed"]
        bucket = a.cases[i]["category"]
        if pa and pb:
            both += 1
        elif pa:
            a_only += 1
            a_only_by[bucket] += 1
        elif pb:
            b_only += 1
            b_only_by[bucket] += 1
        else:
            neither += 1
    lines = [f"\nper-case agreement over {len(shared)} cases scored by both",
             f"  {'both pass':<34}{both:5d}",
             f"  {f'only {a.label} passes':<34}{a_only:5d}",
             f"  {f'only {b.label} passes':<34}{b_only:5d}",
             f"  {'neither passes':<34}{neither:5d}",
             f"  {f'net {b.label} - {a.label}':<34}{b_only - a_only:+5d}"]
    if a_only_by:
        lines.append(f"\n  categories only {a.label} gets right (top 8)")
        for cat, n in a_only_by.most_common(8):
            lines.append(f"    {cat:<44} {n}")
    if b_only_by:
        lines.append(f"\n  categories only {b.label} gets right (top 8)")
        for cat, n in b_only_by.most_common(8):
            lines.append(f"    {cat:<44} {n}")
    return lines


def build_report(runs: list[Run]) -> list[str]:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("1000-case benchmark — pf/tests.combined.yaml")
    lines.append("=" * 78)

    lines.append("\nrun integrity")
    for r in runs:
        flag = ""
        if r.truncated:
            flag = f"  << only {r.n} cases! expected {DATASET_SIZE} — DO NOT QUOTE"
        err = f"  {r.errors} provider error(s) excluded" if r.errors else ""
        lines.append(f"  {r.label:<28} {r.n:>5} cases{err}{flag}")

    lines.append("\nheadline (provider errors excluded from the denominator)")
    for r in runs:
        p, t = r.rate()
        lines.append(f"  {r.label:<28} {_pct(p, t)}  ({p}/{t})")

    lines.extend(table(runs, "call vs no-call",
                       lambda c: "wants a tool call" if c["wants_call"]
                       else "wants NO call (refusal/ablation)", width=32))
    lines.extend(table(runs, "by rounds (1 = single-turn)",
                       lambda c: c["rounds"], order=[1, 2, 3, 4, 5, 6], width=30))
    lines.extend(table(runs, "by family", lambda c: c["family"], width=30))

    # The subset a published-number comparison may legitimately use.
    shared = comparable_ids()
    if shared:
        lines.extend(table(
            runs,
            f"vs the published v4 set: the {len(shared)} byte-identical cases only",
            lambda c: ("byte-identical to published v4" if c["id"] in shared
                       else "new or regenerated"), width=34))
    lines.extend(table(
        runs, "by conversation mechanism (571 cases)", lambda c: c["mechanism"],
        order=["progressive", "correction", "distractor", "switch",
               "exact_output", "token_address"], width=30))

    # Rounds within each mechanism — the degradation curve, which is the whole
    # reason the slice spans 2-6 rounds rather than sitting at one length.
    for mech in ("progressive", "correction", "distractor", "switch",
                 "exact_output", "token_address"):
        has = any(any(c["mechanism"] == mech for c in r.scored()) for r in runs)
        if not has:
            continue
        lines.extend(table(
            runs, f"  {mech}: accuracy by round",
            lambda c, m=mech: c["rounds"] if c["mechanism"] == m else None,
            order=[2, 3, 4, 5, 6], width=30))
    lines.extend(table(runs, "by conversation mechanism x call/no-call",
                       lambda c: (f"{c['mechanism']} "
                                  f"({'call' if c['wants_call'] else 'NO call'})")
                       if c["mechanism"] else None, width=34))

    # Pairwise where it is informative. With four runs the interesting pairs are
    # each base-vs-fine-tune (what training did) and fine-tune-vs-fine-tune (what
    # the base model did, on identical training data) — not all six combinations.
    pairs = _pairs_of_interest(runs)
    for a, b in pairs:
        lines.extend(disagreement(a, b))
    if not pairs and len(runs) == 2:
        lines.extend(disagreement(runs[0], runs[1]))
    return lines


def _pairs_of_interest(runs: list[Run]) -> list[tuple[Run, Run]]:
    """base->fine-tune within a family, then fine-tune across families.

    Matched on the label, which is why the configs use stable labels: a run whose
    label does not say which family and whether it is tuned cannot be paired, and
    silently dropping it is better than guessing wrong.
    """
    def find(pred):
        return [r for r in runs if pred(r.label)]

    out: list[tuple[Run, Run]] = []
    for family in ("gemma", "qwen"):
        base = find(lambda l, f=family: f in l and "ft" not in l and "base" in l)
        tuned = find(lambda l, f=family: f in l and "ft" in l)
        if base and tuned:
            out.append((base[0], tuned[0]))
    gft = find(lambda l: "gemma" in l and "ft" in l)
    qft = find(lambda l: "qwen" in l and "ft" in l)
    if gft and qft:
        out.append((gft[0], qft[0]))
    return out


def as_json(runs: list[Run]) -> dict:
    def slice_(run: Run, key):
        out = {}
        for k, cases in _grouped(run, key).items():
            if k is None:
                continue
            p, t = run.rate(cases)
            out[str(k)] = {"passed": p, "total": t}
        return out

    return {
        "dataset": "pf/tests.combined.yaml",
        "dataset_size": DATASET_SIZE,
        "runs": [
            {
                "label": r.label,
                "exports": [str(p) for p in r.paths],
                "cases": r.n,
                "truncated": r.truncated,
                "errors": r.errors,
                "passed": r.rate()[0],
                "scored": r.rate()[1],
                "by_rounds": slice_(r, lambda c: c["rounds"]),
                "by_family": slice_(r, lambda c: c["family"]),
                "by_mechanism": slice_(r, lambda c: c["mechanism"] or None),
                "by_category": slice_(r, lambda c: c["category"]),
                "by_mechanism_round": slice_(
                    r, lambda c: f"{c['mechanism']}-{c['rounds']}r"
                    if c["mechanism"] else None),
                "call_split": slice_(
                    r, lambda c: "call" if c["wants_call"] else "nocall"),
            }
            for r in runs
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("exports", nargs="*", type=Path,
                    help="promptfoo -o JSON files, one run each")
    ap.add_argument("--run", action="append", default=[],
                    metavar="LABEL=GLOB",
                    help="one run assembled from several exports, e.g. "
                         "--run 'base=runs/gemma4-e4b-base.part*.out.json'")
    ap.add_argument("--label", action="append", default=None,
                    help="override a positional run's label, in order")
    ap.add_argument("--json", type=Path, default=None, help="also write the JSON")
    args = ap.parse_args()

    runs: list[Run] = []
    for spec in args.run:
        label, _, pattern = spec.partition("=")
        paths = sorted(Path().glob(pattern)) if not Path(pattern).exists() \
            else [Path(pattern)]
        if not paths:
            ap.error(f"--run {spec!r} matched no files")
        runs.extend(split_providers(label, paths))
    for i, path in enumerate(args.exports):
        label = (args.label[i] if args.label and i < len(args.label)
                 else path.stem.replace(".out", ""))
        runs.extend(split_providers(label, [path]))
    if not runs:
        ap.error("give at least one export or --run LABEL=GLOB")

    print("\n".join(build_report(runs)))

    # Compact per-model summary — the form that goes into a commit message.
    print("\n" + "=" * 78)
    print("SUMMARY (overall | tool-call cases | no-call cases)")
    print("=" * 78)
    for r in runs:
        p, t = r.rate()
        cp, ct = r.rate([c for c in r.scored() if c["wants_call"]])
        np_, nt = r.rate([c for c in r.scored() if not c["wants_call"]])
        flag = "  << TRUNCATED, DO NOT QUOTE" if r.truncated else ""
        print(f"  {r.label:<30} {_pct(p, t)} ({p}/{t})   "
              f"call {_pct(cp, ct)} ({cp}/{ct})   "
              f"no-call {_pct(np_, nt)} ({np_}/{nt}){flag}")
    if args.json:
        args.json.write_text(json.dumps(as_json(runs), indent=2))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
