#!/usr/bin/env python3
"""Field-level failure anatomy for a frozen promptfoo run.

    uv run python scripts/analyze_failures.py runs/gemma4-e4b-ft-v4-appprompt.part*.out.json

Answers three questions the aggregate tables cannot:

  * what SHAPE is each failure — no call, spurious call, or wrong arguments;
  * which ARGUMENT is wrong, per field;
  * where the wrong value CAME FROM — a value the user really said (so the model picked
    the wrong turn), a near-miss corruption of the right literal (so it damaged a copy),
    or something absent from the conversation entirely (so it invented it).

The last distinction is the whole point: "wrong turn" and "corrupted copy" are different
defects with different fixes, and they are indistinguishable in a per-category accuracy
table.

IMPORTANT: gold is read from the DATASET, never from the run export. promptfoo redacts
any metadata field named `token` and truncates long `to` values, so the export's
`expected_calls` cannot be compared against — see CLAUDE.md.
"""
from __future__ import annotations

import argparse
import collections
import difflib
import glob
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "pf" / "tests.combined.yaml"
#: An emitted literal this similar to the gold one is a corrupted COPY, not a different
#: value. Real errors sit at edit distance 1-2 in a 42-character address (ratio ~0.98);
#: two genuinely different addresses from the same generator share a prefix at best.
CORRUPTION_RATIO = 0.90


def load_dataset(path: Path) -> tuple[dict, dict]:
    doc = yaml.safe_load(path.read_text())
    tests = doc["tests"] if isinstance(doc, dict) else doc
    gold, convo = {}, {}
    for t in tests:
        md = t.get("metadata") or {}
        v = t.get("vars") or {}
        cid = md.get("id")
        gold[cid] = md.get("expected_calls")
        msgs = v.get("messages")
        convo[cid] = (
            " \n".join(m.get("content", "") for m in msgs if isinstance(m, dict))
            if msgs else str(v.get("user_message", ""))
        )
    return gold, convo


def load_run(patterns: list[str]) -> list[dict]:
    out = []
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            for r in json.load(open(p))["results"]["results"]:
                md = r.get("testCase", {}).get("metadata", {})
                out.append(dict(id=md.get("id"), category=md.get("category", "?"),
                                ok=bool(r.get("success")),
                                output=(r.get("response") or {}).get("output", "")))
    return out


def parse_call(output: str) -> tuple[str | None, dict | None]:
    """The provider normalises every dialect to `[{"name", "arguments"}]`."""
    try:
        v = json.loads(output)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            args = v[0].get("arguments")
            if isinstance(args, str):
                args = json.loads(args)
            return v[0].get("name"), (args or {})
    except Exception:
        pass
    return None, None


def classify_value(field: str, want: str, got: str, conversation: str) -> str:
    if got.lower() in conversation.lower():
        return "a value the user DID say (wrong turn)"
    if difflib.SequenceMatcher(None, want.lower(), got.lower()).ratio() >= CORRUPTION_RATIO:
        return "a CORRUPTED copy of the right literal"
    return "a value absent from the conversation"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="one or more *.out.json (globs allowed)")
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--examples", type=int, default=4)
    args = ap.parse_args()

    gold, convo = load_dataset(args.dataset)
    rows = load_run(args.runs)
    scored = [r for r in rows if r["id"] in gold]
    passed = sum(1 for r in scored if r["ok"])
    print(f"{len(scored)} cases joined to {args.dataset.name} "
          f"({len(rows) - len(scored)} unjoined)")
    print(f"accuracy {passed}/{len(scored)} = {100 * passed / max(len(scored), 1):.1f}%\n")

    shapes = collections.Counter()
    fields = collections.Counter()
    sources = collections.Counter()
    by_cat = collections.Counter()
    examples = collections.defaultdict(list)

    for r in scored:
        if r["ok"]:
            continue
        want = gold[r["id"]]
        name, emitted = parse_call(r["output"])
        called = emitted is not None
        if not want:
            shapes["spurious call (gold = no call)" if called else "other"] += 1
            continue
        if not called:
            q = r["output"].strip().endswith("?")
            shapes[f"wanted a call, produced none ({'question' if q else 'prose'})"] += 1
            continue
        shapes["called, WRONG arguments"] += 1
        g = want[0]
        if name and name != g.get("tool"):
            fields["WRONG tool"] += 1
        for f, gv in g.items():
            if f == "tool":
                continue
            ev = emitted.get(f)
            if ev is None:
                fields[f"MISSING {f}"] += 1
                continue
            if str(ev).lower() == str(gv).lower():
                continue
            fields[f"WRONG {f}"] += 1
            src = classify_value(f, str(gv), str(ev), convo.get(r["id"], ""))
            sources[f"{f}: {src}"] += 1
            by_cat[r["category"]] += 1
            if len(examples[f"{f}: {src}"]) < args.examples:
                examples[f"{f}: {src}"].append((r["category"], str(gv), str(ev)))

    def table(title: str, counter: collections.Counter, limit: int = 20) -> None:
        print(f"== {title} ==")
        for k, v in counter.most_common(limit):
            print(f"  {v:5d}  {k}")
        print()

    table("failure shapes", shapes)
    table("which argument is wrong", fields)
    table("where the wrong value came from", sources)
    table("wrong-argument cases by category", by_cat, limit=12)
    print("== examples ==")
    for k, ex in sorted(examples.items()):
        print(f"\n{k}")
        for cat, gv, ev in ex:
            print(f"   [{cat}]\n     gold {gv}\n     emit {ev}")


if __name__ == "__main__":
    main()
