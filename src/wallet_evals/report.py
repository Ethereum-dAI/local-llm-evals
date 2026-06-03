"""Aggregate CaseResults into overall + per-slice accuracy."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from wallet_evals.runner import CaseResult


def _bucket(results: list[CaseResult], key) -> dict[str, dict[str, Any]]:
    agg: dict[str, list[int]] = defaultdict(list)
    for r in results:
        for k in key(r):
            agg[k].append(r.score)
    out: dict[str, dict[str, Any]] = {}
    for k, scores in agg.items():
        n = len(scores)
        passed = sum(scores)
        out[k] = {"n": n, "passed": passed, "accuracy": round(passed / n, 4) if n else 0.0}
    return out


def aggregate(results: list[CaseResult]) -> dict[str, Any]:
    n = len(results)
    passed = sum(r.score for r in results)
    return {
        "overall": {"n": n, "passed": passed, "accuracy": round(passed / n, 4) if n else 0.0},
        "by_level": _bucket(results, lambda r: [r.level]),
        "by_protocol": _bucket(results, lambda r: [r.protocol]),
        "by_query_type": _bucket(results, lambda r: [r.query_type or "none"]),
        "by_difficulty": _bucket(results, lambda r: [r.difficulty]),
        "by_language": _bucket(results, lambda r: [r.language]),
        "by_capability": _bucket(results, lambda r: r.requires),
    }


def format_summary(report: dict[str, Any]) -> str:
    lines = []
    o = report["overall"]
    lines.append(f"Overall: {o['passed']}/{o['n']} = {o['accuracy'] * 100:.1f}%")
    for section in ("by_level", "by_protocol", "by_query_type", "by_difficulty", "by_language", "by_capability"):
        lines.append(f"\n[{section}]")
        for k, v in sorted(report[section].items()):
            lines.append(f"  {k:<24} {v['passed']}/{v['n']} = {v['accuracy'] * 100:.1f}%")
    return "\n".join(lines)
