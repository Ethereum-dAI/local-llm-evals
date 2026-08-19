"""A multi-provider export must become one run PER ARM, not one arm silently.

`Run.cases` is keyed by case id so a re-run chunk replaces rather than double-counts.
That is right for the chunked single-model runs it was written for, and wrong for every
A/B config in this repo: those put 2-3 arms in ONE export, so each arm overwrote the
previous one and the report showed the LAST arm's results as the whole run — at a case
count of exactly 1000, so the truncation guard stayed silent too.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location(
        "report_1000", ROOT / "scripts" / "report_1000.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _row(case_id: str, provider: str, passed: bool) -> dict:
    return {
        "provider": {"id": f"file://x:{provider}", "label": provider},
        "success": passed,
        "failureReason": 0,
        "response": {"output": "[]"},
        "gradingResult": {"reason": ""},
        "testCase": {"vars": {}, "metadata": {
            "id": case_id, "category": "generated-transfer-pos",
            "expected_calls": [{"tool": "transfer"}]}},
    }


def _export(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "e.out.json"
    p.write_text(json.dumps({"results": {"results": rows}}))
    return p


def test_two_arms_in_one_export_become_two_runs(tmp_path):
    m = _mod()
    # Same two case ids under both arms: arm A gets both right, arm B gets both wrong.
    rows = [_row("c1", "armA", True), _row("c2", "armA", True),
            _row("c1", "armB", False), _row("c2", "armB", False)]
    runs = m.split_providers("ignored", [_export(tmp_path, rows)])
    assert [r.label for r in runs] == ["armA", "armB"]
    assert [r.n for r in runs] == [2, 2], "each arm must keep all of its own cases"
    passed = {r.label: sum(c["passed"] for c in r.cases.values()) for r in runs}
    assert passed == {"armA": 2, "armB": 0}, \
        "arms must not overwrite each other — this is the bug the split exists for"


def test_single_provider_export_keeps_the_callers_label(tmp_path):
    """Existing invocations pass one model per export and name it themselves; the
    split must not rename those or every recorded table's labels change."""
    m = _mod()
    rows = [_row("c1", "only", True), _row("c2", "only", False)]
    runs = m.split_providers("my-label", [_export(tmp_path, rows)])
    assert len(runs) == 1
    assert runs[0].label == "my-label"
    assert runs[0].n == 2


def test_provider_falls_back_to_id_when_unlabelled(tmp_path):
    m = _mod()
    rows = [_row("c1", "a", True), _row("c2", "b", True)]
    for r in rows:
        del r["provider"]["label"]
    runs = m.split_providers("x", [_export(tmp_path, rows)])
    assert sorted(r.label for r in runs) == ["file://x:a", "file://x:b"]
