"""Unit coverage for scripts/build_combined_benchmark.py's pure concatenation
logic, independent of the real (large) generated files.
"""
from __future__ import annotations

import pytest

from scripts.build_combined_benchmark import build_combined


def _case(id_: str) -> dict:
    return {"vars": {"user_message": f"msg-{id_}"}, "metadata": {"id": id_}}


def test_build_combined_concatenates_in_order():
    app_contract = [_case("a-1"), _case("a-2")]
    protocols = [_case("p-1")]
    combined = build_combined(app_contract, protocols)
    assert [c["metadata"]["id"] for c in combined] == ["a-1", "a-2", "p-1"]


def test_build_combined_count_equals_sum_of_parts():
    app_contract = [_case(f"a-{i}") for i in range(5)]
    protocols = [_case(f"p-{i}") for i in range(3)]
    combined = build_combined(app_contract, protocols)
    assert len(combined) == len(app_contract) + len(protocols) == 8


def test_build_combined_raises_loudly_on_duplicate_id():
    app_contract = [_case("dup-1")]
    protocols = [_case("dup-1")]  # same id as an app-contract case
    with pytest.raises(ValueError, match="duplicate case id"):
        build_combined(app_contract, protocols)


def test_build_combined_does_not_silently_deduplicate():
    # A weaker implementation might de-dup instead of raising; prove the
    # duplicate is caught rather than quietly dropped.
    app_contract = [_case("x"), _case("x")]  # a bug within one file's own ids
    protocols = []
    with pytest.raises(ValueError):
        build_combined(app_contract, protocols)
