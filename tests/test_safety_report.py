from scripts.safety_report import (
    _results_from_output,
    format_report,
    summarize_safety,
)


def _result(provider, cid, category, expected_calls, success):
    return {
        "provider": {"id": f"openrouter:vendor/{provider}"},
        "success": success,
        "testCase": {
            "metadata": {
                "id": cid,
                "category": category,
                "expected_calls": expected_calls,
            }
        },
    }


def test_false_execution_rate_counts_failed_no_call_cases():
    results = [
        # gpt: 2 no-call adversarial, one fired (failed), one safe (passed).
        _result("gpt", "adv-injection-001", "adversarial", [], success=True),
        _result("gpt", "adv-scam-001", "adversarial", [], success=False),
        # a non-adversarial case must be ignored entirely.
        _result("gpt", "swap-en-001", "truePositiveSwap", [{"tool": "swap"}], True),
    ]
    stats = summarize_safety(results)
    assert stats["gpt"]["no_call_total"] == 2
    assert stats["gpt"]["false_executions"] == 1
    assert stats["gpt"]["false_execution_rate"] == 0.5


def test_confused_deputy_reported_separately():
    results = [
        _result("gemma", "adv-injection-002", "adversarial", [{"tool": "executeTx"}], True),
        _result("gemma", "adv-injection-001", "adversarial", [], success=True),
    ]
    stats = summarize_safety(results)
    assert stats["gemma"]["deputy_total"] == 1
    assert stats["gemma"]["deputy_passed"] == 1
    # The deputy case (non-empty gold) is NOT folded into the no-call total.
    assert stats["gemma"]["no_call_total"] == 1


def test_zero_division_safe_when_no_no_call_cases():
    stats = summarize_safety(
        [_result("m", "adv-injection-002", "adversarial", [{"tool": "executeTx"}], True)]
    )
    assert stats["m"]["false_execution_rate"] == 0.0


def test_results_from_output_unwraps_promptfoo_nesting():
    nested = {"results": {"results": [1, 2, 3]}}
    assert _results_from_output(nested) == [1, 2, 3]


def test_format_report_handles_empty():
    assert "No adversarial cases" in format_report({})
