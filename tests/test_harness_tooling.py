from __future__ import annotations

import importlib.util
import json
import pytest
import yaml

from pathlib import Path
from scripts.build_combined_benchmark import build_combined
from scripts.convert_recognition import convert_case, LOOKUP
from scripts.safety_report import (
    _results_from_output,
    format_report,
    summarize_safety,
)
from wallet_evals.intents import to_base_units

# ============================================================================
# test_harness_tooling
# ============================================================================
#
# The harness's own tooling: provider retry, report splitting, config pairing.
#
# Three small suites merged into one file. None is about the models or the data; each
# guards a piece of measurement machinery that has already produced a wrong number
# once, which is why they exist at all.

def _provider():
    path = Path(__file__).resolve().parents[1] / "pf" / "provider_functiongemma.py"
    spec = importlib.util.spec_from_file_location("pf_provider_retry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_has_call_reads_the_translators_own_contract():
    """`raw_output_to_scoreable` returns an OpenAI-shaped LIST when a call was parsed and
    the prose verbatim otherwise, so the retry trigger must key off that, not re-parse."""
    m = _provider()
    assert m._has_call(json.dumps([{"name": "transfer", "arguments": "{}"}]))
    assert m._has_call(json.dumps([{"name": "swap", "arguments": "{}"},
                                   {"name": "transfer", "arguments": "{}"}]))
    assert not m._has_call("What token do you mean?")
    assert not m._has_call("[]")
    assert not m._has_call("")


def test_has_call_rejects_a_list_that_is_not_tool_calls():
    """Prose that happens to be valid JSON must not read as a call, or the retry would
    silently never fire for it."""
    m = _provider()
    assert not m._has_call(json.dumps(["transfer", "swap"]))
    assert not m._has_call(json.dumps([{"tool": "transfer"}]))
    assert not m._has_call(json.dumps({"name": "transfer"}))


def test_retry_nudge_restates_the_refusal_option():
    """Without this sentence the nudge is a pure bias toward acting, which is exactly how
    a refusal case gets converted into a transaction on the second attempt."""
    m = _provider()
    text = m.RETRY_NUDGE.lower()
    assert "refused" in text and "no tool call" in text
    assert "do not ask" in text


def test_retry_is_off_unless_a_config_asks_for_it():
    """It changes app behaviour, so it must never be the default — a run with it on is not
    app parity."""
    src = (Path(__file__).resolve().parents[1] / "pf"
           / "provider_functiongemma.py").read_text()
    assert 'config.get("retry_on_no_call")' in src, "must be opt-in via config"
    assert 'config.get("retry_on_no_call", True)' not in src, "must not default to on"


def test_retry_result_is_kept_only_when_it_produced_a_call():
    """A second refusal must not overwrite the first answer: replacing it would hide a
    refusal behind a second helping of prose, and would also mask the failure shape the
    analysis depends on."""
    src = (Path(__file__).resolve().parents[1] / "pf"
           / "provider_functiongemma.py").read_text()
    assert "if _has_call(output2):" in src


def test_exactly_one_retry_no_loop():
    """A loop would keep pushing until it got a call, which scores well by destroying the
    refusals — the opposite of what this is for."""
    src = (Path(__file__).resolve().parents[1] / "pf"
           / "provider_functiongemma.py").read_text()
    body = src[src.index("if config.get(\"retry_on_no_call\")"):]
    body = body[:body.index("result: dict[str, Any]")]
    assert "while" not in body, "retry must not loop"
    assert body.count("_has_call(output2)") == 1


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


PAIRS = {
    "promptfooconfig.v5-vs-base.remote.yaml": ("gemma4-e4b-base",
                                               "gemma4-e4b-ft-v5"),
}


MODEL_SOURCE_KEYS = {"model_path", "repo_id", "filename", "revision", "remote_url"}


def _config(name: str) -> dict:
    path = ROOT / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    return yaml.safe_load(path.read_text())


@pytest.mark.parametrize("name", sorted(PAIRS))
def test_pair_differs_only_in_the_model_source(name):
    cfg = _config(name)
    labels = PAIRS[name]
    by_label = {p["label"]: p["config"] for p in cfg["providers"]}
    for label in labels:
        assert label in by_label, f"{name} has no provider labelled {label!r}"
    a, b = (by_label[l] for l in labels)
    differing = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    stray = differing - MODEL_SOURCE_KEYS
    assert not stray, (
        f"{name}: {labels[0]} and {labels[1]} differ in {sorted(stray)} — a "
        f"base-vs-fine-tune delta must come from the weights alone")
    assert differing, f"{name}: the two providers are identical; one loads the wrong model"


@pytest.mark.parametrize("name", sorted(PAIRS))
def test_pair_has_no_unknown_top_level_keys(name):
    """promptfoo rejects unrecognised top-level keys. A leftover YAML anchor block
    would fail the run rather than the suite, which is the expensive way to find out."""
    allowed = {"description", "prompts", "providers", "defaultTest", "tests",
               "env", "outputPath", "sharing", "defaultTestOptions", "scenarios",
               "derivedMetrics", "extensions", "commandLineOptions", "redteam",
               "evaluateOptions", "tags", "metadata"}
    stray = set(_config(name)) - allowed
    assert not stray, f"{name} has unknown top-level key(s): {sorted(stray)}"


@pytest.mark.parametrize("name", sorted(PAIRS))
def test_pair_scores_the_full_benchmark_by_default(name):
    cfg = _config(name)
    assert "tests.combined.yaml" in cfg["tests"], \
        f"{name} does not default to the 1000-case benchmark"
    assert cfg["defaultTest"]["assert"][0]["value"] == "file://pf/assert.py"


def test_a_remote_pair_names_a_distinct_pod_per_arm():
    """The failure this catches: both arms pointing at the SAME `remote_url` env var,
    which serves one model under two labels and reports a 0-point delta as a result."""
    cfg = _config("promptfooconfig.v5-vs-base.remote.yaml")
    urls = [p["config"].get("remote_url") for p in cfg["providers"]]
    assert all(urls), "every remote arm needs a remote_url"
    assert len(set(urls)) == len(urls), f"arms share a pod: {urls}"

# ============================================================================
# test_safety_report
# ============================================================================

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

# ============================================================================
# test_build_combined_benchmark
# ============================================================================
#
# Unit coverage for scripts/build_combined_benchmark.py's pure concatenation
# logic, independent of the real (large) generated files.

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

# ============================================================================
# test_convert
# ============================================================================

def test_to_base_units_eth():
    assert to_base_units("0.1", 18) == "100000000000000000"


def test_to_base_units_usdc():
    assert to_base_units("100", 6) == "100000000"


def test_to_base_units_integer():
    assert to_base_units("5", 18) == "5000000000000000000"


def test_convert_null_tool_to_empty_calls():
    raw = {"id": "ambiguous-001", "user_message": "Send ETH to bob.eth", "category": "ambiguous",
           "language": "english", "expected_tool": None, "expected_args": None, "notes": None}
    case, manual = convert_case(raw)
    assert manual is None
    assert case["expected_calls"] == []
    assert case["level"] == "intent"


def test_convert_native_transfer():
    raw = {"id": "transfer-en-001", "user_message": "Send 0.1 ETH to vitalik.eth",
           "category": "truePositiveTransfer", "language": "english", "expected_tool": "transfer",
           "expected_args": {"to": {"kind": "exact", "value": "vitalik.eth"},
                             "amount": {"kind": "exact", "value": "0.1"},
                             "token": {"kind": "exact", "value": "ETH"}}, "notes": None}
    case, manual = convert_case(raw)
    assert manual is None
    call = case["expected_calls"][0]
    assert call["tool"] == "transfer"
    assert "chainId" not in call  # app tools declare none
    assert call["to"] == "vitalik.eth"
    assert call["amount"] == "0.1"
    assert call["token"] == "ETH"


def test_convert_erc20_transfer():
    raw = {"id": "transfer-en-002", "user_message": "Send 100 USDC to vitalik.eth",
           "category": "truePositiveTransfer", "language": "english", "expected_tool": "transfer",
           "expected_args": {"to": {"kind": "exact", "value": "vitalik.eth"},
                             "amount": {"kind": "exact", "value": "100"},
                             "token": {"kind": "exact", "value": "USDC"}}, "notes": None}
    case, manual = convert_case(raw)
    call = case["expected_calls"][0]
    assert call["tool"] == "transfer"
    assert "chainId" not in call  # app tools declare none
    assert call["to"] == "vitalik.eth"
    assert call["amount"] == "100"
    assert call["token"] == "USDC"


def test_incomplete_swap_routed_to_manual():
    raw = {"id": "swap-en-001", "user_message": "Swap 100 USDC for ETH", "category": "truePositiveSwap",
           "language": "english", "expected_tool": "swap",
           "expected_args": {"from_token": {"kind": "exact", "value": "USDC"}}, "notes": None}
    case, manual = convert_case(raw)
    assert case is None
    assert manual == "swap-en-001"


def test_convert_swap_exact_in():
    raw = {"id": "swap-en-001", "user_message": "Swap 100 USDC for DAI", "category": "truePositiveSwap",
           "language": "english", "expected_tool": "swap",
           "expected_args": {"from_token": {"kind": "exact", "value": "USDC"},
                             "to_token": {"kind": "exact", "value": "DAI"},
                             "amount": {"kind": "exact", "value": "100"},
                             "amount_side": {"kind": "exact", "value": "input"}}, "notes": None}
    case, manual = convert_case(raw)
    assert manual is None
    assert case["protocol"] == "uniswap"
    call = case["expected_calls"][0]
    assert call["tool"] == "swap"
    assert "chainId" not in call  # app tools declare none
    assert call["from_token"] == "USDC"
    assert call["to_token"] == "DAI"
    assert call["amount"] == "100"
    assert call["amount_side"] == "input"


def test_convert_swap_native_eth_from_token():
    raw = {"id": "swap-en-002", "user_message": "Swap 1 ETH for USDC", "category": "truePositiveSwap",
           "language": "english", "expected_tool": "swap",
           "expected_args": {"from_token": {"kind": "exact", "value": "ETH"},
                             "to_token": {"kind": "exact", "value": "USDC"},
                             "amount": {"kind": "exact", "value": "1"},
                             "amount_side": {"kind": "exact", "value": "input"}}, "notes": None}
    case, manual = convert_case(raw)
    call = case["expected_calls"][0]
    assert call["from_token"] == "ETH"
    assert call["amount"] == "1"


def test_convert_swap_exact_output_to_manual():
    raw = {"id": "swap-en-003", "user_message": "Buy 1 ETH with USDC", "category": "truePositiveSwap",
           "language": "english", "expected_tool": "swap",
           "expected_args": {"from_token": {"kind": "exact", "value": "USDC"},
                             "to_token": {"kind": "exact", "value": "ETH"},
                             "amount": {"kind": "exact", "value": "1"},
                             "amount_side": {"kind": "exact", "value": "output"}}, "notes": None}
    case, manual = convert_case(raw)
    assert case is None
    assert manual == "swap-en-003"


def test_amount_all_routed_to_manual():
    raw = {"id": "transfer-en-003", "user_message": "Send all my ETH to vitalik.eth",
           "category": "truePositiveTransfer", "language": "english", "expected_tool": "transfer",
           "expected_args": {"to": {"kind": "exact", "value": "vitalik.eth"},
                             "amount": {"kind": "exact", "value": "all"},
                             "token": {"kind": "exact", "value": "ETH"}}, "notes": None}
    case, manual = convert_case(raw)
    assert case is None
    assert manual == "transfer-en-003"


def test_ens_recipient_flagged_as_ens_resolution():
    raw = {"id": "transfer-en-001", "user_message": "Send 0.1 ETH to vitalik.eth",
           "category": "truePositiveTransfer", "language": "english", "expected_tool": "transfer",
           "expected_args": {"to": {"kind": "exact", "value": "vitalik.eth"},
                             "amount": {"kind": "exact", "value": "0.1"},
                             "token": {"kind": "exact", "value": "ETH"}}, "notes": None}
    case, _ = convert_case(raw)
    assert case["requires"] == ["ens_resolution"]


def test_erc20_to_ens_carries_both_flags():
    raw = {"id": "transfer-en-002", "user_message": "Send 100 USDC to vitalik.eth",
           "category": "truePositiveTransfer", "language": "english", "expected_tool": "transfer",
           "expected_args": {"to": {"kind": "exact", "value": "vitalik.eth"},
                             "amount": {"kind": "exact", "value": "100"},
                             "token": {"kind": "exact", "value": "USDC"}}, "notes": None}
    case, _ = convert_case(raw)
    assert case["requires"] == ["ens_resolution", "token_address_lookup"]


def test_raw_address_recipient_has_no_ens_flag():
    raw = {"id": "transfer-en-004", "user_message": "Send 1 ETH to 0x1111111111111111111111111111111111111111",
           "category": "truePositiveTransfer", "language": "english", "expected_tool": "transfer",
           "expected_args": {"to": {"kind": "exact", "value": "0x1111111111111111111111111111111111111111"},
                             "amount": {"kind": "exact", "value": "1"},
                             "token": {"kind": "exact", "value": "ETH"}}, "notes": None}
    case, _ = convert_case(raw)
    assert case["requires"] == []


def test_unknown_ens_accepted_as_unresolved():
    raw = {"id": "transfer-en-005", "user_message": "Send 1 ETH to bob.eth",
           "category": "truePositiveTransfer", "language": "english", "expected_tool": "transfer",
           "expected_args": {"to": {"kind": "exact", "value": "bob.eth"},
                             "amount": {"kind": "exact", "value": "1"},
                             "token": {"kind": "exact", "value": "ETH"}}, "notes": None}
    case, manual = convert_case(raw)
    # App contract: unresolved recipients (including unknown ENS) are accepted
    assert manual is None
    call = case["expected_calls"][0]
    assert call["to"] == "bob.eth"
    # bob.eth is not in LOOKUP, so no ens_resolution flag (only known ENS names get it)
    assert case["requires"] == []


def test_difficulty_derived_swap_and_multilingual_medium():
    swap_raw = {"id": "swap-en-001", "user_message": "Swap 100 USDC for DAI", "category": "truePositiveSwap",
                "language": "english", "expected_tool": "swap",
                "expected_args": {"from_token": {"kind": "exact", "value": "USDC"},
                                  "to_token": {"kind": "exact", "value": "DAI"},
                                  "amount": {"kind": "exact", "value": "100"},
                                  "amount_side": {"kind": "exact", "value": "input"}}, "notes": None}
    case, _ = convert_case(swap_raw)
    assert case["difficulty"] == "medium"

    it_raw = {"id": "transfer-it-001", "user_message": "Invia 1 ETH a vitalik.eth",
              "category": "multilingualTransfer", "language": "italian", "expected_tool": "transfer",
              "expected_args": {"to": {"kind": "exact", "value": "vitalik.eth"},
                                "amount": {"kind": "exact", "value": "1"},
                                "token": {"kind": "exact", "value": "ETH"}}, "notes": None}
    case, _ = convert_case(it_raw)
    assert case["difficulty"] == "medium"

    en_raw = {"id": "transfer-en-001", "user_message": "Send 0.1 ETH to vitalik.eth",
              "category": "truePositiveTransfer", "language": "english", "expected_tool": "transfer",
              "expected_args": {"to": {"kind": "exact", "value": "vitalik.eth"},
                                "amount": {"kind": "exact", "value": "0.1"},
                                "token": {"kind": "exact", "value": "ETH"}}, "notes": None}
    case, _ = convert_case(en_raw)
    assert case["difficulty"] == "easy"


def test_convert_transfer_emits_app_contract_gold():
    raw = {
        "id": "transfer-en-001", "user_message": "Send 0.1 ETH to vitalik.eth",
        "category": "truePositiveTransfer", "language": "english",
        "expected_tool": "transfer",
        "expected_args": {
            "to": {"kind": "exact", "value": "vitalik.eth"},
            "amount": {"kind": "exact", "value": "0.1"},
            "token": {"kind": "exact", "value": "ETH"},
        },
        "notes": None,
    }
    case, manual = convert_case(raw)
    assert manual is None
    assert case["expected_calls"] == [{
        "tool": "transfer", "to": "vitalik.eth",
        "amount": "0.1", "token": "ETH",
    }]
