from wallet_evals.runner import CaseResult
from wallet_evals.report import aggregate, format_summary


def _r(case_id, score, **kw):
    base = dict(model="m", level="payload", protocol="transfer", query_type="one_shot",
                difficulty="easy", language="english", requires=[])
    base.update(kw)
    return CaseResult(case_id=case_id, repeat=0, score=score, **base)


def test_overall_accuracy():
    report = aggregate([_r("a", 1), _r("b", 0), _r("c", 1)])
    assert report["overall"]["n"] == 3
    assert report["overall"]["accuracy"] == round(2 / 3, 4)


def test_slice_by_protocol():
    results = [_r("a", 1, protocol="transfer"), _r("b", 0, protocol="uniswap"), _r("c", 1, protocol="uniswap")]
    report = aggregate(results)
    assert report["by_protocol"]["transfer"] == {"n": 1, "passed": 1, "accuracy": 1.0}
    assert report["by_protocol"]["uniswap"] == {"n": 2, "passed": 1, "accuracy": 0.5}


def test_slice_by_capability():
    results = [_r("a", 1, requires=["dex_routing"]), _r("b", 0, requires=["dex_routing", "multi_step"])]
    report = aggregate(results)
    assert report["by_capability"]["dex_routing"] == {"n": 2, "passed": 1, "accuracy": 0.5}
    assert report["by_capability"]["multi_step"] == {"n": 1, "passed": 0, "accuracy": 0.0}


def test_format_summary_is_string_with_overall():
    report = aggregate([_r("a", 1), _r("b", 0)])
    text = format_summary(report)
    assert "Overall" in text
    assert "50" in text  # 50% accuracy somewhere
