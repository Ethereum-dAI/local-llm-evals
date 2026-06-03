from wallet_evals.schema import Case, ParsedToolCall, ParsedTurn
from wallet_evals.runner import run_eval, CaseResult


class _ScriptedAdapter:
    """Returns a preset ParsedTurn per user_message."""
    model = "fake/model"

    def __init__(self, by_message):
        self._by_message = by_message

    def run(self, user_message: str) -> ParsedTurn:
        return self._by_message[user_message]


def _native_transfer_case():
    return Case(id="t1", user_message="send", level="payload", language="english",
                category="truePositiveTransfer", query_type="one_shot", protocol="transfer",
                difficulty="easy", requires=[],
                expected_calls=[{"tool": "executeTx", "chainId": "1", "to": "0xabc",
                                 "value": "100", "function": None, "args": []}], notes=None)


def test_runner_scores_pass_and_fail():
    case_pass = _native_transfer_case()
    case_fail = _native_transfer_case()
    case_fail.id = "t2"
    case_fail.user_message = "wrong"
    adapter = _ScriptedAdapter({
        "send": ParsedTurn(tool_calls=[ParsedToolCall(name="executeTx", chainId="1", to="0xabc", value="100", function=None, args=[])]),
        "wrong": ParsedTurn(tool_calls=[]),
    })
    results = run_eval([case_pass, case_fail], adapter, repeats=1)
    by_id = {(r.case_id, r.repeat): r for r in results}
    assert by_id[("t1", 0)].score == 1
    assert by_id[("t2", 0)].score == 0
    assert all(isinstance(r, CaseResult) for r in results)


def test_runner_repeats():
    case = _native_transfer_case()
    adapter = _ScriptedAdapter({"send": ParsedTurn(tool_calls=[])})
    results = run_eval([case], adapter, repeats=3)
    assert len(results) == 3
    assert {r.repeat for r in results} == {0, 1, 2}


def test_runner_records_metadata_for_slicing():
    case = _native_transfer_case()
    adapter = _ScriptedAdapter({"send": ParsedTurn(tool_calls=[])})
    r = run_eval([case], adapter, repeats=1)[0]
    assert r.protocol == "transfer"
    assert r.level == "payload"
    assert r.model == "fake/model"
