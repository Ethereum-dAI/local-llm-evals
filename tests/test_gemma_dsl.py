from wallet_evals.gemma_dsl import parse_gemma_tool_calls


def test_parses_single_quoted_call():
    raw = '<|tool_call>call:executeTx{chainId:<|"|>1<|"|>,to:<|"|>0xabc<|"|>,value:<|"|>0<|"|>}<tool_call|>'
    calls = parse_gemma_tool_calls(raw)
    assert len(calls) == 1
    name, fields = calls[0]
    assert name == "executeTx"
    assert fields == {"chainId": "1", "to": "0xabc", "value": "0"}


def test_parses_args_as_json_string_value():
    raw = '<|tool_call>call:executeTx{chainId:<|"|>1<|"|>,args:<|"|>["0xspender","100"]<|"|>}<tool_call|>'
    calls = parse_gemma_tool_calls(raw)
    name, fields = calls[0]
    assert fields["args"] == '["0xspender","100"]'


def test_parses_multiple_calls():
    raw = (
        '<|tool_call>call:executeTx{to:<|"|>0x1<|"|>}<tool_call|>'
        '<|tool_call>call:executeTx{to:<|"|>0x2<|"|>}<tool_call|>'
    )
    calls = parse_gemma_tool_calls(raw)
    assert [f["to"] for _, f in calls] == ["0x1", "0x2"]


def test_no_tool_call_returns_empty():
    assert parse_gemma_tool_calls("I cannot help with that.") == []


def test_bare_unquoted_value():
    raw = "<|tool_call>call:readTx{chainId:1,to:0xabc}<tool_call|>"
    calls = parse_gemma_tool_calls(raw)
    _, fields = calls[0]
    assert fields == {"chainId": "1", "to": "0xabc"}
