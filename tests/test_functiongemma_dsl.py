"""FunctionGemma-270m emits the same call:NAME{...} DSL family as the shipped
on-device Gemma 4, but with different delimiters:

    <start_function_call>call:NAME{key:<escape>value<escape>,...}<end_function_call>

(see https://ai.google.dev/gemma/docs/functiongemma/function-calling-with-hf).
These tests pin the FUNCTIONGEMMA dialect through the shared parser.
"""
from wallet_evals.gemma_dsl import FUNCTIONGEMMA, parse_gemma_tool_calls


def test_parses_single_quoted_call():
    raw = (
        "<start_function_call>call:executeTx{"
        "chainId:<escape>1<escape>,to:<escape>0xabc<escape>,value:<escape>0<escape>"
        "}<end_function_call>"
    )
    calls = parse_gemma_tool_calls(raw, FUNCTIONGEMMA)
    assert len(calls) == 1
    name, fields = calls[0]
    assert name == "executeTx"
    assert fields == {"chainId": "1", "to": "0xabc", "value": "0"}


def test_parses_args_as_json_string_value():
    raw = (
        "<start_function_call>call:executeTx{"
        'chainId:<escape>1<escape>,args:<escape>["0xspender","100"]<escape>'
        "}<end_function_call>"
    )
    calls = parse_gemma_tool_calls(raw, FUNCTIONGEMMA)
    _, fields = calls[0]
    assert fields["args"] == '["0xspender","100"]'


def test_parses_multiple_calls():
    raw = (
        "<start_function_call>call:executeTx{to:<escape>0x1<escape>}<end_function_call>"
        "<start_function_call>call:executeTx{to:<escape>0x2<escape>}<end_function_call>"
    )
    calls = parse_gemma_tool_calls(raw, FUNCTIONGEMMA)
    assert [f["to"] for _, f in calls] == ["0x1", "0x2"]


def test_no_tool_call_returns_empty():
    assert parse_gemma_tool_calls("I cannot help with that.", FUNCTIONGEMMA) == []


def test_bare_unquoted_value():
    raw = "<start_function_call>call:readTx{chainId:1,to:0xabc}<end_function_call>"
    calls = parse_gemma_tool_calls(raw, FUNCTIONGEMMA)
    _, fields = calls[0]
    assert fields == {"chainId": "1", "to": "0xabc"}


def test_bare_json_array_with_internal_commas_kept_whole():
    """The 270M model emits `args` as a BARE array (no <escape> wrapping); a
    multi-element array has internal commas that must NOT split the value."""
    raw = (
        "<start_function_call>call:executeTx{"
        'to:<escape>0xtoken<escape>,args:["0xrecipient","1000000"],value:<escape>0<escape>'
        "}<end_function_call>"
    )
    calls = parse_gemma_tool_calls(raw, FUNCTIONGEMMA)
    _, fields = calls[0]
    assert fields["args"] == '["0xrecipient","1000000"]'
    assert fields["value"] == "0"


def test_bare_single_element_array():
    """The exact shape seen from the raw model: args:[0.1]."""
    raw = "<start_function_call>call:readTx{to:<escape>vitalik.eth<escape>,args:[0.1]}<end_function_call>"
    calls = parse_gemma_tool_calls(raw, FUNCTIONGEMMA)
    _, fields = calls[0]
    assert fields["args"] == "[0.1]"


def test_default_dialect_unchanged():
    """The default (no dialect arg) must still be the legacy on-device Gemma 4
    format, so parsing.py and the existing suite stay green."""
    raw = '<|tool_call>call:executeTx{to:<|"|>0xabc<|"|>}<tool_call|>'
    calls = parse_gemma_tool_calls(raw)
    assert calls[0][1]["to"] == "0xabc"
