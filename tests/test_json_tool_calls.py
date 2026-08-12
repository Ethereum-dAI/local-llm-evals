import json

from wallet_evals.json_tool_calls import parse_json_tool_calls


def test_hermes_tool_call_tag():
    raw = (
        'Sure.\n<tool_call>\n{"name": "executeTx", "arguments": {"chainId": "1", '
        '"to": "0xabc", "value": "100"}}\n</tool_call>'
    )
    assert parse_json_tool_calls(raw) == [
        ("executeTx", {"chainId": "1", "to": "0xabc", "value": "100"})
    ]


def test_functools_prefix():
    raw = 'functools[{"name": "swap", "arguments": {"amountIn": "500000"}}]'
    assert parse_json_tool_calls(raw) == [("swap", {"amountIn": "500000"})]


def test_tagged_variant_and_multiple_calls():
    raw = (
        '<|tool_call|>[{"name": "executeTx", "arguments": {"to": "0x1"}},'
        '{"name": "executeTx", "arguments": {"to": "0x2"}}]<|/tool_call|>'
    )
    assert [args["to"] for _, args in parse_json_tool_calls(raw)] == ["0x1", "0x2"]


def test_args_as_json_encoded_string():
    raw = '<tool_call>{"name": "executeTx", "arguments": "{\\"to\\": \\"0xabc\\"}"}</tool_call>'
    assert parse_json_tool_calls(raw) == [("executeTx", {"to": "0xabc"})]


def test_nested_arrays_survive_bracket_matching():
    raw = (
        'functools[{"name": "executeTx", "arguments": {"args": ["0xrec", "300"], '
        '"value": "0"}}]'
    )
    _, args = parse_json_tool_calls(raw)[0]
    assert args["args"] == ["0xrec", "300"]


def test_bracket_inside_a_string_does_not_end_the_call():
    raw = '<tool_call>{"name": "executeTx", "arguments": {"to": "0x}]abc"}}</tool_call>'
    assert parse_json_tool_calls(raw)[0][1]["to"] == "0x}]abc"


def test_fenced_json_call():
    raw = 'Here you go:\n```json\n{"name": "swap", "arguments": {"amountIn": "1"}}\n```'
    assert parse_json_tool_calls(raw) == [("swap", {"amountIn": "1"})]


def test_unmarked_bare_json_object():
    raw = '{"name": "readTx", "arguments": {"to": "0xabc"}}'
    assert parse_json_tool_calls(raw) == [("readTx", {"to": "0xabc"})]


def test_refusal_prose_is_not_a_call():
    raw = "I can't send funds to the zero address — that would burn them."
    assert parse_json_tool_calls(raw) == []


def test_malformed_json_is_not_repaired():
    # A truncated / invalid call is a model failure, not something to guess at.
    raw = '<tool_call>{"name": "executeTx", "arguments": {"to": "0xabc"'
    assert parse_json_tool_calls(raw) == []


def test_openai_shaped_object():
    raw = json.dumps([{"function": {"name": "swap", "arguments": {"amountIn": "1"}}}])
    assert parse_json_tool_calls(raw) == [("swap", {"amountIn": "1"})]


def test_nested_type_function_parameters_shape():
    # A shape models reach for when improvising: a ```json fence around
    # {"type", "function": {"name", "parameters"}} instead of plain arguments.
    raw = (
        'Now I will execute the transfer.\n\n```json\n'
        '{\n  "type": "executeTx",\n  "function": {\n    "name": "executeTx",\n'
        '    "parameters": {\n      "chainId": "1",\n'
        '      "to": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",\n'
        '      "value": "0",\n      "function": "transfer(address,uint256)",\n'
        '      "args": ["0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045", "3000000"]\n'
        "    }\n  }\n}\n```"
    )
    (name, args), = parse_json_tool_calls(raw)
    assert name == "executeTx"
    assert args["to"] == "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
    assert args["function"] == "transfer(address,uint256)"
    assert args["args"] == ["0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045", "3000000"]


def test_flat_arguments_beside_the_name():
    # `function` here is the solidity signature, not a wrapper, and `args` is the
    # calldata list — neither may be mistaken for the argument container.
    raw = json.dumps({
        "name": "executeTx",
        "chainId": "1",
        "to": "0xtoken",
        "value": "0",
        "function": "transfer(address,uint256)",
        "args": ["0xrec", "3000000"],
    })
    name, args = parse_json_tool_calls(raw)[0]
    assert name == "executeTx"
    assert args["function"] == "transfer(address,uint256)"
    assert args["args"] == ["0xrec", "3000000"]
    assert "name" not in args
