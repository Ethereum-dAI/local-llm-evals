from __future__ import annotations

import importlib
import json

from scripts.generate_finetune_data import _reasoning_text
from wallet_evals.functiongemma import decode_prompt, raw_output_to_scoreable
from wallet_evals.functiongemma import raw_output_to_scoreable
from wallet_evals.gemma_dsl import FUNCTIONGEMMA, parse_gemma_tool_calls
from wallet_evals.gemma_dsl import parse_gemma_tool_calls
from wallet_evals.intents import LOOKUP
from wallet_evals.json_tool_calls import parse_json_tool_calls
from wallet_evals.parsing import UndecodableArgs, parse_turn
from wallet_evals.schema import Case, ExpectedCall
from wallet_evals.scorer import score_case

# ============================================================================
# test_functiongemma_dsl
# ============================================================================
#
# FunctionGemma-270m emits the same call:NAME{...} DSL family as the shipped
# on-device Gemma 4, but with different delimiters:
#
#     <start_function_call>call:NAME{key:<escape>value<escape>,...}<end_function_call>
#
# (see https://ai.google.dev/gemma/docs/functiongemma/function-calling-with-hf).
# These tests pin the FUNCTIONGEMMA dialect through the shared parser.

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

# ============================================================================
# test_gemma_dsl
# ============================================================================
#
# Suffixed on merge, to clear a name this file shared with an
# earlier section: test_parses_single_quoted_call -> test_parses_single_quoted_call_gemma_dsl, test_parses_args_as_json_string_value -> test_parses_args_as_json_string_value_gemma_dsl, test_parses_multiple_calls -> test_parses_multiple_calls_gemma_dsl, test_no_tool_call_returns_empty -> test_no_tool_call_returns_empty_gemma_dsl, test_bare_unquoted_value -> test_bare_unquoted_value_gemma_dsl.

def test_parses_single_quoted_call_gemma_dsl():
    raw = '<|tool_call>call:executeTx{chainId:<|"|>1<|"|>,to:<|"|>0xabc<|"|>,value:<|"|>0<|"|>}<tool_call|>'
    calls = parse_gemma_tool_calls(raw)
    assert len(calls) == 1
    name, fields = calls[0]
    assert name == "executeTx"
    assert fields == {"chainId": "1", "to": "0xabc", "value": "0"}


def test_parses_args_as_json_string_value_gemma_dsl():
    raw = '<|tool_call>call:executeTx{chainId:<|"|>1<|"|>,args:<|"|>["0xspender","100"]<|"|>}<tool_call|>'
    calls = parse_gemma_tool_calls(raw)
    name, fields = calls[0]
    assert fields["args"] == '["0xspender","100"]'


def test_parses_multiple_calls_gemma_dsl():
    raw = (
        '<|tool_call>call:executeTx{to:<|"|>0x1<|"|>}<tool_call|>'
        '<|tool_call>call:executeTx{to:<|"|>0x2<|"|>}<tool_call|>'
    )
    calls = parse_gemma_tool_calls(raw)
    assert [f["to"] for _, f in calls] == ["0x1", "0x2"]


def test_no_tool_call_returns_empty_gemma_dsl():
    assert parse_gemma_tool_calls("I cannot help with that.") == []


def test_bare_unquoted_value_gemma_dsl():
    raw = "<|tool_call>call:readTx{chainId:1,to:0xabc}<tool_call|>"
    calls = parse_gemma_tool_calls(raw)
    _, fields = calls[0]
    assert fields == {"chainId": "1", "to": "0xabc"}

# ============================================================================
# test_functiongemma_provider
# ============================================================================
#
# Pure logic behind the FunctionGemma promptfoo provider (no model load).
#
# The provider itself is thin glue over llama-cpp-python; everything that can be
# tested offline lives in wallet_evals.functiongemma.

def test_tool_call_becomes_openai_shaped_json_string():
    raw = (
        "<start_function_call>call:executeTx{"
        "chainId:<escape>1<escape>,to:<escape>0xabc<escape>,value:<escape>0<escape>"
        "}<end_function_call>"
    )
    out = raw_output_to_scoreable(raw)
    calls = json.loads(out)
    assert len(calls) == 1
    assert calls[0]["name"] == "executeTx"
    assert json.loads(calls[0]["arguments"]) == {"chainId": "1", "to": "0xabc", "value": "0"}


def test_args_array_survives_as_json_string_argument():
    raw = (
        "<start_function_call>call:executeTx{"
        'to:<escape>0xtoken<escape>,function:<escape>transfer(address,uint256)<escape>,'
        'args:<escape>["0xrecipient","1000000"]<escape>'
        "}<end_function_call>"
    )
    calls = json.loads(raw_output_to_scoreable(raw))
    args_obj = json.loads(calls[0]["arguments"])
    # args stays a JSON string; assert.py's _coerce_args decodes it downstream.
    assert json.loads(args_obj["args"]) == ["0xrecipient", "1000000"]


def test_prose_without_tool_call_returned_verbatim():
    """A refusal / clarifying question has no call -> pass the text through so
    refusal cases (expected_calls == []) score and the reason surfaces it."""
    raw = "I can't send funds to a burn address."
    assert raw_output_to_scoreable(raw) == raw


def test_decodes_json_conversation_and_remaps_system_to_developer():
    prompt = json.dumps([
        {"role": "system", "content": "wallet manual"},
        {"role": "user", "content": "send 1 eth"},
    ])
    msgs = decode_prompt(prompt)
    assert msgs[0] == {"role": "developer", "content": "wallet manual"}
    assert msgs[1] == {"role": "user", "content": "send 1 eth"}


def test_plain_string_prompt_becomes_single_user_message():
    msgs = decode_prompt("send 1 eth to vitalik.eth")
    assert msgs == [{"role": "user", "content": "send 1 eth to vitalik.eth"}]

# ============================================================================
# test_functiongemma_scoring
# ============================================================================
#
# End-to-end: a FunctionGemma raw output, run through the provider's
# translation, must score through the UNCHANGED pf/assert.py scorer. This is the
# claim the whole integration rests on — assert.py, tools.json, and the dataset
# stay untouched.

get_assert = importlib.import_module("pf.assert").get_assert  # 'assert' is a keyword


_BASE = {  # fields the Case model requires but that don't affect scoring here
    "level": "payload",
    "language": "english",
    "category": "generated",
    "protocol": "uniswap",
    "difficulty": "medium",
}


def _score(raw: str, metadata: dict) -> dict:
    output = raw_output_to_scoreable(raw)
    ctx = {"test": {"metadata": {**_BASE, **metadata}}, "providerResponse": {"output": output}}
    return get_assert(output, ctx)


def test_correct_swap_call_scores_pass():
    metadata = {
        "id": "gen-swap-pos-0176",
        "expected_calls": [{
            "tool": "swap",
            "chainId": "1",
            "currencyIn": "0x0000000000000000000000000000000000000000",
            "currencyOut": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            "amountIn": "987654320000000000000000",
            "amountOutMinimum": "0",
            "recipient": "<wallet>",
        }],
    }
    raw = (
        "<start_function_call>call:swap{"
        "chainId:<escape>1<escape>,"
        "currencyIn:<escape>0x0000000000000000000000000000000000000000<escape>,"
        "currencyOut:<escape>0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2<escape>,"
        "amountIn:<escape>987654320000000000000000<escape>,"
        "amountOutMinimum:<escape>0<escape>,"
        "recipient:<escape><wallet><escape>"
        "}<end_function_call>"
    )
    assert _score(raw, metadata)["pass"] is True


def test_erc20_transfer_with_args_array_scores_pass():
    metadata = {
        "id": "fg-erc20",
        "expected_calls": [{
            "tool": "executeTx",
            "chainId": "1",
            "to": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
            "value": "0",
            "function": "transfer(address,uint256)",
            "args": ["0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045", "3000000"],
        }],
    }
    raw = (
        "<start_function_call>call:executeTx{"
        "chainId:<escape>1<escape>,"
        "to:<escape>0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48<escape>,"
        "value:<escape>0<escape>,"
        "function:<escape>transfer(address,uint256)<escape>,"
        'args:<escape>["0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045","3000000"]<escape>'
        "}<end_function_call>"
    )
    assert _score(raw, metadata)["pass"] is True


def test_prose_refusal_scores_pass_on_refusal_case():
    metadata = {"id": "gen-refusal-0001", "expected_calls": []}
    raw = "I can't send funds to a burn address — that would destroy them."
    assert _score(raw, metadata)["pass"] is True

# ============================================================================
# test_json_tool_calls
# ============================================================================

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

# ============================================================================
# test_parsing
# ============================================================================

def test_native_tool_calls():
    native = [
        {
            "name": "executeTx",
            "arguments": '{"chainId":"1","to":"0xABC","value":"0","function":"approve(address,uint256)","args":["0xspender","100"]}',
        }
    ]
    turn = parse_turn(content=None, native_tool_calls=native, raw_text="")
    assert len(turn.tool_calls) == 1
    call = turn.tool_calls[0]
    assert call.name == "executeTx"
    assert call.chainId == "1"
    assert call.function == "approve(address,uint256)"
    assert call.args == ["0xspender", "100"]


def test_native_with_nested_tuple_args():
    native = [{"name": "executeTx", "arguments": '{"chainId":"1","to":"0xR","args":[["0xa","0xb","3000"]]}'}]
    turn = parse_turn(content=None, native_tool_calls=native, raw_text="")
    assert turn.tool_calls[0].args == [["0xa", "0xb", "3000"]]


def test_dsl_fallback_decodes_args_json():
    raw = '<|tool_call>call:executeTx{chainId:<|"|>1<|"|>,to:<|"|>0xABC<|"|>,args:<|"|>["0xspender","100"]<|"|>}<tool_call|>'
    turn = parse_turn(content=None, native_tool_calls=None, raw_text=raw)
    call = turn.tool_calls[0]
    assert call.to == "0xABC"
    assert call.args == ["0xspender", "100"]


def test_no_call_returns_empty_tool_calls():
    turn = parse_turn(content="I need more info.", native_tool_calls=None, raw_text="I need more info.")
    assert turn.tool_calls == []
    assert turn.content == "I need more info."


def test_undecodable_args_are_reported_not_silently_emptied():
    """A parser miss must not look like "the model emitted no args".

    Those two were the same value ([]) until the element-quoted Gemma array
    showed what it costs: 130 protocol cases where the base model answered
    correctly were reported as `args: expected [...] got []`, indistinguishable
    from no answer at all, and the one-directional loss landed entirely on the
    baseline the fine-tune was being compared against.
    """
    native = [{"name": "executeTx", "arguments": '{"chainId":"1","to":"0xABC","args":"not-json"}'}]
    turn = parse_turn(content=None, native_tool_calls=native, raw_text="")
    args = turn.tool_calls[0].args

    assert args != [], "an undecodable payload must not read as an empty list"
    assert len(args) == 1 and isinstance(args[0], UndecodableArgs)
    assert "not-json" in repr(args[0])


def test_genuinely_absent_args_are_still_an_empty_list():
    native = [{"name": "executeTx", "arguments": '{"chainId":"1","to":"0xABC","args":""}'}]
    turn = parse_turn(content=None, native_tool_calls=native, raw_text="")
    assert turn.tool_calls[0].args == []


def test_string_null_function_normalized_to_none():
    native = [{"name": "executeTx", "arguments": '{"chainId":"1","to":"0xabc","value":"100","function":"null","args":[]}'}]
    turn = parse_turn(content=None, native_tool_calls=native, raw_text="")
    assert turn.tool_calls[0].function is None


def test_empty_string_function_normalized_to_none():
    native = [{"name": "executeTx", "arguments": '{"chainId":"1","to":"0xabc","value":"100","function":"","args":[]}'}]
    turn = parse_turn(content=None, native_tool_calls=native, raw_text="")
    assert turn.tool_calls[0].function is None


def test_real_function_preserved():
    native = [{"name": "executeTx", "arguments": '{"chainId":"1","to":"0xabc","function":"transfer(address,uint256)","args":["0xdef","1"]}'}]
    turn = parse_turn(content=None, native_tool_calls=native, raw_text="")
    assert turn.tool_calls[0].function == "transfer(address,uint256)"


def test_app_swap_fields_round_trip():
    # A model emitting the app's actual swap contract (from_token/to_token/
    # amount_side, no currencyIn/currencyOut/amountIn) must have those fields
    # survive parsing — previously _build_call dropped all three, so every real
    # app-swap eval run would score 0 regardless of what the model emitted.
    native = [{
        "name": "swap",
        "arguments": '{"chainId":"1","from_token":"USDC","to_token":"ETH",'
                     '"amount":"100","amount_side":"input"}',
    }]
    turn = parse_turn(content=None, native_tool_calls=native, raw_text="")
    call = turn.tool_calls[0]
    assert call.from_token == "USDC"
    assert call.to_token == "ETH"
    assert call.amount_side == "input"

    case = Case(
        id="t", user_message="x", level="payload", language="english",
        category="truePositiveSwap", query_type="one_shot", protocol="uniswap",
        difficulty="easy", requires=[],
        expected_calls=[ExpectedCall(
            tool="swap", chainId="1", amount="100",
            from_token="USDC", to_token="ETH", amount_side="input",
        )],
        notes=None,
    )
    assert score_case(case, turn) == 1

# ============================================================================
# test_reasoning_trace
# ============================================================================
#
# Unit tests for the rewritten `<think>` trace builder (Step 0 of the
# app-contract fine-tune regeneration).
#
# The old `_reasoning_text` computed a base-unit derivation ("USDC has 6
# decimals, so 0.000002 USDC = 2 base units...") that would directly contradict
# an app-contract target, which emits the HUMAN-unit amount verbatim. These
# tests assert the rewritten trace only ever talks about the human-unit call the
# app contract actually takes, for both families it now covers
# (transfer/swap), and that the deterministic separator-note path fires
# correctly for the `separator` bucket.

_FORBIDDEN = ("base units", "wei", "decimals")


_TOKEN_CONTRACT_ADDRESSES = [
    meta["address"] for meta in LOOKUP["tokens"].values() if meta.get("address")
]


def _assert_app_contract_shape(trace: str, amount: str) -> None:
    assert amount in trace, f"trace must carry the human-unit amount: {trace!r}"
    lowered = trace.lower()
    for phrase in _FORBIDDEN:
        assert phrase not in lowered, f"trace must not mention {phrase!r}: {trace!r}"
    for addr in _TOKEN_CONTRACT_ADDRESSES:
        assert addr not in trace, f"trace must not name a token contract address: {trace!r}"


def test_transfer_trace_is_app_contract_shaped():
    intent = {"action": "transfer", "amount": "42.5", "token": "USDC",
              "recipient": "vitalik.eth"}
    trace = _reasoning_text(intent)
    _assert_app_contract_shape(trace, "42.5")
    assert "transfer" in trace
    assert "vitalik.eth" in trace  # recipient copied verbatim, not resolved


def test_swap_trace_is_app_contract_shaped():
    intent = {"action": "swap", "amount": "7.25", "from_token": "USDC",
              "to_token": "ETH"}
    trace = _reasoning_text(intent)
    _assert_app_contract_shape(trace, "7.25")
    assert "swap" in trace
    assert 'amount_side "input"' in trace or "amount_side" in trace


def test_transfer_trace_never_mentions_erc20_contract_address():
    """The old trace's whole ERC-20 branch named `meta['address']`; the new one
    must never resurrect it for a non-native token like USDC/DAI/WETH."""
    for tok in ("USDC", "DAI", "WETH"):
        intent = {"action": "transfer", "amount": "1.23", "token": tok,
                  "recipient": "vitalik.eth"}
        trace = _reasoning_text(intent)
        assert LOOKUP["tokens"][tok]["address"] not in trace


def test_separator_note_names_the_grouping_and_states_plain_decimal():
    """When `surface_amount` differs from `amount` (the `separator` bucket),
    the trace must name the separator and state the plain decimal — the exact
    failure mode Step 2 exists to close."""
    intent = {"action": "transfer", "amount": "20000", "token": "ETH",
              "recipient": "vitalik.eth", "surface_amount": "20,000"}
    trace = _reasoning_text(intent)
    assert "20,000" in trace  # names the comma-grouped surface form
    assert "20000" in trace  # states the plain decimal
    _assert_app_contract_shape(trace, "20000")


def test_no_separator_note_when_surface_amount_matches_amount():
    """A normal (non-separator) case must not spuriously claim a separator was
    present when the surface never carried one."""
    intent = {"action": "transfer", "amount": "5", "token": "ETH",
              "recipient": "vitalik.eth"}
    trace = _reasoning_text(intent)
    assert "separator" not in trace.lower()
    assert "strip" not in trace.lower()  # the separator-stripping instruction is absent


def test_reasoning_text_deterministic():
    intent = {"action": "swap", "amount": "3.5", "from_token": "ETH", "to_token": "DAI"}
    assert _reasoning_text(intent) == _reasoning_text(dict(intent))
