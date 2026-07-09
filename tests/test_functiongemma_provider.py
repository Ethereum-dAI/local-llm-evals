"""Pure logic behind the FunctionGemma promptfoo provider (no model load).

The provider itself is thin glue over llama-cpp-python; everything that can be
tested offline lives in wallet_evals.functiongemma.
"""
import json

from wallet_evals.functiongemma import decode_prompt, raw_output_to_scoreable


# --- raw_output_to_scoreable: FunctionGemma text -> what assert.py can score ---

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


# --- decode_prompt: promptfoo prompt string -> messages with developer role ---

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
