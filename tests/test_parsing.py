from wallet_evals.parsing import parse_turn


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


def test_malformed_args_json_falls_back_to_empty_list():
    native = [{"name": "executeTx", "arguments": '{"chainId":"1","to":"0xABC","args":"not-json"}'}]
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
