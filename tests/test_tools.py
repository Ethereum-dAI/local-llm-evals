from wallet_evals.tools import SYSTEM_PROMPT, TOOLS, tool_names


def test_system_prompt_matches_production_prefix():
    assert SYSTEM_PROMPT.startswith("You are the local AI inside a macOS Ethereum wallet app.")
    assert "you MUST call the corresponding tool" in SYSTEM_PROMPT
    assert "Never invent recipient addresses" in SYSTEM_PROMPT


def test_tools_are_execute_read_and_swap():
    assert tool_names() == ["executeTx", "readTx", "swap"]


def test_executeTx_schema_shape():
    execute = next(t for t in TOOLS if t["function"]["name"] == "executeTx")
    props = execute["function"]["parameters"]["properties"]
    assert set(["chainId", "to", "value", "function", "args"]).issubset(props.keys())
    assert props["args"]["type"] == "array"
    assert set(execute["function"]["parameters"]["required"]) == {"chainId", "to", "args"}
