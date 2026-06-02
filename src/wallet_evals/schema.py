"""Pydantic models for the unified eval schema (spec §3)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ToolName = Literal["executeTx", "readTx"]


class ExpectedCall(BaseModel):
    """One gold on-chain call in a case's expected_calls sequence."""

    tool: ToolName
    chainId: str
    to: str
    value: str = "0"
    function: str | None = None
    args: list[Any] = Field(default_factory=list)


class Case(BaseModel):
    """A single eval case (intent or payload level)."""

    id: str
    user_message: str
    level: Literal["intent", "payload"]
    language: str
    category: str
    query_type: str | None = None
    protocol: str
    difficulty: str
    requires: list[str] = Field(default_factory=list)
    expected_calls: list[ExpectedCall] = Field(default_factory=list)
    notes: str | None = None


class ParsedToolCall(BaseModel):
    """A tool call extracted and normalized from a model turn."""

    name: str
    chainId: str | None = None
    to: str | None = None
    value: str | None = None
    function: str | None = None
    args: list[Any] = Field(default_factory=list)


class ParsedTurn(BaseModel):
    """The normalized result of one model response."""

    content: str | None = None
    tool_calls: list[ParsedToolCall] = Field(default_factory=list)


class Dataset(BaseModel):
    schema_: str = Field(alias="schema", default="evals-local-llm/cases/v1")
    cases: list[Case]
