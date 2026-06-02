"""Pydantic models for the unified eval schema (spec §3)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

_PREVIEW_WIDTH = 72

ToolName = Literal["executeTx", "readTx"]


class PreviewContext(BaseModel):
    """Optional context passed through the preview hierarchy."""

    model_config = ConfigDict(frozen=True)

    call_index: int = 0
    source: Path | str = ""


class Previewable(BaseModel, ABC):
    """Models that can render a human-readable preview string."""

    @abstractmethod
    def format_preview(self, ctx: PreviewContext | None = None) -> str: ...


class ExpectedCall(Previewable):
    """One gold on-chain call in a case's expected_calls sequence."""

    tool: ToolName
    chainId: str
    to: str
    value: str = "0"
    function: str | None = None
    args: list[Any] = Field(default_factory=list)

    def format_preview(self, ctx: PreviewContext | None = None) -> str:
        ctx = ctx or PreviewContext()
        lines = [
            f"  expected call #{ctx.call_index + 1}: {self.tool} -> {self.to}",
            f"      chainId={self.chainId} value={self.value} function={self.function}",
            f"      args={self.args}",
        ]
        return "\n".join(lines)


class Case(Previewable):
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

    def format_preview(self, ctx: PreviewContext | None = None) -> str:
        lines = [
            f"\n[{self.id}]  level={self.level}  protocol={self.protocol}",
            f"  user: {self.user_message}",
        ]
        if not self.expected_calls:
            lines.append("  expected: (no tool call)")
        else:
            lines.extend(
                call.format_preview(PreviewContext(call_index=i))
                for i, call in enumerate(self.expected_calls)
            )
        return "\n".join(lines)


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


class Dataset(Previewable):
    schema_: str = Field(alias="schema", default="evals-local-llm/cases/v1")
    cases: list[Case]

    def format_preview(self, ctx: PreviewContext | None = None) -> str:
        ctx = ctx or PreviewContext()
        source = ctx.source or "<unknown>"
        sep = "=" * _PREVIEW_WIDTH
        lines = [
            sep,
            f"DATASET: {source}  ({len(self.cases)} cases)",
            sep,
        ]
        lines.extend(case.format_preview() for case in self.cases)
        return "\n".join(lines)
