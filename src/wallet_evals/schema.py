"""Pydantic models for the unified eval schema (spec §3)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_PREVIEW_WIDTH = 72

ToolName = Literal["executeTx", "readTx", "swap", "shield", "unshield", "transfer"]

# Tools whose arguments mirror the macOS app's own ToolDefinitions verbatim:
# a HUMAN-unit `amount` plus a token SYMBOL, not a base-unit payload.
PRIVACY_TOOLS = ("shield", "unshield")
HUMAN_UNIT_TOOLS = ("transfer", "swap", "shield", "unshield")

# Tools whose schema carries a `token` field. Swap names its sides with
# from_token/to_token instead, so `token` is not part of its schema.
TOKEN_TOOLS = ("transfer", "shield", "unshield")


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
    # Optional: executeTx/readTx (the base-unit protocol contract) carry it, the
    # app tools do NOT — wallet-macos's ToolDefinitions declares no chainId on
    # transfer/swap and never reads one; chain comes from activeChain.id.
    #
    # `_call_matches` skips the check whenever gold omits it, which is right for
    # the app tools and wrong for the builder ones: an executeTx gold that lost
    # its chainId would silently stop being checked rather than fail. The
    # validator below closes that — the field is optional by TOOL, not by
    # accident.
    chainId: str | None = None
    to: str | None = None
    value: str = "0"
    function: str | None = None
    args: list[Any] = Field(default_factory=list)
    # Swap intent fields (None for executeTx/readTx).
    currencyIn: str | None = None
    currencyOut: str | None = None
    amountIn: str | None = None
    amountOutMinimum: str | None = None
    recipient: str | None = None
    # RAILGUN privacy fields (None for every other tool). `amount` is human units.
    amount: str | None = None
    token: str | None = None
    # App-contract swap fields (None for every other tool). Symbols, not addresses.
    from_token: str | None = None
    to_token: str | None = None
    amount_side: str | None = None

    @model_validator(mode="after")
    def _builder_tools_must_carry_a_chain(self) -> "ExpectedCall":
        """`executeTx`/`readTx` gold must state its chainId.

        Gold is computed, so a missing one means a builder changed, not that a
        case is unusual — and because the matcher skips the comparison whenever
        gold omits the field, the failure mode is a check that quietly stops
        happening rather than a test that goes red.
        """
        if self.tool in ("executeTx", "readTx") and self.chainId is None:
            raise ValueError(
                f"{self.tool} gold must carry a chainId — without it the scorer "
                "skips the chain check entirely instead of failing"
            )
        return self

    def as_parsed_call(self) -> "ParsedToolCall":
        """This gold call as if a model had emitted it — used to assert that every
        gold self-scores to 1. A straight field copy: it must add no defaulting of
        its own, or it would mask a scorer bug."""
        return ParsedToolCall(name=self.tool, **self.model_dump(exclude={"tool"}))

    def format_preview(self, ctx: PreviewContext | None = None) -> str:
        ctx = ctx or PreviewContext()
        if self.tool in PRIVACY_TOOLS:
            line = (f"  expected call #{ctx.call_index + 1}: {self.tool} "
                    f"(chainId={self.chainId}) amount={self.amount} token={self.token}")
            return line if self.to is None else f"{line} to={self.to}"
        if self.tool == "transfer":
            return (f"  expected call #{ctx.call_index + 1}: transfer "
                    f"(chainId={self.chainId}) amount={self.amount} "
                    f"token={self.token} to={self.to}")
        if self.tool == "swap":
            return (f"  expected call #{ctx.call_index + 1}: swap "
                    f"(chainId={self.chainId}) amount={self.amount} "
                    f"{self.from_token} -> {self.to_token}")
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
    # Swap intent fields (None for executeTx/readTx).
    currencyIn: str | None = None
    currencyOut: str | None = None
    amountIn: str | None = None
    amountOutMinimum: str | None = None
    recipient: str | None = None
    # RAILGUN privacy fields (None for every other tool). `amount` is human units.
    amount: str | None = None
    token: str | None = None
    # App-contract swap fields (None for every other tool). Symbols, not addresses.
    from_token: str | None = None
    to_token: str | None = None
    amount_side: str | None = None


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
