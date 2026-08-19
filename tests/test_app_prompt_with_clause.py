"""The wallet's prompt AFTER `ToolDefinitions.safetyClause` landed, and its exact delta.

`pf/app_contract_reference.json` is the LIVE reference: it is what `APP_SYSTEM` is read
from, so every recorded number on this benchmark was measured against those bytes.
`pf/app_contract_reference.with_clause.json` is the dump taken from the wallet branch
that adds the clause (`wallet-eval prompt-dump`), kept here so the app's new bytes are
in this repo rather than only in the other one.

**It is deliberately NOT wired into any run.** Swapping it in would make
`PROMPT_VARIANT=none` mean "clause on", silently changing what every existing config
measures, and it carries a third tool that `pf/tools.app.json` does not offer. The
decision to re-baseline is a separate, explicit one; these tests pin the relationship in
the meantime so neither file can drift unnoticed.

What they establish, and why each matters:

  * the clause in the app's dump is byte-identical to `SAFETY_FULL` — so the clause-on
    numbers in results/ describe the string the app now sends, not a near-miss;
  * it is appended as a suffix joined by ONE space — the concatenation `augment()`
    produces, and the one the models were scored on;
  * the ONLY other change from the live reference is the pre-existing
    `top_up_bundler` sentence, which is exercised by 0 of the 1000 cases. That bounds
    the re-measurement question: the prompt delta that matters is the clause, which is
    already measured on three models.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from pf.prompt import APP_SYSTEM
from pf.prompt_candidates import SAFETY_FULL

ROOT = Path(__file__).resolve().parents[1]
WITH_CLAUSE = json.loads((ROOT / "pf" / "app_contract_reference.with_clause.json").read_text())

#: The system turn the clause-on arms were scored on: APP_SYSTEM + " " + SAFETY_FULL.
MEASURED_SYSTEM_CHARS = 2110


def test_the_clause_in_the_app_dump_is_byte_identical() -> None:
    assert SAFETY_FULL in WITH_CLAUSE["systemPrompt"]


def test_the_clause_is_a_suffix_joined_by_one_space() -> None:
    system = WITH_CLAUSE["systemPrompt"]
    assert system.endswith(SAFETY_FULL)
    assert system.endswith(f" {SAFETY_FULL}")
    # ...and not by a newline, which would be a different string from the measured one.
    assert not system.endswith(f"\n{SAFETY_FULL}")


def test_the_only_other_delta_is_the_pre_existing_top_up_sentence() -> None:
    """If this fails, the app's prompt changed in some way BEYOND the clause, and the
    recorded numbers no longer bound the difference — re-measure rather than reasoning
    about it."""
    prefix = WITH_CLAUSE["systemPrompt"][: -len(SAFETY_FULL) - 1]
    assert prefix.startswith(APP_SYSTEM)
    extra = prefix[len(APP_SYSTEM):]
    assert extra == (
        "\nFor a bundler top-up, call top_up_bundler with only the requested amount; "
        "never invent or request a destination address."
    ), extra


def test_the_top_up_tool_is_exercised_by_no_case_in_the_frozen_set() -> None:
    """Which is what makes the delta above ignorable for now. The frozen 1000 are all
    transfer/swap/refusal, so a third tool in the prompt changes the tool list the model
    sees but nothing it is asked to do."""
    assert WITH_CLAUSE["toolNames"] == ["transfer", "swap", "top_up_bundler"]
    cases = yaml.safe_load((ROOT / "pf" / "tests.combined.yaml").read_text())
    tools_wanted = {
        call["tool"]
        for c in cases
        for call in (c["metadata"].get("expected_calls") or [])
    }
    assert tools_wanted == {"transfer", "swap"}, tools_wanted


def test_the_live_reference_still_has_no_clause() -> None:
    """The guard on the footgun: if someone swaps the with-clause dump in as the live
    reference, `PROMPT_VARIANT=none` starts meaning "clause on" and every recorded
    number silently describes a prompt it was not measured under."""
    assert SAFETY_FULL not in APP_SYSTEM
    assert len(APP_SYSTEM) == 533
    assert len(f"{APP_SYSTEM} {SAFETY_FULL}") == MEASURED_SYSTEM_CHARS
