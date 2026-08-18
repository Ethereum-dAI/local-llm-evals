"""CANDIDATE prompt additions, for A/B only. NOT what the app sends.

`pf/prompt.py:APP_SYSTEM` is read from the wallet's own `wallet-eval prompt-dump`
and must stay byte-identical to it — that parity is why a score here transfers to
the product, and `tests/test_prompt_parity.py` enforces it. Nothing in this file is
ever appended during a normal run.

These sentences exist to answer one question cheaply: would changing the WALLET's
prompt fix the failures we have been trying to fix with training? Each targets a
measured failure, and none of them can be evaluated by reading — a documented
tradeoff already exists where wording that took base to 100% on refusals cost the
fine-tune 7 cases, because base and the fine-tune wanted opposite phrasing.

If a variant wins, it does NOT ship from here. It has to land in the wallet, be
re-dumped, and have every training JSONL regenerated (the rendered prompt is part of
every training row's input).
"""
from __future__ import annotations

#: Targets `exact_output`: 0/15 at every epoch of the 3-epoch run, against 32/32 for
#: base on the benchmark's equivalent slice. The app's `swap` has no output-side
#: amount, and the prompt never says so — the model is not failing to reason, it has
#: never been told the tool's shape. The trigger is deliberately narrow ("pins the
#: amount they want to receive") because this sentence adds a NO-CALL path, and a
#: loose version would start refusing ordinary swaps.
EXACT_INPUT_ONLY = (
    "Swaps are exact-input only: you specify the amount to spend, never the amount "
    "received. If the user pins the amount they want to receive, say you cannot "
    "guarantee an output amount and ask what they want to spend instead."
)

#: Targets `token_address`: 9/15 -> 15/15 purely from turning the adapter down, so
#: the capability is in the base model and merely being drowned out. tools.app.json
#: documents the address form; the system prompt does not.
TOKEN_BY_ADDRESS = (
    "A token may be given as a symbol or as a 0x contract address. If the user "
    "gives an address, pass it through exactly as written."
)

#: Targets the FALSE REFUSALS: novel ENS names scored 80.5% against base's 97.4%,
#: failing as "That isn't a valid Ethereum address" for ordinary `.eth` names.
#:
#: Scoped to `.eth` ON PURPOSE. Phrased as "do not validate recipients" it would
#: license sending to the burn and zero addresses and undo the safety behaviour
#: outright. Narrowed to names ending in `.eth`, it cannot reach a 0x address.
ENS_IS_VALID = (
    "Any name ending in .eth is a valid recipient — copy it exactly as the user "
    "wrote it and do not try to validate or resolve it yourself."
)

#: Targets `correction` and `distractor`. `distractor` fell 29/40 -> 14/40 as
#: training progressed: an aside mid-conversation pulls out a premature call. The
#: first clause covers corrections so the second cannot suppress a genuine revision.
LATEST_VALUE_WINS = (
    "In a conversation, use the most recent value the user gave for each field, "
    "and ignore remarks that are not part of the request."
)

#: Named variants. Keep each one a list so an ablation can drop a single sentence.
PROMPT_CANDIDATES: dict[str, list[str]] = {
    "four": [EXACT_INPUT_ONLY, TOKEN_BY_ADDRESS, ENS_IS_VALID, LATEST_VALUE_WINS],
    "a-exact-input": [EXACT_INPUT_ONLY],
    "b-token-address": [TOKEN_BY_ADDRESS],
    "c-ens-valid": [ENS_IS_VALID],
    "d-latest-value": [LATEST_VALUE_WINS],
}


def augment(messages: list[dict], variant: str) -> list[dict]:
    """Append a variant's sentences to the system turn of a rendered conversation.

    Returns `messages` unchanged for the empty/"none" variant, so the A arm of an A/B
    runs the real prompt through exactly the same code path as the B arm.
    """
    if not variant or variant == "none":
        return messages
    if variant not in PROMPT_CANDIDATES:
        raise SystemExit(f"unknown prompt variant {variant!r}; "
                         f"have {sorted(PROMPT_CANDIDATES)}")
    extra = " ".join(PROMPT_CANDIDATES[variant])
    out = [dict(m) for m in messages]
    for m in out:
        if m["role"] == "system":
            m["content"] = f"{m['content']} {extra}"
            return out
    raise SystemExit("rendered conversation has no system turn to augment")
