# ACT_NOT_ASK on the base model — REJECTED

**A prompt clause cannot fix base's biggest failure bucket. The clause was read and
ignored, and on its own it costs safety.**

## The bucket it targeted

Base's 74 non-safety failures on the frozen 1000-case set decompose as:

| shape | count |
| --- | --- |
| **expected a call, made NONE** | **40** |
| wrong arguments | 30 |
| expected no call, made one | 4 |

All 40 begin with a `<|channel>thought` trace, and it is **not truncation** — median
completion 256 tokens, max 849, none within 175 tokens of the 1024 cap. The model reasons
to the correct answer and then asks a question:

- "I think you want to swap 987654.32 USDC for DAI. Is that correct?"
- "What token are you referring to when you say 'iT'?"  (a deliberately typo'd surface)
- "How much ETH would you like to trade for WETH?"      (output amount was given)

The same shape holds on the dev slice — 11 of 13 control failures, every one ending in a
question mark — which is what made it safe to design against dev rather than against the
frozen benchmark where the pattern was first noticed.

## Result: no effect on accuracy, negative on safety

| arm | dev accuracy (145) | refusals (30) |
| --- | --- | --- |
| `none` | 133/145 = 91.7% | 18/30 = 60.0% |
| `act` | 133/145 = 91.7% | **16/30 = 53.3%** |
| `safety+act` | 133/145 = 91.7% | 26/30 = 86.7% |

Accuracy is **identical to the case** across all three arms, with +3/−3 and +5/−5 flips.
`act` alone broke two refusals (`wrong-chain-address`, `prompt-injection`) and fixed none
— the "this never overrides a refusal" carve-out was not enough to stop a
bias-toward-acting from leaking into cases that needed refusing. `safety+act` simply
reproduces safety-full, so `act` contributes nothing it does not cost.

## The clause was genuinely read

Checked rather than assumed, because "identical scores" is exactly what a silently
unapplied variant looks like:

| arm | median prompt tokens |
| --- | --- |
| `none` | 799 |
| `act` | 902 |
| `safety+act` | 1300 |

And the targeted behaviour is untouched: **9 no-call failures in every arm, and in the
`act` arm all 9 still end in a question mark.** The model read the instruction and asked
anyway.

## Two consequences beyond this clause

**1. The dev slice has an accuracy noise floor of about ±5 cases (±3.5 points).** Three
arms with materially different prompts all scored exactly 133/145 while individually
flipping up to 5 cases each way. That retroactively downgrades the safety A/B's *accuracy*
claim: `safety-full`'s +7/−2 (net +5) sits inside this envelope, so it is **not** an
established accuracy gain. What survives is the weaker but sufficient claim that it does
not COST accuracy — plus its refusal gain, which is one-directional (+9/−0, +8/−0) and
therefore not noise.

**2. Safety-full replicated on a second device.** This run was on the rented GPU and gave
86.7% refusals at +8/−0, against 86.7% at +9/−0 locally. That is the same conclusion from
different hardware, which is stronger evidence than either run alone.

## Where that leaves the prompt-only path

The safety clause is worth roughly +10 of base's 19 test-set safety failures, so about
90.7% -> ~92% overall. The 40-case bucket is the only one big enough to reach 95%, and it
does not respond to wording. Remaining hypothesis: the reasoning pass itself produces the
hesitation, making the lever `enable_thinking` (an app SETTING, mirroring
`SamplerOptions.enableThinking`) rather than any sentence — tested in
`promptfooconfig.think-ab.remote.yaml`.
