# THE DELIVERABLE: base + safety-full on the frozen 1000-case benchmark

**95% is not reached. Base tops out at 91.0% overall, and the safety clause does not raise
the total — it REALLOCATES errors from dangerous ones to harmless ones. Ship it anyway.**

Both arms run on the same rented GPU so the comparison is device-controlled, 2000
generations, 52m43s.

| arm | overall | task (951) | safety (49) |
| --- | --- | --- | --- |
| control (what the wallet sends today) | 908/1000 = **90.8%** | **92.4%** | 29/49 = 59.2% |
| **+ safety-full** | 910/1000 = **91.0%** | 91.0% | 45/49 = **91.8%** |

Net +2 overall from +38 fixed and -36 broken. With 74 flips the SD of a difference is about
sqrt(74) ~= 8.6 cases, so **the overall number did not move.**

## What actually happened: a reallocation, visible in the failure shapes

| failure shape | control | + safety-full |
| --- | --- | --- |
| spurious call (acted when it should not) | 24 | **7** |
| expected a call, made NONE | 36 | **56** |
| wrong arguments | 32 | 27 |

The clause makes the model more hesitant. That converts **17 spurious calls into
non-actions**, which is exactly what the refusal slice rewards, and simultaneously pushes
**20 more task cases into "asked instead of acting"**, which the task slice punishes. Safety
+16, task -13, total flat.

This is the finding the dev slice could not deliver. On 145 cases the accuracy cost was
invisible (net +2, inside a +/-5 floor); on 951 task cases -13 is a ~1.7 sigma effect and,
crucially, it comes with a *mechanism* — the no-call count rising by exactly the amount the
spurious calls fell. It is a real cost, not noise.

## Safety by kind — and a correction

| kind | control | + safety-full |
| --- | --- | --- |
| burn-send | 0/4 | **4/4** |
| zero-send | 0/4 | **4/4** |
| prompt-injection | 1/3 | **3/3** |
| roleplay-jailbreak | 1/3 | **3/3** |
| malformed-address | 0/3 | **2/3** |
| unlimited-approval | 2/3 | 3/3 |
| negative-amount | 2/3 | 3/3 |
| impersonation-scam | 2/3 | 3/3 |
| unverified-token-swap | 2/4 | **1/4** |
| the other 6 kinds | 3/3 or 4/4 | unchanged |

**Correction to an earlier claim of mine.** I reported that `malformed-address` was "not
prompt-fixable" because it scored 0/2 in every arm of the dev A/B. On the frozen set it goes
**0/3 -> 2/3**. The dev slice had two cases of that kind; two cases cannot support that
conclusion and I should not have drawn it. App-side validation is still the better fix for
malformed input, but it is not true that the prompt cannot help.

`unverified-token-swap` genuinely regressed (2/4 -> 1/4) and remains the one kind the clause
hurts.

## Recommendation: ship it, because the two error types are not equally costly

The headline totals are flat, so a naive reading says the clause is pointless. That reading
is wrong for a wallet:

- a **spurious call** on a burn-address send destroys funds irreversibly;
- a **no-call** on a garbled request is a clarifying question the user answers.

Trading 13 of the second for 16 of the first — including burn-send and zero-send going from
0/8 to 8/8 combined — is a good trade at any sane exchange rate. **Do not judge this change
on the overall percentage**; judge it on the error classes, which is why this report leads
with the shapes rather than the total.

## What this closes

95% overall is **not reachable on base by configuration**. Tested and rejected: an
act-not-ask clause (no effect), thinking off (much worse), temperature 0.0-0.8 (inert),
few-shot exemplars (-11 accuracy), retry-on-no-call (nothing on top of safety-full), and
swapping in Qwen3-8B (equal accuracy, refusals 33.3% vs 60.0%). The only lever that moves
anything is this clause, and it moves *which* errors happen rather than how many.

The remaining task errors are 56 no-call and 27 wrong-argument. Their root cause is input
quality, not reasoning: 37 of 40 in the original bucket carried the typo mutator, and ~60%
had already worked out the correct answer before asking. That points at deterministic input
normalisation in the app — mapping near-miss surfaces onto KNOWN vocabulary only
("nitO" -> into, "becmoe" -> become rather than a recipient name) — which is a place where a
few lines of Swift beat any amount of prompting.
