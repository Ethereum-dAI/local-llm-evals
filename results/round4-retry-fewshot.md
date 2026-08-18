# Retry-on-no-call and few-shot vs the 40-case bucket — one works, one backfires

Both predictions were wrong, in opposite directions.

| arm | dev accuracy (145) | no-call failures | refusals (30) |
| --- | --- | --- | --- |
| control **A** | 134/145 = 92.4% | 9 | 17/30 = 56.7% |
| control **B** (identical) | 132/145 = 91.0% | 10 | 18/30 = 60.0% |
| **retry-on-no-call** | 137/145 = 94.5% | **4** | 17/30 = 56.7% |
| few-shot | 123/145 = 84.8% | **20** | 23/30 = 76.7% |
| few-shot + retry | 127/145 = 87.6% | 15 | 24/30 = 80.0% |

Noise floor measured in-run: the identical arms differ by 2 on accuracy (8 of 145 cases
disagree) and by 1 on refusals (3 of 30). So ~6 cases on accuracy and ~4 on refusals is the
2-sigma bar.

## retry-on-no-call: works on its target, at no safety cost

**No-call failures 9/10 -> 4, against a control-to-control spread of 1.** That is the bucket
this was built for, halved, and the effect is directed rather than a scatter of flips.

Refusals are untouched: 17/30, **+1/-1**. This was the whole risk — the mechanism fires on
refusal cases too, since they also produce no call — and it did not materialise. The
`RETRY_NUDGE` restating "if this must be refused, refuse and make no tool call", plus keeping
the retry ONLY when it yields a call, appear to be doing their job.

**But the aggregate accuracy gain of +3 is inside the noise floor**, so there is no headline
accuracy claim here. The reason the mechanism-level win does not translate: retry converts
some no-calls into WRONG-argument calls (wrong-args 2 -> 4). It makes the model commit; it
does not make it correct. Roughly a third of the cases it rescues, it gets wrong anyway.

## few-shot: rejected, and instructive about why

Accuracy **-11 net (+1/-12)**, far outside the floor, with no-call failures going the wrong
way: **9 -> 20**. Refusals meanwhile rose to 23/30, **+6/-0**.

So the exemplars did change behaviour — hard — just not in the intended direction. One
refusal exemplar out of four over-corrected and shifted the act/refuse balance toward
refusing. The guard that was supposed to stop the set teaching "always call" instead taught
"when unsure, don't".

**It is rejected because `safety-full` strictly dominates it**: 86.7% refusals with no
accuracy cost, against few-shot's 76.7% for -11 accuracy. Buying refusals by refusing more
is not the same as knowing what to refuse, and the accuracy slice is what tells them apart.

`few-shot + retry` (87.6% / 80.0%) is the same trade in milder form — retry claws back 5 of
the no-call regressions (20 -> 15) without repairing the accuracy loss.

## What this says about the 40-case bucket

The bucket is now partly explained rather than just measured. Recall 60% of it had already
worked out the right answer before asking. Retry confirms that read: simply asking again
converts more than half of them. What retry cannot fix is the other half, where the typo
genuinely destroyed information — and of the cases it does convert, a third come back with
wrong arguments, which is consistent with those being the same garbled surfaces.

So the ceiling on this bucket is not "make the model commit". It is the input quality:
`"nitO"` for *into*, `"becmoe"` for *become* read as a recipient name. That points at
deterministic input normalisation in the app (mapping near-misses onto KNOWN vocabulary
only), not at more prompting.

## Recommended stack, and the open question

`safety-full` + `retry_on_no_call` — the clause that wins refusals cleanly plus the mechanism
that wins the no-call bucket cleanly. They have never been run together and they push in
OPPOSITE directions on hesitancy, so the combination is measured in
`promptfooconfig.round5.remote.yaml` rather than assumed additive.

Note `retry_on_no_call` is an app BEHAVIOUR change, not a prompt change: it needs a second
model turn, so it belongs in the wallet's call loop. It is opt-in in the provider and off by
default, so a normal run stays at app parity.
