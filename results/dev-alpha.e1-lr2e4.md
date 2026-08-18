# Tier 1 — post-hoc LoRA alpha scaling on the 1-epoch Gemma-4 fine-tune

`finetune/modal_eval_gemma4.py --tag e1-lr2e4 --dataset pf/tests.dev.yaml
--alpha-scales "1.0,0.75,0.5"` (Modal app `ap-QCToB66DisLrsk12GcYhmr`). 145 OOD dev
cases, real scorer, one A100. `alpha_scale` multiplies every LoRA layer's scaling
factor at LOAD time — no retraining, no new data, ~294 layers touched.

## Result

| adapter | dev accuracy |
| --- | --: |
| e3 run, epoch 3 — **what the pipeline actually shipped** | 53.1% (77/145) |
| e3 run, epoch 1 | 67.6% (98/145) |
| e1 run, alpha 1.0 | 70.3% (102/145) |
| e1 run, alpha 0.75 | 80.7% (117/145) |
| **e1 run, alpha 0.5** | **82.8% (120/145)** |

**+29.7 points over the shipped artifact, none of it from new data.** The 1-epoch
schedule alone is worth +2.7 over epoch 1 of the 3-epoch run (70.3 vs 67.6) — not
the same weights, since the LR schedule decays to zero at one epoch.

## What alpha scaling is doing

| mechanism | α=1.0 | α=0.75 | α=0.5 |
| --- | --: | --: | --: |
| `correction` | 28/40 | 31/40 | **37/40** |
| `distractor` | 29/40 | 30/40 | **33/40** |
| `token_address` | 9/15 | 10/15 | **15/15** |
| `exact_output` | 1/15 | **11/15** | 0/15 |
| `progressive` | 12/12 | 12/12 | 12/12 |
| `switch` | 23/23 | 23/23 | 23/23 |

| by rounds | α=1.0 | α=0.75 | α=0.5 |
| --- | --: | --: | --: |
| 2 | 9/15 | **13/15** | 10/15 |
| 3 | 31/42 | 36/42 | **37/42** |
| 4 | 25/37 | **31/37** | 28/37 |
| 5 | 21/26 | 21/26 | **24/26** |
| 6 | 16/25 | 16/25 | **21/25** |

`token_address` 9/15 -> 15/15 and 6-round 16/25 -> 21/25 are the clean wins: both
are pass-through fidelity and depth, exactly what over-training was destroying.
**These capabilities were never lost — the adapter was drowning out a base model
that still had them.** That is the mechanism behind the whole regression, and it
argues narrow-distribution training over classic overfitting: nothing was
forgotten, the adapter was simply too loud relative to what it had learned.

## `exact_output` is NON-MONOTONIC and unexplained

1/15 -> 11/15 -> 0/15 across α=1.0/0.75/0.5. A monotonic knob producing a
non-monotonic response on one mechanism is a warning sign, not a curiosity, and it
is NOT yet explained. Two candidates, neither tested:

  * α=0.5 may sit in a blend region where the model reverts to the base behaviour
    of emitting an ordinary input-side swap — plausible, but the pure base model
    scored 32/32 on the benchmark's equivalent slice, so "closer to base" ought to
    help rather than hurt;
  * 15 cases is a small denominator, and a single behavioural flip moves it from 11
    to 0. That magnitude of swing on one mechanism is what a threshold effect looks
    like.

Do not pick α on the strength of the overall number alone while this is open. α=0.5
wins overall (82.8% vs 80.7%) but is the WORST of the three on `exact_output`, and
that mechanism is a correctness-of-refusal behaviour: emitting a swap the user did
not ask for is a worse product failure than the two extra points are worth.
**α=0.75 is the defensible choice until the non-monotonicity is understood.**

## Caveat on the selector

α was chosen ON the dev set, which makes dev a fitted hyperparameter here rather
than a clean holdout for this one decision. The frozen 1000-case benchmark is
still untouched and remains the number to report; a candidate should be scored
there once, at the end, not iterated against.
