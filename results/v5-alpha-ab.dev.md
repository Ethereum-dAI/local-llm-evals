# v5 alpha selection on the 145-case OOD dev set — and the generalization probe passed

`scripts/run_v5_alpha_ab.sh` → `promptfooconfig.v5-alpha-ab.remote.yaml`. Two rented
GPUs (RTX 4000 Ada SFF at $0.18/hr, RTX A4500 at $0.19/hr), both Q4_K_M, both sha256
verified against the hash the export recorded before a single case ran. 290 generations,
**1m28s, 0 provider errors.**

| model | dev accuracy (145) |
| --- | --: |
| **v5, alpha 0.75** | **97.2%** (141/145) |
| v5, alpha 1.0 | 95.2% (138/145) |
| base E4B | 91.0% (132/145) |
| v4, best alpha (0.5) | 82.8% (120/145) |
| v4, as shipped (epoch 3, alpha 1.0) | 53.1% (77/145) |

**v5 clears base by 6.2 points on the selector it was gated against.** Both alphas beat
base, so this is not an artifact of the alpha choice.

## The two pure data holes are closed, and `switch` says it generalized

| mechanism | v4 epoch 1 | v4 epoch 3 | **v5 alpha 0.75** | training rows added |
| --- | --: | --: | --: | --: |
| `exact_output` | 0/15 | 0/15 | **15/15** | 90 |
| `token_address` | 9/15 | 6/15 | **15/15** | 80 |
| `distractor` | 29/40 | 14/40 | **40/40** | 150 |
| `correction` | 27/40 | 25/40 | 36/40 | 120 |
| `progressive` | 10/12 | 9/12 | **12/12** | 60 |
| `switch` | 23/23 | 23/23 | **23/23** | **0 — HELD OUT** |

`exact_output` was never learned at any epoch of v4 because v4 had **zero** rows of it;
it is now solved. `distractor`, which absorbed nearly all of v4's over-training damage
(29/40 → 14/40), is perfect.

**The `switch` row is the one that matters most.** `HELD_OUT_MECHANISMS` keeps `switch`
out of training on purpose, as a probe: if trained mechanisms improve while `switch`
collapses, the model is memorising shapes rather than learning the contract. `switch`
stayed at 23/23 while every trained mechanism rose. So the gains are not shape
memorisation — which is exactly the failure the probe was built to detect, and the
reason it was worth sacrificing a mechanism to build.

## Why alpha 0.75, when the margin is only 3 cases

3 of 145 is about 1.5σ and would not be decisive on its own. The reason to pick it is
that the pattern is **one-directional**: alpha 0.75 is greater than or equal to alpha
1.0 on *every* mechanism and on the no-call slice (15/15 vs 14/15), losing nowhere. Net
deltas of this size are noise; a consistent sign across independent buckets is not.

It also matches the direction already measured on v4, where scaling the adapter down
recovered pass-through fidelity (`token_address` 9/15 → 15/15) — and the sanity generate
showed the same mechanism directly: at alpha 1.0 the merged model lowercased both
addresses, applying the case normalisation the training targets use for `to` onto
`token` as well, while alpha 0.75 passed both through exactly as the user wrote them.
That difference is cosmetic for scoring (the scorer folds 0x-address case), but it is a
visible marker of how hard the adapter is overriding the base model.

Note `exact_output` is **monotonic** here — 15/15 at 0.75 and 14/15 at 1.0 — where v4's
was non-monotonic and unexplained (1/15 → 11/15 → 0/15 across 1.0/0.75/0.5). v4 had no
`exact_output` training rows at all, so that curve was measuring the base model bleeding
through at various strengths rather than any learned behaviour. With 90 rows the
mechanism is learned and the knob behaves.

## Caveat on the selector, stated because it is load-bearing

Alpha was chosen ON dev, which makes dev a fitted hyperparameter for this one decision.
That is the whole reason the frozen 1000-case benchmark is scored **once**, afterwards,
for the winner only. Scoring both alphas there and taking the better one would convert
the test set into a hyperparameter and the headline into a selection artifact.

Also: dev deliberately over-weights the new conversation mechanisms and contains no
single-turn or safety cases, so its shape is not the benchmark's. 97.2% here does not
predict 97.2% there — only the ordering transfers.
