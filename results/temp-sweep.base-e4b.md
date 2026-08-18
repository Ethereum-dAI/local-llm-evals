# Temperature sweep on base Gemma-4 E4B — no effect, and the noise floor measured

**Temperature is inert across 0.0-0.8 on this task. Keep 0.2, which is app parity.**

| arm | dev accuracy (145) | refusals (30) | median completion |
| --- | --- | --- | --- |
| 0.2 — control **A** | 129/145 = 89.0% | 18/30 = 60.0% | 195 tok |
| 0.2 — control **B** (identical) | 129/145 = 89.0% | 16/30 = 53.3% | 181 tok |
| 0.0 — greedy | 130/145 = 89.7% | 18/30 = 60.0% | 180 tok |
| 0.8 | 133/145 = 91.7% | 17/30 = 56.7% | 232 tok |

## The headline is the control-to-control gap

Arms 1 and 2 were byte-identical. They landed on the **same** score, 129/145 — and
**disagreed on 12 of 145 individual cases**. On the refusal slice the same pair differed by
2 in score.

That converts the noise floor from a hand-wave into a number. With ~12 cases flipping
between identical runs, the standard deviation of a *difference* in counts is about
sqrt(12) ~= 3.5 cases, so a difference has to clear roughly 7 cases (2 sigma) before it
means anything on this slice.

Against that bar:

- greedy: net **+1** — nothing.
- 0.8: net **+4**, from +8 fixed and -4 broken. That is ~1.1 sigma. It is the highest number
  in the table and it is still **not a result**.

Reporting 0.8 as "91.7% vs 89.0%, a 2.7-point win" would have been wrong, and without the
duplicate arm it would have looked entirely reasonable.

## Why the duplicate arm exists

Control scores on this slice, all with an unchanged prompt, have now come in at 132, 133,
128 and 129/129 of 145 across separate runs. An earlier A/B reported safety-full as
"91.0% -> 94.5% accuracy" on a net +5; that claim was withdrawn once `ACT_NOT_ASK` showed
three materially different prompts all scoring exactly 133/145 while individually flipping
up to 5 cases. Measuring the floor *inside* each run is the fix, and it costs one arm.

Note the flips are not only a sampling artefact: at `-j 8` llama-server batches
continuously, which changes floating-point reduction order, so even temperature 0.0 is not
bitwise reproducible across runs. Greedy removes sampling, not batching.

## Consequences

1. **Leave temperature at 0.2.** No setting in 0.0-0.8 beats it, so there is no reason to
   ask the wallet to change one.
2. **Greedy shows no benefit**, so there is no tension with Qwen3's card forbidding greedy
   decoding when comparing models — neither model needs it.
3. **The 145-case dev slice cannot resolve effects below ~5 points.** Anything smaller has
   to be measured on the 1000-case set or across repeated runs. This is the binding
   constraint on every remaining prompt-level experiment, and it is why the surviving
   claim for safety-full is its one-directional refusal gain (+9/-0, +8/-0) rather than
   any accuracy delta.
