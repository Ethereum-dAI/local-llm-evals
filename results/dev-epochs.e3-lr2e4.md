# Phase 0c — epochs vs OOD accuracy for the Gemma-4 fine-tune (`e3-lr2e4`)

Scored with `finetune/modal_eval_gemma4.py --tag e3-lr2e4 --dataset pf/tests.dev.yaml`
(Modal app `ap-vSyeF6Jv0lHFU8voqlDH3f`, A100, bf16 base + LoRA adapter, the repo's
own deterministic scorer). 145 out-of-distribution dev cases, disjoint from both the
training rows and the frozen 1000-case benchmark by construction.

Training run: 1768 rows, 1592 train / 176 holdout (10%), LR 2e-4, 3 epochs.

## The headline: validation loss is anti-correlated with capability

| checkpoint | `eval_loss` | dev accuracy |
| --- | --: | --: |
| epoch 1 (`checkpoint-100`) | 0.0068 | **67.6%** (98/145) |
| epoch 2 (`checkpoint-200`) | 0.0022 | 57.2% (83/145) |
| epoch 3 (`checkpoint-300`) | 0.0016 | 53.1% (77/145) |
| `adapter-e3-lr2e4` (the shipped artifact) | 0.0016 | 53.1% (77/145) |

The last row is a consistency check, and it passed to the case: the saved adapter
reproduces `checkpoint-300` exactly (77/145). Two independent paths to the same
weights — a raw trainer checkpoint vs. the saved adapter directory — give
identical verdicts across all 145 cases, so the curve is not an artifact of how
adapters are loaded, and `load_best_model_at_end` did genuinely restore epoch 3.
The artifact the pipeline would have shipped is therefore the *worst* of the three.

`eval_loss` falls monotonically; dev accuracy falls monotonically. Every epoch of
apparent improvement cost real capability — **14.5 points across the run**.

Two conclusions, both acted on:

1. **Train 1 epoch, not 3.** The 14.5 points are free, and the run is 3x cheaper.
2. **`metric_for_best_model="eval_loss"` must go.** It is not merely uninformative
   here, it reliably selects the *worst* of the three checkpoints. Phase 0b wired
   `load_best_model_at_end` to it, which is therefore actively harmful as
   configured. The dev set has to be the selector.

Note what the loss level does *not* show. The targets are template-generated tool
calls with near-zero entropy given the prompt, so a well-fit model *should* reach
~0.002 — the low value is not itself evidence of overfitting, and Unsloth's
"training loss below 0.2 means overfitting" heuristic reaches the right conclusion
for the wrong reason. The usable signal is the *rate of change*: past epoch 1
further loss reduction buys nothing in-distribution (already solved at 0.0068) and
is paid for entirely in generalization.

## Where the loss went, by mechanism

| mechanism | epoch 1 | epoch 2 | epoch 3 |
| --- | --: | --: | --: |
| `distractor` | 29/40 | 15/40 | **14/40** |
| `token_address` | 9/15 | 9/15 | **6/15** |
| `correction` | 27/40 | 26/40 | 25/40 |
| `progressive` | 10/12 | 10/12 | 9/12 |
| `switch` | 23/23 | 23/23 | 23/23 |
| `exact_output` | 0/15 | 0/15 | 0/15 |

| by rounds | epoch 1 | epoch 2 | epoch 3 |
| --- | --: | --: | --: |
| 2 rounds | 8/15 | 7/15 | 5/15 |
| 3 rounds | 29/42 | 27/42 | 26/42 |
| 4 rounds | 25/37 | 20/37 | 18/37 |
| 5 rounds | 21/26 | 17/26 | 17/26 |
| 6 rounds | 15/25 | 12/25 | 11/25 |

- `distractor` absorbs nearly all the damage — a 35-point collapse in one epoch.
  Over-training specifically destroys the ability to ignore irrelevant
  conversational content.
- `switch` is immune (23/23 throughout).
- `exact_output` is 0/15 at **every** epoch, so it was never learned at all. No
  stopping rule can recover it; it is a pure training-data gap. This matches the
  frozen benchmark, where ft-v4 scored 0/32 on this mechanism against base's 32/32.

## Do not compare these numbers to the 1000-case benchmark

The dev set deliberately over-weights the new mechanisms, so its shallow bucket is
dense with `exact_output`/`token_address` cases and 2-round is the *worst* depth
bucket — the opposite of the benchmark's shape. That is composition, not
capability. Only within-dev comparisons across checkpoints are meaningful here.
