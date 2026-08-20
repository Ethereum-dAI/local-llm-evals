# THE DELIVERABLE: ft-v5 scores 94.9% on the frozen 1000 — above base AND above gpt-5

Both on-device arms served from rented GPUs in the same run so the comparison is
device-controlled; both Q4_K_M; both at the sampling that produced every recorded number
on this benchmark (T=0.2, max_tokens 1024, n_ctx 4096). 2000 generations, **16m49s, 0
provider errors.** gpt-5 ran separately via OpenRouter over the same 1000 cases and the
same `pf/prompt.py:render`.

| model | overall | wants a call (880) | wants NO call (120) | safety (49) |
| --- | --: | --: | --: | --: |
| **Gemma-4 E4B ft-v5** (4B, Q4_K_M, on-device) | **94.9%** (949/1000) | **95.2%** | **92.5%** | 81.6% |
| gpt-5 (frontier, hosted) | 92.8% (928/1000) | 94.1% | 83.3% | 69.4% |
| Gemma-4 E4B base (what the wallet ships) | 90.3% (903/1000) | 91.7% | 80.0% | 61.2% |
| ft-v4 (the previous fine-tune) | 78.7% (787/1000) | 79.8% | 70.8% | 93.9% |

**v5 − base = +4.6 points.** 100 cases flipped (73 only-v5, 27 only-base), so the SD of a
difference is about sqrt(100) = 10 cases and +46 is roughly 4.6 sigma. This is not the
noise floor; contrast the safety-clause experiment, where net +2 on 74 flips was
correctly withdrawn.

Base re-ran at 90.3% here against 90.7% on the local Metal run — 4 cases, inside the
measured cross-device agreement. Re-running it rather than quoting the frozen local
number is what makes the +4.6 a device-controlled measurement instead of a plausible one.

## What actually changed: the act-versus-ask decision, which nothing else fixed

| failure shape | base | **ft-v5** | gpt-5 | ft-v4 |
| --- | --: | --: | --: | --: |
| wanted a call, produced NONE | 45 | **1** | 51 | 74 |
| called, WRONG arguments | 28 | 41 | **1** | 104 |
| spurious call (gold = no call) | 24 | **9** | 20 | 35 |
| **total failures** | **97** | **51** | 72 | 213 |

v5 took base's largest failure bucket from **45 to 1**. That bucket is the one this whole
effort could not move any other way: an act-not-ask prompt clause did nothing,
retry-on-no-call added nothing on top of the safety clause, few-shot exemplars made it
worse (9 -> 20), and **gpt-5 has 51 of them** — more than base. Asking instead of acting
is not a small-model deficiency and it is not promptable; it is a behaviour that has to
be trained.

And it did not simply learn to always act: spurious calls fell 24 -> 9, `ablation` went
85.7% -> **100% (28/28)**, and `exact_output` held at 32/32. It learned when to act and
when not to, in the same pass.

The cost is 13 more wrong-argument calls (28 -> 41), which is the expected shape: some
hesitation converts into action that is wrong rather than absent. Worth it at 46 net
cases — but note gpt-5 makes **one** argument error in 1000, so argument fidelity is
where v5's remaining headroom is, not the decision.

## The depth collapse is fixed, and that was v4's defining failure

| rounds | base | **ft-v5** | gpt-5 | ft-v4 |
| --- | --: | --: | --: | --: |
| 1 | 87.2% | 92.3% | 90.2% | 95.8% |
| 2 | 95.2% | **98.2%** | 91.6% | 82.4% |
| 3 | 94.0% | **98.0%** | 96.0% | 68.0% |
| 4 | 92.7% | **97.3%** | 95.5% | 66.4% |
| 5 | 81.2% | 87.5% | **96.2%** | 50.0% |
| 6 | 82.4% | 92.2% | **96.1%** | 49.0% |

v4 fell to 49% at six rounds. v5 is at 92.2%, ten points above base. gpt-5 still wins at
five rounds (96.2% vs 87.5%), so deep-conversation robustness remains the one axis where
the frontier model is genuinely ahead.

## By mechanism — the trained holes closed and one held-out regression

| mechanism | base | **ft-v5** | gpt-5 | v5 training rows |
| --- | --: | --: | --: | --: |
| `distractor` | 72.3% (73/101) | **91.1%** (92/101) | 94.1% | 150 |
| `token_address` | 87.5% (21/24) | **100%** (24/24) | 91.7% | 80 |
| `correction` | 90.9% (140/154) | **96.1%** (148/154) | 90.3% | 120 |
| `exact_output` | 100% (32/32) | 100% (32/32) | 96.9% | 90 |
| `progressive` | 98.2% | 98.2% | 99.1% | 60 |
| `switch` | **99.3%** (148/149) | 97.3% (145/149) | 98.7% | **0 — HELD OUT** |

`distractor` is the headline mechanism win: +19 cases, and by round it goes 59.3% -> 81.5%
at five rounds and 64.7% -> 88.2% at six.

**The one real regression is `switch`, and it is the held-out mechanism.** It cost 3 net
cases, all at depth (5-round 100% -> 92.3%, 6-round 100% -> 88.2%). On the dev set
`switch` held at 23/23, so this only surfaced on the larger sample — 149 cases against
dev's 23. It is small and it is the honest price of the trade, but it is the direction the
probe exists to detect, and it means the generalization claim should be stated as "the
gains transfer, with a measurable few-case cost to an untrained mechanism at depth"
rather than as "no cost at all".

## Safety: much better than base, still short of v4

61.2% -> **81.6%** (30/49 -> 40/49), against gpt-5's 69.4% and v4's 93.9%. So v5 gives
back some of v4's refusal advantage in exchange for 171 task cases. Two things follow:

- **v5 dominates base+safety-clause on both halves.** That arm scored 91.0% overall /
  91.0% task / 91.8% safety and required editing the wallet's prompt. v5 is 94.9% / 95.2%
  / 81.6% with the prompt **untouched** — better overall and better at the task, worse at
  refusals, and free of a prompt change.
- Safety is still the slice with the least training support: ~4 rows per category out of
  2288. If refusals matter more than the 10-point task gap, the two are combinable —
  nothing tested `v5 + SAFETY_FULL` together, and that is the obvious next experiment.

gpt-5 failing this slice at 69.4% remains the strongest evidence the residue is a missing
prompt clause rather than model capability.

## A shared ceiling worth noting

The arithmetic slice is **93.8% (75/80) for all three models** — base, v5 and gpt-5,
identical. Fine-tuning did not move it and neither does frontier scale, which suggests
those 5 cases are a property of the slice rather than of any model.

## What produced v5

Same recipe as v4 with three changes, each motivated by a measurement rather than a guess:

1. **Data**: 2288 rows, adding the 500 multi-round conversation rows v4 had none of —
   150 `distractor`, 120 `correction`, 90 `exact_output`, 80 `token_address`, 60
   `progressive` — plus 20 prose-answer rehearsal rows. `switch` deliberately excluded.
2. **1 epoch, not 3.** Measured: epochs 2-3 cost 14.5 points of OOD accuracy while
   `eval_loss` fell monotonically, so `metric_for_best_model="eval_loss"` reliably
   selected the worst of three checkpoints.
3. **LoRA alpha 0.75 at merge time**, selected on the 145-case dev set (97.2% vs 95.2%
   at alpha 1.0) — chosen on a one-directional pattern across mechanisms, not on the
   3-case total.

Trained on one rented A40 in ~18 minutes. Final training loss 0.2407, eval_loss 0.0377 —
*higher* than v4's 0.0068, because the holdout now contains the hard conversation rows,
and because loss is anti-correlated with capability here anyway.

**Artifacts** (private, `ef-dai-team/gemma-4-E4B-wallet-ft-v5`):

| file | size | sha256 |
| --- | --- | --- |
| `gemma-4-E4B-wallet-ft-a075.Q4_K_M.gguf` (**shipped candidate**) | 5.34 GB | `40332b62f282336d92a94dc4147ecc44e83c0e11496ac2e5738d8ef342d1b09c` |
| `gemma-4-E4B-wallet-ft-a1.Q4_K_M.gguf` | 5.34 GB | `fbfa02abd0e7f0c1b96cf965d1da393aee063dfdd8132f4688749f90e9b9b1df` |
| `adapter/` | LoRA r=16 | — |

Every serve pod verified the sha256 of the file it downloaded against these before
scoring a single case.

**Cost: about $0.76 of RunPod** (one A40 at $0.44/hr for training, four serve pods at
$0.18-0.19/hr) plus roughly $7 of OpenRouter for gpt-5's 1.26M tokens. Budget was $5 of
RunPod.

## Reading the merged report

`runs/final-3way.out.json` carries all three models as columns over all 1000 cases, 1000
of 1000 rows populated. Imported as `eval-Mu4-2026-08-19T02:30:20`:

```bash
npx promptfoo view          # then pick that eval
uv run python scripts/report_1000.py runs/v5-vs-base.out.json runs/gpt5-1000.out.json
```
