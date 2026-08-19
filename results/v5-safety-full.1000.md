# v5 + SAFETY_FULL: they DO stack — 95.9% safety at no cost to the total

Both arms on ONE pod (RTX 4000 Ada, $0.20/hr), so the weights, device, quantization and
sampling are identical and `prompt_variant` is the only variable. 2000 generations,
20m09s, **0 provider errors**. Verified before the run that the variant is really wired:
the safety arm's system turn is 2110 chars against the control's 533, and `augment()`
raises on an unknown name rather than silently serving the control.

| arm | overall | wants a call (880) | wants NO call (120) | safety (49) |
| --- | --: | --: | --: | --: |
| v5 control | **95.2%** (952/1000) | 95.6% | 92.5% | 81.6% (40/49) |
| **v5 + SAFETY_FULL** | 95.1% (951/1000) | 94.7% | **98.3%** | **95.9% (47/49)** |

**Net −1 on 35 flips (18 / 17) is 0.17 sigma. The total did not move.** And the safety
slice gained 7 cases with **nothing in it breaking**:

| refusal kind | control | + clause |
| --- | --: | --: |
| `burn-send` | 2/4 | **4/4** |
| `unverified-token-swap` | 2/4 | **4/4** |
| `wrong-chain-address` | 2/3 | **3/3** |
| `negative-amount` | 2/3 | **3/3** |
| `non-numeric-amount` | 2/3 | **3/3** |
| `malformed-address` | 1/3 | 1/3 |
| the other 9 kinds | 3/3 or 4/4 | unchanged |

Five kinds fixed, **zero regressed** — unlike on base, where the same clause cost
`unverified-token-swap` (2/4 → 1/4). The whole task cost lands in one place:
`single-turn positive` 94.4% → 90.4% (−8). Multi-round conversation held (96.5% → 96.0%),
`ablation` stayed 28/28, and the arithmetic slice was flat.

## This is a smaller, better-behaved trade than the same clause made on base

| | safety gain | task cost | net overall |
| --- | --: | --: | --: |
| base + SAFETY_FULL | +16 | −13 | +2 (flat) |
| **v5 + SAFETY_FULL** | **+7** | −8 | −1 (flat) |

On base the clause worked by making the model hesitant: spurious calls 24 → 7 but
no-calls 36 → 56. v5 had already learned when to act (its no-call bucket is 1), so the
clause has far less hesitancy to add and mostly does the job it names. The mechanism that
made this a hard tradeoff on base is largely spent.

Note the direction of the prior worry. v5 was trained on the app prompt with **no** safety
clause, so 1577 characters of it at inference is off-distribution for the adapter, and the
documented precedent was a fine-tune breaking on wording that helped base. That did not
happen: the cost is 8 single-turn cases, not a collapse.

## The full option set, on one frozen benchmark

| config | overall | task (880) | safety (49) | needs a wallet prompt change |
| --- | --: | --: | --: | --- |
| base (ships today) | 90.3% | 91.7% | 61.2% | no |
| base + SAFETY_FULL | 91.0% | 91.0% | 91.8% | **yes** |
| gpt-5 (hosted) | 92.8% | 94.1% | 69.4% | no |
| ft-v4 | 78.7% | 79.8% | 93.9% | no |
| ft-v5 | **95.2%** | **95.6%** | 81.6% | no |
| **ft-v5 + SAFETY_FULL** | 95.1% | 94.7% | **95.9%** | **yes** |

`ft-v5 + SAFETY_FULL` is the best configuration measured on **both** halves at once: it
beats `base + SAFETY_FULL` by 4.1 points overall while also beating it by 4.1 points on
safety, and its 95.9% safety is above ft-v4's 93.9%, which was the previous best refusal
number this project had produced.

## What adopting it actually costs

Not free, and the cost is not in the eval:

1. **A wallet prompt change.** `APP_SYSTEM` is read from the app's own `wallet-eval
   prompt-dump`, so the clause has to land in `ToolDefinitions.swift` and be re-dumped.
   Nothing may be appended from this repo — that parity is why a score here transfers.
2. **Retraining, eventually.** The rendered prompt is part of every training row's input,
   so once the wallet's prompt changes, v5 was trained against a prompt the app no longer
   sends. Measured, that mismatch costs about 1 case net today, so it is not urgent — but
   a v6 trained WITH the clause in its rendered input is the strictly better artifact and
   is the obvious next build.
3. `malformed-address` stays 1/3 in both arms. App-side validation is the right fix for
   malformed input; no prompt reaches it.

If a prompt change is unacceptable, **ft-v5 alone at 95.2% / 81.6% is still the best
no-prompt-change option by a wide margin** — 4.9 points of overall accuracy and 20.4
points of safety over what the wallet ships today.

## A free stability datum

The control arm re-measured v5 at 952/1000 against the 949/1000 recorded the previous
run — the same GGUF, a different pod, a different hour. **3 cases.** So quote v5's
headline as ~95% rather than to the case, and treat any future difference under ~6 cases
on this benchmark as unresolved.
