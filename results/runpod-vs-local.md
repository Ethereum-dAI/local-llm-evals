# Renting a GPU vs running on the laptop

**4.8x faster in wall clock, same conclusions, but per-case verdicts are not
reproducible across the two. Quote wall clock; never quote per-case latency.**

Identical slice (`pf/tests.dev.safety.yaml`, 30 cases x 3 prompt arms = 90 generations),
identical GGUF at the same pinned revision, identical chat template, identical sampling.
Only the device generating tokens differs.

| | wall clock | per-case median latency | errors |
| --- | --- | --- | --- |
| local, `-j 1` (Metal) | **1107 s** | 11.19 s | 0 |
| RunPod RTX 4000 Ada SFF, `-j 8` | **230 s** | 17.11 s | 0 |

**The win is concurrency, not the card.** Per-case latency is 53% WORSE on the rented
GPU — 8 requests share the device, and this card is not faster than an M-series Metal
backend for one stream. All of the speedup comes from having 8 in flight against
`--parallel 8` slots. A report quoting per-case latency would conclude the opposite of
the truth.

The local number was measured on a quiet box; the remote number was measured while the
laptop was simultaneously running another eval, so 4.8x is a floor rather than a
best case.

## What moved and what did not

Only `/completion` token generation is remote. Prompt rendering (from the same GGUF's own
chat template via a `vocab_only` handle), tool injection, output translation and scoring
all stay local, and the provider POSTs a **fully-rendered prompt** rather than messages —
so the remote server cannot re-apply a chat template and silently change the prompt
bytes. Consequence: a scorer or parser change never needs a redeploy.

## Batched inference is not bitwise reproducible

Agreement on the same slice was **82/90 = 91.1%**, with flips in BOTH directions (4 each
way) and no systematic bias. Arm totals survived intact:

| arm | local | remote |
| --- | --- | --- |
| `safety-none` | 17/30 | 18/30 |
| `safety-full` | 26/30 | 27/30 |
| `safety-min` | 27/30 | 25/30 |

Continuous batching changes floating-point reduction order, so identical inputs at
`--parallel 8` do not have to produce identical tokens even with a pinned seed. Two rules
follow, and they are not optional:

1. **Run both arms of any comparison on the same device.** A control measured locally and
   a treatment measured remotely differ by ~9% of cases for free.
2. **Do not compare case-by-case across devices.** Aggregates are stable to about a point;
   individual verdicts are not.

## Cost

$0.18/hr for a 20 GB card, and the price-ranked picker walks candidates by price with a
capacity fallback (the cheapest card is frequently unavailable). Total for all of this
work: well under a dollar. The dominant cost is not GPU time, it is the **5 GB model
download** — 3 to 17 minutes depending on how hard the HF CDN throttles, and it is paid
again on every fresh pod. A restart (`PATCH` + `/restart`) keeps `/workspace` and so
keeps the model, which is the cheap way to change llama-server flags.

**Always `down --all`.** An idle pod bills exactly like a busy one.
