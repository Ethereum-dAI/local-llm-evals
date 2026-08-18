# What ft-v4 actually gets wrong — a failure anatomy, not an overfitting story

Reconstructed from the frozen 1000-case runs (`runs/gemma4-e4b-base.part*.out.json`
and `runs/gemma4-e4b-ft-v4-appprompt.part*.out.json`) by re-joining every case to its
real gold in `pf/tests.combined.yaml`. Same runs that produced base 90.7% / ft-v4 78.7%.

## Read this first: the exported gold is NOT usable for field-level analysis

promptfoo redacts any metadata field **named** `token` as if it were a secret.
`expected_calls[].token` comes back as the literal string `"[REDACTED]"` in **459 of
1000** cases, and the exported `to` values are truncated as well. Comparing an emitted
argument against the export's gold therefore invents mismatches — it told me there
were 52 wrong `token` values when the real number is 22, and it hid 21 of the 27
recipient errors by making the correct answer unrecognisable.

The run-time verdicts are unaffected (scoring used the real gold in-process). Only
post-hoc analysis breaks. **Always re-join to `pf/tests.combined.yaml` by
`metadata.id`.**

## The shape of the gap

| failure shape | base | ft-v4 |
| --- | --: | --: |
| called, WRONG arguments | 30 | **104** |
| wanted a call, produced NONE | 40 | **74** |
| spurious call (gold = no call) | 23 | 35 |
| **total failures / 1000** | **93** | **213** |

The 120-case regression is **62% wrong-arguments**. That is the bucket to explain.

## Wrong arguments decompose into exactly two mechanisms

Field-level errors, with the source of the wrong value recovered by searching the
conversation text for it:

| | base | ft-v4 |
| --- | --: | --: |
| `from_token` wrong | 4 | **38** |
| `to` (recipient) wrong | 6 | **27** |
| `token` wrong | 2 | **22** |
| `amount` wrong | 20 | 19 |
| `to_token` wrong | 0 | **10** |
| wrong TOOL chosen | 1 | 7 |

### Mechanism 1 — it uses the wrong TURN's value (60 cases, 52 in correction/distractor)

For every field except the recipient, the value ft-v4 emits is overwhelmingly a value
**the user really did say** — just not the one that should win:

| field | wrong value was said earlier | wrong value never appears |
| --- | --: | --: |
| `from_token` | 32 | 6 |
| `token` | 22 | 0 |
| `amount` | 14 | 5 |
| `to_token` | 9 | 1 |

52 of those 60 sit in `correction` and `distractor` — the two mechanisms that exist to
test *which* turn wins. This is a turn-tracking failure, not memorisation of the
training vocabulary and not invention. (I checked the memorisation hypothesis and it
does not survive: `from_token` in training is 55% USDC / 41% ETH, yet the errors run
`gold USDC -> emitted ETH`, i.e. toward the wallet's native default, not the training
mode.)

### Mechanism 2 — it corrupts long literals while copying them (21 cases)

**21 of the 27 wrong recipients are a single-character corruption of the correct
address**, at edit distance 1-2:

```
gold  0xf4192bf24df925a8201411432b2263a9bbbe39ce
emit  0xf4192bf24df925a8201411432b2263a9bbe39ce      (dropped a 'b')
emit  0xf4192bf24df925a8201411432b2263a9bbem39ce     (inserted 'm' — not even hex)
gold  0x52d52a177db687acc09744dcf19592db95c51b67
emit  0x52d52a177db687acc09744dc f19592db95c51b67    (inserted a SPACE)
gold  0xf13a370c86dedb53d13c77262c9dc58efe49b435
emit  0xf13a370c86dedeb53d13c77262c9dc58efe49b435    (doubled 'e')
```

Base makes this error essentially never (5 of its 6 recipient errors are a different
literal the user genuinely said). So **fine-tuning damaged verbatim pass-through of
long literals** — a capability the base model has. `pay.acme.eth -> pay.me.eth` is the
same defect on an ENS name.

This is the same capability alpha-scaling recovered: `token_address` went 9/15 -> 15/15
purely by turning the adapter down (`results/dev-alpha.e1-lr2e4.md`). Both levers point
at the same thing.

## The 74 no-calls are mostly the SAME behaviour base has

57 of 74 are a plain clarifying question — base's own dominant failure mode, just more
of it (base: 40). Only **4** are the memorised refusal template misfiring:

```
[safety-refusal-malformed-address, 4 identical training rows]
  "That isn't a valid Ethereum address — it needs to be 0x followed by 40 hex characters..."
emitted on generated-transfer-pos cases whose address is perfectly valid.
```

Two of the four generalised the template to a field it was never trained on — *"That
isn't how I take amount"*, a string that appears in none of the 2288 training rows. Real
overreach from 4 rows, but small: do not build a strategy around 4 cases.

## Two pure data holes, both already filled in the v5 data (never trained)

| mechanism | ft-v4 on frozen set | v4 training rows | v5 training rows |
| --- | --: | --: | --: |
| `exact_output` | **0/32** | 0 | 90 |
| `token_address` | 6/24 | 0 | 80 |
| `distractor` | 30/101 | 0 | 150 |
| `correction` | 99/154 | 0 | 120 |

`exact_output` at 0/15 for *every* epoch was never a stopping-rule problem — there was
nothing to learn from. And a surface-feature census refutes a third hypothesis I had:
thousands separators (10.5% of train vs 6.4% of the benchmark), leading zeros (17.9% vs
16.1%) and 40-hex addresses (22.4% vs 33.1%) are all well represented in training, so
the false objections are not caused by unseen surfaces.

## What this means for a v5 attempt

The two dominant mechanisms — wrong turn's value (60) and corrupted literal copy (21) —
are both *capability erosion* rather than missing knowledge: base has both abilities and
the adapter overrides them. That is consistent with the measured alpha result (+29.7
points with no new data) and it is a more tractable problem than "the model is overfit".
