# gpt-5 on the frozen 1000 — 92.8%, and its residual is not argument extraction

`promptfooconfig.gpt5-appcontract.yaml` via OpenRouter, T=0.1, max_tokens 4096,
`pf/tools.app.json` (all 1000 cases are wallet-path: 476 uniswap / 475 transfer / 49
safety). 26m14s at `-j 8`, **0 provider errors**, 1,260,886 tokens (625,629 prompt /
635,257 completion, of which 561,344 reasoning) — about $7 of API.

| model | overall | wants a call (880) | wants NO call (120) |
| --- | --: | --: | --: |
| **gpt-5** | **92.8%** (928/1000) | **94.1%** | 83.3% |
| base E4B | 90.7% (907/1000) | 92.0% | 80.8% |
| ft-v4 | 78.7% (787/1000) | 79.8% | 70.8% |

## The frontier is 2.1 points above a 4B on-device model

That is the headline and it is about the CONTRACT, not the models. On the old builder
contract gpt-5 scored 96-98% while E4B-base scored 9.8%, because base-unit arithmetic
was the discriminator. The app contract removed exactly that — human decimal amounts,
two tools, verbatim recipients — so the gap collapsed to 2.1 points. Do not read
"base is nearly gpt-5" as a statement about base; read it as the ceiling this dataset
can express.

## Where gpt-5's 72 failures actually are

| failure shape | gpt-5 | base | ft-v4 |
| --- | --: | --: | --: |
| called, WRONG arguments | **1** | 30 | 104 |
| wanted a call, produced none | 51 | 40 | 74 |
| spurious call (gold = no call) | 20 | 23 | 35 |

**One** wrong argument in 1000 cases, and it is a formatting slip rather than a
misreading — `999,999.99` where the gold is `999999.99`, the user's own thousands
separators passed straight through. Everything else gpt-5 gets wrong is a DECISION:
51 times it asked instead of acting, 20 times it acted when it should have refused.

So the benchmark's arguments are solvable, and what remains discriminating is (a) the
act-versus-ask judgement and (b) knowing what to refuse. Both of the on-device models'
residuals sit in the same two buckets — base is 30 wrong-args away from gpt-5's
argument fidelity, ft-v4 is 104 away.

## gpt-5 fails the safety slice too, which localises that gap in the PROMPT

Safety refusals: gpt-5 **69.4% (34/49)**, base 61.2%, ft-v4 93.9%. A 12-case smoke run
happened to draw 7 refusal cases first and gpt-5 failed 5 — sending 50 USDC to
`0x…dEaD`, 100 DAI to the zero address, and swapping into an unknown `0x…bEEF` token,
each as a clean well-formed tool call.

The app's own system prompt contains **no safety clause at all**. A frontier model
under-refusing exactly where base under-refuses is strong evidence that this is a
missing instruction rather than a small-model capability limit — and it is consistent
with `SAFETY_FULL` lifting base's safety slice 59.2% -> 91.8% on this same set.
The fine-tune remains the only artifact that gets there without a prompt change.

## Depth is where gpt-5 is genuinely, structurally better

| rounds | gpt-5 | base | ft-v4 |
| --- | --: | --: | --: |
| 1 | 90.2% | 87.2% | 95.8% |
| 2 | 91.6% | 94.5% | 82.4% |
| 3 | 96.0% | 94.0% | 68.0% |
| 4 | 95.5% | 93.6% | 66.4% |
| 5 | **96.2%** | 85.0% | 50.0% |
| 6 | **96.1%** | 86.3% | 49.0% |

gpt-5 is FLAT from 3 to 6 rounds — it actually improves with context. Base loses ~9
points past 4 rounds and ft-v4 falls off a cliff. Conversation depth, not argument
encoding, is the axis that still separates a frontier model from a 4B one here.
