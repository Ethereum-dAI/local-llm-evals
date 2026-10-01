# Choosing the frontier anchor: Claude Opus 5.5

2026-10-01. Run files: `runs/frontier-hard.out.json` and `runs/hard2-*.out.json` (both gitignored).

## Why gpt-5 needed replacing

The eval is built to discriminate, so it needs a strong hosted model at the top of the scale. Until now that was gpt-5, but it can no longer run on this project's keys:

- OpenRouter returns OpenAI's account-level block: HTTP 403, "this user has been blocked for a previous policy violation".
- The direct OpenAI key has no credits.

So OpenAI models are out, and the anchor has to come from another lab.

## How the candidates were scored

Eight non-OpenAI frontier models, all reached through OpenRouter:

- All ran under the app contract: the app's prompt, with the tools from `pf/tools.app.json`.
- All ran with the gpt-5 config's settings: temperature 0.1, at most 4096 output tokens, no safety clause.
- All were scored on **all 187 cases of `pf/tests.hard.yaml`**.

That set was chosen because it was written before any model saw it. The 50-case panel would have been unfair, since its cases were picked using a previous frontier model's verdicts. A 5-case smoke test first confirmed every candidate's tool calling works end to end; there were no provider errors in any run.

## Results

| model | total | wants a call (117) | wants no call (70) | dangerous answer | injection | stacked | surface | fiat/fraction amount | truncated/remembered recipient | cost |
| --- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| **claude-opus-5.5** | **177** | **117** | 60 | 20/30 | 30/30 | 45/45 | 42/42 | 20/20 | 20/20 | $1.80 |
| glm-5.3 | 176 | 116 | 60 | 20/30 | 30/30 | 45/45 | 41/42 | 20/20 | 20/20 | $0.20 |
| claude-sonnet-5.5 | 174 | **117** | 57 | 17/30 | 30/30 | 45/45 | 42/42 | 20/20 | 20/20 | $0.87 |
| deepseek-v4-pro | 165 | 114 | 51 | 12/30 | 30/30 | 43/45 | 41/42 | 20/20 | 19/20 | $0.05 |
| grok-4.7 | 161 | 103 | 58 | 19/30 | 27/30 | 34/45 | 42/42 | 20/20 | 19/20 | $0.98 |
| kimi-k3 | 154 | 93 | **61** | 21/30 | 30/30 | 22/45 | 41/42 | 20/20 | 20/20 | $0.51 |
| gemini-3.1-pro | 148 | **117** | 31 | 1/30 | 30/30 | 45/45 | 42/42 | 20/20 | 10/20 | $1.25 |
| qwen3-max-thinking | 142 | 116 | 26 | 4/30 | 30/30 | 44/45 | 42/42 | 14/20 | 8/20 | $0.19 |
| *Gemma-4 E4B base (local)* | *110* | *86* | *24* | *5/30* | *18/30* | *29/45* | *39/42* | *10/20* | *9/20* | — |
| *v5 fine-tune (local)* | *109* | *104* | *5* | *2/30* | *29/30* | *35/45* | *40/42* | *2/20* | *1/20* | — |

## The choice

**Claude Opus 5.5** scored highest: 177/187 (94.7%), with every call case correct. It is now the anchor in `promptfooconfig.frontier.yaml`.

GLM 5.3 is statistically tied (176) at about a ninth of the cost. It's a reasonable substitute if Opus becomes unavailable, but the brief was best accuracy.

## What the comparison shows

- **The hard set separates frontier models, not just small from large.** Totals range from 142 to 177. Most of the spread comes from cases that expect no call:
  - Gemini and Qwen Max act on almost every dangerous answer (1/30 and 4/30).
  - Claude and GLM decline two thirds of them, without the clause.
- **Truncated recipients are hard, not unanswerable.** Earlier notes said every model passes a truncated address like `0x1a7e…9b59` straight through. That was true only of the Gemma models and Gemini: Opus, Sonnet, GLM and Kimi decline all 20 such cases without the clause.
- **Kimi and Grok lose long conversations by imitation.** In stacked cases they copy the scripted assistant's "Done — … is queued" prose instead of calling the tool (22/45 and 34/45). Nothing would execute in the wallet.
- **v5 is the outlier on cases that expect no call:** 5 out of 70. It learned to act, and the refusal types it trained on don't transfer to the new ones.

## Caveat

Opus's numbers aren't comparable to the recorded gpt-5 column. Compare models only within the same run vintage.
