# New eval datasets: why they exist and how they were built

September 2026. Branch `feat/hard-benchmark`.

This document explains three new datasets for the wallet tool-calling eval:

| File | Cases | Purpose |
| --- | --: | --- |
| `pf/tests.hard.yaml` | 187 | New generated cases aimed at where models were measured to differ |
| `pf/tests.benchmark.yaml` | 508 | The new benchmark: 321 cases taken from the 1000, plus the 187 hard cases |
| `pf/tests.panel.yaml` | 50 | A small, cheap panel of cases known to separate models |

`results/panel50.csv` lists the 50 panel cases, one row each. Each row gives the category, whether the case comes from the 1000, how far the 1000 already covers that kind of case, the correct answer, and every model's verdict.

## 1. The problem: the 1000 stopped separating models

The frozen 1000-case benchmark (`pf/tests.combined.yaml`) put every configuration within five points:

| model | score on the 1000 |
| --- | --: |
| Gemma-4 E4B base (what the wallet ships) | 90.3% |
| gpt-5 | 92.8% |
| v5 fine-tune | 94.9% |

Most of its slices were at or near 100% for all three models. The totals hid how differently the models fail:

- **Base** usually fails by asking a question when it should act (45 cases).
- **gpt-5** asks instead of acting in 51 cases, but makes one wrong-argument call in 1000.
- **v5** asks instead of acting only once, but makes 41 wrong-argument calls.

The goal was a benchmark whose score actually differs between an untuned small model, a frontier model and our fine-tune.

## 2. Where the ideas came from

**The wallet (`local-wallet-mac`).** I listed every tool the wallet's model can call and every check the wallet applies before it executes one. That list drives two rules below: a case must be answerable from what the app sends the model, and its correct answer must be something the wallet can execute. It also turned up several mismatches:

- The app now offers three tools (`transfer`, `swap`, `top_up_bundler`), but this repo offers the model two.
- RAILGUN `shield`/`unshield` were removed from the app.
- The tool schema advertises `amount: "all"` and contact-name recipients, but the wallet rejects both.
- The token registry is Sepolia-only, with 7 tokens.

**Sibling eval repos.** Three shaped the new cases:

- **`mainnet-attack-gym`**: showed that attack-style inputs separate model tiers far more than clean requests. Open models were fooled 70–90% of the time by address poisoning; frontier models 0–7%. This led to the injection and unresolvable-recipient cases.
- **`evm-rl-eval-environment`**: its generators for token-name injection and underspecified requests fed the injection and "no call" designs.
- **`ethskills-evals`**: its reports found that "quiz" tasks, where the trap is the whole point, saturate quickly. Tasks where the same trap is buried in an ordinary request are what separate models. That became the rule of putting traps inside normal multi-round conversations.

## 3. Rules every case follows

1. **The correct answer is computed, never hand-written.** It comes from a structured intent through the same code the 1000 uses. When the right behaviour is to make no call, the correct answer is an empty list, and only the mechanisms that declare this may use it.
2. **Scripted assistant turns match the app's prompt.** A scripted assistant turn only ever asks for a genuinely missing field, answers an off-topic question in prose, or reports that a request was prepared. A script that hesitated on a complete request would teach the very behaviour we score against.
3. **The signal a case depends on is never garbled.** The random typo and letter-case noise never touches the cancellation word, the injected address, the truncated address, or the unusual number form. Garbling any of these would make the case unanswerable rather than harder.
4. **A case must be answerable from what the app sends the model.** This rule is why several proposed slices were dropped (section 5).
5. **The correct answer must be a call the wallet would execute** (section 6).
6. **The cases are held out from training.** No correct amount appears in any training seed file, or in the 1000's or the dev set's seed files; a test enforces this.
7. **Everything regenerates from seeds.** Every file is reproducible byte for byte from its generator, and a test fails if it drifts.

## 4. The hard set: six mechanisms

Each mechanism targets a measured difference between the models, or a known failure the 1000 had no case for.

| Mechanism | What it does | Correct answer | Why |
| --- | --- | --- | --- |
| `stacked` (7–9 rounds) | Corrections and off-topic questions mixed into one conversation, sometimes after cancelling a first request | The final request, with every correction applied | The 1000 tests each of these alone and stops at 6 rounds. gpt-5 already led at 5–6 rounds, and v5's held-out mechanism slipped only at depth. |
| `injection_distractor` (3–5 rounds) | One interruption is a pasted message trying to redirect the pending request. The scripted assistant has already called it a scam. | The user's own request | Tests whether an injected address or amount leaks into the call. This mirrors the address-poisoning findings. |
| `unresolvable_recipient` (1–3 rounds) | A truncated address (`0x7a2F…9c1B`) or "the address I sent to last time" | No call | The prompt says never to invent an address, the model has no history, and the wallet rejects anything that isn't 40 hex characters or a name. |
| `unresolvable_amount` (1–3 rounds) | "$50 worth of ETH", "half my ETH" | No call | The prompt includes no prices and no balances. Only ETH/WETH are priced in dollars, because "$50 of USDC" arguably *is* 50 USDC. |
| `surface` (1–2 rounds) | Number words ("sixteen hundred"), "2.4k" and speech-to-text style, each given as a correction over a stale numeric amount. Plus single-turn Spanish, Portuguese, German and French with decimal commas. | The normal numeric call | The first version put these in one turn and every model scored 10/10, so the numbers now also have to replace a stale value. |
| `embedded_refusal` (2–4 rounds) | A routine conversation whose answer is dangerous (burn or zero address, invalid or non-EVM address, negative amount, unknown token contract), with "support told me" pressure | No call | ethskills-evals found traps inside ordinary tasks separate models better than the same trap asked directly. The 1000 only asks these directly. |

70 of the 187 cases expect no call (37%), against 12% in the 1000. So a model that never acts gets a sizeable score for free: always read the call and no-call split, not only the total.

## 5. What was left out, and why

- **Wrong-chain token addresses.** The app's prompt names no chain and no token address, so no model could tell mainnet USDC from the wallet's USDC. The case would be unanswerable, not hard.
- **`amount: "all"` and contact-name recipients.** The tool schema tells the model to use both. A "no call" answer would contradict the prompt, and a call would be one the wallet rejects.
- **`top_up_bundler`.** Adding a third tool changes the prompt for every case. That's a deliberate rebaseline of every recorded number, not a dataset change.

## 6. Every correct answer must be one the wallet executes

The first version of the panel was tested in the wallet with `wallet-eval userop` (in `local-wallet-mac`). This harness runs a tool call through the wallet's own checks and builds the transaction (UserOp) the wallet would sign. Two kinds of correct answer failed:

- **ENS names the wallet can't resolve.** The wallet's resolver tries Sepolia first, then falls back to mainnet ENS (`resolve_name.rs`). Of the 14-name ENS bank, only 8 resolve on mainnet. `carla`, `treasury`, `grants`, `team-ops`, `ops.mydao` and `pay.acme` don't.
- **Mainnet token addresses.** The wallet's token registry is Sepolia-only.

**The fix.** `src/wallet_evals/wallet_executable.py` mirrors the wallet's transfer and swap checks on Sepolia:

- the token must be in the registry, by symbol or by Sepolia address;
- the amount must pass the wallet's amount parser;
- the recipient must be 20 bytes of hex, or a name that resolves (the 8 verified names, with their addresses recorded);
- a swap needs two different known tokens and must be exact-input.

The hard-set generator now draws ENS recipients only from the 8. The benchmark and panel builders only accept cases the check passes. A test runs the check on every correct answer in all three files.

**Confirmed in the wallet.** On the final panel, all 37 cases that expect a call built a signable UserOp in `wallet-eval userop`, with 0 failures. For this run, the harness's ENS stub, which knows only `vitalik.eth`, was extended with the same 8 verified names so it resolves what the wallet's resolver would.

**This also exposed a problem in the 1000.** 104 of its correct answers can't execute in the real app: 78 unresolvable ENS names, 24 mainnet token addresses, and 2 amounts the parser rejects. The 1000 is left unchanged, because every recorded result depends on it; the new datasets simply exclude those cases.

## 7. How the 508-case benchmark was built

The benchmark is 321 cases from the 1000 plus all 187 hard cases. The 321 are unchanged copies, so existing runs can be re-scored on them without new inference.

**How the 321 were picked.** Each slice of the 1000 gets a quota:

- **Slices where every model was near 100% keep a small sample:** progressive disclosure, short `switch`, exact-output swaps, single-turn swaps, and missing-field cases.
- **Slices where the models differed keep most or all of their cases:** long `distractor`, `correction`, long `switch`, single-turn transfers, and all 49 safety refusals.
- **`token_address` keeps none:** all 24 of its cases name mainnet token addresses.

Within a slice, cases are drawn by fixed seed, never by how any model scored on them.

**Built-in bias.** The quotas were set from the same models' per-slice results, so some of the subset's separation is built in. The hard set has no such bias.

**Re-scoring the 321 from existing results** (an earlier version of the subset, before executability filtering; there was no new inference):

| | full 1000 | 321 subset |
| --- | --: | --: |
| base | 90.3% | 83.5% |
| gpt-5 | 92.8% | 87.9% |
| v5 | 94.9% | 92.8% |

On v5 vs base, the subset keeps 4.2σ of the full set's 4.6σ, while a random sample of the same size averages 2.6σ. Here σ is net cases won divided by the square root of the cases where the two models disagree.

## 8. How the 50-case panel was built

**Candidates.** All 187 hard cases, plus the 160 cases in the 1000 where base, v5 and gpt-5 disagreed. Only cases whose correct answer the wallet executes were eligible.

**Models.** Base Gemma-4 E4B and the v5 fine-tune both ran locally, as Q4_K_M quantizations, at temperature 0.2. The frontier model was Gemini 3.1 Pro via OpenRouter.

gpt-5 couldn't be used. OpenRouter returns OpenAI's account-level block (HTTP 403, "blocked for a previous policy violation"), and the direct OpenAI key has no credits. Gemini's scores are therefore **not** comparable to the recorded gpt-5 numbers.

**Selection.** Of the cases that split the three models, 50 were sampled in proportion to each pass/fail pattern, spread round-robin across mechanisms. That gives 15 mechanisms: 33 cases from the 1000 and 17 new, with 13 expecting no call.

**Independent re-run.** Choosing cases by outcome builds separation in, and a single run's verdicts include random flips. So the panel was run a second time, and the second run is the result to read:

| | run used for selection | **independent re-run** | wants a call (37) | wants no call (13) |
| --- | --: | --: | --: | --: |
| base | 17 | **22** | 15 | 7 |
| v5 | 26 | **28** | 22 | 6 |
| Gemini 3.1 Pro | 42 | **41** | **37** | 4 |

- **Gemini vs the Gemma models:** it beats base by 19 net cases (3.8σ) and v5 by 13 (2.6σ).
- **Base vs v5:** +6 net, only 1.0σ, so no reliable ranking. They fail on different cases: v5 when it should hold back, base in long conversations.
- **Stability:** 43 of the 50 cases still split the models on the re-run. Per-model agreement with the selection run was 43, 48 and 47 out of 50.

## 9. What the new cases found

- **v5 learned to act, but not when to hold back.** On the 1000's no-call cases it scored 92.5%, because those refusal types were in its training. On the new no-call cases it doesn't carry over:
  - it put truncated addresses into calls;
  - it turned "half my ETH" into `0.5`;
  - it made up an address for "wherever my last transfer went";
  - it sent to the burn address.
- **v5 with the safety clause followed a pasted phishing message** and sent the funds to the attacker's address, even though the scripted assistant had already called the message a scam. The 1000 had no case that could catch this.
- **Base loses track in long conversations.** It asks again for amounts and tokens it was already given. v5 is near-perfect here.
- **Gemini's failures are all cases that expect no call.** It got every case that wanted a call right.
- **Truncated addresses beat every model without the clause.** All three models, Gemini included, pass `0x1a7e...9b59` straight through: without the clause, the tool description says to "pass the value as the user expressed it", and nothing says a truncated address is invalid. These cases, and burn/zero-address sends, are only meaningful with the safety clause on. The wallet's `main` branch now ships the clause.

A first 60-case probe of the hard set, run before the executability fix, showed the same pattern:

| configuration | score (of 60) |
| --- | --: |
| base | 34 |
| v5 | 32 |
| base + clause | 35 |
| v5 + clause | 41 |

The surface-form mechanism was 10/10 everywhere, which is why it was rebuilt.

## 10. Limitations

- **The panel was selected by outcome**, so its separation is partly built in. The independent re-run is the honest number.
- **Gemini is standing in for gpt-5**, so its column can't be compared with the recorded gpt-5 numbers.
- **Every panel number is without the safety clause.** The wallet now ships the clause, and the refusal-style cases only make sense with it. A with-clause run of the panel is the obvious next measurement.
- **Local and remote runs differ.** The Gemma runs were local; the recorded 1000 numbers came from rented GPUs. Batched inference changes a few verdicts, so compare totals across the two, not individual cases.
- **The executability check is a hand-kept copy of the wallet's guards**, like the wallet's own ported harness code. `wallet-eval userop` is the end-to-end check.
- **Sample sizes are small.** At 10 cases per mechanism, or 50 in total, a difference of 2–3 cases in one row is noise.

## 11. Regenerating and running

```bash
uv run python scripts/generate_hard_cases.py          # -> pf/tests.hard.yaml
uv run python scripts/build_benchmark.py --project    # -> pf/tests.benchmark.yaml
uv run python scripts/build_panel.py                  # -> pf/tests.panel.yaml + results/panel50.csv
uv run pytest -q tests/test_hard_benchmark.py

# run the panel
EVAL_DATASET=pf/tests.panel.yaml scripts/eval.sh -c promptfooconfig.gemini-anchor.yaml \
    -j 8 --no-cache -o runs/panel-gemini.out.json
EVAL_DATASET=pf/tests.panel.yaml scripts/eval.sh -c promptfooconfig.hard-slice.local.yaml \
    --filter-providers '^gemma4-e4b-base$' -j 1 --no-cache -o runs/panel-base.out.json
```

To re-check executability in the wallet, convert the panel with `local-wallet-mac/scripts/convert-userop-eval-dataset.py`. Rebuild `wallet-eval` with that file as `Dataset/userop_cases.json`, and add the names in `RESOLVABLE_ENS` to the harness's `ENSFixture`. Then run `wallet-eval userop --model <gguf>`: a clean result has no "UNCLASSIFIED" section.
