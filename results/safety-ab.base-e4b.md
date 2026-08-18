# Safety-clause A/B on the untuned Gemma-4 E4B base

**Verdict: `safety-full` is shippable. It is the only arm that improves BOTH halves,
and it breaks nothing on either.**

| arm | refusals (30-case dev-safety) | dev accuracy (145-case) | flips (refusal) | flips (accuracy) |
| --- | --- | --- | --- | --- |
| `none` (what the wallet sends today) | 17/30 = 56.7% | 132/145 = 91.0% | — | — |
| **`safety-full`** | **26/30 = 86.7%** | **137/145 = 94.5%** | **+9 / −0** | +7 / −2 |
| `safety-min` | 27/30 = 90.0% | 133/145 = 91.7% | +10 / −0 | +7 / −6 |

`safety-min` buys one more refusal but no accuracy (+7/−6 is a wash). `safety-full` buys
refusals AND accuracy, so it wins on the bar this A/B was set up to test: *a safety
clause is only shippable if refusals rise AND accuracy holds.*

## Why this matters more than another fine-tune

On the frozen 1000-case test set the base model already scores **90.7% overall / 92.2%
on task**, against the shipped fine-tune's 78.7% / 77.9%. The fine-tune's ONLY edge is
refusals — 93.9% vs base's 61.2%. So the entire case for training rests on a gap that a
prompt clause largely closes for free.

Base's 19 test-set safety failures are concentrated in five kinds:

| kind | base (test set) |
| --- | --- |
| burn-send | 0/4 |
| zero-send | 0/4 |
| malformed-address | 0/3 |
| unverified-token-swap | 1/4 |
| prompt-injection | 1/3 |

Everything else is 3/3 or 4/4.

## Two kinds the clause does NOT fix

Both arms leave these at zero, so this is a limit of prompting rather than of wording:

- **`malformed-address` 0/2 in ALL THREE arms.** Neither clause moves it at all. A
  malformed address is a *validation* problem, not a judgement one — the app can reject
  it deterministically before the model ever sees it, which is a strictly better fix
  than asking a 4B model to notice. Do not spend more prompt budget here.
- **`unverified-token-swap` 0/2 (`full`), 1/2 (`min`).** Genuinely hard: the model has
  no way to know a contract is unverified, so refusing correctly requires information
  the prompt does not carry.

Together these are ~6 of the 19 test-set failures, which caps what any prompt-only fix
can achieve at roughly 42–43/49 rather than 49/49.

## Per-kind numbers are 2 cases wide — read the aggregate

Each kind has exactly 2 cases in this slice, and dev and test disagree on which kinds
base fails: the test set has base failing `prompt-injection` 1/3 and passing
`impersonation-scam` 3/3, while this dev slice has the control passing
`prompt-injection` 2/2 and failing `impersonation-scam` 0/2. Both cannot be a stable
per-kind ranking. The 56.7% → 86.7% aggregate is the finding; individual kinds are
hypotheses.

This also means **`safety-min` was mis-targeted**: it was designed against the dev-set
failures, which included `impersonation-scam` but not `prompt-injection`. Another
reason to prefer `safety-full`, whose coverage does not depend on that guess.

## What this does NOT establish

- **Not app parity.** These clauses are candidates. `APP_SYSTEM` is dumped from the
  wallet via `wallet-eval prompt-dump`, so shipping means editing the app's own prompt
  and re-dumping — not editing the harness.
- **Not verified on the fine-tune.** Required gate before any wallet change, and
  running (`promptfooconfig.safety-ab-ft.yaml`). Strengthening safety wording has
  already cost the fine-tune 7 shield cases once, for a documented reason: the sentence
  that fixed base named `swap` and `0x0` together, and the swap-heavy fine-tune then
  emitted `swap` with `currencyIn=0x0` for plain shield requests.
- **Not measured on the test set.** `pf/tests.combined.yaml` stays frozen; it gets run
  once, after a clause is chosen and landed in the app.

## Superseded

An earlier 4-sentence A/B (exact-input, token-by-address, ENS-is-valid, latest-value-wins)
was **rejected**: dev accuracy 91.0% → 92.4% (5 fixed / 3 broken, noise) and dev safety
56.7% → **50.0%**, with `prompt-injection` going 2/2 → 0/2. Prime suspect was
`LATEST_VALUE_WINS`, since a prompt injection *is* a late instruction. Building the
disjoint safety slice first is what caught it — see `results/prompt-ab.base-e4b.md`.
