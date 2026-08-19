# The model the wallet ships TODAY scores 68.6% — below the untuned base

This number did not exist until now, and its absence nearly put a wrong baseline into a
shareable document. `OnboardingSettingsStore.swift:195` pins `LocalAIModel.recommended`
to `ef-dai-team/gemma-4-E4B-wallet-ft`, sha256 `fdf5c30e…`. The **78.7%** recorded
elsewhere in `results/` for "ft-v4" is a DIFFERENT artifact — an unpublished app-contract
retrain, sha256 `2742aad3…` — so neither it nor base described what users actually run.

The serve pod verified sha256 `fdf5c30e86d83c03…` before scoring a case, so this is the
byte-identical artifact the app downloads, not a rebuild of it.

| model | overall | wants a call (880) | wants NO call (120) |
| --- | --: | --: | --: |
| **shipping fine-tune** (`gemma-4-E4B-wallet-ft`) | **68.6%** (684/997) | 70.0% | 58.3% |
| untuned base (`gemma-4-E4B-it` Q4_K_M) | 90.3% | 91.7% | 80.0% |
| ft-v5 | 95.2% | 95.6% | 92.5% |

3 provider errors excluded from the denominator (0.3%) rather than counted as failures.

**The wallet is shipping a fine-tune that is 21.7 points WORSE than no fine-tune at all,
for the contract the app actually sends.** Its model card claims 80.1%, and that claim is
not wrong — it was measured on the 307-case base-unit benchmark, against a base that
scored 9.8% there. The contract changed underneath it.

## Why: it is still answering the contract it was trained for

| failure shape | shipping ft | base | ft-v5 |
| --- | --: | --: | --: |
| wanted a call, produced NONE | **147** (110 prose + 37 question) | 45 | 1 |
| called, WRONG arguments | 119 | 28 | 41 |
| spurious call (gold = no call) | 50 | 24 | 9 |

Two mechanisms, and it is worth being precise about their relative size:

1. **Over-refusal dominates.** 147 no-calls, 110 of them prose rather than a question —
   it explains itself instead of acting. Verbatim: *"I will not call a tool because the
   user did not specify the input amount, and the `swap` tool requires an exact input
   amount."* That is a base-unit-contract habit: under that contract the amount had to be
   exact and pre-converted, so refusing an underspecified amount was correct.

2. **Base-unit emission is real but a minority.** Of 48 wrong-amount errors, 6 are
   inflated by 1e5 or more — unmistakable base-unit conversion:

   | gold | emitted | factor |
   | --- | --- | --: |
   | `88.5` | `88.5e18` | 1e18 |
   | `6.02` | `6020000` | 1e6 |
   | `0.0303` | `30300` | 1e6 |

   The 1e6 cases are USDC decimals, the 1e18 one is wei, and one output is the literal
   string `88.5e18` — the model narrating the conversion rather than performing it.

The remaining argument errors are the ordinary mix (24 corrupted recipient literals, 20
wrong-turn amounts), so this is not ONLY a contract mismatch — the model is also weaker at
the mechanics than base.

## What this does to the recommendation

It makes the upgrade urgent rather than merely attractive. Against what users have today,
ft-v5 is **+26.6 points** (68.6% -> 95.2%), and even reverting to the *untuned base* would
be +21.7. Any of the three moves is an improvement; shipping v5 is the largest.

It also retires a comparison this project kept making. "The fine-tune buys +70 points over
base" was true of the base-unit contract and has been false since the app contract landed.
Fine-tune-versus-base is not a fixed fact about the models; it is a fact about the contract
they are scored against, and the contract is now the app's.
