"""Assemble the deployable trees from the harness's canonical sources.

Hugging Face repos have to be flat and self-contained: the Space needs
`prompt.py` and `wallet_evals/` sitting next to `app.py`, and the dataset repo
needs the training scripts sitting next to the data. The obvious way to get that
is to commit copies under `space/` — which is what this used to do, and those
copies silently drift the moment anyone edits `pf/` or `src/`.

So nothing under `space/` duplicates a file that exists elsewhere in the repo.
The copies are materialised on demand, here, from one source of truth each.
`tests/test_space_staging.py` fails if a duplicate is ever committed again.

    uv run python space/stage.py gradio     # -> space/build/gradio
    uv run python space/stage.py dataset    # -> space/build/dataset
    uv run python space/stage.py static     # -> space/build/static
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPACE = ROOT / "space"
BUILD = SPACE / "build"

# The subset of wallet_evals the Space actually imports. Deliberately explicit:
# copying the whole package would drag in generation/protocol code the Space has
# no use for, and `intents.py` pulls the fixture tree with it.
WALLET_EVALS_MODULES = (
    "__init__.py",
    "functiongemma.py",
    "gemma_dsl.py",
    "json_tool_calls.py",
    "parsing.py",
    "promptfoo.py",
    "schema.py",
    "scorer.py",
)

# (source relative to repo root, destination relative to the staged tree)
GRADIO_FILES: tuple[tuple[str, str], ...] = (
    ("space/app.py", "app.py"),
    ("space/requirements.txt", "requirements.txt"),
    ("space/README.md", "README.md"),
    ("space/data/benchmark.json", "data/benchmark.json"),
    ("space/data/eval_cases.json", "data/eval_cases.json"),
    ("pf/prompt.py", "prompt.py"),
    ("pf/tools.json", "tools.json"),
    ("pf/tools.app.json", "tools.app.json"),
    # prompt.py reads the app's contract dump at import time.
    ("pf/app_contract_reference.json", "app_contract_reference.json"),
    # app.py imports `scoring`; `assert` is a Python keyword so the harness file
    # can only be imported by path. Renaming on copy is the whole difference.
    ("pf/assert.py", "scoring.py"),
    *tuple((f"src/wallet_evals/{m}", f"wallet_evals/{m}") for m in WALLET_EVALS_MODULES),
)

# The dataset repo has to stand on its own: someone who downloads it should be
# able to regenerate the JSONLs and retrain without the (private) harness. That
# means shipping the generators plus the exact modules they import — traced, not
# guessed — under the harness's own layout.
#
# The `src/` prefix is load-bearing. `wallet_evals/intents.py` reads
# `datasets/lookup.json` relative to its own grandparent, so flattening the
# package to the repo root makes that path miss by exactly one level. A bundled
# `pyproject.toml` (space/dataset_pyproject.toml) puts `src` on the path.
GENERATOR_MODULES = (
    "src/wallet_evals/__init__.py",
    # The multi-round conversation builders. generate_finetune_data.py imports these
    # for the 500 training rows that close the depth collapse, so the published tree
    # cannot regenerate its own data without them —
    # test_published_dataset_regenerates_its_own_data caught exactly that.
    "src/wallet_evals/conversations.py",
    "src/wallet_evals/finetune.py",
    "src/wallet_evals/gemma_dsl.py",
    "src/wallet_evals/generation.py",
    "src/wallet_evals/intents.py",
    "src/wallet_evals/protocols/__init__.py",
    "src/wallet_evals/protocols/aave.py",
    "src/wallet_evals/protocols/safe.py",
    # The prose-answer rehearsal turns. Same reason as conversations.py above: the
    # generator imports it, so the published tree cannot rebuild its own data
    # without it. That is now three modules this manifest needed and an import
    # trace did not supply.
    "src/wallet_evals/rehearsal.py",
)

# Never publishable: every held-out case file, and the seeds that reconstruct them.
# `test_space_staging.py` enforces this list against the manifest, by path AND by
# content hash, so a rename cannot slip through.
#
# This list used to name only the original 307-case set. The benchmark has since
# grown to 1000 cases across several files and gained a dev set, and neither was
# covered — so the guard was silently protecting the least important of them. The
# two that matter most are the newest: `tests.combined.yaml` is the number we
# REPORT, and `tests.dev.yaml` is the selector every checkpoint decision runs
# through. Publishing either would invalidate the results rather than merely leak
# some data.
EVAL_SET_FILES = (
    "pf/tests.generated.yaml",
    "pf/tests.protocols.yaml",
    "pf/tests.yaml",
    "pf/tests.combined.yaml",
    "pf/tests.conversations.yaml",
    "pf/tests.app-contract.yaml",
    "pf/tests.refusals.yaml",
    "pf/tests.dev.yaml",
    "datasets/seeds.yaml",
    "datasets/seeds.conversations.yaml",
    "datasets/seeds.arithmetic.yaml",
    "datasets/seeds.dev.yaml",
)

DATASET_FILES: tuple[tuple[str, str], ...] = (
    ("space/dataset_card.md", "README.md"),
    # Mirrors the harness paths the generators read, so they run unmodified.
    ("space/dataset_pyproject.toml", "pyproject.toml"),
    ("pf/prompt.py", "pf/prompt.py"),
    ("pf/tools.json", "pf/tools.json"),
    ("pf/tools.app.json", "pf/tools.app.json"),
    ("pf/app_contract_reference.json", "pf/app_contract_reference.json"),
    # Read at import time by wallet_evals/intents.py — invisible to an import
    # trace, and the first thing that broke when this tree was tested standalone.
    ("datasets/lookup.json", "datasets/lookup.json"),
    ("datasets/finetune_seeds.yaml", "datasets/finetune_seeds.yaml"),
    ("datasets/protocols/safe.finetune.fixtures.json",
     "datasets/protocols/safe.finetune.fixtures.json"),
    ("datasets/protocols/aave.finetune.fixtures.json",
     "datasets/protocols/aave.finetune.fixtures.json"),
    ("scripts/generate_finetune_data.py", "scripts/generate_finetune_data.py"),
    ("scripts/generate_gemma4_finetune_data.py", "scripts/generate_gemma4_finetune_data.py"),
    ("finetune/_bundled.py", "scripts/_bundled.py"),
    ("finetune/modal_finetune.py", "scripts/modal_finetune.py"),
    ("finetune/modal_finetune_gemma4.py", "scripts/modal_finetune_gemma4.py"),
    ("finetune/modal_export.py", "scripts/modal_export.py"),
    ("finetune/modal_export_gemma4.py", "scripts/modal_export_gemma4.py"),
    ("finetune/train_functiongemma.py", "scripts/train_functiongemma.py"),
    # modal_export.py's post-merge sanity check — previously not published at all,
    # which broke the export job outright.
    ("finetune/diag_sample.jsonl", "data/diag_sample.jsonl"),
    ("data_for_finetune/functiongemma_train.jsonl", "data/functiongemma_train.jsonl"),
    # FOUR app-contract variants, because "the training data" is genuinely four
    # files and publishing one of them is how the repo ended up shipping the
    # superseded base-unit set for months:
    #
    #   *_train.jsonl                1768 rows, WALLET ONLY — the default mix
    #   *_train.with-protocol.jsonl  1863 rows, + 95 Aave/Safe builder rows
    #
    # The with-protocol files are what gemma-4-E4B-wallet-ft-v4 and
    # qwen3-8b-wallet-ft-v4 actually trained on, so without them neither published
    # model is reproducible. The wallet-only files are the current default, because
    # the builder rows teach a second tool contract (executeTx, base units) opposed
    # to the app's own. Both are published and labelled rather than leaving the
    # reader to guess which produced which artifact.
    ("data_for_finetune/gemma4_train.jsonl", "data/gemma4_train.jsonl"),
    ("data_for_finetune/gemma4_train.with-protocol.jsonl",
     "data/gemma4_train.with-protocol.jsonl"),
    ("data_for_finetune/qwen_train.jsonl", "data/qwen_train.jsonl"),
    ("data_for_finetune/qwen_train.with-protocol.jsonl",
     "data/qwen_train.with-protocol.jsonl"),
    # Qwen's encoder. Its absence made qwen3-8b-wallet-ft-v4 unreproducible from the
    # published tree even though its data was byte-identical to gemma's in content.
    ("scripts/generate_qwen_finetune_data.py", "scripts/generate_qwen_finetune_data.py"),
    *tuple((m, m) for m in GENERATOR_MODULES),
)

# The static report. It used to upload straight from `space/static/`, which was
# true for as long as every file it needed lived there. The benchmark chart does
# not — `charts/chart-scores.jpg` is also what `scripts/export_chart_images.py`
# writes and what gets pasted into decks — so it gets staged like everything else
# rather than copied under space/.
STATIC_FILES: tuple[tuple[str, str], ...] = (
    ("space/static/README.md", "README.md"),
    ("space/static/index.html", "index.html"),
    ("space/static/data.json", "data.json"),
    # Both charts are exports of `scripts/export_chart_images.py`, which writes
    # into charts/. The headline one was once uploaded straight to the Space and
    # existed nowhere else in this repo, so the next `--delete "*"` upload removed
    # it. Listing it here is what makes it survive a redeploy.
    ("charts/02-overall-accuracy.jpg", "overall-accuracy.jpg"),
    ("charts/chart-scores.jpg", "chart-scores.jpg"),
)

TARGETS = {"gradio": GRADIO_FILES, "dataset": DATASET_FILES, "static": STATIC_FILES}


class MissingSources(FileNotFoundError):
    """A manifest entry points at a file that isn't there.

    Carries the paths as data (`.missing`) rather than only in the message,
    because the staging tests must distinguish two very different causes:

      * a gitignored build product (the fine-tuning JSONLs) — legitimately
        absent on a fresh clone, so the test skips;
      * a tracked file the manifest still names after it was deleted — a real
        breakage, so the test must FAIL.

    Both used to raise a plain FileNotFoundError that every caller skipped on.
    That hid a manifest still pointing at two deleted railgun files: the three
    tests guaranteeing "the dataset repo reproduces itself" skipped silently
    while `space/deploy.sh` could not publish at all, and the message blamed
    the gitignored JSONLs for it.
    """

    def __init__(self, target: str, missing: list[str]) -> None:
        self.target = target
        self.missing = list(missing)
        super().__init__(
            f"{target}: missing source file(s): {', '.join(missing)}. "
            "The fine-tuning JSONLs are gitignored — regenerate them with "
            "scripts/generate_finetune_data.py and scripts/generate_gemma4_finetune_data.py. "
            "Anything else here is a stale manifest entry."
        )


def stage(target: str, dest: Path | None = None) -> Path:
    """Copy `target`'s file list into `dest`, replacing whatever was there."""
    files = TARGETS[target]
    dest = Path(dest) if dest else BUILD / target

    missing = [src for src, _ in files if not (ROOT / src).is_file()]
    if missing:
        raise MissingSources(target, missing)

    if dest.exists():
        shutil.rmtree(dest)
    for src, rel in files:
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / src, out)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("target", choices=sorted(TARGETS))
    ap.add_argument("--out", type=Path, default=None,
                    help="destination (default: space/build/<target>)")
    args = ap.parse_args()

    dest = stage(args.target, args.out)
    files = sorted(p for p in dest.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    print(f"{args.target}: {len(files)} files, {total / 1024:.0f} KB -> {dest}")
    for p in files:
        print(f"  {p.relative_to(dest)}")


if __name__ == "__main__":
    main()
