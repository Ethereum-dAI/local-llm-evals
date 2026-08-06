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

`space/static/` is uploaded directly and deliberately has no staging step: it
duplicates nothing, so there is nothing to assemble.
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
    "src/wallet_evals/finetune.py",
    "src/wallet_evals/gemma_dsl.py",
    "src/wallet_evals/generation.py",
    "src/wallet_evals/intents.py",
    "src/wallet_evals/protocols/__init__.py",
    "src/wallet_evals/protocols/aave.py",
    "src/wallet_evals/protocols/railgun.py",
    "src/wallet_evals/protocols/safe.py",
)

# Never publishable: the 307 held-out cases and the seeds/scorer that reconstruct
# them. `test_space_staging.py` enforces this list against the manifest.
EVAL_SET_FILES = (
    "pf/tests.generated.yaml",
    "pf/tests.protocols.yaml",
    "pf/tests.yaml",
    "datasets/seeds.yaml",
)

DATASET_FILES: tuple[tuple[str, str], ...] = (
    ("space/dataset_card.md", "README.md"),
    # Mirrors the harness paths the generators read, so they run unmodified.
    ("space/dataset_pyproject.toml", "pyproject.toml"),
    ("pf/prompt.py", "pf/prompt.py"),
    ("pf/tools.json", "pf/tools.json"),
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
    ("data_for_finetune/gemma4_train.jsonl", "data/gemma4_train.jsonl"),
    *tuple((m, m) for m in GENERATOR_MODULES),
)

TARGETS = {"gradio": GRADIO_FILES, "dataset": DATASET_FILES}


def stage(target: str, dest: Path | None = None) -> Path:
    """Copy `target`'s file list into `dest`, replacing whatever was there."""
    files = TARGETS[target]
    dest = Path(dest) if dest else BUILD / target

    missing = [src for src, _ in files if not (ROOT / src).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{target}: missing source file(s): {', '.join(missing)}. "
            "The fine-tuning JSONLs are gitignored — regenerate them with "
            "scripts/generate_finetune_data.py and scripts/generate_gemma4_finetune_data.py."
        )

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
