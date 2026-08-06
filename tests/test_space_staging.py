"""Guard the Space/dataset staging: one copy of everything, and it stays in sync.

The Space and dataset repos need flat self-contained trees, which used to be
committed as copies under `space/`. Copies drift. `space/stage.py` materialises
them from the canonical sources instead, and these tests fail if either half of
that arrangement breaks — a duplicate creeps back into git, or a staged file
stops matching its source.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "space"))
import stage as staging  # noqa: E402


def _tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    return [ROOT / p for p in out.split("\0") if p]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_no_committed_file_is_duplicated_anywhere():
    """No tracked file may be a byte-for-byte copy of another tracked file.

    This is the rule the Space kept breaking: 16 duplicates of pf/, src/ and
    finetune/ files were committed under space/ and dataset_repo/.
    """
    by_digest: dict[str, list[Path]] = collections.defaultdict(list)
    for path in _tracked_files():
        if not path.is_file():
            continue  # deleted in the working tree
        if path.stat().st_size == 0:
            continue  # empty files collide trivially and harmlessly
        by_digest[_digest(path)].append(path)

    dupes = {
        d: sorted(str(p.relative_to(ROOT)) for p in paths)
        for d, paths in by_digest.items() if len(paths) > 1
    }
    assert not dupes, (
        "these tracked files are byte-identical copies of each other — keep one "
        "and materialise the rest with space/stage.py:\n"
        + "\n".join(f"  {' == '.join(v)}" for v in dupes.values())
    )


@pytest.mark.parametrize("target", sorted(staging.TARGETS))
def test_staged_tree_matches_its_sources(target, tmp_path):
    """Every staged file is byte-identical to the source it was copied from."""
    try:
        dest = staging.stage(target, tmp_path / target)
    except FileNotFoundError as e:
        pytest.skip(str(e))  # the gitignored JSONLs aren't present

    for src, rel in staging.TARGETS[target]:
        out = dest / rel
        assert out.is_file(), f"{target}: stage.py did not produce {rel}"
        assert _digest(out) == _digest(ROOT / src), \
            f"{target}: staged {rel} differs from {src}"

    staged = {str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file()}
    assert staged == {rel for _, rel in staging.TARGETS[target]}, \
        f"{target}: staged tree has files the manifest does not list"


def test_dataset_never_publishes_the_eval_set():
    """The 307 held-out cases stay private — that is what makes the scores mean
    anything. Guards the manifest against the files that would leak them, either
    directly or by making them regenerable."""
    sources = {src for src, _ in staging.DATASET_FILES}
    leaked = sources & set(staging.EVAL_SET_FILES)
    assert not leaked, f"dataset manifest would publish the eval set: {sorted(leaked)}"

    # Belt and braces: catch a rename by comparing content, not just paths.
    eval_digests = {
        _digest(ROOT / f): f for f in staging.EVAL_SET_FILES if (ROOT / f).is_file()
    }
    for src, rel in staging.DATASET_FILES:
        path = ROOT / src
        if path.is_file() and _digest(path) in eval_digests:
            pytest.fail(f"{rel} is a copy of {eval_digests[_digest(path)]}")


def test_published_dataset_regenerates_its_own_data(tmp_path):
    """The staged dataset must reproduce its JSONLs byte-for-byte, on its own.

    This is the claim the dataset card makes, so it gets tested rather than
    asserted. It has caught two distinct gaps that an import trace cannot see:
    a module missing from the manifest, and `datasets/lookup.json`, which
    `wallet_evals/intents.py` reads at import time.

    Uses PYTHONPATH rather than the bundled pyproject.toml so the suite stays
    offline; the pyproject is what gives a real downloader the same sys.path.
    """
    try:
        dest = staging.stage("dataset", tmp_path / "dataset")
    except FileNotFoundError as e:
        pytest.skip(str(e))

    env = {**os.environ, "PYTHONPATH": str(dest / "src")}
    for script, published in (("generate_finetune_data.py", "data/functiongemma_train.jsonl"),
                              ("generate_gemma4_finetune_data.py", "data/gemma4_train.jsonl")):
        out = tmp_path / f"regen-{script}.jsonl"
        proc = subprocess.run(
            [sys.executable, str(dest / "scripts" / script), "--out", str(out)],
            cwd=dest, env=env, capture_output=True, text=True)
        assert proc.returncode == 0, \
            f"published tree cannot run {script}:\n{proc.stderr[-1500:]}"
        assert _digest(out) == _digest(dest / published), \
            f"{script} did not reproduce {published} byte-for-byte"


def test_published_modal_jobs_find_their_data_in_both_layouts(tmp_path):
    """`bundled()` must resolve in the dataset layout and the harness layout.

    Every Modal job was silently broken in the published repo because it assumed
    the harness's directory names; each one died at `add_local_file`.
    """
    sys.path.insert(0, str(ROOT / "finetune"))
    try:
        from _bundled import bundled
    finally:
        sys.path.remove(str(ROOT / "finetune"))

    candidates = (
        ("data_for_finetune/functiongemma_train.jsonl", "data/functiongemma_train.jsonl"),
        ("data_for_finetune/gemma4_train.jsonl", "data/gemma4_train.jsonl"),
        ("finetune/diag_sample.jsonl", "data/diag_sample.jsonl"),
    )
    try:
        dest = staging.stage("dataset", tmp_path / "dataset")
    except FileNotFoundError as e:
        pytest.skip(str(e))

    for layout in (ROOT, dest):
        for cands in candidates:
            bundled(layout, *cands)  # raises if neither location exists


SPACE_IMPORTS = ("prompt", "scoring", "wallet_evals.scorer", "wallet_evals.parsing",
                 "wallet_evals.promptfoo", "wallet_evals.functiongemma",
                 "wallet_evals.gemma_dsl")


def test_staged_gradio_app_has_every_module_it_imports(tmp_path):
    """The staged Space imports cleanly, so the wallet_evals subset is complete.

    Catches the failure mode the manifest invites: adding an import to app.py
    without adding the module it needs to WALLET_EVALS_MODULES. Gradio and
    llama_cpp aren't installed for the offline suite, so app.py itself is not
    imported — everything it pulls from the harness is.

    This runs in a fresh interpreter on purpose. `wallet_evals` is pip-installed
    into the dev venv and the rest of the suite has already imported it, so an
    in-process check would resolve `src/`, silently pass, and prove nothing about
    the staged tree.
    """
    dest = staging.stage("gradio", tmp_path / "gradio")
    probe = (
        "import json, importlib\n"
        f"names = {SPACE_IMPORTS!r}\n"
        "print(json.dumps({n: importlib.import_module(n).__file__ for n in names}))\n"
    )
    proc = subprocess.run([sys.executable, "-c", probe], cwd=tmp_path,
                          env={**os.environ, "PYTHONPATH": str(dest)},
                          capture_output=True, text=True)
    assert proc.returncode == 0, f"staged Space failed to import:\n{proc.stderr}"

    origins = json.loads(proc.stdout)
    for name, origin in origins.items():
        assert Path(origin).is_relative_to(dest), \
            f"{name} resolved to {origin}, not the staged tree — staging is incomplete"
