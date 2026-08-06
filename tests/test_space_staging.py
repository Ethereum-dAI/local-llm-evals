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
