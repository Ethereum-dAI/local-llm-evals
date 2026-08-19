#!/usr/bin/env python3
"""Rent a GPU, fine-tune Gemma-4 v5 on it, and leave the GGUFs on HuggingFace.

Modal is out (no credits), so this replaces finetune/modal_finetune_gemma4.py +
modal_export_gemma4.py with a RunPod pod. It reuses runpod_serve_gguf.py's REST
plumbing rather than re-deriving it — those four traps (Cloudflare 1010 on urllib's
default UA, router mode, per-slot context, /health lying) were expensive to find.

    uv run --with runpod,huggingface_hub python scripts/runpod_train_gemma4.py bundle
    uv run --with runpod python scripts/runpod_train_gemma4.py up
    uv run --with runpod python scripts/runpod_train_gemma4.py logs --pod-id <id>
    uv run --with runpod python scripts/runpod_train_gemma4.py down --pod-id <id>

WHY A BUNDLE ON HF instead of pushing files into the pod: RunPod gives no inbound file
channel to a pod created over REST, and the container is created from a bash string,
which cannot carry 9 MB of JSONL. The private dataset repo is already the designated
home for the training data (see CLAUDE.md), so the pod pulls its job from there with
an HF token and pushes the artifacts back the same way. NOTHING from the eval set is
ever in the bundle — `bundle` refuses to upload a path outside JOB_FILES.

WHY NO CUDA IMAGE: the only llama.cpp binary this pod needs is `llama-quantize`, which
is CPU-only, and torch's own wheels carry the CUDA runtime for training. So a plain
python image is enough — the same choice modal_export_gemma4.py made. Serving the
result is a separate, already-proven pod (runpod_serve_gguf.py) on the llama.cpp CUDA
image.
"""
from __future__ import annotations

import argparse
import shlex
import sys
import time
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runpod_serve_gguf import (  # noqa: E402
    _api_key, _fetch_pod_log, _get, _rank_gpus, _rest,
)

#: Plain python, not a CUDA image: see the module docstring. `slim` keeps the pull fast;
#: git/build-essential/cmake are apt-installed in the boot script.
IMAGE = "python:3.11-slim"
NAME_PREFIX = "wallet-ft-train"
LOG_PORT = 8081
#: E4B LoRA is ~17 GB of weights before optimizer state and activations; the A100-40GB
#: Modal recipe was already close to its limit, so ask for 40+ and let price decide.
MIN_VRAM_GB = 40
#: 150 GB because the export peaks at ~55 GB of transients (16 base + 16 merged bf16 +
#: 16 f16 GGUF + 4.5 Q4_K_M) on top of a 16 GB HF cache, and a second alpha scale
#: churns another 32 GB through the same space.
VOLUME_GB = 150
CONTAINER_GB = 40

JOB_REPO = "ef-dai-team/wallet-tool-calling-ft"
JOB_PREFIX = "runpod-job"
OUT_REPO = "ef-dai-team/gemma-4-E4B-wallet-ft-v5"

#: The ONLY files that may go into the bundle. The 1000-case benchmark and every other
#: `pf/tests.*.yaml` stay off the Hub — `test_dataset_never_publishes_the_eval_set`
#: guards the dataset manifest by path and content hash, and this list is the same
#: promise for this ad-hoc upload path.
JOB_FILES = {
    "gemma4_train.jsonl": "data_for_finetune/gemma4_train.jsonl",
    "app_contract_reference.json": "pf/app_contract_reference.json",
    "runpod_train_gemma4.py": "finetune/runpod_train_gemma4.py",
}
ROOT = Path(__file__).resolve().parents[1]


def _hf_token() -> str:
    """The token the pod uses for both the job bundle and the artifact push."""
    import os
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if tok:
        return tok
    cached = Path.home() / ".cache" / "huggingface" / "token"
    if cached.is_file():
        return cached.read_text().strip()
    raise SystemExit("no HF token: set HF_TOKEN or run `hf auth login`")


def bundle(args) -> None:
    from huggingface_hub import HfApi
    api = HfApi(token=_hf_token())
    for name, rel in JOB_FILES.items():
        src = ROOT / rel
        if not src.is_file():
            raise SystemExit(f"missing {rel}")
        if "tests." in src.name:
            raise SystemExit(f"refusing to upload an eval file: {rel}")
        api.upload_file(path_or_fileobj=str(src), repo_id=JOB_REPO,
                        repo_type="dataset", path_in_repo=f"{JOB_PREFIX}/{name}")
        print(f"[bundle] {rel} -> {JOB_REPO}:{JOB_PREFIX}/{name} "
              f"({src.stat().st_size/1e6:.1f} MB)", flush=True)


def _boot_script(job_args: str) -> str:
    """One bash script, run blind. Every step appends to a log served over HTTP.

    The log server starts BEFORE anything that can fail and runs for the pod's whole
    life — RunPod's REST API has no logs endpoint, so a pod that dies during `pip
    install` is otherwise indistinguishable from a pod that never booted.
    """
    dl = (
        "from huggingface_hub import hf_hub_download; import os, shutil; "
        "os.makedirs('/workspace/job', exist_ok=True); "
        f"[shutil.copy(hf_hub_download('{JOB_REPO}', f'{JOB_PREFIX}/'+n, "
        f"repo_type='dataset', token=os.environ['HF_TOKEN']), '/workspace/job/'+n) "
        f"for n in {sorted(JOB_FILES)!r}]"
    )
    return "\n".join([
        "set -u",
        "mkdir -p /workspace/logs /workspace/job",
        "LOG=/workspace/logs/train.log",
        ': > "$LOG"',
        'log() { echo "[boot] $*" >> "$LOG" 2>/dev/null; }',
        'log "=== pod boot $(date -u) ==="',
        # Always-on log server, started first.
        f'(cd /workspace/logs && exec python3 -m http.server {LOG_PORT} >/dev/null 2>&1) &',
        'log "log server up"',
        "export HF_HOME=/workspace/hf",
        "export PIP_ROOT_USER_ACTION=ignore",
        'log "apt…"',
        "apt-get update -qq >> \"$LOG\" 2>&1 && "
        "apt-get install -y -qq git build-essential cmake curl ca-certificates "
        ">> \"$LOG\" 2>&1",
        'log "pip (unsloth pulls its own CUDA torch — do not pre-install one)…"',
        'pip install --no-cache-dir -q unsloth huggingface_hub sentencepiece gguf '
        'protobuf numpy >> "$LOG" 2>&1',
        'log "pip rc=$?"',
        'log "nvidia-smi: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1)"',
        # llama-quantize only: the k-quant step convert_hf_to_gguf cannot do. CPU-only,
        # so no nvcc and no CMAKE_CUDA_ARCHITECTURES needed here.
        'log "llama.cpp…"',
        "git clone --depth 1 https://github.com/ggml-org/llama.cpp /llama.cpp "
        ">> \"$LOG\" 2>&1",
        "cmake -S /llama.cpp -B /llama.cpp/build -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF "
        ">> \"$LOG\" 2>&1",
        "cmake --build /llama.cpp/build --target llama-quantize -j >> \"$LOG\" 2>&1",
        'log "llama-quantize: $(ls -l /llama.cpp/build/bin/llama-quantize 2>&1)"',
        'log "fetching job bundle…"',
        f"python3 -c {shlex.quote(dl)} >> \"$LOG\" 2>&1",
        'log "job: $(ls -l /workspace/job 2>&1 | tr \'\\n\' \' \')"',
        'log "=== starting job ==="',
        f'python3 -u /workspace/job/runpod_train_gemma4.py {job_args} >> "$LOG" 2>&1',
        'log "=== job EXITED rc=$? ==="',
        # Keep the log reachable after the job ends; the caller terminates the pod.
        "sleep 86400",
    ])


def up(args) -> None:
    import runpod
    runpod.api_key = _api_key()
    job_args = (f"--repo {args.out_repo} --epochs {args.epochs} --lr {args.lr} "
                f"--alphas {args.alphas}" + (" --mlp-only" if args.mlp_only else ""))
    script = _boot_script(job_args)
    print(f"[train] job: {job_args}", flush=True)

    pod, errors = None, []
    for price, vram, name, gid, cloud in _rank_gpus(
            runpod, args.gpu, args.max_price, min_vram_gb=MIN_VRAM_GB):
        code, body = _rest("POST", "/pods", {
            "name": f"{NAME_PREFIX}-{int(time.time())}",
            "imageName": IMAGE,
            "gpuTypeIds": [gid],
            "cloudType": cloud,
            "gpuCount": 1,
            "containerDiskInGb": CONTAINER_GB,
            "volumeInGb": VOLUME_GB,
            "volumeMountPath": "/workspace",
            "ports": [f"{LOG_PORT}/http"],
            "env": {"HF_TOKEN": _hf_token()},
            "dockerEntrypoint": ["/bin/sh", "-c"],
            "dockerStartCmd": [script],
        })
        if code in (200, 201) and isinstance(body, dict) and body.get("id"):
            pod = body
            print(f"[train] got {name} {vram}GB {cloud} at ${price:.2f}/hr", flush=True)
            break
        errors.append(f"{name} {cloud} (${price:.2f}): HTTP {code} {body}")
        print(f"[train] {name}/{cloud} unavailable, trying next", flush=True)
    if pod is None:
        raise SystemExit("[train] no capacity:\n  " + "\n  ".join(errors))

    pod_id = pod["id"]
    print(f"[train] pod {pod_id}", flush=True)
    print(f"[train] log:  https://{pod_id}-{LOG_PORT}.proxy.runpod.net/train.log",
          flush=True)
    print(f"[train] TERMINATE WITH: uv run --with runpod python "
          f"{Path(__file__).name} down --pod-id {pod_id}", flush=True)
    print(f"RUNPOD_POD_ID={pod_id}")


def logs(args) -> None:
    url = f"https://{args.pod_id}-{LOG_PORT}.proxy.runpod.net/train.log"
    try:
        with _get(url, timeout=30) as fh:
            text = fh.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        # A 403 from the proxy means NO LISTENER on that port yet, not a rejection —
        # the pod is still pulling the image or apt/pip has not reached the server.
        print(f"[train] proxy {e.code}: nothing listening on {LOG_PORT} yet")
        return
    except Exception as e:
        print(f"[train] {type(e).__name__}: {e}")
        return
    print(text[-args.tail:] if args.tail else text)


def down(args) -> None:
    import runpod
    runpod.api_key = _api_key()
    if args.pod_id:
        runpod.terminate_pod(args.pod_id)
        print(f"[train] terminated {args.pod_id}")
        return
    for pod in runpod.get_pods() or []:
        if str(pod.get("name", "")).startswith(NAME_PREFIX):
            runpod.terminate_pod(pod["id"])
            print(f"[train] terminated {pod['id']} ({pod['name']})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("bundle").set_defaults(fn=bundle)

    u = sub.add_parser("up")
    u.add_argument("--gpu", default=None, help="substring preference, e.g. 'A40'")
    u.add_argument("--max-price", type=float, default=1.00)
    u.add_argument("--epochs", type=float, default=1)
    u.add_argument("--lr", type=float, default=2e-4)
    u.add_argument("--alphas", default="1.0,0.75")
    u.add_argument("--mlp-only", action="store_true")
    u.add_argument("--out-repo", default=OUT_REPO)
    u.set_defaults(fn=up)

    l = sub.add_parser("logs")
    l.add_argument("--pod-id", required=True)
    l.add_argument("--tail", type=int, default=4000)
    l.set_defaults(fn=logs)

    d = sub.add_parser("down")
    d.add_argument("--pod-id", default=None)
    d.set_defaults(fn=down)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
