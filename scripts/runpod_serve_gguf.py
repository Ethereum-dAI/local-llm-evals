"""Rent a cheap RunPod GPU, serve the wallet's GGUF over HTTP, print its URL.

Pairs with `remote_url` in pf/provider_functiongemma.py: the prompt is still rendered
locally from the same GGUF's own chat template, and the output is still translated and
scored locally. Only token generation moves, so this cannot change what is measured
beyond the device — and `promptfooconfig.phi4-mini.modal.yaml` exists in this repo
precisely because "the device changed the measurement" is a real risk worth a control.

Uses the official llama.cpp CUDA server image, so there is no build step: a
llama-cpp-python CUDA wheel would cost 10-15 minutes of compile per pod, which is most
of the run time we are trying to save.

    # start (prints RUNPOD_LLAMA_URL=...)
    uv run --with runpod python scripts/runpod_serve_gguf.py up
    # tear down — ALWAYS, it bills by the hour
    uv run --with runpod python scripts/runpod_serve_gguf.py down --pod-id <id>
    uv run --with runpod python scripts/runpod_serve_gguf.py down --all

COST. Defaults to the cheapest 24 GB card (RTX A5000, ~$0.16/hr). A 525-generation
A/B is minutes of GPU time, so the bill is cents — but only if the pod is terminated.
`down --all` kills every pod this script created, and is safe to run repeatedly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

#: The exact GGUF the wallet ships, at the pinned revision the benchmark uses.
HF_REPO = "ggml-org/gemma-4-E4B-it-GGUF"
HF_FILE = "gemma-4-E4B-it-Q4_K_M.gguf"
HF_REVISION = "1762c8e8713f"
IMAGE = "ghcr.io/ggml-org/llama.cpp:server-cuda"
PORT = 8080
#: Tagged so `down --all` can find them. A pod that outlives the run is the only way
#: this gets expensive.
NAME_PREFIX = "wallet-eval-gguf"
#: Enough VRAM for a 5 GB Q4_K_M plus a 4k context, with headroom. Chosen by PRICE at
#: run time rather than from a hardcoded list: `get_gpus()` advertises far more cards
#: than an account can actually launch (this one is offered 48 and can create 12 — the
#: cheap consumer cards are all unavailable), so a fixed preference order goes stale
#: silently and then fails at create time.
MIN_VRAM_GB = 20


def _api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if key:
        return key
    # The repo keeps it in .env at the top level (gitignored); worktrees do not have
    # their own copy, so walk up until one is found.
    here = Path(__file__).resolve()
    for parent in here.parents:
        env = parent / ".env"
        if env.is_file():
            for line in env.read_text().splitlines():
                if line.startswith("RUNPOD_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("RUNPOD_API_KEY not in the environment or any parent .env")


def _pick_gpu(runpod, prefer: str | None = None) -> str:
    """The cheapest launchable GPU with enough VRAM.

    Prices come from `get_gpu()` per card; both secure and community are considered
    and the lower wins, since for a minutes-long batch job there is no reason to pay
    for secure capacity.
    """
    candidates = []
    for entry in runpod.get_gpus():
        try:
            detail = runpod.get_gpu(entry["id"])
        except Exception:
            continue
        vram = detail.get("memoryInGb") or 0
        prices = [p for p in (detail.get("securePrice"), detail.get("communityPrice"))
                  if p]
        if vram >= MIN_VRAM_GB and prices:
            candidates.append((min(prices), vram, detail["displayName"], entry["id"]))
    if not candidates:
        raise SystemExit(f"no launchable GPU with >={MIN_VRAM_GB}GB VRAM")
    candidates.sort()
    if prefer:
        for price, vram, name, gid in candidates:
            if prefer.lower() in name.lower():
                print(f"[runpod] gpu: {name} {vram}GB ${price:.2f}/hr (requested)",
                      flush=True)
                return gid
        print(f"[runpod] {prefer!r} not available, falling back to cheapest",
              flush=True)
    price, vram, name, gid = candidates[0]
    others = ", ".join(f"{n} ${p:.2f}" for p, v, n, _ in candidates[1:4])
    print(f"[runpod] gpu: {name} {vram}GB ${price:.2f}/hr  (next: {others})",
          flush=True)
    return gid


def _pod_state(runpod, pod_id: str) -> str:
    """One-line pod state, for diagnosing a proxy that will not answer.

    A 403 from `*.proxy.runpod.net` does NOT mean the server rejected us — it is what
    the proxy returns while the pod has no listener registered on that port. So the
    useful question is never "what did the proxy say" but "is the pod actually running
    and is the port up", which only this call can answer.
    """
    try:
        pod = runpod.get_pod(pod_id) or {}
    except Exception as e:
        return f"get_pod failed: {type(e).__name__}: {e}"
    runtime = pod.get("runtime") or {}
    ports = runtime.get("ports") or []
    return (f"desired={pod.get('desiredStatus')} "
            f"uptime={runtime.get('uptimeInSeconds')}s "
            f"ports={[(p.get('privatePort'), p.get('isIpPublic')) for p in ports]}")


def _wait_healthy(url: str, deadline_s: float, runpod=None,
                  pod_id: str | None = None) -> bool:
    """Poll /health until the server reports ready.

    llama-server returns 503 while it is still loading the model, so a plain
    connection success is not readiness — it would send the first request into a
    loading server and fail the case.
    """
    end = time.time() + deadline_s
    last = ""
    tick = 0
    while time.time() < end:
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=10) as fh:
                body = json.loads(fh.read().decode())
            if body.get("status") == "ok":
                return True
            last = str(body)
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
        except Exception as e:  # connection refused while the container boots
            last = f"{type(e).__name__}"
        # Report the POD's state alongside the probe result, every ~60s. Without this
        # a failed bring-up is just "403 for 15 minutes" with no way to tell a pod
        # still queueing for capacity from a container that exited on startup.
        if runpod is not None and pod_id and tick % 4 == 0:
            print(f"[runpod] probe={last}  {_pod_state(runpod, pod_id)}", flush=True)
        tick += 1
        time.sleep(15)
    print(f"[runpod] not healthy within {deadline_s:.0f}s (last: {last})", flush=True)
    if runpod is not None and pod_id:
        print(f"[runpod] final state: {_pod_state(runpod, pod_id)}", flush=True)
    return False


def up(args) -> None:
    import runpod
    runpod.api_key = _api_key()

    # --model-url, NOT -hf. `-hf` resolves against the repo's `main` branch, and this
    # Q4_K_M was DELETED from main — it exists only at the pinned commit, where the
    # benchmark's own provider fetches it (main returns 404 EntryNotFound, verified).
    # A revision-pinned resolve URL is the only form that gets the byte-identical
    # GGUF the wallet ships, and downloading inside the pod uses RunPod's network
    # instead of pushing 5 GB from here.
    #
    # -ngl 99 offloads every layer; -c must cover the longest prompt plus generation
    # (1133 + 1024 measured, 4096 configured). --parallel gives concurrent slots so
    # promptfoo can run -j >1 instead of serialised behind one worker.
    model_url = (f"https://huggingface.co/{HF_REPO}/resolve/{HF_REVISION}/{HF_FILE}")
    cmd = (
        f"--model-url {model_url} --host 0.0.0.0 --port {PORT} "
        f"-ngl 99 -c {args.n_ctx} --parallel {args.parallel} --cont-batching"
    )
    pod = runpod.create_pod(
        name=f"{NAME_PREFIX}-{int(time.time())}",
        image_name=IMAGE,
        gpu_type_id=_pick_gpu(runpod, args.gpu),
        cloud_type="ALL",
        gpu_count=1,
        container_disk_in_gb=args.disk,
        ports=f"{PORT}/http",
        docker_args=cmd,
        env={"HF_HUB_ENABLE_HF_TRANSFER": "1"},
    )
    pod_id = pod["id"]
    url = f"https://{pod_id}-{PORT}.proxy.runpod.net"
    print(f"[runpod] pod {pod_id} created", flush=True)
    print(f"[runpod] TERMINATE WITH: uv run --with runpod python "
          f"{Path(__file__).name} down --pod-id {pod_id}", flush=True)
    print(f"[runpod] waiting for the model to load (5 GB download + load)…",
          flush=True)
    if not _wait_healthy(url, args.wait, runpod=runpod, pod_id=pod_id):
        if args.keep_on_failure:
            print(f"[runpod] KEEPING pod {pod_id} for inspection — it is BILLING. "
                  f"Logs: https://www.runpod.io/console/pods  then `down --pod-id "
                  f"{pod_id}`", flush=True)
        else:
            print("[runpod] giving up; terminating so it cannot bill idle", flush=True)
            runpod.terminate_pod(pod_id)
        raise SystemExit(1)
    print(f"[runpod] READY", flush=True)
    print(f"RUNPOD_POD_ID={pod_id}")
    print(f"RUNPOD_LLAMA_URL={url}")


def down(args) -> None:
    import runpod
    runpod.api_key = _api_key()
    if args.pod_id:
        runpod.terminate_pod(args.pod_id)
        print(f"[runpod] terminated {args.pod_id}")
        return
    killed = 0
    for pod in runpod.get_pods() or []:
        if str(pod.get("name", "")).startswith(NAME_PREFIX):
            runpod.terminate_pod(pod["id"])
            print(f"[runpod] terminated {pod['id']} ({pod['name']})")
            killed += 1
    print(f"[runpod] {killed} pod(s) terminated")


def status(args) -> None:
    import runpod
    runpod.api_key = _api_key()
    for pod in runpod.get_pods() or []:
        print(f"{pod['id']:22} {pod.get('desiredStatus'):10} {pod.get('name')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("up")
    u.add_argument("--n-ctx", type=int, default=4096)
    u.add_argument("--parallel", type=int, default=8,
                   help="concurrent slots; the point of renting a GPU is that "
                        "promptfoo can then run -j >1 instead of serialised")
    u.add_argument("--disk", type=int, default=30)
    u.add_argument("--gpu", default=None,
                   help="substring of a preferred GPU name; falls back to the "
                        "cheapest with enough VRAM if it is unavailable")
    u.add_argument("--wait", type=float, default=900)
    u.add_argument("--keep-on-failure", action="store_true",
                   help="do NOT terminate when the server never becomes healthy, so "
                        "the pod's own logs can be read in the RunPod console. Costs "
                        "money until torn down by hand — `down --all`.")
    u.set_defaults(func=up)
    d = sub.add_parser("down")
    d.add_argument("--pod-id")
    d.add_argument("--all", action="store_true")
    d.set_defaults(func=down)
    s = sub.add_parser("status")
    s.set_defaults(func=status)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
