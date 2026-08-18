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

THREE failures were resolved here, and the FIRST DIAGNOSIS OF ONE OF THEM WAS WRONG.
Read this before trusting any 403 from this script.

1. **Cloudflare blocks urllib.** RunPod fronts pod ports with Cloudflare, which rejects
   the default `Python-urllib/3.x` User-Agent with HTTP 403 and body `error code: 1010`.
   The identical request via curl returns 200. This is the ACTUAL cause of the "403
   forever" that was previously attributed to a dead container, and it is expensive:
   `_wait_healthy` probed a perfectly healthy server for 1500s, timed out, and
   TERMINATED the pod along with its 5 GB download. Every urllib call here now sends
   `_UA`.

   Two signals were misread on the way, and both are worthless — do not use them:
     * `uptime=None` from the legacy SDK persisted for the whole life of a pod that was
       running and serving. It is not evidence the container died.
     * `ports=[...]` comes from the pod CONFIGURATION, not from a listening process, so
       ports appearing means nothing either.

2. **llama-server starts in ROUTER mode** whenever no model is specified at startup, and
   `--model-url` does not count as specifying one. `/health` answered `{"status":"ok"}`
   throughout — for the ROUTER, not for a loaded model — while `/completion` failed with
   "model name is missing from the request". Fixed by downloading the GGUF in-container
   and passing `-m <path>`. `-hf` is not an alternative: this Q4_K_M was deleted from the
   repo's main branch and exists only at the pinned revision.

3. **`-c` is TOTAL context, divided across `--parallel` slots.** `-c 4096 --parallel 8`
   served 512 tokens per slot, and `/props` confirmed it. Our prompts reach 1133 tokens
   plus 1024 of generation, so every case would have been silently truncated — a full
   1000-case run of plausible-looking garbage. `--n-ctx` now means PER-SLOT and is
   multiplied by `--parallel`, and `_served_model` refuses a slot under MIN_SLOT_CTX.

Pod creation goes through REST v1 (`dockerEntrypoint`/`dockerStartCmd` as separate
arrays) rather than the legacy `docker_args` string. That was originally done to settle
whether `docker_args` replaces or appends to the image ENTRYPOINT — a question that
turned out NOT to be the bug, since the first pod did start and serve. The REST form is
kept anyway because it states entrypoint and command explicitly instead of relying on
either reading, but do not go on believing `docker_args` was broken. It was not.

The lesson that generalises: `/health` is a liveness check on a process, never evidence
that the right model is loaded and usable. `_wait_healthy` now requires
`_served_model()`, which checks BOTH the model and its slot context.

NOT A BLOCKER for any measurement. The same eval runs locally against the same GGUF —
see promptfooconfig.safety-ab.yaml — and the local path is what produced every number
in results/. This exists only to make the LARGER runs (the 1000-case benchmark, the
fine-tune verification) minutes instead of hours.

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
#: Pod creation goes through REST v1, not the legacy SDK — only REST exposes
#: `dockerEntrypoint`/`dockerStartCmd` separately. GPU pricing still comes from the
#: legacy SDK, which is the only one that lists cards at all.
REST_BASE = "https://rest.runpod.io/v1"
PORT = 8080
#: A SECOND port serving /tmp, so `curl <url>/llama.log` works even while
#: llama-server is running — see _container_script.
LOG_PORT = 8081
#: Longest measured prompt is 1133 tokens and generation is capped at 1024, so a
#: slot below this silently truncates cases. Enforced in `_served_model`.
MIN_SLOT_CTX = 2560
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


def _rank_gpus(runpod, prefer: str | None = None,
               max_price: float = 0.60) -> list[tuple]:
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
        if vram < MIN_VRAM_GB:
            continue
        # One candidate per (card, cloud). REST's cloudType has no "ALL", so the cloud
        # is part of the choice now and a card that is busy on community may still be
        # launchable on secure — worth trying both rather than only the cheaper one.
        for cloud, price in (("COMMUNITY", detail.get("communityPrice")),
                             ("SECURE", detail.get("securePrice"))):
            if price:
                candidates.append(
                    (price, vram, detail["displayName"], entry["id"], cloud))
    candidates = [c for c in candidates if c[0] <= max_price]
    if not candidates:
        raise SystemExit(f"no launchable GPU with >={MIN_VRAM_GB}GB VRAM under "
                         f"${max_price:.2f}/hr — raise --max-price deliberately "
                         f"rather than letting this pick an H200")
    candidates.sort()
    if prefer:
        candidates.sort(key=lambda c: (prefer.lower() not in c[2].lower(), c[0]))
    shown = ", ".join(f"{n}/{c[0]} ${p:.2f}" for p, v, n, _, c in candidates[:5])
    print(f"[runpod] candidates in price order: {shown}", flush=True)
    return candidates


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


#: Cloudflare fronts every pod port and rejects urllib's default `Python-urllib/3.x`
#: with HTTP 403 + `error code: 1010`. EVERY urllib call in this file needs this, and
#: leaving it off does not look like a blocked client — it looks like a dead pod, which
#: is exactly how it cost one 5 GB download (see the header).
_UA = {"User-Agent": "wallet-evals/1.0"}


def _get(url: str, timeout: float = 15.0):
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=_UA), timeout=timeout)


def _served_model(url: str) -> str | None:
    """The model llama-server is actually serving, or None if it is serving nothing.

    Reads /props rather than /v1/models because a single-model server reports its path
    there directly; the router reports `role=router`, `model_path=none`, and an empty
    model list, which is the state this exists to catch.
    """
    try:
        with _get(url.rstrip("/") + "/props") as fh:
            # strict=False: /props embeds the model's chat template, and Qwen's carries
            # raw newlines, which a strict parser rejects as control characters — a
            # readiness check that fails on the model's own metadata would look exactly
            # like a pod that never came up.
            props = json.loads(fh.read().decode(), strict=False)
    except Exception:
        return None
    if props.get("role") == "router":
        return None
    path = props.get("model_path")
    if not path or path == "none":
        return None
    # PER-SLOT context, and it must actually fit a case. llama-server divides -c across
    # --parallel slots, so a wrong -c yields a loaded model serving 512-token slots that
    # truncates every prompt (ours reach 1133 + 1024 generation) while answering
    # /health, /props and /completion perfectly happily. Readiness has to mean "can do
    # the work", not "is running".
    slot_ctx = (props.get("default_generation_settings") or {}).get("n_ctx") or 0
    if slot_ctx and slot_ctx < MIN_SLOT_CTX:
        print(f"[runpod] model loaded but slot context is {slot_ctx} < {MIN_SLOT_CTX} "
              f"— prompts would be TRUNCATED; raise --n-ctx or lower --parallel",
              flush=True)
        return None
    return f"{path} (slot ctx {slot_ctx})"


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
            with _get(url.rstrip("/") + "/health", timeout=10) as fh:
                body = json.loads(fh.read().decode())
            if body.get("status") == "ok":
                # /health ok is NOT enough. A model-less llama-server starts in ROUTER
                # mode and answers /health with exactly this, while every /completion
                # fails "model name is missing from the request" — a green check for the
                # router process. So require evidence that a MODEL is loaded before
                # calling the pod ready, or the eval sends 435 requests into a server
                # that cannot answer one.
                served = _served_model(url)
                if served:
                    print(f"[runpod] model loaded: {served}", flush=True)
                    return True
                last = "health ok but NO MODEL loaded (router mode?)"
            else:
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


#: The container's start command. Bound as an explicit ENTRYPOINT + CMD pair through
#: the REST API rather than the legacy `docker_args` string, which is what the earlier
#: attempt got wrong: `docker_args` is ambiguous about whether it REPLACES or APPENDS
#: to the image ENTRYPOINT, and the two readings need opposite argument strings. REST
#: v1 takes `dockerEntrypoint` and `dockerStartCmd` as separate arrays, so setting both
#: explicitly is correct under either reading and there is nothing left to guess.
ENTRYPOINT = ["/bin/sh", "-c"]


def _container_script(model_url: str, n_ctx: int, parallel: int) -> str:
    """Download the GGUF, serve it single-model, and keep the log readable throughout.

    `-m <local path>`, NOT `--model-url`. llama-server enters ROUTER mode whenever no
    model is specified at startup, and `--model-url` does not count as specifying one:
    build b10481 came up with `"role":"router"`, an EMPTY `/v1/models`, and `/health`
    still answering `{"status":"ok"}` — a green check for the router process, not for a
    loaded model. `/completion` then rejects every request with "model name is missing".
    `-hf` is not an alternative either: this Q4_K_M was deleted from the repo's main
    branch and exists only at the pinned revision, which `-hf` cannot address.

    The log server on LOG_PORT is deliberately started BEFORE anything can fail and
    runs for the pod's whole life. RunPod's REST API has no logs endpoint (verified
    against its own openapi.json), so without this a misconfigured container is just
    "403 forever" with the reason locked inside it — and a crash-only log server cannot
    explain a process that is running but wrong, which is exactly what happened here.
    """
    dl = (f'curl -fL --retry 3 --retry-delay 5 -o "$MODEL" "{model_url}"',
          f'wget -q -O "$MODEL" "{model_url}"',
          f'python3 -c \'import urllib.request,sys;urllib.request.urlretrieve(sys.argv[1],sys.argv[2])\' "{model_url}" "$MODEL"')
    return "\n".join([
        "set -u",
        "LOG=/tmp/llama.log",
        "MODEL=/workspace/model.gguf",
        # Progress meter goes to its OWN file: it is genuinely useful (it is how the
        # HF CDN throttling from 65MB/s to 2.5MB/s became visible) but 9KB of carriage
        # returns in llama.log buries the four lines that explain a failure.
        "DLLOG=/tmp/download.log",
        ': > "$LOG"',
        'log() { echo "$*" >> "$LOG" 2>/dev/null; }',
        'log "=== wallet-eval pod boot $(date -u) ==="',
        'log "tools: curl=$(command -v curl) wget=$(command -v wget) python3=$(command -v python3)"',
        # Always-on, so the log is readable while llama-server is UP and wrong.
        f'if command -v python3 >/dev/null 2>&1; then (cd /tmp && exec python3 -m http.server {LOG_PORT} >/dev/null 2>&1) & log "log server on {LOG_PORT}"; else log "NO python3 - no log server"; fi',
        'if [ ! -s "$MODEL" ]; then',
        f'  if command -v curl >/dev/null 2>&1; then log "downloading with curl (progress -> download.log)"; {dl[0]} >> "$DLLOG" 2>&1',
        f'  elif command -v wget >/dev/null 2>&1; then log "downloading with wget"; {dl[1]} >> "$DLLOG" 2>&1',
        f'  elif command -v python3 >/dev/null 2>&1; then log "downloading with python3"; {dl[2]} >> "$DLLOG" 2>&1',
        '  else log "NO DOWNLOAD TOOL IN IMAGE"; fi',
        "fi",
        'log "model: $(ls -l \"$MODEL\" 2>&1)"',
        # -c is the TOTAL context, DIVIDED across --parallel slots: llama-server gives
        # each slot n_ctx/n_parallel. `-c 4096 --parallel 8` therefore serves 512 tokens
        # per slot, and /props reported exactly that. Our longest prompt is 1133 tokens
        # plus 1024 of generation (~2157), so every single case would have been silently
        # truncated — a full 1000-case run of plausible-looking garbage. Multiply here so
        # `n_ctx` means PER-SLOT context, which is the only meaning the caller cares
        # about and the one that matches the local provider's `n_ctx`.
        f'/app/llama-server -m "$MODEL" --host 0.0.0.0 --port {PORT} -ngl 99 -c {n_ctx * parallel} --parallel {parallel} --cont-batching >> "$LOG" 2>&1',
        "rc=$?",
        'log "=== llama-server EXITED rc=$rc ==="',
        # 8080 is free again now, so re-bind it with the log too: a caller who only
        # knows the eval URL still gets the reason.
        "cd /tmp",
        f"if command -v python3 >/dev/null 2>&1; then exec python3 -m http.server {PORT}; else sleep 86400; fi",
    ])


def _rest(method: str, path: str, payload: dict | None = None) -> tuple[int, object]:
    req = urllib.request.Request(
        REST_BASE + path, method=method,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {_api_key()}",
                 "Content-Type": "application/json", **_UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as fh:
            body = fh.read().decode()
            return fh.status, (json.loads(body) if body.strip() else None)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:600]


def _fetch_pod_log(pod_id: str) -> str | None:
    """Pull /llama.log back through the proxy — see `_container_script`.

    Uses LOG_PORT, which serves for the pod's whole life, so this also works when
    llama-server is UP but serving the wrong thing. A crash-only log server could not
    explain the router-mode failure, because nothing had crashed.
    """
    url = f"https://{pod_id}-{LOG_PORT}.proxy.runpod.net/llama.log"
    try:
        with _get(url, timeout=20) as fh:
            return fh.read().decode(errors="replace")
    except Exception:
        return None


def up(args) -> None:
    import runpod
    runpod.api_key = _api_key()

    # --model-url, NOT -hf. `-hf` resolves against the repo's `main` branch, and this
    # Q4_K_M was DELETED from main — it exists only at the pinned commit, where the
    # benchmark's own provider fetches it (main returns 404 EntryNotFound, verified).
    # A revision-pinned resolve URL is the only form that gets the byte-identical
    # GGUF the wallet ships, and downloading inside the pod uses RunPod's network
    # instead of pushing 5 GB from here.
    # --gguf lets one pod serve a different QUANTIZATION of the same model, which is
    # how "is 4-bit quantization itself costing accuracy?" gets answered. Q8_0 (8.0 GB)
    # and BF16 (15.1 GB) both exist at this revision; Q4_K_M does NOT exist on `main`,
    # which is why the revision is pinned.
    model_url = (f"https://huggingface.co/{args.repo}/resolve/{args.revision}/{args.gguf}")
    script = _container_script(model_url, args.n_ctx, args.parallel)

    # Walk the price-ordered list. "There are no longer any instances available with
    # the requested specifications" is routine for the cheap cards — capacity comes
    # and goes minute to minute — so failing on the first choice would make this
    # script work only by luck. Any card here runs a 5 GB Q4_K_M identically; only
    # the price differs.
    pod = None
    errors: list[str] = []
    for price, vram, name, gid, cloud in _rank_gpus(runpod, args.gpu, args.max_price):
        code, body = _rest("POST", "/pods", {
            "name": f"{NAME_PREFIX}-{int(time.time())}",
            "imageName": IMAGE,
            "gpuTypeIds": [gid],
            "cloudType": cloud,
            "gpuCount": 1,
            "containerDiskInGb": args.disk,
            "ports": [f"{PORT}/http", f"{LOG_PORT}/http"],
            "dockerEntrypoint": ENTRYPOINT,
            "dockerStartCmd": [script],
        })
        if code in (200, 201) and isinstance(body, dict) and body.get("id"):
            pod = body
            print(f"[runpod] got {name} {vram}GB {cloud} at ${price:.2f}/hr",
                  flush=True)
            break
        errors.append(f"{name} {cloud} (${price:.2f}): HTTP {code} {body}")
        print(f"[runpod] {name}/{cloud} unavailable, trying next", flush=True)
    if pod is None:
        raise SystemExit("[runpod] no capacity on any candidate:\n  "
                         + "\n  ".join(errors))
    pod_id = pod["id"]
    url = f"https://{pod_id}-{PORT}.proxy.runpod.net"
    print(f"[runpod] pod {pod_id} created", flush=True)
    print(f"[runpod] TERMINATE WITH: uv run --with runpod python "
          f"{Path(__file__).name} down --pod-id {pod_id}", flush=True)
    print("[runpod] waiting for the model to load (5 GB download + load)…",
          flush=True)
    if not _wait_healthy(url, args.wait, runpod=runpod, pod_id=pod_id):
        log = _fetch_pod_log(pod_id)
        if log:
            print("[runpod] ---- container log (llama-server died) ----", flush=True)
            print(log[-4000:], flush=True)
            print("[runpod] ---- end container log ----", flush=True)
        else:
            print("[runpod] no log served either — the ENTRYPOINT itself never ran",
                  flush=True)
        if args.keep_on_failure:
            print(f"[runpod] KEEPING pod {pod_id} — it is BILLING. "
                  f"`down --pod-id {pod_id}` when done", flush=True)
        else:
            print("[runpod] terminating so it cannot bill idle", flush=True)
            runpod.terminate_pod(pod_id)
        raise SystemExit(1)
    print("[runpod] READY", flush=True)
    print(f"[runpod] log: https://{pod_id}-{LOG_PORT}.proxy.runpod.net/llama.log",
          flush=True)
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
    u.add_argument("--n-ctx", type=int, default=4096,
                   help="context PER SLOT. Multiplied by --parallel for llama-server's "
                        "-c, which is a total that gets divided across slots.")
    u.add_argument("--parallel", type=int, default=8,
                   help="concurrent slots; the point of renting a GPU is that "
                        "promptfoo can then run -j >1 instead of serialised")
    u.add_argument("--disk", type=int, default=30)
    u.add_argument("--repo", default=HF_REPO,
                   help="HF repo to serve. Defaults to the wallet's own GGUF repo; set it "
                        "to compare a different MODEL (e.g. Qwen/Qwen3-8B-GGUF).")
    u.add_argument("--revision", default=HF_REVISION,
                   help="git revision in --repo. The wallet's Q4_K_M only exists at the "
                        "pinned commit; other repos usually want 'main'.")
    u.add_argument("--gguf", default=HF_FILE,
                   help="GGUF filename in the pinned repo/revision. Raise --disk and "
                        "lower --parallel for the bigger quants (Q8_0 is 8.0 GB, BF16 "
                        "15.1 GB) — VRAM holds the weights AND n_ctx*parallel of KV.")
    u.add_argument("--max-price", type=float, default=0.60,
                   help="hard ceiling in $/hr. Any of these cards runs a 5 GB "
                        "Q4_K_M the same, so there is no reason to fall back onto "
                        "an H200 because the cheap ones were busy.")
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
