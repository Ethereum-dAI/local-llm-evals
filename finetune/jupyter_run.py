"""Run Python code on a remote Jupyter kernel (e.g. a Kaggle GPU notebook) and
stream its stdout/stderr back — the Kaggle equivalent of `colab exec`.

Talks the Jupyter kernel websocket protocol directly (no local kernel needed):
GET /api to seed cookies, pick/create a kernel, open /api/kernels/<id>/channels,
send one execute_request, print iopub stream/result/error until the kernel goes
idle. Auth is the JWT baked into the Kaggle proxy base URL.

Usage (run with the project venv + deps):
    uv run --with websocket-client --with requests python finetune/jupyter_run.py \
        --base "$(cat /tmp/kaggle_base.txt)" --file finetune/train_kaggle.py --timeout 3600
    ... --code 'import torch; print(torch.cuda.get_device_name(0))'
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid

import requests
from websocket import create_connection


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Jupyter base URL (with proxy token)")
    ap.add_argument("--kernel", help="kernel id; default = first existing, else create")
    ap.add_argument("--file", help="local .py file to execute remotely")
    ap.add_argument("--code", help="inline code (alternative to --file)")
    ap.add_argument("--timeout", type=float, default=3600)
    a = ap.parse_args()

    base = a.base.rstrip("/")
    s = requests.Session()
    # Best-effort cookie seed; auth is the JWT in the path, so this is optional
    # and the Kaggle proxy's HTTP endpoints are flaky — never let it be fatal.
    for _ in range(3):
        try:
            s.get(base + "/api", timeout=15)
            break
        except Exception as e:
            print(f"[jupyter_run] seed GET retry: {e}", file=sys.stderr)

    kid = a.kernel
    if not kid:
        kernels = s.get(base + "/api/kernels", timeout=30).json()
        kid = (kernels[0]["id"] if kernels else
               s.post(base + "/api/kernels", json={"name": "python3"}, timeout=60).json()["id"])
    print(f"[jupyter_run] kernel {kid}", file=sys.stderr)

    code = open(a.file).read() if a.file else (a.code or "")
    ws_base = base.replace("https://", "wss://").replace("http://", "ws://")
    cookies = "; ".join(f"{c.name}={c.value}" for c in s.cookies)
    ws = create_connection(f"{ws_base}/api/kernels/{kid}/channels",
                           cookie=cookies or None, timeout=a.timeout,
                           max_size=None, enable_multithread=True)

    msg_id = uuid.uuid4().hex
    ws.send(json.dumps({
        "header": {"msg_id": msg_id, "username": "kc", "session": uuid.uuid4().hex,
                   "msg_type": "execute_request", "version": "5.3", "date": ""},
        "parent_header": {}, "metadata": {},
        "content": {"code": code, "silent": False, "store_history": True,
                    "user_expressions": {}, "allow_stdin": False, "stop_on_error": True},
        "channel": "shell", "buffers": [],
    }))

    deadline = time.time() + a.timeout
    ws.settimeout(90)
    err = False
    while time.time() < deadline:
        try:
            raw = ws.recv()
        except Exception:
            continue  # idle recv timeout during long compute; keep waiting
        if not raw:
            continue
        m = json.loads(raw)
        if m.get("parent_header", {}).get("msg_id") != msg_id:
            continue
        t, c = m.get("msg_type"), m.get("content", {})
        if t == "stream":
            sys.stdout.write(c.get("text", ""))
            sys.stdout.flush()
        elif t in ("execute_result", "display_data"):
            txt = c.get("data", {}).get("text/plain")
            if txt:
                print(txt)
        elif t == "error":
            err = True
            print("\n".join(c.get("traceback", [])), file=sys.stderr)
        elif t == "status" and c.get("execution_state") == "idle":
            break
    ws.close()
    sys.exit(1 if err else 0)


if __name__ == "__main__":
    main()
