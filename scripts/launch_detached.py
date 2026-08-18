"""Start a long job in its own session, so nothing upstream can reap it.

Why this exists: a 2 h eval cannot live inside a Claude Code Bash call — that tool
caps `timeout` at 600000 ms (10 min) and kills the task at the limit. The first
attempt lost gemma4-base's chunk 3 at 110/250 cases that way, with no error in any
log, which looks exactly like a crash. An earlier attempt with `nohup ... &` inside
a background call failed too: the child stayed in the caller's process group and
went down with it.

`start_new_session=True` calls setsid(2) in the child, so it leads a new session
with no controlling terminal and is not in the caller's process group. Signals sent
to that group do not reach it.

    uv run python scripts/launch_detached.py scripts/run_all_four.sh

Prints the pid and the log path, then exits immediately. Poll the log; the job
outlives this process, the tool call, and the session.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(f"usage: {Path(sys.argv[0]).name} <script> [args...]")
    cmd = sys.argv[1:]
    logs = ROOT / "runs"
    logs.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    log = logs / f"detached.{Path(cmd[0]).stem}.{stamp}.log"

    with open(log, "wb") as fh:
        proc = subprocess.Popen(
            cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # setsid: new session, not our process group
        )
    print(f"pid   {proc.pid}")
    print(f"log   {log.relative_to(ROOT)}")
    print(f"cmd   {' '.join(cmd)}")
    # A job that dies instantly (bad path, syntax error) should be visible now, not
    # discovered as silence an hour later.
    time.sleep(3)
    if proc.poll() is not None:
        print(f"\nEXITED IMMEDIATELY with {proc.returncode}:")
        print(log.read_text()[-2000:])
        sys.exit(1)
    print("\nstill running after 3s — detached OK")


if __name__ == "__main__":
    main()
