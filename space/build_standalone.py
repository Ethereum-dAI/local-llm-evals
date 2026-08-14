"""Freeze index.html + data.json into one self-contained file.

The report normally loads `data.json` via `fetch()`, which needs an HTTP
server (or at least `file://` fetch support, which browsers block). To hand
the report to someone as a single file — email attachment, Slack upload, a
tab opened straight from disk — `data.json` has to be embedded in the page
instead of fetched.

This script does the minimum needed for that:
  1. Read the built `data.json` (run `space/build_static.py` first).
  2. Inline it as a `<script id="eval-data" type="application/json">` blob,
     with every `</` escaped to `<\\/` so a `</script>` sequence inside any
     recorded model output can't prematurely close the tag and corrupt the
     page (model outputs are attacker-adjacent free text, not benchmark
     authored HTML).
  3. Replace the `fetch("data.json").then(r => r.json())` call with a
     `Promise.resolve(...)` that reads and parses that embedded blob instead,
     keeping the rest of the `.then(...).catch(...)` chain untouched.

Run from the repo root, after build_static.py:
  uv run python space/build_standalone.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

STATIC = Path(__file__).resolve().parent / "static"
SRC_HTML = STATIC / "index.html"
SRC_DATA = STATIC / "data.json"
OUT = STATIC / "index.standalone.html"

FETCH_CALL = 'fetch("data.json").then((r) => r.json())'
EMBEDDED_READ = (
    'Promise.resolve('
    'JSON.parse(document.getElementById("eval-data").textContent))'
)


def main() -> None:
    if not SRC_DATA.exists():
        sys.exit(f"{SRC_DATA} not found — run space/build_static.py first")

    html = SRC_HTML.read_text()
    data_json = SRC_DATA.read_text()

    # Re-serialize to guarantee compact, valid JSON regardless of how
    # data.json was formatted, then escape "</" so no byte sequence in the
    # blob can close the surrounding <script> tag early.
    payload = json.loads(data_json)
    embedded = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")

    if FETCH_CALL not in html:
        sys.exit(f"expected fetch call not found in {SRC_HTML}: {FETCH_CALL!r}")
    html = html.replace(FETCH_CALL, EMBEDDED_READ)

    data_script = (
        f'<script id="eval-data" type="application/json">{embedded}</script>\n'
    )
    if "</head>" not in html:
        sys.exit(f"no </head> in {SRC_HTML} to inline the data blob before")
    html = html.replace("</head>", data_script + "</head>", 1)

    if "fetch(" in html:
        sys.exit("fetch( still present in standalone output — embedding failed")

    OUT.write_text(html)
    size = OUT.stat().st_size / 1024
    print(f"{OUT.relative_to(SRC_HTML.parent.parent.parent)}  {size:.0f} KB")


if __name__ == "__main__":
    main()
