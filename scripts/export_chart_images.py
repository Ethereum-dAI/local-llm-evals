#!/usr/bin/env python
"""Export the three eval charts as standalone JPEGs.

    uv run python scripts/chart_data.py -o chartdata.json
    uv run python scripts/export_chart_images.py chartdata.json --outdir charts

Reuses the exact drawing code behind the shared HTML page, so an image and the
page can never disagree about a number. Each chart is wrapped in its own SVG
canvas with a title, subtitle and legend baked in — an image gets forwarded and
pasted into decks with no surrounding page to explain it, so it has to caption
itself.

Rendering goes through rsvg-convert rather than a headless browser: the charts
are pure SVG, so a browser adds a dependency and a variable (text wrapping,
window sizing) without adding fidelity. Colours are literal hex here, not the
page's CSS custom properties, because librsvg does not resolve var() — the light
palette is inlined and every canvas is painted opaque white, since JPEG has no
alpha and would otherwise composite the transparent ground to black.
"""
from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_eval_charts import SERIES, headline, modes_chart, nice, panel  # noqa: E402

SCALE = 2          # render at 2x for crisp text when scaled down in a deck
JPEG_QUALITY = 92

# Light-theme literals mirroring build_eval_charts.CSS. Fonts are concrete
# families, not the page's ui-* stacks, which librsvg cannot resolve.
STYLE = """
.bar.s1{fill:#2a78d6}.bar.s2{fill:#eb6834}.bar.s3{fill:#1baf7a}
.grid{stroke:#ECEEF2;stroke-width:1}
.sep{stroke:#E2E5EA;stroke-width:1}
.whisk{stroke:#555C66;stroke-width:1.5}
.pt{font-size:11px;font-weight:600;fill:#16181C;text-anchor:middle;font-family:Helvetica,Arial,sans-serif}
.pn{font-size:9.5px;fill:#868E9A;text-anchor:middle;font-family:Menlo,monospace}
.pn.warn{fill:#eb6834}
.ax{font-size:9px;fill:#868E9A;text-anchor:end;font-family:Menlo,monospace}
.ax.mid{text-anchor:middle}
.xl{font-size:9.5px;fill:#555C66;text-anchor:middle;font-family:Menlo,monospace}
.yl{font-size:12px;fill:#555C66;text-anchor:end;font-family:Menlo,monospace}
.yl.big{font-size:14px;fill:#16181C;font-family:Helvetica,Arial,sans-serif;font-weight:600}
.vl{font-size:10.5px;font-weight:700;fill:#16181C;text-anchor:middle;font-family:Menlo,monospace}
.vl.start{text-anchor:start}.vl.big{font-size:15px}
.sub{font-size:10px;fill:#868E9A;font-family:Menlo,monospace}
.sub.start{text-anchor:start}
.h1{font-size:19px;font-weight:600;fill:#16181C;font-family:Georgia,serif}
.h2{font-size:12px;fill:#555C66;font-family:Helvetica,Arial,sans-serif}
.lg{font-size:11px;fill:#555C66;font-family:Helvetica,Arial,sans-serif}
.fo{font-size:9.5px;fill:#868E9A;font-family:Menlo,monospace}
"""


def header(title: str, sub: str, w: int, y0: int = 26,
           labels: dict | None = None) -> tuple[str, int]:
    """Title + subtitle + legend. Returns (svg, y of the content below).

    labels must match whatever the chart itself prints, or the legend and the
    bars end up calling the same model two different things.
    """
    o = [f'<text class="h1" x="24" y="{y0}">{html.escape(title)}</text>',
         f'<text class="h2" x="24" y="{y0+20}">{html.escape(sub)}</text>']
    y = y0 + 44
    x = 24
    for key, lab, cls in SERIES:
        name = (labels or {}).get(key, lab)
        fill = {"s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a"}[cls]
        o.append(f'<rect x="{x}" y="{y-9}" width="10" height="10" rx="3" fill="{fill}"/>')
        o.append(f'<text class="lg" x="{x+16}" y="{y}">{html.escape(name)}</text>')
        x += 20 + int(len(name) * 6.1) + 18
    return "".join(o), y + 16


def footer(w: int, h: int, intervals: bool = True) -> str:
    # Only claim intervals on charts that actually draw them — the failure-mode
    # chart plots raw counts, and a provenance line that overstates the method is
    # worse than none, especially on an image that travels without its page.
    ci = " &#183; 95% Wilson intervals" if intervals else ""
    return (f'<text class="fo" x="24" y="{h-12}">307-case dev set &#183; binary '
            f"deterministic scorer{ci} &#183; 2026-08-10</text>")


def canvas(w: int, h: int, body: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}">'
            f"<style>{STYLE}</style>"
            f'<rect width="{w}" height="{h}" fill="#FFFFFF"/>'
            f"{body}</svg>")


def nest(svg: str, x: float, y: float, w: int, h: int) -> str:
    """Embed a chart svg (which carries its own viewBox) at a position."""
    inner = svg.split(">", 1)[1].rsplit("</svg>", 1)[0]
    vb = svg.split('viewBox="', 1)[1].split('"', 1)[0]
    return (f'<svg x="{x}" y="{y}" width="{w}" height="{h}" viewBox="{vb}">'
            f"{inner}</svg>")


# Both local models are Q4_K_M — read from each GGUF's own general.file_type
# (enum 15), not inferred from the filename. Naming the quant in the label matters
# because "Gemma-4 E4B" alone is ambiguous: the unquantized model is a different
# artifact from the one the wallet actually ships and this eval actually scored.
QUANT_LABELS = {
    "gemma4-e4b-base": "Gemma-4 E4B Q4_K_M",
    "gemma4-e4b-ft": "Fine-tuned Gemma-4 E4B Q4_K_M",
    "gpt-5": "gpt-5",
}


def build_overall(D, show_ci: bool = True) -> tuple[str, int, int]:
    pad_l, chart_w = 250, 800
    w = chart_w + 40
    sub = ("One tool call per request, scored exactly. 307 cases, same prompt, "
           "no cached responses.")
    head, y = header("Overall accuracy on the wallet tool-calling eval", sub, w,
                     labels=QUANT_LABELS)
    ch = headline(D["overall"], w=chart_w, show_ci=show_ci,
                  labels=QUANT_LABELS, pad_l=pad_l)
    ch_h = 10 + 54 * 3 + 26
    body = head + nest(ch, 12, y, chart_w, ch_h)
    h = y + ch_h + 34
    return canvas(w, h, body + footer(w, h, intervals=show_ci)), w, h


def build_modes(D) -> tuple[str, int, int]:
    modes = D["failure_modes"]
    order = [m for m in ["wrong amount", "wrong args", "wrong recipient",
                         "wrong token", "wrong tool", "no tool call",
                         "acted when it should refuse"]
             if any(modes[k].get(m) for k, _, _ in SERIES)]
    w = 740
    head, y = header("What each model gets wrong",
                     "Failure counts by kind. Denominators are identical, so these "
                     "are counts, not rates.", w)
    ch = modes_chart(order, modes)
    ch_h = 8 + 62 * len(order) + 8
    body = head + nest(ch, 12, y, 680, ch_h)
    h = y + ch_h + 34
    return canvas(w, h, body + footer(w, h, intervals=False)), w, h


def build_categories(D) -> tuple[str, int, int]:
    cats = D["categories"]
    cols, pw, ph = 4, 236, 202
    rows = (len(cats) + cols - 1) // cols
    w = 24 * 2 + cols * pw
    head, y = header("Accuracy by eval category",
                     "Fixed 0-100 axis on every panel so panels stay comparable. "
                     "Refusal panels have n = 1-2 - read them as direction, not rate.",
                     w)
    cells = []
    for i, c in enumerate(cats):
        cx = 24 + (i % cols) * pw
        cy = y + (i // cols) * ph
        cells.append(nest(panel(c["category"], c["models"], c["n"]), cx, cy, pw, ph))
    h = y + rows * ph + 34
    return canvas(w, h, head + "".join(cells) + footer(w, h)), w, h


def render(svg: str, w: int, stem: str, outdir: Path) -> Path:
    svg_p, png_p, jpg_p = (outdir / f"{stem}.svg", outdir / f"{stem}.png",
                           outdir / f"{stem}.jpg")
    svg_p.write_text(svg)
    subprocess.run(["rsvg-convert", "-w", str(w * SCALE), str(svg_p), "-o", str(png_p)],
                   check=True)
    subprocess.run(["sips", "-s", "format", "jpeg", "-s", "formatOptions",
                    str(JPEG_QUALITY), str(png_p), "--out", str(jpg_p)],
                   check=True, stdout=subprocess.DEVNULL)
    png_p.unlink()
    svg_p.unlink()
    return jpg_p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--outdir", default="charts")
    ap.add_argument("--no-ci", action="store_true",
                    help="drop the Wilson whiskers from the overall chart")
    ap.add_argument("--only", help="render just one chart by stem prefix, e.g. 02")
    a = ap.parse_args()
    D = json.loads(Path(a.data).read_text())
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    jobs = [("01-accuracy-by-category", build_categories),
            ("02-overall-accuracy", lambda d: build_overall(d, show_ci=not a.no_ci)),
            ("03-failure-modes", build_modes)]
    if a.only:
        jobs = [j for j in jobs if j[0].startswith(a.only)]
    for stem, fn in jobs:
        svg, w, h = fn(D)
        p = render(svg, w, stem, out)
        dims = subprocess.run(["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(p)],
                              capture_output=True, text=True).stdout
        px = [l.split(":")[-1].strip() for l in dims.splitlines() if ":" in l][-2:]
        print(f"{p.resolve()}   {'x'.join(px)} px   "
              f"{p.stat().st_size//1024} KB")


if __name__ == "__main__":
    main()
