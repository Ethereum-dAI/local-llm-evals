#!/usr/bin/env python
"""Render the three sign-off charts as one self-contained, shareable HTML page.

    uv run python scripts/chart_data.py -o chartdata.json
    uv run python scripts/build_eval_charts.py chartdata.json -o eval-charts.html

Self-contained by requirement, not preference: this is served through
htmlpreview.github.io off a gist, so there is no origin to load assets from.
No external CSS, fonts, or scripts — everything inline.

Aggregate only. Category pass rates and failure counts carry no case text and no
gold answers, so the page can be shared where the dataset itself cannot.

Palette is the validated 3-slot categorical set; colour means the same model in
every chart. Light-mode aqua is under the 3:1 contrast gate, so the relief rule
applies — every bar carries a direct label AND the page ships a table view.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH_IMG = REPO / "charts" / "chart-scores.jpg"

SERIES = [
    ("gemma4-e4b-base", "Stock E4B", "s1"),
    ("gemma4-e4b-ft", "Fine-tuned E4B", "s2"),
    ("gpt-5", "gpt-5", "s3"),
]
SHORT = {"gemma4-e4b-base": "stock", "gemma4-e4b-ft": "ft", "gpt-5": "gpt-5"}
LOW_N = 5  # below this, a per-category rate is not worth reading as a rate


def esc(s) -> str:
    return html.escape(str(s))


def tip(label: str, body: str) -> str:
    """data-tip payload for the shared hover layer."""
    return f'data-tip="{esc(label)}&#10;{esc(body)}"'


def nice(cat: str) -> str:
    return (cat.replace("generated-", "")
               .replace("safety-refusal-", "refuse: ")
               .replace("multiturn-", "multi-turn "))


def panel(cat: str, models: dict, n: int, w=236, h=202) -> str:
    # pad_t must clear the subtitle AND the value label of a 100% bar. At p = 1.0
    # the bar top is the plot top, its label sits 6px above that, and the "n =" line
    # is at y = 25 — so anything under ~46 collides (ablation-recipient, where two
    # models score 100%, is the case that shows it). Plot height is held at 126 by
    # growing h in step, so panels stay the same size as before.
    pad_l, pad_r, pad_t, pad_b = 30, 8, 46, 30
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    thin = n < LOW_N
    o = [f'<svg viewBox="0 0 {w} {h}" role="img" '
         f'aria-label="{esc(nice(cat))}, n={n}">']
    o.append(f'<text x="{w/2:.0f}" y="14" class="pt">{esc(nice(cat))}</text>')
    o.append(f'<text x="{w/2:.0f}" y="25" class="pn{" warn" if thin else ""}">'
             f'n = {n}{" — too few to read as a rate" if thin else ""}</text>')
    for g in (0, 25, 50, 75, 100):
        y = pad_t + ph - ph * g / 100
        o.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w-pad_r}" y2="{y:.1f}" class="grid"/>')
        o.append(f'<text x="{pad_l-6}" y="{y+3:.1f}" class="ax">{g}</text>')
    slot = pw / len(SERIES)
    bw = min(30, slot - 12)          # >=2px surface gap between adjacent bars
    for i, (key, lab, cls) in enumerate(SERIES):
        m = models[key]
        cx = pad_l + slot * i + slot / 2
        bh = ph * m["p"]
        y = pad_t + ph - bh
        t = tip(f'{lab} · {nice(cat)}',
                f'{m["k"]}/{m["n"]} = {m["p"]*100:.1f}%  (95% CI '
                f'{m["lo"]*100:.0f}–{m["hi"]*100:.0f})')
        o.append(f'<rect x="{cx-bw/2:.1f}" y="{y:.1f}" width="{bw:.1f}" '
                 f'height="{max(bh,0.8):.1f}" rx="4" class="bar {cls}" {t}/>')
        lo, hi = pad_t + ph - ph * m["lo"], pad_t + ph - ph * m["hi"]
        o.append(f'<line x1="{cx:.1f}" y1="{hi:.1f}" x2="{cx:.1f}" y2="{lo:.1f}" class="whisk"/>')
        for yy in (hi, lo):
            o.append(f'<line x1="{cx-4:.1f}" y1="{yy:.1f}" x2="{cx+4:.1f}" y2="{yy:.1f}" class="whisk"/>')
        o.append(f'<text x="{cx:.1f}" y="{hi-6:.1f}" class="vl">{m["p"]*100:.0f}</text>')
        o.append(f'<text x="{cx:.1f}" y="{h-10:.0f}" class="xl">{esc(SHORT[key])}</text>')
    o.append("</svg>")
    return "".join(o)


def headline(overall: dict, w=680, rowh=54, show_ci: bool = True,
             labels: dict | None = None, pad_l: int = 150) -> str:
    """Overall accuracy, one bar per model.

    show_ci=False drops the Wilson whiskers. Worth doing deliberately: with all
    307 cases scored and the three intervals far apart, the intervals add nothing
    to the ranking and cost a reader who has to be told what they are. Keep them
    wherever a difference is small enough that someone might over-read it.

    labels overrides the display name per model key (e.g. to name the quant), and
    pad_l must grow with it — the name is right-anchored at pad_l - 12.
    """
    pad_r, pad_t = 96, 10
    names = {k: (labels or {}).get(k, lab) for k, lab, _ in SERIES}
    h = pad_t + rowh * len(SERIES) + 26
    pw = w - pad_l - pad_r
    o = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Overall accuracy">']
    for g in (0, 25, 50, 75, 100):
        x = pad_l + pw * g / 100
        o.append(f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{h-24}" class="grid"/>')
        o.append(f'<text x="{x:.1f}" y="{h-8}" class="ax mid">{g}%</text>')
    for i, (key, lab, cls) in enumerate(SERIES):
        m = overall[key]
        y = pad_t + rowh * i + rowh / 2
        bh = 26
        name = names[key]
        t = tip(name, f'{m["k"]}/{m["n"]} = {m["p"]*100:.1f}%  (95% CI '
                      f'{m["lo"]*100:.1f}–{m["hi"]*100:.1f})')
        o.append(f'<text x="{pad_l-12}" y="{y+5:.1f}" class="yl big">{esc(name)}</text>')
        o.append(f'<rect x="{pad_l}" y="{y-bh/2:.1f}" width="{max(pw*m["p"],1):.1f}" '
                 f'height="{bh}" rx="4" class="bar {cls}" {t}/>')
        if show_ci:
            x1, x2 = pad_l + pw * m["lo"], pad_l + pw * m["hi"]
            o.append(f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" class="whisk"/>')
            for xx in (x1, x2):
                o.append(f'<line x1="{xx:.1f}" y1="{y-5:.1f}" x2="{xx:.1f}" y2="{y+5:.1f}" class="whisk"/>')
        # Labels clear the whisker when it is drawn, the bar end when it is not.
        lx = pad_l + pw * (m["hi"] if show_ci else m["p"]) + 10
        o.append(f'<text x="{lx:.1f}" y="{y+1:.1f}" class="vl start big">'
                 f'{m["p"]*100:.1f}%</text>')
        o.append(f'<text x="{lx:.1f}" y="{y+13:.1f}" class="sub start">'
                 f'{m["k"]}/{m["n"]}</text>')
    o.append("</svg>")
    return "".join(o)


def modes_chart(order: list, modes: dict, w=680, rowh=62) -> str:
    # pad_l holds the longest mode name, right-anchored at pad_l-12. The longest
    # is "acted when it should refuse" — 27 chars of 12px mono, ~194px — which was
    # clipped off the left edge at the previous 190.
    pad_l, pad_r, pad_t = 220, 52, 8
    h = pad_t + rowh * len(order) + 8
    pw = w - pad_l - pad_r
    mx = max(max(modes[k].get(c, 0) for k, _, _ in SERIES) for c in order) or 1
    o = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Failure modes by model">']
    for i, c in enumerate(order):
        gy = pad_t + rowh * i
        if i:
            o.append(f'<line x1="{pad_l-178}" y1="{gy:.1f}" x2="{w-pad_r}" y2="{gy:.1f}" class="sep"/>')
        o.append(f'<text x="{pad_l-12}" y="{gy+rowh/2+4:.1f}" class="yl">{esc(c)}</text>')
        for j, (key, lab, cls) in enumerate(SERIES):
            v = modes[key].get(c, 0)
            bh = 14
            y = gy + 9 + j * (bh + 2)      # 2px surface gap between bars
            t = tip(lab, f'{v} failure{"" if v == 1 else "s"} — {c}')
            o.append(f'<rect x="{pad_l}" y="{y}" width="{max(pw*v/mx,0.8):.1f}" '
                     f'height="{bh}" rx="4" class="bar {cls}" {t}/>')
            o.append(f'<text x="{pad_l+pw*v/mx+7:.1f}" y="{y+11}" class="vl start">{v}</text>')
    o.append("</svg>")
    return "".join(o)


CSS = """
:root{
  --bg:#FAFAF9; --card:#FFFFFF; --ink:#16181C; --ink2:#555C66; --ink3:#868E9A;
  --rule:#E2E5EA; --grid:#ECEEF2; --accent:#2a78d6;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a;
  --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  --serif:ui-serif,Georgia,"Times New Roman",serif;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#0F1115; --card:#171A1F; --ink:#E9ECEF; --ink2:#A4ADB8; --ink3:#737D89;
  --rule:#262B32; --grid:#212630; --accent:#3987e5;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70;
}}
:root[data-theme="dark"]{
  --bg:#0F1115; --card:#171A1F; --ink:#E9ECEF; --ink2:#A4ADB8; --ink3:#737D89;
  --rule:#262B32; --grid:#212630; --accent:#3987e5;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
  font-size:15px;line-height:1.55;-webkit-font-smoothing:antialiased}
.wrap{max-width:1000px;margin:0 auto;padding:46px 24px 88px;
  display:flex;flex-direction:column;gap:42px}
header{border-bottom:2px solid var(--ink);padding-bottom:20px;
  display:flex;flex-direction:column;gap:11px}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--accent)}
h1{font-family:var(--serif);font-size:clamp(27px,3.8vw,38px);font-weight:600;
  margin:0;line-height:1.13;text-wrap:balance;letter-spacing:-.01em}
.lede{margin:0;color:var(--ink2);max-width:70ch}
figure{margin:0;background:var(--card);border:1px solid var(--rule);
  display:flex;flex-direction:column}
figcaption{padding:16px 20px 0;display:flex;flex-direction:column;gap:6px}
h2{font-family:var(--serif);font-size:21px;font-weight:600;margin:0;
  letter-spacing:-.005em}
.say{color:var(--ink2);font-size:14px;max-width:74ch;margin:0}
.say b{color:var(--ink);font-weight:600}
.legend{display:flex;gap:16px;flex-wrap:wrap;padding:12px 20px 0;
  font-size:12.5px;color:var(--ink2)}
.legend i{display:inline-block;width:10px;height:10px;border-radius:3px;
  margin-right:6px;vertical-align:-1px}
.plot{padding:14px 20px 18px;overflow-x:auto}
.grid-sm{display:grid;grid-template-columns:repeat(auto-fill,minmax(236px,1fr));
  gap:6px}
svg{display:block;max-width:100%;height:auto;overflow:visible}
/* The benchmark chart is a light-mode screenshot from another harness, so it
   keeps its own white ground; the border makes that read as an inset panel
   rather than a hole when the page is in dark mode. */
.plot img{display:block;width:100%;height:auto;background:#fff;
  border:1px solid var(--rule);border-radius:4px}
.bar{transition:opacity .12s}
.bar:hover{opacity:.72;cursor:default}
.bar.s1{fill:var(--s1)} .bar.s2{fill:var(--s2)} .bar.s3{fill:var(--s3)}
.grid{stroke:var(--grid);stroke-width:1}
.sep{stroke:var(--rule);stroke-width:1}
.whisk{stroke:var(--ink2);stroke-width:1.5}
.pt{font-size:11px;font-weight:600;fill:var(--ink);text-anchor:middle}
.pn{font-size:9.5px;fill:var(--ink3);text-anchor:middle;font-family:var(--mono)}
.pn.warn{fill:var(--s2)}
.ax{font-size:9px;fill:var(--ink3);text-anchor:end;font-family:var(--mono)}
.ax.mid{text-anchor:middle}
.xl{font-size:9.5px;fill:var(--ink2);text-anchor:middle;font-family:var(--mono)}
.yl{font-size:12px;fill:var(--ink2);text-anchor:end;font-family:var(--mono)}
.yl.big{font-size:14px;fill:var(--ink);font-family:var(--sans);font-weight:600}
.vl{font-size:10.5px;font-weight:700;fill:var(--ink);text-anchor:middle;
  font-family:var(--mono);font-variant-numeric:tabular-nums}
.vl.start{text-anchor:start} .vl.big{font-size:15px}
.sub{font-size:10px;fill:var(--ink3);font-family:var(--mono)}
.sub.start{text-anchor:start}
table{border-collapse:collapse;width:100%;font-size:13px;min-width:560px}
th{font-family:var(--mono);font-size:10px;letter-spacing:.08em;
  text-transform:uppercase;color:var(--ink2);text-align:right;padding:9px 12px;
  border-bottom:1px solid var(--rule);font-weight:600;white-space:nowrap}
th:first-child{text-align:left}
td{padding:7px 12px;border-bottom:1px solid var(--grid);text-align:right;
  font-family:var(--mono);font-variant-numeric:tabular-nums}
td:first-child{text-align:left}
details{background:var(--card);border:1px solid var(--rule)}
summary{padding:13px 20px;cursor:pointer;font-family:var(--mono);font-size:12px;
  letter-spacing:.06em;text-transform:uppercase;color:var(--ink2)}
summary:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
.tbl-scroll{padding:0 20px 18px;overflow-x:auto}
#tip{position:fixed;pointer-events:none;opacity:0;transition:opacity .1s;
  background:var(--ink);color:var(--bg);padding:7px 10px;border-radius:5px;
  font-family:var(--mono);font-size:11.5px;line-height:1.45;white-space:pre;
  z-index:9;max-width:min(300px,90vw)}
.note{background:var(--card);border-left:3px solid var(--accent);
  padding:15px 19px;font-size:14px;color:var(--ink2);max-width:76ch}
.note strong{color:var(--ink)}
footer{border-top:1px solid var(--rule);padding-top:16px;color:var(--ink3);
  font-family:var(--mono);font-size:11.5px;line-height:1.7}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""

JS = """
(function(){
 var tip=document.getElementById('tip');
 function show(e){var t=e.target.getAttribute&&e.target.getAttribute('data-tip');
  if(!t){hide();return;} tip.textContent=t; tip.style.opacity='1';
  var r=tip.getBoundingClientRect(), x=e.clientX+14, y=e.clientY+14;
  if(x+r.width>innerWidth-8) x=e.clientX-r.width-14;
  if(y+r.height>innerHeight-8) y=e.clientY-r.height-14;
  tip.style.left=x+'px'; tip.style.top=y+'px';}
 function hide(){tip.style.opacity='0';}
 document.addEventListener('mousemove',show);
 document.addEventListener('mouseleave',hide);
})();
"""

LEGEND = ('<div class="legend">'
          '<span><i style="background:var(--s1)"></i>Stock E4B &mdash; ships on-device today</span>'
          '<span><i style="background:var(--s2)"></i>Fine-tuned E4B &mdash; launch candidate</span>'
          '<span><i style="background:var(--s3)"></i>gpt-5 &mdash; frontier anchor</span>'
          "</div>")


def bench_figure(img: Path) -> str:
    """The general-capability sanity check, embedded as a data URI.

    Not drawn from `chartdata.json` like the other three: these are
    lm-evaluation-harness runs, not this harness's, so the image is the record.
    Inlined rather than linked because the page has no origin to load from.
    Missing file -> the figure is simply omitted, so the page still builds on a
    clone without `charts/`.
    """
    if not img.exists():
        return ""
    b64 = base64.b64encode(img.read_bytes()).decode()
    return (
        '<figure><figcaption><h2>Sanity check: the fine-tune did not cost'
        " general capability</h2>"
        '<p class="say">Tool calling is a narrow skill, and the usual worry about'
        " fine-tuning for one is that everything else regresses. It did not."
        " Across six standard benchmarks nothing collapses: <b>MMLU is 4.4 points"
        " higher</b> after fine-tuning, <b>TruthfulQA 2.2 lower</b>, and the other"
        " four move by about a point or less &mdash; so the 12.7% &rarr; 80.1%"
        " above is a skill that was added, not traded for. The Q4_K_M columns are"
        " the quantized on-device builds: quantization costs 1&ndash;4.6 points"
        " depending on the benchmark, and costs the two models about the same, so"
        " the base-vs-fine-tune comparison survives it. Different harness"
        " (lm-evaluation-harness 0.4.12), not the 307 cases.</p>"
        f'</figcaption><div class="plot"><img src="data:image/jpeg;base64,{b64}"'
        ' alt="Grouped bar chart. MMLU, GSM8K, HellaSwag, ARC-Challenge,'
        " Winogrande and TruthfulQA scores for four variants: gemma-4-E4B-it,"
        " gemma-4-E4B-wallet-ft, and the Q4_K_M quantization of each. Within each"
        ' benchmark all four bars are close; no variant collapses."></div></figure>')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--benchmarks", type=Path, default=BENCH_IMG,
                    help="general-capability chart image (omitted if missing)")
    a = ap.parse_args()
    D = json.loads(Path(a.data).read_text())
    cats, over, modes = D["categories"], D["overall"], D["failure_modes"]

    P = ['<div id="tip"></div>', '<div class="wrap">', "<header>",
         '<div class="eyebrow">Wallet tool-calling eval &middot; 307 cases &middot; 2026-08-10</div>',
         "<h1>Can a fine-tuned on-device model do the wallet's tool calling?</h1>",
         '<p class="lede">Every case asks one thing: turn a natural-language request'
         " into the correct structured tool call. Scoring is binary and deterministic"
         " &mdash; the arguments match the computed gold answer, or they do not."
         " Three models, the same 307 cases, the same prompt, run fresh with no"
         " cached responses. Bars carry 95% Wilson intervals; hover any bar for"
         " counts.</p>", "</header>"]

    # 02 first — the headline frames everything
    P.append(
        '<figure><figcaption><h2>Overall accuracy</h2>'
        '<p class="say">The fine-tune takes the on-device model from <b>12.7% to'
        " 80.1%</b> &mdash; a 6.3&times; improvement that closes most of the gap to"
        " gpt-5. The three intervals do not overlap, so the ordering is solid.</p>"
        f'</figcaption>{LEGEND}<div class="plot">{headline(over)}</div></figure>')

    # Sanity check, straight after the headline: the number above is only worth
    # reading if the model is still a general model. Sits here rather than in an
    # appendix because that objection lands on the headline, not on the details.
    P.append(bench_figure(a.benchmarks))

    # 01 — small multiples
    panels = "".join(f"<div>{panel(c['category'], c['models'], c['n'])}</div>"
                     for c in cats)
    P.append(
        '<figure><figcaption><h2>Accuracy by category</h2>'
        '<p class="say">The fine-tune matches gpt-5 on the <b>ablation</b> categories'
        " (where the right answer is to ask a question rather than guess) and trails"
        " it most on <b>multi-turn</b> requests, where the needed detail arrives"
        " across several messages. Panels share a fixed 0&ndash;100 axis so they stay"
        " comparable; the four refusal panels have <b>n = 1&ndash;2</b> and are marked"
        " accordingly &mdash; read those as direction, never as a rate.</p>"
        f'</figcaption>{LEGEND}<div class="plot"><div class="grid-sm">{panels}</div>'
        "</div></figure>")

    # 03 — failure modes
    order = [m for m in ["wrong amount", "wrong args", "wrong recipient",
                         "wrong token", "wrong tool", "no tool call",
                         "acted when it should refuse"]
             if any(modes[k].get(m) for k, _, _ in SERIES)]
    P.append(
        '<figure><figcaption><h2>What each model gets wrong</h2>'
        '<p class="say">Failure counts by kind &mdash; the chart that explains the'
        " headline. The stock model's errors are overwhelmingly <b>wrong amount</b>"
        " (147): it does not reliably convert a human amount into base units, which"
        " is precisely the skill this eval was built to isolate. Fine-tuning cuts"
        " that to 35. gpt-5's remaining failures are mostly <b>no tool call</b> (9)"
        " &mdash; it asks a clarifying question or declines instead of acting, which"
        " is over-caution rather than miscalculation.</p>"
        f'</figcaption>{LEGEND}<div class="plot">{modes_chart(order, modes)}</div>'
        "</figure>")

    # Table view — the contrast-relief requirement, and the numbers behind the charts
    rows = ['<tr><th>Category</th><th>n</th>'
            + "".join(f"<th>{esc(l)}</th>" for _, l, _ in SERIES) + "</tr>"]
    for c in cats:
        cells = "".join(
            f'<td>{c["models"][k]["k"]}/{c["models"][k]["n"]} '
            f'({c["models"][k]["p"]*100:.0f}%)</td>' for k, _, _ in SERIES)
        rows.append(f'<tr><td>{esc(nice(c["category"]))}</td><td>{c["n"]}</td>{cells}</tr>')
    tot = "".join(f'<td><b>{over[k]["k"]}/{over[k]["n"]} '
                  f'({over[k]["p"]*100:.1f}%)</b></td>' for k, _, _ in SERIES)
    rows.append(f'<tr><td><b>Overall</b></td><td><b>307</b></td>{tot}</tr>')
    P.append('<details><summary>Table view &mdash; every number in the charts</summary>'
             f'<div class="tbl-scroll"><table>{"".join(rows)}</table></div></details>')

    P.append(
        '<div class="note"><strong>Two caveats worth stating plainly.</strong>'
        " First, the eval phrasings are drawn from the same 26 surface templates the"
        " model was fine-tuned on &mdash; the amounts and recipients differ (so the"
        " arithmetic is genuinely new), but this does not yet measure robustness to"
        " arbitrary user phrasing. Second, the fine-tune refuses burn-address sends,"
        " zero-address sends and unknown-spender approvals, but still executes"
        " unverified-token swaps (0 of 2) where gpt-5 refuses both. The training set"
        " contains exactly one example per safety category, which is the likely"
        " cause and a cheap fix.</div>")

    P.append('<footer>307-case dev set &middot; binary deterministic scorer &middot;'
             " 95% Wilson intervals &middot; no cached responses &middot; stock model"
             " SHA-verified against the GGUF the wallet ships &middot; palette"
             " validated for colour-vision deficiency in both themes</footer>")
    P.append("</div>")

    # A COMPLETE document, not a fragment: this is served raw through
    # htmlpreview.github.io, where nothing wraps it. The viewport meta is what
    # makes it readable on a phone, which is where a forwarded link often lands.
    doc = ('<!doctype html>\n<html lang="en">\n<head>\n'
           '<meta charset="utf-8">\n'
           '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
           "<title>Wallet tool-calling eval — model comparison</title>\n"
           f"<style>{CSS}</style>\n</head>\n<body>\n"
           + "\n".join(P)
           + f"\n<script>{JS}</script>\n</body>\n</html>\n")
    Path(a.out).write_text(doc)
    print(f"wrote {a.out} ({len(doc)//1024} KB)")


if __name__ == "__main__":
    main()
