"""Small server-side SVG charts. Colours come from CSS classes so they follow the page tokens."""
import numpy as np
from markupsafe import Markup


def _ticks(vmax, n=4):
    step = max(1, int(np.ceil(vmax / n)))
    if vmax > 40: step = int(np.ceil(vmax / n / 10.0) * 10)
    return list(range(0, int(vmax) + 1, step))


def histogram(counts, bins, med=None, q1=None, q3=None, w=640, h=240, label="Salary (LPA)"):
    ml, mr, mt, mb = 40, 10, 12, 42
    pw, ph = w - ml - mr, h - mt - mb
    top = bins[-1]; ymax = max(max(counts), 1) * 1.12
    X = lambda v: ml + v / top * pw
    Y = lambda c: mt + ph * (1 - c / ymax)
    o = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="Distribution of salaries">']
    for t in _ticks(ymax):
        o.append(f'<line class="c-grid" x1="{ml}" x2="{w - mr}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/><text class="c-tick" x="{ml - 8}" y="{Y(t) + 4:.1f}" text-anchor="end">{t}</text>')
    if q1 is not None and q3 is not None and q3 > q1:
        o.append(f'<rect class="c-range" x="{X(min(q1, top)):.1f}" y="{mt}" width="{max(X(min(q3, top)) - X(min(q1, top)), 2):.1f}" height="{ph}"/>')
    bw = pw * (bins[1] - bins[0]) / top
    for i, c in enumerate(counts):
        if c: o.append(f'<rect class="c-bar" x="{X(bins[i]) + 1:.1f}" y="{Y(c):.1f}" width="{bw - 2:.1f}" height="{ph * c / ymax:.1f}" rx="2"><title>{c} postings at {bins[i]}–{bins[i + 1]} LPA</title></rect>')
    if med is not None:
        o.append(f'<line class="c-med" x1="{X(min(med, top)):.1f}" x2="{X(min(med, top)):.1f}" y1="{mt}" y2="{mt + ph}"/>')
    for t in range(0, int(top) + 1, 10):
        o.append(f'<text class="c-tick" x="{X(t):.1f}" y="{h - 22}" text-anchor="middle">{t}{"+" if t == top else ""}</text>')
    o.append(f'<text class="c-axis" x="{ml + pw / 2:.0f}" y="{h - 4}" text-anchor="middle">{label}</text></svg>')
    return Markup("".join(o))


def experience_curve(xs, curve, exp_min, exp_max, mid, w=640, h=260):
    ml, mr, mt, mb = 40, 10, 12, 42
    pw, ph = w - ml - mr, h - mt - mb
    ymax = float(np.ceil(min(curve[:, 2].max(), 150) / 10) * 10)
    X = lambda v: ml + v / 25 * pw
    Y = lambda v: mt + ph * (1 - min(v, ymax) / ymax)
    o = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="Estimated salary by experience">']
    step = 10 if ymax <= 60 else 20 if ymax <= 120 else 30
    for t in range(0, int(ymax) + 1, step):
        o.append(f'<line class="c-grid" x1="{ml}" x2="{w - mr}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/><text class="c-tick" x="{ml - 8}" y="{Y(t) + 4:.1f}" text-anchor="end">{t}</text>')
    o.append(f'<rect class="c-exp" x="{X(exp_min):.1f}" y="{mt}" width="{max(X(exp_max) - X(exp_min), 2):.1f}" height="{ph}"/>')
    up = " ".join(f"{X(x):.1f},{Y(v):.1f}" for x, v in zip(xs, curve[:, 2]))
    dn = " ".join(f"{X(x):.1f},{Y(v):.1f}" for x, v in zip(xs[::-1], curve[::-1, 0]))
    o.append(f'<polygon class="c-band" points="{up} {dn}"/>')
    o.append(f'<polyline class="c-line" fill="none" points="{" ".join(f"{X(x):.1f},{Y(v):.1f}" for x, v in zip(xs, curve[:, 1]))}"/>')
    o.append(f'<circle class="c-dot" cx="{X(exp_min):.1f}" cy="{Y(mid):.1f}" r="5"/>')
    for t in range(0, 26, 5):
        o.append(f'<text class="c-tick" x="{X(t):.1f}" y="{h - 22}" text-anchor="middle">{t}</text>')
    o.append(f'<text class="c-axis" x="{ml + pw / 2:.0f}" y="{h - 4}" text-anchor="middle">Minimum experience asked (years)</text></svg>')
    return Markup("".join(o))


def band_ranges(bands, w=640, row=44):
    """Horizontal range bars: middle half of salaries with the median marked, one row per experience band."""
    if not bands: return Markup("")
    ml, mr = 84, 44
    h = row * len(bands) + 34
    vmax = max(b["q3"] for b in bands) * 1.1
    vmax = float(np.ceil(vmax / 5) * 5)
    pw = w - ml - mr
    X = lambda v: ml + v / vmax * pw
    o = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="Salary range by experience band">']
    for t in np.linspace(0, vmax, 6):
        o.append(f'<line class="c-grid" x1="{X(t):.1f}" x2="{X(t):.1f}" y1="6" y2="{h - 28}"/><text class="c-tick" x="{X(t):.1f}" y="{h - 10}" text-anchor="middle">{t:g}</text>')
    for i, b in enumerate(bands):
        y = 10 + i * row
        o.append(f'<text class="c-label" x="{ml - 12}" y="{y + 20}" text-anchor="end">{b["label"]}</text>')
        o.append(f'<rect class="c-span" x="{X(b["q1"]):.1f}" y="{y + 6}" width="{max(X(b["q3"]) - X(b["q1"]), 3):.1f}" height="20" rx="4"><title>{b["n"]} postings</title></rect>')
        o.append(f'<line class="c-med" x1="{X(b["med"]):.1f}" x2="{X(b["med"]):.1f}" y1="{y + 2}" y2="{y + 30}"/>')
        o.append(f'<text class="c-val" x="{X(b["q3"]) + 8:.1f}" y="{y + 21}">{b["med"]:.0f}</text>')
    o.append("</svg>")
    return Markup("".join(o))


def pulse(counts, bins, med, q1, q3, w=1200, h=250):
    """Wide salary ribbon for the hero: bars inside the middle half are bright, the rest dim; the median is marked."""
    ml, mr, mt, mb = 0, 0, 40, 34
    pw, ph = w - ml - mr, h - mt - mb
    top = bins[-1]; ymax = max(max(counts), 1)
    X = lambda v: ml + min(v, top) / top * pw
    bw = pw * (bins[1] - bins[0]) / top
    o = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Salary distribution of matching postings, median {med:.1f} LPA">']
    for i, c in enumerate(counts):
        if not c: continue
        hh = max(ph * c / ymax, 3)
        inside = bins[i + 1] > q1 and bins[i] < q3
        o.append(f'<rect class="p-bar{"" if inside else " dim"}" style="--i:{i}" x="{X(bins[i]) + 3:.1f}" y="{mt + ph - hh:.1f}" width="{bw - 6:.1f}" height="{hh:.1f}" rx="5"><title>{c} postings at {bins[i]}-{bins[i + 1]} LPA</title></rect>')
    mx = X(med)
    o.append(f'<line class="p-med" x1="{mx:.1f}" x2="{mx:.1f}" y1="{mt - 14}" y2="{mt + ph}"/>')
    anchor = "end" if mx > w - 200 else "start"
    o.append(f'<text class="p-lbl" x="{mx + (-10 if anchor == "end" else 10):.1f}" y="{mt - 18}" text-anchor="{anchor}">Median {med:.1f} LPA</text>')
    for t in range(0, int(top) + 1, 10):
        o.append(f'<text class="p-tick" x="{X(t):.1f}" y="{h - 10}" text-anchor="{"start" if t == 0 else "end" if t == top else "middle"}">{t}{"+" if t == top else ""}{" LPA" if t == 0 else ""}</text>')
    o.append("</svg>")
    return Markup("".join(o))
