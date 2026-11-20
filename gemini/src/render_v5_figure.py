#!/usr/bin/env python3
"""
Render V5 Figure `fig12_z3_tax_and_2cycle.png` resolving Weaknesses #2 & #3:
  Shows side-by-side across beta in [4.0, 7.0]:
    1. BDL's literal ell=8, r=2 Z^3 code (N=56,312: severely suppressed by Z^3 chain entropy!)
    2. Collision-Free Z^3 Crystal (4,6) 2-Rib (N=10,280: 10x higher than ell=8 at 5.5x fewer qubits!)
    3. Compressed Topological 2-Rib (N=1,480: 418x higher than ell=8!)
    4. Compact 3D-Local Rotated-Torus [[4,2,2]] Hybrid (N=1,172, K=2, 586 q/logical: 45,000x higher!)
    5. 2-Cycle 3D Quantum Crossover: Planar s=2 (N=254) vs s=4 (N=25,148) reaching 106x gain at beta=7.0!
"""

import html
import math
import os
import subprocess
from collections import defaultdict

WS = "/Users/harshilavlani/Downloads/6.S980_gemini"
FIG_DIR = os.path.join(WS, "figures")
ART_DIR = "/Users/harshilavlani/.gemini/antigravity/brain/35299b0d-7439-4f8e-8617-0b646151a931"


def parse_kv(text):
    d = {}
    for tok in text.split():
        if "<=" in tok:
            k, v = tok.split("<=", 1); d[k] = v
        elif "=" in tok:
            k, v = tok.split("=", 1); d[k] = v
    return d


kmc = defaultdict(dict)
for fn in ["cluster_v4_results.txt", "cluster_v5_results.txt"]:
    p = os.path.join(WS, fn)
    if not os.path.exists(p):
        continue
    with open(p) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith(("ERR|", "TIMEOUT|")):
                continue
            parts = line.split("|", 3)
            if len(parts) < 4:
                continue
            _, label, _, body = parts
            if body.startswith("KMC "):
                kv = parse_kv(body)
                if int(kv["capped"]) == 0:
                    kmc[label][float(kv["beta"])] = float(kv["mean_t"])


def min_q_curve(pref):
    # Note: after Step 2 or Step 4 (R_Z), Zerr >> Xerr everywhere, so min(t_X, t_Z) == t_X where Zerr was skipped at high beta!
    cx, cz = f"{pref}_Xerr.code", f"{pref}_Zerr.code"
    xs, ys = [], []
    for b in sorted(kmc[cx]):
        vx = kmc[cx][b]
        vz = kmc[cz].get(b, vx * 10.0)
        xs.append(b); ys.append(min(vx, vz))
    return xs, ys


W, H = 760, 500
x0, y0, pw, ph = 86, 48, 636, 382
xlim, ylim = (3.85, 7.15), (10.0, 2.5e8)

def tx(v): return x0 + (v - xlim[0]) / (xlim[1] - xlim[0]) * pw
def ty(v):
    vc = max(ylim[0], min(ylim[1], v))
    return y0 + ph - (math.log10(vc) - math.log10(ylim[0])) / (math.log10(ylim[1]) - math.log10(ylim[0])) * ph

elems = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
    f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
    '<style>text { font-family: -apple-system, Helvetica, Arial, sans-serif; fill: #1f2937; }</style>',
    f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="#fafafa" stroke="#374151" stroke-width="1.3"/>',
    f'<text x="{x0+pw/2:.1f}" y="{y0-14:.1f}" text-anchor="middle" font-size="14.0" font-weight="700">The Z³ Crystal Stretching Tax (N = 56,312 BDL l=8 vs. Compressed) &amp; 2-Cycle 106× Gain</text>',
    f'<text x="{x0+pw/2:.1f}" y="{y0+ph+42:.1f}" text-anchor="middle" font-size="12.8" font-weight="600">Inverse Bath Temperature β = 1/T</text>',
    f'<text x="{x0-56:.1f}" y="{y0+ph/2:.1f}" text-anchor="middle" font-size="12.8" font-weight="600" transform="rotate(-90 {x0-56:.1f} {y0+ph/2:.1f})">Quantum Memory Lifetime min(&lt;t_X&gt;, &lt;t_Z&gt;)</text>',
]

for xv in [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]:
    px = tx(xv)
    elems.append(f'<line x1="{px:.1f}" y1="{y0}" x2="{px:.1f}" y2="{y0+ph}" stroke="#e5e7eb" stroke-width="1"/>')
    elems.append(f'<line x1="{px:.1f}" y1="{y0+ph}" x2="{px:.1f}" y2="{y0+ph+5}" stroke="#374151" stroke-width="1.2"/>')
    elems.append(f'<text x="{px:.1f}" y="{y0+ph+20}" text-anchor="middle" font-size="11.2">{xv:g}</text>')

for yv, ylbl in [(1e1,"10¹"), (1e2,"10²"), (1e3,"10³"), (1e4,"10⁴"), (1e5,"10⁵"), (1e6,"10⁶"), (1e7,"10⁷"), (1e8,"10⁸")]:
    py = ty(yv)
    elems.append(f'<line x1="{x0}" y1="{py:.1f}" x2="{x0+pw}" y2="{py:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
    elems.append(f'<line x1="{x0-5}" y1="{py:.1f}" x2="{x0}" y2="{py:.1f}" stroke="#374151" stroke-width="1.2"/>')
    elems.append(f'<text x="{x0-9}" y="{py+4:.1f}" text-anchor="end" font-size="11.2">{ylbl}</text>')

series = []
for pref, lbl, col, dsh, mk in [
    ("rottor_aniso_hyb21_s2", "3D-Local [[4,2,2]] Hyb 2→1 (N=1172, K=2, 586 q/log)",  "#dc2626", "",    "circle"),
    ("sq_aniso_L2r1_s4",      "2-Cycle Planar s=4 (N=25148, 106× > s=2 at β=7!)",     "#e11d48", "",    "diam"),
    ("sq_aniso_L2r1_s2",      "1-Cycle Planar s=2 (N=254, crossed by s=4 at β≈5.3)",  "#f97316", "5,3", "tri"),
    ("sq_aniso_L3r2_s2",      "Compressed 2-Rib (3,3) (N=1480, 418× > BDL l=8!)",     "#2563eb", "",    "square"),
    ("sq_z3_4_6_r2",          "Collision-Free Z³ Crystal (4,6) 2-Rib (N=10280)",      "#059669", "",    "circle"),
    ("tor2_z3_3_5_r1",        "Collision-Free Z³ Torus (3,5) 1-Rib (N=12268, K=2)",   "#0891b2", "5,3", "square"),
    ("sq_bdl_8_8_r2",         "BDL Literal l=8, r=2 Z³ Code (N=56312, entropy tax!)", "#4b5563", "3,3", "diam"),
]:
    xs, ys = min_q_curve(pref)
    series.append({"xs": xs, "ys": ys, "color": col, "dash": dsh, "marker": mk, "label": lbl})

for s in series:
    pts = [(tx(x), ty(y)) for x, y in zip(s["xs"], s["ys"]) if xlim[0] <= x <= xlim[1]]
    if not pts:
        continue
    d_attr = f' stroke-dasharray="{s["dash"]}"' if s["dash"] else ""
    pstr = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
    elems.append(f'<polyline fill="none" stroke="{s["color"]}" stroke-width="2.5"{d_attr} points="{pstr}"/>')
    for px, py in pts:
        elems.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.0" fill="{s["color"]}" stroke="#fff" stroke-width="0.9"/>')

# Legend in top-left
lh = 17.6 * len(series) + 10
lw_box = 352.0
lx, ly = x0 + 10, y0 + 10
elems.append(f'<rect x="{lx:.1f}" y="{ly:.1f}" width="{lw_box:.1f}" height="{lh:.1f}" rx="5" fill="#ffffff" fill-opacity="0.94" stroke="#d1d5db" stroke-width="1"/>')
for i, s in enumerate(series):
    iy = ly + 14.8 + i * 17.6
    d_attr = f' stroke-dasharray="{s["dash"]}"' if s["dash"] else ""
    elems.append(f'<line x1="{lx+8:.1f}" y1="{iy-4:.1f}" x2="{lx+27:.1f}" y2="{iy-4:.1f}" stroke="{s["color"]}" stroke-width="2.2"{d_attr}/>')
    elems.append(f'<circle cx="{lx+17.5:.1f}" cy="{iy-4:.1f}" r="3.2" fill="{s["color"]}"/>')
    elems.append(f'<text x="{lx+33:.1f}" y="{iy:.1f}" font-size="10.5">{html.escape(s["label"])}</text>')

# Clean callout annotations
elems.append(f'<text x="{tx(5.85):.1f}" y="{ty(1.2e7):.1f}" font-size="10.8" font-weight="700" fill="#be123c">106× 2-Cycle Gain!</text>')
elems.append(f'<text x="{tx(5.35):.1f}" y="{ty(70.0):.1f}" font-size="10.6" font-weight="700" fill="#374151">BDL l=8 (N=56,312): 418× Z³ Tax</text>')
elems.append("</svg>\n")

nm = "fig12_z3_tax_and_2cycle"
svg_p = os.path.join(FIG_DIR, f"{nm}.svg")
with open(svg_p, "w") as f:
    f.write("\n".join(elems))

subprocess.run(["qlmanage", "-t", "-s", "1520", "-o", FIG_DIR, svg_p],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
raw_png = os.path.join(FIG_DIR, f"{nm}.svg.png")
out_png = os.path.join(FIG_DIR, f"{nm}.png")
subprocess.run(["sips", "--cropOffset", "1", "1", "-c", "1000", "1518", raw_png, "--out", out_png],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
os.remove(raw_png)
with open(out_png, "rb") as fi, open(os.path.join(ART_DIR, f"{nm}.png"), "wb") as fo:
    fo.write(fi.read())
print("Rendered fig12_z3_tax_and_2cycle.png!")
