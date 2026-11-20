#!/usr/bin/env python3
"""
Render 3 crisp V3 publication figures (760x500 SVG -> 1518x1000 PNG) for the Two-Track
(3D-Local vs Non-Local qLDPC) & Three-Simulation-Regime synthesis.
"""

import html
import math
import os
import subprocess
from collections import defaultdict

WS = "/Users/harshilavlani/Downloads/6.S980_gemini"
FIG_DIR = os.path.join(WS, "figures")
ART_DIR = "/Users/harshilavlani/.gemini/antigravity/brain/35299b0d-7439-4f8e-8617-0b646151a931"
os.makedirs(FIG_DIR, exist_ok=True)


def parse_kv(text):
    d = {}
    for tok in text.split():
        if "<=" in tok:
            k, v = tok.split("<=", 1); d[k] = v
        elif "=" in tok:
            k, v = tok.split("=", 1); d[k] = v
    return d


kmc_uni = defaultdict(dict)
kmc_jgrad = defaultdict(dict)
mfqec = defaultdict(dict)
singleshot = defaultdict(dict)

with open(os.path.join(WS, "cluster_v3_results.txt")) as f:
    for raw in f:
        line = raw.strip()
        if not line or line.startswith(("ERR|", "TIMEOUT|")):
            continue
        suite, label, _, body = line.split("|", 3)
        kv = parse_kv(body)
        if suite == "KMC_UNI" and body.startswith("KMC "):
            kmc_uni[label][float(kv["beta"])] = float(kv["mean_t"])
        elif suite == "KMC_JGRAD" and body.startswith("KMC "):
            kmc_jgrad[label][float(kv["beta"])] = float(kv["mean_t"])
        elif suite == "MFQEC" and body.startswith("MFQEC "):
            mfqec[label][float(kv["p_gate"])] = float(kv["P_fail"])
        elif suite == "SINGLESHOT" and body.startswith("SINGLESHOT "):
            singleshot[label][float(kv["p_data"])] = float(kv["P_fail"])


def save_single_plot(svg_path, title, xlabel, ylabel, xlim, ylim, xticks, yticks,
                     series, xlog=False, ylog=False, legend_pos="tl"):
    W, H = 760, 500
    x0, y0, pw, ph = 86, 48, 636, 382
    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<style>text { font-family: -apple-system, Helvetica, Arial, sans-serif; fill: #1f2937; }</style>',
        f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="#fafafa" stroke="#374151" stroke-width="1.3"/>',
        f'<text x="{x0 + pw/2:.1f}" y="{y0 - 14:.1f}" text-anchor="middle" font-size="14.2" font-weight="700">{html.escape(title)}</text>',
        f'<text x="{x0 + pw/2:.1f}" y="{y0 + ph + 42:.1f}" text-anchor="middle" font-size="13.0" font-weight="600">{html.escape(xlabel)}</text>',
        f'<text x="{x0 - 56:.1f}" y="{y0 + ph/2:.1f}" text-anchor="middle" font-size="13.0" font-weight="600" transform="rotate(-90 {x0 - 56:.1f} {y0 + ph/2:.1f})">{html.escape(ylabel)}</text>',
    ]

    def tx(v):
        lv = math.log10(v) if xlog else v
        l0 = math.log10(xlim[0]) if xlog else xlim[0]
        l1 = math.log10(xlim[1]) if xlog else xlim[1]
        return x0 + (lv - l0) / (l1 - l0) * pw

    def ty(v):
        vc = max(ylim[0], min(ylim[1], v))
        lv = math.log10(vc) if ylog else vc
        l0 = math.log10(ylim[0]) if ylog else ylim[0]
        l1 = math.log10(ylim[1]) if ylog else ylim[1]
        return y0 + ph - (lv - l0) / (l1 - l0) * ph

    for xv, xlbl in xticks:
        px = tx(xv)
        elems.append(f'<line x1="{px:.1f}" y1="{y0}" x2="{px:.1f}" y2="{y0+ph}" stroke="#e5e7eb" stroke-width="1"/>')
        elems.append(f'<line x1="{px:.1f}" y1="{y0+ph}" x2="{px:.1f}" y2="{y0+ph+5}" stroke="#374151" stroke-width="1.2"/>')
        elems.append(f'<text x="{px:.1f}" y="{y0+ph+20}" text-anchor="middle" font-size="11.2">{html.escape(xlbl)}</text>')

    for yv, ylbl in yticks:
        py = ty(yv)
        elems.append(f'<line x1="{x0}" y1="{py:.1f}" x2="{x0+pw}" y2="{py:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        elems.append(f'<line x1="{x0-5}" y1="{py:.1f}" x2="{x0}" y2="{py:.1f}" stroke="#374151" stroke-width="1.2"/>')
        elems.append(f'<text x="{x0-9}" y="{py+4:.1f}" text-anchor="end" font-size="11.2">{html.escape(ylbl)}</text>')

    leg_items = []
    for s in series:
        pts = [(tx(x), ty(y)) for x, y in zip(s["xs"], s["ys"]) if xlim[0] <= x <= xlim[1]]
        if not pts:
            continue
        d_attr = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
        lw = s.get("lw", 2.4)
        pstr = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        elems.append(f'<polyline fill="none" stroke="{s["color"]}" stroke-width="{lw}"{d_attr} points="{pstr}"/>')
        mk = s.get("marker", "circle")
        for px, py in pts:
            if mk == "circle":
                elems.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.0" fill="{s["color"]}" stroke="#fff" stroke-width="0.9"/>')
            elif mk == "square":
                elems.append(f'<rect x="{px-3.6:.1f}" y="{py-3.6:.1f}" width="7.2" height="7.2" fill="{s["color"]}" stroke="#fff" stroke-width="0.9"/>')
            elif mk == "tri":
                elems.append(f'<polygon points="{px:.1f},{py-4.8:.1f} {px-4.3:.1f},{py+3.6:.1f} {px+4.3:.1f},{py+3.6:.1f}" fill="{s["color"]}" stroke="#fff" stroke-width="0.9"/>')
            elif mk == "diam":
                elems.append(f'<polygon points="{px:.1f},{py-4.8:.1f} {px-4.4:.1f},{py:.1f} {px:.1f},{py+4.8:.1f} {px+4.4:.1f},{py:.1f}" fill="{s["color"]}" stroke="#fff" stroke-width="0.9"/>')
        if s.get("label"):
            leg_items.append(s)

    if leg_items:
        lh = 17.8 * len(leg_items) + 10
        lw_box = max(len(s["label"]) for s in leg_items) * 6.15 + 38
        lx = (x0 + 10) if "l" in legend_pos else (x0 + pw - lw_box - 10)
        ly = (y0 + 10) if "t" in legend_pos else (y0 + ph - lh - 10)
        elems.append(f'<rect x="{lx:.1f}" y="{ly:.1f}" width="{lw_box:.1f}" height="{lh:.1f}" rx="5" fill="#ffffff" fill-opacity="0.93" stroke="#d1d5db" stroke-width="1"/>')
        for i, s in enumerate(leg_items):
            iy = ly + 14.8 + i * 17.8
            d_attr = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
            elems.append(f'<line x1="{lx+8:.1f}" y1="{iy-4:.1f}" x2="{lx+27:.1f}" y2="{iy-4:.1f}" stroke="{s["color"]}" stroke-width="2.3"{d_attr}/>')
            elems.append(f'<circle cx="{lx+17.5:.1f}" cy="{iy-4:.1f}" r="3.2" fill="{s["color"]}"/>')
            elems.append(f'<text x="{lx+33:.1f}" y="{iy:.1f}" font-size="10.6">{html.escape(s["label"])}</text>')

    elems.append("</svg>\n")
    with open(svg_path, "w") as f:
        f.write("\n".join(elems))


def min_sec_v3(pref):
    xs, ys = [], []
    cx, cz = f"{pref}_Xerr.code", f"{pref}_Zerr.code"
    for b in (3.0, 4.0, 5.0, 6.0, 7.0):
        vx = kmc_uni[cx].get(b)
        vz = kmc_uni[cz].get(b)
        if vx is not None and vz is not None:
            xs.append(b); ys.append(min(vx, vz))
        elif vx is not None and b >= 6.0 and kmc_uni[cz].get(5.0, 0) > 10 * kmc_uni[cx].get(5.0, 1e30):
            xs.append(b); ys.append(vx)
    return xs, ys

# =============================================================================
# FIG 6: 3D-Local vs Non-Local Ultra-Compact Frontier + Michnicki Trap
# =============================================================================
s6 = []
for pref, lbl, col, dsh, mk in [
    ("nonloc_ibm_twogross_288",   "[Non-Local] IBM Two-Gross [[288,12,18]] (24 q/log, Δ=6)", "#16a34a", "6,3", "diam"),
    ("nonloc_ibm_gross_144",      "[Non-Local] IBM Gross [[144,12,12]] (12 q/log, Δ=5)",     "#0891b2", "6,3", "tri"),
    ("nonloc_rp2_aniso_hyb21_s2", "[Non-Local] RP² Aniso Hyb 2→1 (N=1758, K=1, +27% vs V2)", "#dc2626", "",    "circle"),
    ("loc3d_tor2_aniso_hyb21_s2", "[3D-Local] T² Aniso Hyb 2→1 (N=2344, K=2, +50% vs V2!)",  "#ea580c", "",    "square"),
    ("nonloc_rp2_aniso_L2r1_s2",  "[Non-Local] RP² Aniso 1-Rib (N=762, K=1, 3.4× > V2!)",    "#2563eb", "",    "circle"),
    ("nonloc_hyp_g3_aniso_s2",    "[Non-Local] Hyperbolic {4,6} Sheaf (N=3048, K=6, 508 q/l)","#7c3aed","",    "tri"),
    ("loc3d_tor2_aniso_L2r1_s2",  "[3D-Local] T² Aniso 1-Rib (N=1016, K=2, 508 q/log)",      "#0d9488", "",    "square"),
    ("loc3d_tor2_sparse_L2r1_s2", "[3D-Local] Over-Pruned Sparse T² (N=974 — Barrier Collapse!)","#6b7280","3,3","diam"),
]:
    xs, ys = min_sec_v3(pref)
    s6.append({"xs": xs, "ys": ys, "color": col, "dash": dsh, "marker": mk, "label": lbl})

save_single_plot(
    os.path.join(FIG_DIR, "fig6_3d_vs_nonlocal_frontier.svg"),
    "Ultra-Compact 3D-Local vs Non-Local Thermal Frontier & Pruning Collapse",
    "Inverse Bath Temperature β = 1/T", "Two-Sector Memory Lifetime min(<t_X>, <t_Z>)",
    (2.85, 7.15), (4.0, 5.0e11),
    [(3,"3"), (4,"4"), (5,"5"), (6,"6"), (7,"7")],
    [(1e1,"10¹"), (1e3,"10³"), (1e5,"10⁵"), (1e7,"10⁷"), (1e9,"10⁹"), (1e11,"10¹¹")],
    s6, ylog=True, legend_pos="tl"
)

# =============================================================================
# FIG 7: Gate-Based Digital Measurement-Free QEC (Hong-Guo-Lucas Nat Comm 2025)
# =============================================================================
s7 = []
for tag, lbl, col, dsh, mk in [
    ("NL_4DToric_486",        "[Non-Local] 4D (2,2) Toric (N=486, K=6, qdeg=4 — 0.0% at pg≤5e-4!)", "#16a34a", "",    "circle"),
    ("NL_RP2_Aniso_762_Jgrad","[Non-Local] RP² Aniso Graded (N=762, K=1, qdeg=4)",                  "#2563eb", "",    "square"),
    ("3D_Layer_1853",         "[3D-Local] 3D Layer [[9,1,3]] (N=1853, K=1)",                        "#ea580c", "6,3", "tri"),
    ("3D_Tor2_Sparse_974",    "[3D-Local] T² Compact Sheaf (N=974, K=2)",                           "#0891b2", "",    "diam"),
    ("2D_Toric_800",          "[2D-Local] 2D Toric L=20 (N=800, K=2)",                              "#6b7280", "3,3", "circle"),
    ("NL_IBM_Gross_144",      "[Non-Local] IBM Gross [[144,12,12]] (qdeg=3 — Gate-Fault Cascade!)", "#dc2626", "5,3", "square"),
    ("NL_IBM_2Gross_288",     "[Non-Local] IBM Two-Gross [[288,12,18]] (qdeg=3 — 99.5% @ pg=2e-3)", "#991b1b", "3,3", "tri"),
]:
    pts = sorted(mfqec[tag].items())
    # plot against p_gate in % (map 0 -> 0.015% on log axis or linear 0..0.8%)
    s7.append({"xs": [p*100 for p, _ in pts], "ys": [v for _, v in pts],
               "color": col, "dash": dsh, "marker": mk, "label": lbl})

save_single_plot(
    os.path.join(FIG_DIR, "fig7_mfqec_digital_cooling.svg"),
    "Digital Gate-Based Measurement-Free QEC (150 Sweeps, p_env=0.2%/sweep)",
    "Cooling-Circuit Gate Fault Rate p_gate (%)", "Logical Failure Prob. After 150 Readout-Free Sweeps",
    (-0.02, 0.83), (0.0, 1.05),
    [(0.0,"0%"), (0.1,"0.1%"), (0.2,"0.2%"), (0.4,"0.4%"), (0.6,"0.6%"), (0.8,"0.8%")],
    [(0.0,"0.0"), (0.2,"0.2"), (0.4,"0.4"), (0.6,"0.6"), (0.8,"0.8"), (1.0,"1.0")],
    s7, legend_pos="br"
)

# =============================================================================
# FIG 8: 20-Round Single-Shot Active QEC Under Noisy Syndrome Measurements
# =============================================================================
s8 = []
for tag, lbl, col, dsh, mk in [
    ("NL_4DToric_486",         "[Non-Local] 4D (2,2) Toric (N=486, K=6, M_meta=84)",        "#16a34a", "",    "circle"),
    ("NL_IBM_2Gross_288",      "[Non-Local] IBM Two-Gross [[288,12,18]] (0.0% ≤ 1.2%!)",    "#0891b2", "",    "diam"),
    ("NL_IBM_Gross_144",       "[Non-Local] IBM Gross [[144,12,12]] (0.0% ≤ 0.6%)",         "#2563eb", "6,3", "square"),
    ("NL_RP2_Aniso_762",       "[Non-Local] RP² Aniso Sheaf (N=762, K=1, M_meta=23)",       "#dc2626", "",    "tri"),
    ("3D_Layer_1853",          "[3D-Local] 3D Layer [[9,1,3]] (N=1853, K=1)",               "#ea580c", "6,3", "circle"),
    ("3D_Tor2_Hyb_2224",       "[3D-Local] T² Compact Hyb Sheaf (N=2224, K=2, M_meta=58)",  "#7c3aed", "",    "square"),
    ("2D_Toric_800",           "[2D-Local] 2D Toric L=20 (N=800, M_meta=1)",                "#6b7280", "3,3", "diam"),
    ("3D_Toric_String_1029",   "[3D-Local] 3D Toric String Sector (N=1029 — No Single-Shot!)","#991b1b","2,3","tri"),
]:
    pts = sorted(singleshot[tag].items())
    s8.append({"xs": [p*100 for p, _ in pts], "ys": [v for _, v in pts],
               "color": col, "dash": dsh, "marker": mk, "label": lbl})

save_single_plot(
    os.path.join(FIG_DIR, "fig8_singleshot_qec.svg"),
    "20-Round Single-Shot Active QEC Under Noisy Syndromes (p_data = q_meas)",
    "Per-Round Data & Measurement Error Rate p_data = q_meas (%)", "Cumulative Logical Failure P_fail (20 Single-Shot Rounds)",
    (0.05, 2.08), (0.0, 1.02),
    [(0.2,"0.2%"), (0.4,"0.4%"), (0.8,"0.8%"), (1.2,"1.2%"), (1.6,"1.6%"), (2.0,"2.0%")],
    [(0.0,"0.0"), (0.2,"0.2"), (0.4,"0.4"), (0.6,"0.6"), (0.8,"0.8"), (1.0,"1.0")],
    s8, legend_pos="tl"
)

for nm in ("fig6_3d_vs_nonlocal_frontier", "fig7_mfqec_digital_cooling", "fig8_singleshot_qec"):
    svg_p = os.path.join(FIG_DIR, f"{nm}.svg")
    subprocess.run(["qlmanage", "-t", "-s", "1520", "-o", FIG_DIR, svg_p],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    raw_png = os.path.join(FIG_DIR, f"{nm}.svg.png")
    out_png = os.path.join(FIG_DIR, f"{nm}.png")
    subprocess.run(["sips", "--cropOffset", "1", "1", "-c", "1000", "1518", raw_png, "--out", out_png],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    os.remove(raw_png)
    with open(out_png, "rb") as fi, open(os.path.join(ART_DIR, f"{nm}.png"), "wb") as fo:
        fo.write(fi.read())

print("Rendered and exported all 3 V3 high-DPI PNG figures!")
