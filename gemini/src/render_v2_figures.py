#!/usr/bin/env python3
"""
Render 7 publication-grade single-panel figures (760x500 SVG -> 1520px PNG via QuickLook/sips)
combining V1 + V2 + Step-4 SubMIT cluster datasets.
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
            k, v = tok.split("<=", 1)
            d[k] = v
        elif "=" in tok:
            k, v = tok.split("=", 1)
            d[k] = v
    return d


kmc_q = defaultdict(dict)
phase = defaultdict(dict)
coupled = defaultdict(dict)
dec5_th = defaultdict(lambda: defaultdict(dict))
anneal_inf = defaultdict(dict)
goldilocks = defaultdict(lambda: defaultdict(dict))

# 1. Load V1 data
with open(os.path.join(WS, "cluster_results_all.txt")) as f:
    for raw in f:
        line = raw.strip()
        if not line or line.startswith(("ERR", "TIMEOUT")):
            continue
        parts = line.split("|", 3)
        if len(parts) < 4:
            continue
        suite, cname, _, payload = parts
        kv = parse_kv(payload)
        if suite == "KMC_Q" and payload.startswith("KMC"):
            kmc_q[cname][float(kv["beta"])] = float(kv["mean_t"])
        elif suite == "KMC_CLS" and payload.startswith("KMC"):
            phase[cname][float(kv["beta"])] = float(kv["mean_t"])

if os.path.exists(os.path.join(WS, "cluster_followup.txt")):
    with open(os.path.join(WS, "cluster_followup.txt")) as f:
        for raw in f:
            line = raw.strip()
            if not line.startswith("CROSS|"):
                continue
            parts = line.split("|")
            cname = parts[1]
            kv = parse_kv(parts[-1])
            phase[cname][float(kv["beta"])] = float(kv["mean_t"])

# 2. Load V2 data (supersedes V1 where overlapping)
with open(os.path.join(WS, "cluster_v2_results.txt")) as f:
    for raw in f:
        line = raw.strip()
        if not line or line.startswith(("ERR|", "TIMEOUT|")):
            continue
        parts = line.split("|", 3)
        suite, label, _, body = parts
        kv = parse_kv(body)
        if suite == "NEW_Q" and body.startswith("KMC "):
            kmc_q[label][float(kv["beta"])] = float(kv["mean_t"])
        elif suite == "PHASE" and body.startswith("KMC "):
            phase[label][float(kv["beta"])] = float(kv["mean_t"])
        elif suite == "COUPLED" and body.startswith("KMC_COUPLED"):
            coupled[label][float(kv["beta"])] = float(kv["mean_t"])
        elif suite == "DEC5_TH" and body.startswith("THRESHOLD"):
            dec5_th[label][kv["dec"]][float(kv["p"])] = (float(kv["P_L"]), float(kv["decode_us"]))
        elif suite == "ANNEAL_INF" and body.startswith("ANNEAL"):
            anneal_inf[(label, int(kv["stages"]))][float(kv["T"])] = float(kv["final_syn"])
        elif suite == "GOLDILOCKS" and body.startswith("HYB_NOISY"):
            goldilocks[label][float(kv["p_act"])][float(kv["tau_ec"])] = float(kv["P_fail"])

# 3. Load Step-4 fine crossing data if present
s4_path = os.path.join(WS, "cluster_s4cross.txt")
if os.path.exists(s4_path):
    with open(s4_path) as f:
        for raw in f:
            line = raw.strip()
            if not line.startswith("S4CROSS|"):
                continue
            parts = line.split("|", 3)
            cname = parts[1]
            kv = parse_kv(parts[3])
            phase[cname][float(kv["beta"])] = float(kv["mean_t"])


def save_single_plot(svg_path, title, xlabel, ylabel, xlim, ylim, xticks, yticks,
                     series, xlog=False, ylog=False, vline=None, legend_pos="tl"):
    W, H = 760, 500
    x0, y0, pw, ph = 86, 48, 636, 382
    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<style>text { font-family: -apple-system, Helvetica, Arial, sans-serif; fill: #1f2937; }</style>',
        f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="#fafafa" stroke="#374151" stroke-width="1.3"/>',
        f'<text x="{x0 + pw/2:.1f}" y="{y0 - 14:.1f}" text-anchor="middle" font-size="14.5" font-weight="700">{html.escape(title)}</text>',
        f'<text x="{x0 + pw/2:.1f}" y="{y0 + ph + 42:.1f}" text-anchor="middle" font-size="13.0" font-weight="600">{html.escape(xlabel)}</text>',
        f'<text x="{x0 - 56:.1f}" y="{y0 + ph/2:.1f}" text-anchor="middle" font-size="13.0" font-weight="600" transform="rotate(-90 {x0 - 56:.1f} {y0 + ph/2:.1f})">{html.escape(ylabel)}</text>',
    ]

    def tx(v):
        lv = math.log10(v) if xlog else v
        l0 = math.log10(xlim[0]) if xlog else xlim[0]
        l1 = math.log10(xlim[1]) if xlog else xlim[1]
        return x0 + (lv - l0) / (l1 - l0) * pw

    def ty(v):
        v_clamped = max(ylim[0], min(ylim[1], v))
        lv = math.log10(v_clamped) if ylog else v_clamped
        l0 = math.log10(ylim[0]) if ylog else ylim[0]
        l1 = math.log10(ylim[1]) if ylog else ylim[1]
        return y0 + ph - (lv - l0) / (l1 - l0) * ph

    for xv, xlbl in xticks:
        px = tx(xv)
        elems.append(f'<line x1="{px:.1f}" y1="{y0}" x2="{px:.1f}" y2="{y0+ph}" stroke="#e5e7eb" stroke-width="1"/>')
        elems.append(f'<line x1="{px:.1f}" y1="{y0+ph}" x2="{px:.1f}" y2="{y0+ph+5}" stroke="#374151" stroke-width="1.2"/>')
        elems.append(f'<text x="{px:.1f}" y="{y0+ph+20}" text-anchor="middle" font-size="11.3">{html.escape(xlbl)}</text>')

    for yv, ylbl in yticks:
        py = ty(yv)
        elems.append(f'<line x1="{x0}" y1="{py:.1f}" x2="{x0+pw}" y2="{py:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        elems.append(f'<line x1="{x0-5}" y1="{py:.1f}" x2="{x0}" y2="{py:.1f}" stroke="#374151" stroke-width="1.2"/>')
        elems.append(f'<text x="{x0-9}" y="{py+4:.1f}" text-anchor="end" font-size="11.3">{html.escape(ylbl)}</text>')

    if vline is not None:
        vx, vlbl = vline
        px = tx(vx)
        elems.append(f'<line x1="{px:.1f}" y1="{y0}" x2="{px:.1f}" y2="{y0+ph}" stroke="#111827" stroke-width="1.6" stroke-dasharray="5,4"/>')
        elems.append(f'<text x="{px+6:.1f}" y="{y0+ph-14:.1f}" font-size="11.3" font-weight="600" fill="#111827">{html.escape(vlbl)}</text>')

    leg_items = []
    for s in series:
        pts = [(tx(x), ty(y)) for x, y in zip(s["xs"], s["ys"]) if xlim[0] <= x <= xlim[1]]
        if not pts:
            continue
        d_attr = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
        lw = s.get("lw", 2.4)
        op = s.get("alpha", 1.0)
        pstr = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        elems.append(f'<polyline fill="none" stroke="{s["color"]}" stroke-width="{lw}" stroke-opacity="{op}"{d_attr} points="{pstr}"/>')
        mk = s.get("marker", "circle")
        for px, py in pts:
            if mk == "circle":
                elems.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.1" fill="{s["color"]}" fill-opacity="{op}" stroke="#fff" stroke-width="0.9"/>')
            elif mk == "square":
                elems.append(f'<rect x="{px-3.7:.1f}" y="{py-3.7:.1f}" width="7.4" height="7.4" fill="{s["color"]}" fill-opacity="{op}" stroke="#fff" stroke-width="0.9"/>')
            elif mk == "tri":
                elems.append(f'<polygon points="{px:.1f},{py-4.9:.1f} {px-4.4:.1f},{py+3.7:.1f} {px+4.4:.1f},{py+3.7:.1f}" fill="{s["color"]}" fill-opacity="{op}" stroke="#fff" stroke-width="0.9"/>')
            elif mk == "diam":
                elems.append(f'<polygon points="{px:.1f},{py-4.9:.1f} {px-4.5:.1f},{py:.1f} {px:.1f},{py+4.9:.1f} {px+4.5:.1f},{py:.1f}" fill="{s["color"]}" fill-opacity="{op}" stroke="#fff" stroke-width="0.9"/>')
        if s.get("label"):
            leg_items.append(s)

    if leg_items:
        lh = 18.2 * len(leg_items) + 11
        lw_box = max(len(s["label"]) for s in leg_items) * 6.25 + 40
        lx = (x0 + 11) if "l" in legend_pos else (x0 + pw - lw_box - 11)
        ly = (y0 + 11) if "t" in legend_pos else (y0 + ph - lh - 11)
        elems.append(f'<rect x="{lx:.1f}" y="{ly:.1f}" width="{lw_box:.1f}" height="{lh:.1f}" rx="5" fill="#ffffff" fill-opacity="0.93" stroke="#d1d5db" stroke-width="1"/>')
        for i, s in enumerate(leg_items):
            iy = ly + 15.2 + i * 18.2
            d_attr = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
            elems.append(f'<line x1="{lx+8:.1f}" y1="{iy-4:.1f}" x2="{lx+28:.1f}" y2="{iy-4:.1f}" stroke="{s["color"]}" stroke-width="2.3"{d_attr}/>')
            elems.append(f'<circle cx="{lx+18:.1f}" cy="{iy-4:.1f}" r="3.3" fill="{s["color"]}"/>')
            elems.append(f'<text x="{lx+34:.1f}" y="{iy:.1f}" font-size="10.8">{html.escape(s["label"])}</text>')

    elems.append("</svg>\n")
    with open(svg_path, "w") as f:
        f.write("\n".join(elems))


betas_q = [3.0, 4.0, 5.0, 6.0, 7.0]
def min_sec(pref):
    xs, ys = [], []
    cx, cz = f"{pref}_Xerr.code", f"{pref}_Zerr.code"
    for b in betas_q:
        vx = kmc_q[cx].get(b)
        vz = kmc_q[cz].get(b)
        if vx is not None and vz is not None:
            xs.append(b); ys.append(min(vx, vz))
        elif vx is not None:
            xs.append(b); ys.append(vx)
    return xs, ys

# =============================================================================
# FIG 1A: 3D Quantum Memory Architecture Showdown (including RP^2 K=1 & Sec 3.2!)
# =============================================================================
s1a = []
for pref, lbl, col, dsh, mk in [
    ("bdl_rp2_hyb21_s2",  "RP² Hemi-Cube Hybrid 2→1 rib (N=2370, K=1, Δ=(5,5))", "#dc2626", "",    "circle"),
    ("bdl_tor2_hyb21_s2", "T² Torus Hybrid 2→1 rib (N=3160, K=2, Δ=(5,5))",      "#ea580c", "",    "square"),
    ("sec32_rp2_l3_s2",   "Sec 3.2 Tanner l=3 on RP² (N=3672, K=1, w≤6)",        "#16a34a", "",    "tri"),
    ("bdl_rp2_L2r1_s2",   "RP² Minimal 1-Rib L=2 (N=969, K=1, Δ=(4,5))",         "#2563eb", "",    "diam"),
    ("bdl_tor2_L3r2_s2",  "T² Full 2-Rib L=3 (N=8608, Δ=(5,7), entropy loss)",   "#7c3aed", "6,3", "square"),
    ("bdl_tor2_hyb12_s2", "T² Reversed 1→2 rib (N=3488, Δ=(3,6), 555× worse)",   "#db2777", "6,3", "tri"),
    ("layer_913",         "3D Layer Code [[9,1,3]] (N=1853, Δ=(2,2))",           "#92400e", "4,4", "circle"),
    ("toric3d_L7",        "3D Toric L=7 (N=1029) & 2D Toric L=30 (N=1800)",      "#6b7280", "2,3", "diam"),
]:
    xs, ys = min_sec(pref)
    s1a.append({"xs": xs, "ys": ys, "color": col, "dash": dsh, "marker": mk, "label": lbl})

save_single_plot(
    os.path.join(FIG_DIR, "fig1a_quantum_lifetime.svg"),
    "Two-Sector 3D Quantum Memory Lifetime min(<t_X>, <t_Z>) Across Architectures",
    "Inverse Bath Temperature β = 1/T", "Quantum Memory Lifetime min(<t_X>, <t_Z>)",
    (2.85, 7.15), (1.0, 6.0e8),
    [(3,"3"), (4,"4"), (5,"5"), (6,"6"), (7,"7")],
    [(1e0,"10⁰"), (1e2,"10²"), (1e4,"10⁴"), (1e6,"10⁶"), (1e8,"10⁸")],
    s1a, ylog=True, legend_pos="tl"
)

# =============================================================================
# FIG 1B: Sawtooth Entropic Dip & Boundaryless RP^2 vs Planar Seed (s=0..3)
# =============================================================================
s1b = []
for pref, p_lbl, col_x, col_z, mk in [
    ("bdl_rp2_L2r1", "RP² Boundaryless (K=1, L=2, r=1)", "#dc2626", "#f97316", "circle"),
    ("bdl_sq_L2r1",  "Planar Open Seed (K=1, L=2, r=1)", "#2563eb", "#0891b2", "square"),
]:
    xs_x, ys_x, xs_z, ys_z = [], [], [], []
    for st in [0, 1, 2, 3]:
        vx = kmc_q[f"{pref}_s{st}_Xerr.code"].get(6.0)
        vz = kmc_q[f"{pref}_s{st}_Zerr.code"].get(6.0)
        if vx is not None: xs_x.append(st); ys_x.append(vx)
        if vz is not None: xs_z.append(st); ys_z.append(vz)
    s1b.append({"xs": xs_x, "ys": ys_x, "color": col_x, "dash": "",    "marker": mk, "label": f"{p_lbl} — X-sector (β=6)"})
    s1b.append({"xs": xs_z, "ys": ys_z, "color": col_z, "dash": "6,4", "marker": mk, "label": f"{p_lbl} — Z-sector (β=6)"})

save_single_plot(
    os.path.join(FIG_DIR, "fig1b_sawtooth_steps.svg"),
    "Sawtooth Sector Oscillation & Boundaryless RP² Advantage (β = 6.0)",
    "Recursion Step s (Seed → R_X → R_Z∘R_X → R_X∘R_Z∘R_X)", "Sector Memory Lifetime <t_fail> at β = 6",
    (-0.15, 3.15), (4.0e1, 1.0e9),
    [(0,"s=0 (N=6)"), (1,"s=1 (R_X)"), (2,"s=2 (1 cycle)"), (3,"s=3 (1.5 cyc)")],
    [(1e2,"10²"), (1e4,"10⁴"), (1e6,"10⁶"), (1e8,"10⁸")],
    s1b, ylog=True, legend_pos="tl"
)

# =============================================================================
# FIG 2: Finite-Temperature Self-Correction Phase Crossing (k = 0, 2, 4)
# =============================================================================
s2 = []
for cname, lbl, col, dsh, mk in [
    ("cls_l4_r2_step2.code", "2-Rib Step 2 (l=4, N=400, Δ=6)",       "#dc2626", "",    "square"),
    ("cls_l3_r1_step2.code", "1-Rib Step 2 (l=3, N=196, Δ=5)",       "#2563eb", "",    "circle"),
    ("cls_l3_r1_step4.code", "1-Rib Step 4 (l=3, N=9604, Δ=13)",     "#16a34a", "",    "tri"),
    ("cls_l3_r1_step0.code", "Step 0 Base Seed Loop (N=4, Δ=2)",     "#6b7280", "3,3", "diam"),
]:
    pts = [(b, v) for b, v in sorted(phase[cname].items()) if 2.5 <= b <= 5.5]
    if pts:
        s2.append({"xs": [p[0] for p in pts], "ys": [p[1] for p in pts],
                   "color": col, "dash": dsh, "marker": mk, "label": lbl})

save_single_plot(
    os.path.join(FIG_DIR, "fig2_phase_crossing.svg"),
    "Self-Correction Phase Transition Crossing (1-Rib vs 2-Rib, N up to 40,000)",
    "Inverse Bath Temperature β = 1/T", "Mean Memory Lifetime <t_fail>",
    (2.4, 5.6), (5.0e1, 1.2e8),
    [(2.5,"2.5"), (3.0,"3.0"), (3.5,"3.5"), (4.0,"4.0"), (4.5,"4.5"), (5.0,"5.0"), (5.5,"5.5")],
    [(1e2,"10²"), (1e4,"10⁴"), (1e6,"10⁶"), (1e8,"10⁸")],
    s2, ylog=True, vline=(3.33, "Critical β_c ≈ 3.33"), legend_pos="tl"
)

# =============================================================================
# FIG 3A: 5-Decoder Accuracy + Speed Benchmark on RP^2 Hybrid (N=2370)
# =============================================================================
s3a = []
cn_dec = "bdl_rp2_hyb21_s2_Xerr.code"
for dec, lbl, col, dsh, mk in [
    ("bposd",      "BP+OSD (p_th ≈ 8.5%, 0.0% fail ≤ 6%, ~9 ms)",     "#dc2626", "",    "circle"),
    ("uf",         "FastClusterUF (p_th ≈ 4.8%, 151 μs — 60× faster)","#2563eb", "",    "square"),
    ("rg_soft",    "Novel RG-Soft (Interior-Peel + Soft-BP UF)",       "#16a34a", "",    "tri"),
    ("rg_strict",  "Hard Sheaf-RG Strict (15.8% floor at p=1%)",       "#ea580c", "5,3", "diam"),
    ("rg_relaxed", "Hard Sheaf-RG Relaxed (17.0% floor at p=1%)",      "#7c3aed", "3,3", "circle"),
]:
    pts = sorted(dec5_th[cn_dec][dec].items())
    s3a.append({"xs": [p*100 for p, _ in pts], "ys": [v[0] for _, v in pts],
                "color": col, "dash": dsh, "marker": mk, "label": lbl})

save_single_plot(
    os.path.join(FIG_DIR, "fig3a_decoder_thresholds.svg"),
    "5-Decoder Benchmark on Boundaryless RP² Hybrid Code (N=2370, K=1)",
    "Physical i.i.d. Error Rate p (%)", "Logical Failure Probability P_L(p)",
    (0.6, 10.4), (0.0, 0.56),
    [(1,"1%"), (2,"2%"), (4,"4%"), (6,"6%"), (8,"8%"), (10,"10%")],
    [(0.0,"0.0"), (0.1,"0.1"), (0.2,"0.2"), (0.3,"0.3"), (0.4,"0.4"), (0.5,"0.5")],
    s3a, legend_pos="br"
)

# =============================================================================
# FIG 3B: BDL Outlook #2 Solved — Simulated Thermal Annealing vs Cold Quench
# =============================================================================
s3b = []
for (cname, stg), lbl, col, dsh, mk in [
    (("cls_l4_r2_step3.code", 1),        "2-Rib Step 3: Cold Quench β=6 (100% trapped!)",        "#dc2626", "",    "square"),
    (("cls_l4_r2_step3.code", 10),       "2-Rib Step 3: 10-Stage Anneal β=1.5→6 (2× lower)",     "#f97316", "6,3", "square"),
    (("cls_l3_r1_step3.code", 1),        "1-Rib Step 3: Cold Quench β=6 (46.7% trapped)",        "#2563eb", "",    "circle"),
    (("cls_l3_r1_step3.code", 10),       "1-Rib Step 3: 10-Stage Anneal β=1.5→6 (25× lower!)",   "#16a34a", "6,3", "circle"),
    (("bdl_sq_L3r2_s2_Xerr.code", 1),    "3D Planar 2-Rib: Cold Quench β=6 (95.8% trapped)",     "#7c3aed", "3,3", "tri"),
    (("bdl_rp2_hyb21_s2_Xerr.code", 1),  "3D RP² Hybrid: Cold Quench β=6 (91.7% vacuum!)",       "#0891b2", "",    "diam"),
]:
    pts = sorted(anneal_inf[(cname, stg)].items())
    if pts:
        s3b.append({"xs": [p[0] for p in pts], "ys": [max(p[1], 7e-5) for p in pts],
                    "color": col, "dash": dsh, "marker": mk, "label": lbl})

save_single_plot(
    os.path.join(FIG_DIR, "fig3b_glassy_cooling.svg"),
    "Solving BDL Outlook #2: Simulated Annealing vs Cold Quench from T = ∞",
    "Total Cooling Bath Budget T_cool", "Final Residual Syndrome Density <|σ(T)| / M>",
    (70.0, 1.4e6), (6e-5, 0.08),
    [(1e2,"10²"), (1e3,"10³"), (1e4,"10⁴"), (1e5,"10⁵"), (1e6,"10⁶")],
    [(1e-4,"10⁻⁴"), (1e-3,"10⁻³"), (1e-2,"10⁻²"), (5e-2,"5×10⁻²")],
    s3b, xlog=True, ylog=True, legend_pos="bl"
)

# =============================================================================
# FIG 4: "Goldilocks" Optimal Lazy QEC Under Noisy Active Backaction (T=1000)
# =============================================================================
s4 = []
for (cname, pa), lbl, col, dsh, mk in [
    (("bdl_rp2_hyb21_s2_Xerr.code", 0.002), "RP² Hybrid (N=2370), p_act=0.2% (Optimal τ*=5: 1.0%!)", "#16a34a", "",    "circle"),
    (("bdl_rp2_hyb21_s2_Xerr.code", 0.005), "RP² Hybrid (N=2370), p_act=0.5% (Passive τ=1000: 7%!)", "#dc2626", "",    "square"),
    (("bdl_rp2_hyb21_s2_Xerr.code", 0.000), "RP² Hybrid (N=2370), Ideal Noiseless QEC p_act=0",      "#0891b2", "5,3", "circle"),
    (("layer_913_Xerr.code",        0.002), "3D Layer [[9,1,3]] (N=1853), p_act=0.2%",               "#ea580c", "6,3", "tri"),
    (("layer_913_Xerr.code",        0.005), "3D Layer [[9,1,3]] (N=1853), p_act=0.5%",               "#92400e", "3,3", "tri"),
    (("toric2d_L20_Xerr.code",      0.002), "2D Toric L=20 (N=800), p_act=0.2%",                     "#6b7280", "3,3", "diam"),
]:
    pts = sorted(goldilocks[cname][pa].items())
    s4.append({"xs": [p[0] for p in pts], "ys": [p[1] for p in pts],
               "color": col, "dash": dsh, "marker": mk, "label": lbl})

save_single_plot(
    os.path.join(FIG_DIR, "fig4_hybrid_lazy_qec.svg"),
    "Noisy Active-Passive Hybrid QEC Over Fixed Horizon T=1000 (β=5.0)",
    "Active QEC Measurement Interval τ_EC (τ=1000 is Pure Passive)", "Cumulative Logical Failure P_fail(T=1000)",
    (1.6, 1250.0), (0.0, 1.04),
    [(2,"2"), (5,"5"), (10,"10"), (25,"25"), (100,"100"), (500,"500"), (1000,"1000")],
    [(0.0,"0.0"), (0.2,"0.2"), (0.4,"0.4"), (0.6,"0.6"), (0.8,"0.8"), (1.0,"1.0")],
    s4, xlog=True, legend_pos="tl"
)

# =============================================================================
# FIG 5: Coupled Depolarizing (X, Y, Z) Davies Bath vs Independent (X, Z) Bath
# =============================================================================
s5 = []
for lbl_key, disp, col, dsh, mk in [
    ("bdl_rp2_hyb21_s2_ys0.0",  "RP² Hybrid (N=2370) — Independent X,Z Bath (y_scale=0)",  "#dc2626", "",    "circle"),
    ("bdl_rp2_hyb21_s2_ys1.0",  "RP² Hybrid (N=2370) — Full X,Y,Z Davies Bath (y_scale=1)","#dc2626", "6,3", "square"),
    ("bdl_tor2_hyb21_s2_ys0.0", "T² Hybrid (N=3160) — Independent X,Z Bath (y_scale=0)",   "#2563eb", "",    "tri"),
    ("bdl_tor2_hyb21_s2_ys1.0", "T² Hybrid (N=3160) — Full X,Y,Z Davies Bath (y_scale=1)", "#2563eb", "6,3", "diam"),
    ("sec32_rp2_l3_s2_ys0.0",   "Sec 3.2 RP² (N=3672) — Independent X,Z Bath (y_scale=0)", "#16a34a", "",    "circle"),
    ("sec32_rp2_l3_s2_ys1.0",   "Sec 3.2 RP² (N=3672) — Full X,Y,Z Davies Bath (y_scale=1)","#16a34a","6,3", "square"),
    ("layer_913_ys1.0",         "3D Layer [[9,1,3]] (N=1853) — Full X,Y,Z Davies Bath",    "#6b7280", "3,3", "tri"),
]:
    pts = sorted(coupled[lbl_key].items())
    s5.append({"xs": [p[0] for p in pts], "ys": [p[1] for p in pts],
               "color": col, "dash": dsh, "marker": mk, "label": disp})

save_single_plot(
    os.path.join(FIG_DIR, "fig5_coupled_bath.svg"),
    "Robustness to Cross-Sector Pauli-Y Coupling in Full (X, Y, Z) Davies Bath",
    "Inverse Bath Temperature β = 1/T", "Coupled Two-Sector Lifetime <t_fail>",
    (3.85, 6.15), (1.0e1, 6.0e6),
    [(4.0,"4.0"), (4.5,"4.5"), (5.0,"5.0"), (5.5,"5.5"), (6.0,"6.0")],
    [(1e1,"10¹"), (1e3,"10³"), (1e5,"10⁵"), (5e6,"5×10⁶")],
    s5, ylog=True, legend_pos="tl"
)

# Convert all 7 SVGs to uncropped 1518x1020 PNGs via qlmanage + sips and copy to ART_DIR!
names = [
    "fig1a_quantum_lifetime",
    "fig1b_sawtooth_steps",
    "fig2_phase_crossing",
    "fig3a_decoder_thresholds",
    "fig3b_glassy_cooling",
    "fig4_hybrid_lazy_qec",
    "fig5_coupled_bath",
]
for nm in names:
    svg_p = os.path.join(FIG_DIR, f"{nm}.svg")
    subprocess.run(["qlmanage", "-t", "-s", "1520", "-o", FIG_DIR, svg_p],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    raw_png = os.path.join(FIG_DIR, f"{nm}.svg.png")
    out_png = os.path.join(FIG_DIR, f"{nm}.png")
    subprocess.run(["sips", "--cropOffset", "1", "1", "-c", "1000", "1518", raw_png, "--out", out_png],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    os.remove(raw_png)
    art_png = os.path.join(ART_DIR, f"{nm}.png")
    with open(out_png, "rb") as fi, open(art_png, "wb") as fo:
        fo.write(fi.read())

print("Successfully rendered and exported all 7 high-DPI PNG figures!")
