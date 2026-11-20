#!/usr/bin/env python3
"""
Render 6 crisp single-panel publication figures (760x480 viewBox, zero cropping under QuickLook/sips).
"""

import math
import os
import html
from collections import defaultdict

WS = "/Users/harshilavlani/Downloads/6.S980_gemini"
FIG_DIR = os.path.join(WS, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def parse_kv(text):
    d = {}
    for tok in text.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            d[k] = v
    return d


kmc_q = defaultdict(dict)
thresh = defaultdict(list)
cool = defaultdict(list)
hybrid = defaultdict(list)

with open(os.path.join(WS, "cluster_results_all.txt")) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith(("ERR", "TIMEOUT")):
            continue
        parts = line.split("|", 3)
        if len(parts) < 4:
            continue
        suite, cname, _, payload = parts
        kv = parse_kv(payload)
        if suite == "KMC_Q" and payload.startswith("KMC"):
            kmc_q[cname][float(kv["beta"])] = (
                int(kv["N"]), float(kv["mean_t"]), float(kv["sem_t"]),
                int(kv["capped"]), int(kv["trials"])
            )
        elif suite == "THRESH" and payload.startswith("THRESHOLD"):
            thresh[(cname, kv["dec"])].append((float(kv["p"]), float(kv["P_L"]), float(kv["sem"])))
        elif suite == "COOL" and payload.startswith("COOL"):
            cool[(cname, float(kv["beta"]))].append((float(kv["t"]), float(kv["syn_density"])))
        elif suite == "HYBRID" and payload.startswith("HYBRID"):
            hybrid[cname].append((float(kv["tau_ec"]), float(kv["P_fail"])))

cross = defaultdict(dict)
with open(os.path.join(WS, "cluster_followup.txt")) as f:
    for line in f:
        line = line.strip()
        if not line.startswith("CROSS|"):
            continue
        parts = line.split("|")
        cname = parts[1]
        kv = parse_kv(parts[-1])
        cross[cname][float(kv["beta"])] = (
            int(kv["N"]), float(kv["mean_t"]), float(kv["sem_t"]),
            int(kv["capped"]), int(kv["trials"])
        )


def save_single_plot(path, title, xlabel, ylabel, xlim, ylim, xticks, yticks,
                     series, xlog=False, ylog=False, vline=None, legend_pos="tl"):
    W, H = 760, 500
    x0, y0, pw, ph = 86, 48, 636, 382
    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<style>text { font-family: -apple-system, Helvetica, Arial, sans-serif; fill: #1f2937; }</style>',
        f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="#fafafa" stroke="#374151" stroke-width="1.3"/>',
        f'<text x="{x0 + pw/2:.1f}" y="{y0 - 14:.1f}" text-anchor="middle" font-size="15" font-weight="700">{html.escape(title)}</text>',
        f'<text x="{x0 + pw/2:.1f}" y="{y0 + ph + 42:.1f}" text-anchor="middle" font-size="13.2" font-weight="600">{html.escape(xlabel)}</text>',
        f'<text x="{x0 - 56:.1f}" y="{y0 + ph/2:.1f}" text-anchor="middle" font-size="13.2" font-weight="600" transform="rotate(-90 {x0 - 56:.1f} {y0 + ph/2:.1f})">{html.escape(ylabel)}</text>',
    ]

    def tx(v):
        lv = math.log10(v) if xlog else v
        l0 = math.log10(xlim[0]) if xlog else xlim[0]
        l1 = math.log10(xlim[1]) if xlog else xlim[1]
        return x0 + (lv - l0) / (l1 - l0) * pw

    def ty(v):
        lv = math.log10(v) if ylog else v
        l0 = math.log10(ylim[0]) if ylog else ylim[0]
        l1 = math.log10(ylim[1]) if ylog else ylim[1]
        return y0 + ph - (lv - l0) / (l1 - l0) * ph

    for xv, xlbl in xticks:
        px = tx(xv)
        elems.append(f'<line x1="{px:.1f}" y1="{y0}" x2="{px:.1f}" y2="{y0+ph}" stroke="#e5e7eb" stroke-width="1"/>')
        elems.append(f'<line x1="{px:.1f}" y1="{y0+ph}" x2="{px:.1f}" y2="{y0+ph+5}" stroke="#374151" stroke-width="1.2"/>')
        elems.append(f'<text x="{px:.1f}" y="{y0+ph+20}" text-anchor="middle" font-size="11.5">{html.escape(xlbl)}</text>')

    for yv, ylbl in yticks:
        py = ty(yv)
        elems.append(f'<line x1="{x0}" y1="{py:.1f}" x2="{x0+pw}" y2="{py:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        elems.append(f'<line x1="{x0-5}" y1="{py:.1f}" x2="{x0}" y2="{py:.1f}" stroke="#374151" stroke-width="1.2"/>')
        elems.append(f'<text x="{x0-9}" y="{py+4:.1f}" text-anchor="end" font-size="11.5">{html.escape(ylbl)}</text>')

    if vline is not None:
        vx, vlbl = vline
        px = tx(vx)
        elems.append(f'<line x1="{px:.1f}" y1="{y0}" x2="{px:.1f}" y2="{y0+ph}" stroke="#111827" stroke-width="1.6" stroke-dasharray="5,4"/>')
        elems.append(f'<text x="{px+6:.1f}" y="{y0+ph-14:.1f}" font-size="11.5" font-weight="600" fill="#111827">{html.escape(vlbl)}</text>')

    leg_items = []
    for s in series:
        pts = [(tx(x), ty(y)) for x, y in zip(s["xs"], s["ys"])]
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
                elems.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.2" fill="{s["color"]}" fill-opacity="{op}" stroke="#fff" stroke-width="0.9"/>')
            elif mk == "square":
                elems.append(f'<rect x="{px-3.8:.1f}" y="{py-3.8:.1f}" width="7.6" height="7.6" fill="{s["color"]}" fill-opacity="{op}" stroke="#fff" stroke-width="0.9"/>')
            elif mk == "tri":
                elems.append(f'<polygon points="{px:.1f},{py-5.0:.1f} {px-4.5:.1f},{py+3.8:.1f} {px+4.5:.1f},{py+3.8:.1f}" fill="{s["color"]}" fill-opacity="{op}" stroke="#fff" stroke-width="0.9"/>')
            elif mk == "diam":
                elems.append(f'<polygon points="{px:.1f},{py-5.0:.1f} {px-4.6:.1f},{py:.1f} {px:.1f},{py+5.0:.1f} {px+4.6:.1f},{py:.1f}" fill="{s["color"]}" fill-opacity="{op}" stroke="#fff" stroke-width="0.9"/>')
        if s.get("label"):
            leg_items.append(s)

    if leg_items:
        lh = 19 * len(leg_items) + 12
        lw_box = max(len(s["label"]) for s in leg_items) * 6.45 + 42
        lx = (x0 + 12) if "l" in legend_pos else (x0 + pw - lw_box - 12)
        ly = (y0 + 12) if "t" in legend_pos else (y0 + ph - lh - 12)
        elems.append(f'<rect x="{lx:.1f}" y="{ly:.1f}" width="{lw_box:.1f}" height="{lh:.1f}" rx="5" fill="#ffffff" fill-opacity="0.93" stroke="#d1d5db" stroke-width="1"/>')
        for i, s in enumerate(leg_items):
            iy = ly + 16 + i * 19
            d_attr = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
            elems.append(f'<line x1="{lx+9:.1f}" y1="{iy-4:.1f}" x2="{lx+29:.1f}" y2="{iy-4:.1f}" stroke="{s["color"]}" stroke-width="2.4"{d_attr}/>')
            elems.append(f'<circle cx="{lx+19:.1f}" cy="{iy-4:.1f}" r="3.5" fill="{s["color"]}"/>')
            elems.append(f'<text x="{lx+36:.1f}" y="{iy:.1f}" font-size="11.2">{html.escape(s["label"])}</text>')

    elems.append("</svg>\n")
    with open(path, "w") as f:
        f.write("\n".join(elems))


betas = [3.0, 4.0, 5.0, 6.0, 7.0]
def min_sec(pref):
    xs, ys = [], []
    for b in betas:
        cx, cz = f"{pref}_Xerr.code", f"{pref}_Zerr.code"
        if b in kmc_q[cx] and b in kmc_q[cz]:
            xs.append(b); ys.append(min(kmc_q[cx][b][1], kmc_q[cz][b][1]))
        elif b in kmc_q[cx]:
            xs.append(b); ys.append(kmc_q[cx][b][1])
    return xs, ys

# 1a: True Two-Sector 3D Quantum Memory Lifetime
s1a = []
for pref, lbl, col, dsh, mk in [
    ("bdl_tor2_hyb21_s2", "3D BDL Torus Hybrid 2->1 rib (N=3160, Δ=(5,5))", "#dc2626", "",    "circle"),
    ("bdl_tor3_L2r1_s2",  "3D BDL 3x3 Torus 1-rib (N=2907, Δ=(4,5))",       "#ea580c", "",    "square"),
    ("bdl_tor2_L2r1_s2",  "3D BDL 2x2 Torus 1-rib (N=1292, Δ=(3,5))",       "#16a34a", "",    "tri"),
    ("bdl_sq_L3r2_s2",    "3D BDL Planar 2-rib L=3 (N=2152, Δ=(3,4))",      "#2563eb", "",    "diam"),
    ("bdl_sq_L3r1_s2",    "3D BDL Planar 1-rib L=3 (N=1271, Δ=(2,3))",      "#7c3aed", "6,4", "circle"),
    ("layer_913",         "3D Layer Code [[9,1,3]] (N=1853, Δ=(2,2))",      "#92400e", "6,4", "square"),
    ("toric3d_L7",        "3D Toric L=7 (N=1029, min=Z-string sector)",     "#6b7280", "3,3", "tri"),
    ("toric2d_L30",       "2D Toric L=30 (N=1800, Δ=(2,2))",                "#0891b2", "3,3", "diam"),
]:
    xs, ys = min_sec(pref)
    s1a.append({"xs": xs, "ys": ys, "color": col, "dash": dsh, "marker": mk, "label": lbl})
save_single_plot(
    os.path.join(FIG_DIR, "fig1a_quantum_lifetime.svg"),
    "True Two-Sector 3D Quantum Memory Lifetime min(<t_X>, <t_Z>)",
    "Inverse Bath Temperature β = 1/T", "Quantum Memory Lifetime min(<t_X>, <t_Z>)",
    (2.85, 7.15), (1.0, 2.5e8),
    [(3,"3"), (4,"4"), (5,"5"), (6,"6"), (7,"7")],
    [(1e0,"10⁰"), (1e2,"10²"), (1e4,"10⁴"), (1e6,"10⁶"), (1e8,"10⁸")],
    s1a, ylog=True, legend_pos="tl"
)

# 1b: Sawtooth Entropic Dip & Recovery
s1b = []
for b, col, mk in [(4.0, "#2563eb", "circle"), (5.0, "#ea580c", "square"), (6.0, "#dc2626", "tri")]:
    s1b.append({"xs": [1,2,3], "ys": [kmc_q[f"bdl_sq_L2r1_s{s}_Xerr.code"][b][1] for s in [1,2,3]],
                "color": col, "dash": "", "marker": mk, "label": f"X-sector lifetime (β={b:.0f})"})
    s1b.append({"xs": [1,2,3], "ys": [kmc_q[f"bdl_sq_L2r1_s{s}_Zerr.code"][b][1] for s in [1,2,3]],
                "color": col, "dash": "6,4", "marker": mk, "label": f"Z-sector lifetime (β={b:.0f})", "alpha": 0.8})
save_single_plot(
    os.path.join(FIG_DIR, "fig1b_sawtooth_steps.svg"),
    "Alternating R_X / R_Z Sawtooth Entropic Dip & Recovery (1-Rib L=2)",
    "Recursion Step s (Planar Seed → R_X → R_Z → R_X)", "Sector Memory Lifetime <t_fail>",
    (0.8, 3.2), (2.0, 3e7),
    [(1,"s=1 (R_X, N=19)"), (2,"s=2 (R_Z∘R_X, N=323)"), (3,"s=3 (R_X∘R_Z∘R_X, N=2911)")],
    [(1e1,"10¹"), (1e3,"10³"), (1e5,"10⁵"), (1e7,"10⁷")],
    s1b, ylog=True, legend_pos="tl"
)

# 2: Phase Transition Crossing beta_c (1-Rib vs 2-Rib)
s2 = []
for cname, lbl, col, dsh, mk in [
    ("cls_l4_r2_step2.code", "2-Rib Step 2 (l=4, N=400, 1 cycle)",          "#dc2626", "",    "square"),
    ("cls_l3_r1_step2.code", "1-Rib Step 2 (l=3, N=196, 1 cycle)",          "#2563eb", "",    "circle"),
    ("cls_l3_r1_step4.code", "1-Rib Step 4 (l=3, N=9604, ≥capped at β≥5)",  "#16a34a", "6,3", "tri"),
    ("cls_l3_r1_step0.code", "Step 0 Base Loop Seed (N=4)",                 "#6b7280", "3,3", "diam"),
]:
    pts = sorted(cross[cname].items())
    s2.append({"xs": [p[0] for p in pts], "ys": [p[1][1] for p in pts],
               "color": col, "dash": dsh, "marker": mk, "label": lbl})
save_single_plot(
    os.path.join(FIG_DIR, "fig2_phase_crossing.svg"),
    "Finite-Temperature Self-Correction Phase Transition (β_c ≈ 3.3)",
    "Inverse Bath Temperature β = 1/T", "Mean Memory Lifetime <t_fail>",
    (2.85, 5.65), (1.5e2, 1.5e8),
    [(3.0,"3.0"), (3.5,"3.5"), (4.0,"4.0"), (4.5,"4.5"), (5.0,"5.0"), (5.5,"5.5")],
    [(1e2,"10²"), (1e4,"10⁴"), (1e6,"10⁶"), (1e8,"10⁸")],
    s2, ylog=True, vline=(3.33, "Critical β_c ≈ 3.3"), legend_pos="tl"
)

# 3a: Decoder Comparison Thresholds
s3a = []
for cname, dec, lbl, col, dsh, mk in [
    ("bdl_sq_L3r1_s2_Xerr.code", "bposd",      "BP+OSD — 1-Rib (N=1271)",          "#dc2626", "",    "circle"),
    ("bdl_sq_L3r2_s2_Xerr.code", "bposd",      "BP+OSD — 2-Rib (N=2152)",          "#ea580c", "6,3", "square"),
    ("bdl_sq_L3r1_s2_Xerr.code", "uf",         "Cluster UF — 1-Rib (N=1271)",      "#2563eb", "",    "tri"),
    ("bdl_sq_L3r2_s2_Xerr.code", "uf",         "Cluster UF — 2-Rib (N=2152)",      "#0891b2", "6,3", "diam"),
    ("bdl_sq_L3r1_s2_Xerr.code", "rg_strict",  "Hard Sheaf-RG Strict — 1-Rib",     "#16a34a", "3,3", "circle"),
    ("bdl_sq_L3r1_s2_Xerr.code", "rg_relaxed", "Hard Sheaf-RG Relaxed — 1-Rib",    "#7c3aed", "3,3", "square"),
]:
    pts = sorted(thresh[(cname, dec)])
    s3a.append({"xs": [p[0]*100 for p in pts], "ys": [p[1] for p in pts],
                "color": col, "dash": dsh, "marker": mk, "label": lbl})
save_single_plot(
    os.path.join(FIG_DIR, "fig3a_decoder_thresholds.svg"),
    "Decoder Architecture Comparison Under i.i.d. Pauli Noise (L=3)",
    "Physical Error Rate p (%)", "Logical Failure Probability P_L(p)",
    (0.5, 12.5), (0.0, 0.56),
    [(2,"2%"), (4,"4%"), (6,"6%"), (8,"8%"), (10,"10%"), (12,"12%")],
    [(0.0,"0.0"), (0.1,"0.1"), (0.2,"0.2"), (0.3,"0.3"), (0.4,"0.4"), (0.5,"0.5")],
    s3a, legend_pos="br"
)

# 3b: Passive Thermal Cooling & Glassy Pinning
s3b = []
for key, lbl, col, dsh, mk in [
    (("cls_l4_r2_step3.code", 6.0), "2-Rib Step 3: Cold Quench β=6.0 (Glassy Plateau!)", "#dc2626", "",    "square"),
    (("cls_l4_r2_step3.code", 4.0), "2-Rib Step 3: Warm Quench β=4.0 (2x Lower!)",       "#2563eb", "6,4", "square"),
    (("cls_l3_r1_step3.code", 6.0), "1-Rib Step 3: Cold Quench β=6.0 (3.4x Less Trap!)", "#ea580c", "",    "circle"),
    (("cls_l3_r1_step3.code", 4.0), "1-Rib Step 3: Warm Quench β=4.0",                   "#16a34a", "6,4", "circle"),
    (("toric3d_L7_Xerr.code", 6.0), "3D Toric L=7 Membrane: Cold Quench β=6.0 → 0",      "#6b7280", "3,3", "tri"),
]:
    pts = [(t, max(v, 1e-4)) for t, v in sorted(cool[key]) if 0.2 <= t <= 7e7]
    s3b.append({"xs": [p[0] for p in pts], "ys": [p[1] for p in pts],
                "color": col, "dash": dsh, "marker": mk, "label": lbl})
save_single_plot(
    os.path.join(FIG_DIR, "fig3b_glassy_cooling.svg"),
    "Passive Initialization from T=∞: Glassy Rib Pinning (BDL Outlook #2)",
    "Passive Bath Evolution Time t", "Residual Syndrome Density <|σ(t)| / M>",
    (0.2, 8e7), (1e-4, 0.6),
    [(1e0,"10⁰"), (1e2,"10²"), (1e4,"10⁴"), (1e6,"10⁶"), (6.7e7,"7×10⁷")],
    [(1e-4,"10⁻⁴"), (1e-3,"10⁻³"), (1e-2,"10⁻²"), (1e-1,"10⁻¹"), (0.5,"0.5")],
    s3b, xlog=True, ylog=True, legend_pos="tr"
)

# 4: Hybrid Lazy Active-Passive QEC
s4 = []
for cname, lbl, col, dsh, mk in [
    ("bdl_tor3_L2r1_s2_Zerr.code", "3D BDL 3x3 Torus 1-Rib (Z-sector, N=2907)", "#16a34a", "",    "circle"),
    ("bdl_tor3_L2r1_s2_Xerr.code", "3D BDL 3x3 Torus 1-Rib (X-sector, N=2907)", "#dc2626", "",    "square"),
    ("bdl_sq_L3r2_s2_Xerr.code",   "3D BDL Planar 2-Rib L=3 (X-sector, N=2152)","#2563eb", "",    "tri"),
    ("layer_913_Xerr.code",        "3D Layer Code [[9,1,3]] (N=1853)",          "#ea580c", "6,4", "diam"),
    ("toric2d_L20_Xerr.code",      "2D Toric Code L=20 (N=800)",                "#6b7280", "3,3", "circle"),
]:
    pts = sorted(hybrid[cname])
    s4.append({"xs": [p[0] for p in pts], "ys": [p[1] for p in pts],
               "color": col, "dash": dsh, "marker": mk, "label": lbl})
save_single_plot(
    os.path.join(FIG_DIR, "fig4_hybrid_lazy_qec.svg"),
    "Hybrid Lazy Active + Passive Thermal Memory (β = 5.0, 20 Cycles)",
    "Active QEC Readout/Correction Interval τ_EC", "Cumulative Logical Failure Prob. (20 Cycles)",
    (8.0, 6000.0), (0.0, 1.05),
    [(10,"10"), (50,"50"), (200,"200"), (1000,"1000"), (5000,"5000")],
    [(0.0,"0.0"), (0.2,"0.2"), (0.4,"0.4"), (0.6,"0.6"), (0.8,"0.8"), (1.0,"1.0")],
    s4, xlog=True, legend_pos="br"
)
print("Rendered 6 clean single-panel SVGs!")
