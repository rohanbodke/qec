#!/usr/bin/env python3
"""
Render V4 First-Principles Publication Figures (760x500 SVG -> 1518x1000 Retina PNG):
  - fig2_phase_crossing.png          : Uncapped (c=0) Multi-Cycle Self-Correction Phase Crossing (k=0,1,2)
  - fig9_fixed_T_four_classes.png    : Fixed-Temperature <t_mem(N, beta)> vs. N Across All 4 Symmetry Classes
  - fig10_effective_barrier_scaling.png : Empirical Arrhenius Barrier Delta_eff(N) = d(ln<t>)/d(beta) vs. N
  - fig11_gibbs_confinement.png      : Chesi-Loss / BDL Sec. 5 Gibbs Cluster Confinement <|Q_max|/N> vs. N
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


kmc = defaultdict(dict)
gibbs = defaultdict(dict)

with open(os.path.join(WS, "cluster_v4_results.txt")) as f:
    for raw in f:
        line = raw.strip()
        if not line or line.startswith(("ERR|", "TIMEOUT|")):
            continue
        parts = line.split("|", 3)
        if len(parts) < 4:
            continue
        suite, label, _, body = parts
        kv = parse_kv(body)
        if body.startswith("KMC "):
            b = float(kv["beta"])
            kmc[label][b] = {
                "N": int(kv["N"]),
                "mean_t": float(kv["mean_t"]),
                "sem_t": float(kv["sem_t"]),
                "capped": int(kv["capped"]),
            }
        elif body.startswith("GIBBS "):
            b = float(kv["beta"])
            gibbs[label][b] = {
                "N": int(kv["N"]),
                "max_cl_frac": float(kv["max_cl_frac"]),
                "p_fail": float(kv["p_fail"]),
                "p_instab1": float(kv["p_instab1"]),
            }


def fit_slope(bmap, bmin=3.5, bmax=6.0):
    pts = [
        (b, math.log(r["mean_t"]))
        for b, r in sorted(bmap.items())
        if bmin <= b <= bmax and r["capped"] == 0 and r["mean_t"] > 0
    ]
    if len(pts) < 2:
        return None
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    return (sxy / sxx) if sxx > 1e-12 else None


def save_single_plot(svg_path, title, xlabel, ylabel, xlim, ylim, xticks, yticks,
                     series, xlog=False, ylog=False, vline=None, legend_pos="tl", annotations=None):
    W, H = 760, 500
    x0, y0, pw, ph = 86, 48, 636, 382
    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<style>text { font-family: -apple-system, Helvetica, Arial, sans-serif; fill: #1f2937; }</style>',
        f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="#fafafa" stroke="#374151" stroke-width="1.3"/>',
        f'<text x="{x0 + pw/2:.1f}" y="{y0 - 14:.1f}" text-anchor="middle" font-size="14.2" font-weight="700">{html.escape(title)}</text>',
        f'<text x="{x0 + pw/2:.1f}" y="{y0 + ph + 42:.1f}" text-anchor="middle" font-size="12.8" font-weight="600">{html.escape(xlabel)}</text>',
        f'<text x="{x0 - 56:.1f}" y="{y0 + ph/2:.1f}" text-anchor="middle" font-size="12.8" font-weight="600" transform="rotate(-90 {x0 - 56:.1f} {y0 + ph/2:.1f})">{html.escape(ylabel)}</text>',
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
        elems.append(f'<text x="{px:.1f}" y="{y0+ph+20}" text-anchor="middle" font-size="11.2">{html.escape(xlbl)}</text>')

    for yv, ylbl in yticks:
        py = ty(yv)
        elems.append(f'<line x1="{x0}" y1="{py:.1f}" x2="{x0+pw}" y2="{py:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        elems.append(f'<line x1="{x0-5}" y1="{py:.1f}" x2="{x0}" y2="{py:.1f}" stroke="#374151" stroke-width="1.2"/>')
        elems.append(f'<text x="{x0-9}" y="{py+4:.1f}" text-anchor="end" font-size="11.2">{html.escape(ylbl)}</text>')

    if vline is not None:
        vx, vlbl = vline
        px = tx(vx)
        elems.append(f'<line x1="{px:.1f}" y1="{y0}" x2="{px:.1f}" y2="{y0+ph}" stroke="#111827" stroke-width="1.5" stroke-dasharray="5,4"/>')
        elems.append(f'<text x="{px+6:.1f}" y="{y0+ph-14:.1f}" font-size="11.0" font-weight="600" fill="#111827">{html.escape(vlbl)}</text>')

    leg_items = []
    for s in series:
        pts_raw = []
        for i, (x, y) in enumerate(zip(s["xs"], s["ys"])):
            if xlim[0] <= x <= xlim[1]:
                se = s["sems"][i] if s.get("sems") else 0.0
                pts_raw.append((x, y, se))
        if not pts_raw:
            continue
        pts = [(tx(x), ty(y)) for x, y, _ in pts_raw]
        # Optional SEM error bars
        for x, y, se in pts_raw:
            if se > 0.0:
                px = tx(x)
                y_hi = ty(y + se)
                y_lo = ty(max(ylim[0] * 1.01, y - se))
                elems.append(f'<line x1="{px:.1f}" y1="{y_hi:.1f}" x2="{px:.1f}" y2="{y_lo:.1f}" stroke="{s["color"]}" stroke-width="1.3" stroke-opacity="0.75"/>')
        d_attr = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
        lw = s.get("lw", 2.4)
        op = s.get("alpha", 1.0)
        pstr = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        elems.append(f'<polyline fill="none" stroke="{s["color"]}" stroke-width="{lw}" stroke-opacity="{op}"{d_attr} points="{pstr}"/>')
        mk = s.get("marker", "circle")
        for px, py in pts:
            if mk == "circle":
                elems.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.0" fill="{s["color"]}" fill-opacity="{op}" stroke="#fff" stroke-width="0.9"/>')
            elif mk == "square":
                elems.append(f'<rect x="{px-3.6:.1f}" y="{py-3.6:.1f}" width="7.2" height="7.2" fill="{s["color"]}" fill-opacity="{op}" stroke="#fff" stroke-width="0.9"/>')
            elif mk == "tri":
                elems.append(f'<polygon points="{px:.1f},{py-4.8:.1f} {px-4.3:.1f},{py+3.6:.1f} {px+4.3:.1f},{py+3.6:.1f}" fill="{s["color"]}" fill-opacity="{op}" stroke="#fff" stroke-width="0.9"/>')
            elif mk == "diam":
                elems.append(f'<polygon points="{px:.1f},{py-4.8:.1f} {px-4.4:.1f},{py:.1f} {px:.1f},{py+4.8:.1f} {px+4.4:.1f},{py:.1f}" fill="{s["color"]}" fill-opacity="{op}" stroke="#fff" stroke-width="0.9"/>')
        if s.get("label"):
            leg_items.append(s)

    if annotations:
        for ax, ay, txt, col in annotations:
            px, py = tx(ax), ty(ay)
            elems.append(f'<text x="{px:.1f}" y="{py:.1f}" font-size="10.7" font-weight="700" fill="{col}">{html.escape(txt)}</text>')

    if leg_items:
        lh = 17.6 * len(leg_items) + 10
        lw_box = max(len(s["label"]) for s in leg_items) * 6.05 + 38
        lx = (x0 + 10) if "l" in legend_pos else (x0 + pw - lw_box - 10)
        ly = (y0 + 10) if "t" in legend_pos else (y0 + ph - lh - 10)
        elems.append(f'<rect x="{lx:.1f}" y="{ly:.1f}" width="{lw_box:.1f}" height="{lh:.1f}" rx="5" fill="#ffffff" fill-opacity="0.93" stroke="#d1d5db" stroke-width="1"/>')
        for i, s in enumerate(leg_items):
            iy = ly + 14.8 + i * 17.6
            d_attr = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
            elems.append(f'<line x1="{lx+8:.1f}" y1="{iy-4:.1f}" x2="{lx+27:.1f}" y2="{iy-4:.1f}" stroke="{s["color"]}" stroke-width="2.2"{d_attr}/>')
            elems.append(f'<circle cx="{lx+17.5:.1f}" cy="{iy-4:.1f}" r="3.2" fill="{s["color"]}"/>')
            elems.append(f'<text x="{lx+33:.1f}" y="{iy:.1f}" font-size="10.5">{html.escape(s["label"])}</text>')

    elems.append("</svg>\n")
    with open(svg_path, "w") as f:
        f.write("\n".join(elems))


# =============================================================================
# FIG 2 (UPGRADED UNCAPPED): Multi-Cycle Self-Correction Phase Crossing (c=0!)
# =============================================================================
def extract_beta_curve(cname, uncapped_only=True):
    xs, ys, sems = [], [], []
    for b, r in sorted(kmc[cname].items()):
        if uncapped_only and r["capped"] > 0:
            continue
        xs.append(b); ys.append(r["mean_t"]); sems.append(r["sem_t"])
    return xs, ys, sems

s_fig2 = []
for cname, lbl, col, dsh, mk in [
    ("cls_l2_r1_step0.code", "Cycle k=0 Seed (N=4, Δ_eff=2.07)",                   "#6b7280", "3,3", "diam"),
    ("cls_l2_r1_step2.code", "1-Rib Cycle k=1 (l=2, N=100, Δ_eff=5.09)",           "#2563eb", "",    "square"),
    ("cls_l2_r1_step4.code", "1-Rib Cycle k=2 (l=2, N=2500, Δ_eff=11.17, c=0!)",   "#dc2626", "",    "circle"),
    ("cls_l3_r2_step2.code", "Proven 2-Rib Cycle k=1 (l=3, N=256, Δ_eff=5.39)",    "#059669", "6,3", "tri"),
    ("cls_l3_r2_step4.code", "Proven 2-Rib Cycle k=2 (l=3, N=16384, Δ_eff=8.45!)", "#7c3aed", "6,3", "diam"),
    ("cls_l3_r1_step4.code", "Shifted 1-Rib Cycle k=2 (l=3, N=9604, uncapped!)",   "#ea580c", "4,3", "square"),
]:
    xs, ys, sems = extract_beta_curve(cname, uncapped_only=True)
    s_fig2.append({"xs": xs, "ys": ys, "sems": sems, "color": col, "dash": dsh, "marker": mk, "label": lbl})

save_single_plot(
    os.path.join(FIG_DIR, "fig2_phase_crossing.svg"),
    "Uncapped Multi-Cycle Self-Correction Phase Crossings (k = 0 → 1 → 2, Zero Capping)",
    "Inverse Bath Temperature β = 1/T", "Mean Memory Lifetime <t_mem> (Uncapped BKL KMC)",
    (2.4, 5.1), (70.0, 6.0e7),
    [(2.5,"2.5"), (3.0,"3.0"), (3.5,"3.5"), (4.0,"4.0"), (4.5,"4.5"), (5.0,"5.0")],
    [(1e2,"10²"), (1e3,"10³"), (1e4,"10⁴"), (1e5,"10⁵"), (1e6,"10⁶"), (1e7,"10⁷")],
    s_fig2, ylog=True, vline=(3.15, "T_c⁻¹ (k:0→1)"), legend_pos="tl"
)


# =============================================================================
# FIG 9 (FLAGSHIP): Fixed-Temperature <t_mem(N, beta)> vs. N Across All 4 Classes!
# =============================================================================
def extract_N_curve(cnames, beta_target):
    pts = []
    for cn in cnames:
        r = kmc[cn].get(beta_target)
        if r is not None:
            pts.append((r["N"], r["mean_t"], r["sem_t"]))
    pts.sort(key=lambda t: t[0])
    return [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]

# For 3D Quantum BDL two-sector min(t_X, t_Z) across full cycles s=0 -> 2 -> 4 at beta=6.0:
q_2cyc_Ns, q_2cyc_ys = [], []
for s in [0, 2, 4]:
    rx = kmc[f"sq_aniso_L2r1_s{s}_Xerr.code"].get(6.0)
    rz = kmc[f"sq_aniso_L2r1_s{s}_Zerr.code"].get(6.0)
    if rx is not None:
        # At s=4, Zerr at beta=5.0 is already 3.77e6 >> Xerr at beta=6.0 (6.82e4), so min(t_X, t_Z) = t_X exact!
        vz = rz["mean_t"] if rz is not None else 1e9
        q_2cyc_Ns.append(rx["N"])
        q_2cyc_ys.append(min(rx["mean_t"], vz))

# Also RP^2 3D Quantum X-sector s=0 -> 1 -> 2 -> 3 at beta=4.5 (or odd steps s=1->3, even s=0->2)
# Better: plot RP^2 full cycles + odd steps, or sq_aniso_L2r1 Z-sector s=0 -> 2 -> 4 at beta=4.5 (all uncapped c=0!):
# Wait: look at sq_aniso_L2r1_Zerr at beta=4.5: s=0 (2.74) -> s=2 (988.9) -> s=4 (8.47e4), ALL AT BETA=4.5 AND ALL c=0!!
q_z45_Ns, q_z45_ys, q_z45_sems = extract_N_curve(
    [f"sq_aniso_L2r1_s{s}_Zerr.code" for s in [0, 2, 4]], 4.5
)

tor_Ns, tor_ys, tor_sems = extract_N_curve([f"toric2d_L{L}_Xerr.code" for L in [4,6,10,16,24,34]], 4.5)
bb_Ns, bb_ys, bb_sems = extract_N_curve(
    [f"bb_gross_{l}x{m}_Xerr.code" for l,m in [(6,6),(12,6),(18,12),(24,18),(30,18),(36,18),(48,18),(60,18)]], 4.25
)
haah_Ns, haah_ys, haah_sems = extract_N_curve([f"haah_cubic_L{L}_Xerr.code" for L in [3,5,7,9,11,13]], 4.25)
lay_Ns, lay_ys, lay_sems = extract_N_curve(
    ["layer_512_Zerr.code","layer_steane_Zerr.code","layer_913_Zerr.code","layer_13_1_3_Zerr.code","layer_913_ksep2_Zerr.code","layer_15_35_Zerr.code"], 4.5
)
cls_Ns, cls_ys, cls_sems = extract_N_curve([f"cls_l2_r1_step{s}.code" for s in [0, 2, 4]], 4.0)

s_fig9 = [
    {"xs": cls_Ns,   "ys": cls_ys,   "sems": cls_sems,  "color": "#dc2626", "dash": "",    "marker": "circle", "lw": 2.8,
     "label": "Class IV: BDL 2D (k=0,1,2, β=4.0, ↑)"},
    {"xs": q_2cyc_Ns,"ys": q_2cyc_ys,"sems": None,      "color": "#e11d48", "dash": "6,3", "marker": "diam",   "lw": 2.6,
     "label": "Class IV: 3D BDL min(t_X,t_Z) (k=0,1,2, β=6, ↑)"},
    {"xs": q_z45_Ns, "ys": q_z45_ys, "sems": q_z45_sems,"color": "#f97316", "dash": "",    "marker": "tri",    "lw": 2.5,
     "label": "Class IV: 3D BDL Z-Sector (k=0,1,2, β=4.5, ↑)"},
    {"xs": haah_Ns,  "ys": haah_ys,  "sems": haah_sems, "color": "#7c3aed", "dash": "",    "marker": "square", "lw": 2.4,
     "label": "Class II: Haah 3D Cubic A (β=4.25, peaks L*=11 → ↓)"},
    {"xs": bb_Ns,    "ys": bb_ys,    "sems": bb_sems,   "color": "#2563eb", "dash": "",    "marker": "circle", "lw": 2.4,
     "label": "Class I: Scaled Gross BB (K=12, β=4.25 → 1/N ↓)"},
    {"xs": lay_Ns,   "ys": lay_ys,   "sems": lay_sems,  "color": "#92400e", "dash": "5,3", "marker": "square", "lw": 2.2,
     "label": "Class III: 3D Layer Codes (β=4.5, sheet entropy ↓)"},
    {"xs": tor_Ns,   "ys": tor_ys,   "sems": tor_sems,  "color": "#4b5563", "dash": "3,3", "marker": "diam",   "lw": 2.2,
     "label": "Class I: 2D Toric Code (β=4.5, 1/N ↓)"},
]

save_single_plot(
    os.path.join(FIG_DIR, "fig9_fixed_T_four_classes.svg"),
    "Fixed-Temperature Memory Lifetime <t_mem(N, β)> vs. System Size N: Four Symmetry Classes",
    "Physical System Size N (Qubits / Bits, Log Scale)", "Mean Memory Lifetime <t_mem(N, β_fixed)>",
    (1.6, 35000.0), (1.5, 1.5e11),
    [(2,"2"), (10,"10"), (100,"10²"), (1000,"10³"), (10000,"10⁴"), (30000,"3×10⁴")],
    [(1e0,"10⁰"), (1e2,"10²"), (1e4,"10⁴"), (1e6,"10⁶"), (1e8,"10⁸"), (1e10,"10¹⁰")],
    s_fig9, xlog=True, ylog=True, legend_pos="tl",
    annotations=[
        (1500.0, 1.3e8, "L*=11 Bravyi–Haah Turnover ↓", "#6d28d9"),
        (1150.0, 4.2e5, "Gross BB 1/N Decay ↓", "#1d4ed8"),
        (3300.0, 2.2e6, "BDL Exponential ↑", "#b91c1c"),
    ]
)


# =============================================================================
# FIG 10 (FLAGSHIP): Empirical Effective Barrier Delta_eff(N) = d(ln<t>)/d(beta)!
# =============================================================================
def slope_series(cnames, bmin=3.5, bmax=6.0):
    xs, ys = [], []
    for cn in cnames:
        bmap = kmc[cn]
        if not bmap:
            continue
        N = next(iter(bmap.values()))["N"]
        sl = fit_slope(bmap, bmin, bmax)
        if sl is not None:
            xs.append(N); ys.append(sl)
    pairs = sorted(zip(xs, ys))
    return [p[0] for p in pairs], [p[1] for p in pairs]

# Note: for sq_aniso_L2r1_s0..4 Zerr and Xerr (two full cycles!), use low-T uncapped slopes
c_cls_N, c_cls_D = slope_series([f"cls_l2_r1_step{s}.code" for s in [0, 2, 4]], 3.5, 5.0)
c_rp2_N, c_rp2_D = slope_series([f"nonloc_rp2_aniso_L2r1_s{s}_Xerr.code" for s in [0, 1, 3]], 3.5, 6.0)
c_t2_N,  c_t2_D  = slope_series([f"loc3d_tor2_aniso_L2r1_s{s}_Xerr.code" for s in [0, 1, 3]], 3.5, 6.0)
c_sqZ_N, c_sqZ_D = slope_series([f"sq_aniso_L2r1_s{s}_Zerr.code" for s in [0, 2, 4]], 3.5, 6.0)
c_haah_N,c_haah_D= slope_series([f"haah_cubic_L{L}_Xerr.code" for L in [3, 5, 7, 9, 11, 13]], 3.5, 5.0)
c_bb_N,  c_bb_D  = slope_series([f"bb_gross_{l}x{m}_Xerr.code" for l,m in [(6,6),(12,6),(18,12),(24,18),(30,18),(36,18),(48,18),(60,18)]], 3.5, 5.0)
c_lay_N, c_lay_D = slope_series(["layer_512_Xerr.code","layer_steane_Xerr.code","layer_913_Xerr.code","layer_13_1_3_Xerr.code","layer_913_ksep2_Xerr.code","layer_15_35_Xerr.code"], 3.5, 5.0)
c_tor_N, c_tor_D = slope_series([f"toric2d_L{L}_Xerr.code" for L in [4, 6, 10, 16, 24, 34]], 3.5, 5.0)

s_fig10 = [
    {"xs": c_cls_N, "ys": c_cls_D, "color": "#dc2626", "dash": "",    "marker": "circle", "lw": 2.7,
     "label": "Class IV: BDL 2D (k=0,1,2: Δ_eff = 2.1→5.1→11.2, ∝ N^η)"},
    {"xs": c_rp2_N, "ys": c_rp2_D, "color": "#e11d48", "dash": "6,3", "marker": "diam",   "lw": 2.5,
     "label": "Class IV: 3D RP² BDL (s=0,1,3: Δ_eff = 2.1→4.1→9.9, ∝ N^η)"},
    {"xs": c_t2_N,  "ys": c_t2_D,  "color": "#ea580c", "dash": "",    "marker": "tri",    "lw": 2.5,
     "label": "Class IV: 3D-Local T² BDL (s=0,1,3: Δ_eff = 2.0→3.1→8.0)"},
    {"xs": c_sqZ_N, "ys": c_sqZ_D, "color": "#d97706", "dash": "4,3", "marker": "square", "lw": 2.3,
     "label": "Class IV: 3D Planar 2-Cycle (s=0,2,4, N≤25148: → 6.3)"},
    {"xs": c_haah_N,"ys": c_haah_D,"color": "#7c3aed", "dash": "",    "marker": "square", "lw": 2.4,
     "label": "Class II: Haah 3D Cubic A (4.3→9.3→8.8, Θ(log N))"},
    {"xs": c_bb_N,  "ys": c_bb_D,  "color": "#2563eb", "dash": "",    "marker": "circle", "lw": 2.4,
     "label": "Class I: Scaled Gross BB (flat ceiling Δ_eff = 5.4 = O(1))"},
    {"xs": c_lay_N, "ys": c_lay_D, "color": "#92400e", "dash": "5,3", "marker": "tri",    "lw": 2.2,
     "label": "Class III: 3D Layer Codes (N≤7001, Δ_eff = 1.5–2.1 = O(1))"},
    {"xs": c_tor_N, "ys": c_tor_D, "color": "#4b5563", "dash": "3,3", "marker": "diam",   "lw": 2.2,
     "label": "Class I: 2D Toric Code (N≤2312, Δ_eff = 1.2–2.3 = O(1))"},
]

save_single_plot(
    os.path.join(FIG_DIR, "fig10_effective_barrier_scaling.svg"),
    "Empirical Arrhenius Energy Barrier Δ_eff(N) = ∂ ln<t_mem>/∂β Across Symmetry Classes",
    "Physical System Size N (Qubits / Bits, Log Scale)", "Effective Dynamical Barrier Δ_eff(N)",
    (1.8, 35000.0), (0.0, 17.5),
    [(2,"2"), (10,"10"), (100,"10²"), (1000,"10³"), (10000,"10⁴"), (30000,"3×10⁴")],
    [(0,"0"), (2,"2"), (4,"4"), (6,"6"), (8,"8"), (10,"10"), (12,"12"), (14,"14"), (16,"16")],
    s_fig10, xlog=True, ylog=False, legend_pos="tl",
    annotations=[
        (600.0, 4.55, "Gross BB Flat O(1) Ceiling (Δ_eff ≈ 5.4)", "#1d4ed8"),
        (1650.0, 2.35, "Toric & Layer O(1) Ceiling", "#78350f"),
    ]
)


# =============================================================================
# FIG 11: Chesi-Loss / BDL Sec. 5 Equilibrium Gibbs Cluster Percolation vs Confinement!
# =============================================================================
def gibbs_frac_curve(cnames, beta_target):
    pts = []
    for cn in cnames:
        g = gibbs[cn].get(beta_target)
        if g is not None:
            pts.append((g["N"], max(1e-5, g["max_cl_frac"])))
    pts.sort(key=lambda t: t[0])
    return [p[0] for p in pts], [p[1] for p in pts]

gt_N, gt_F = gibbs_frac_curve([f"toric2d_L{L}_Xerr.code" for L in [4, 6, 10, 16, 24, 34]], 4.0)
gl_N, gl_F = gibbs_frac_curve(["layer_512_Xerr.code","layer_steane_Xerr.code","layer_913_Xerr.code","layer_13_1_3_Xerr.code","layer_913_ksep2_Xerr.code","layer_15_35_Xerr.code"], 4.0)
gc_N, gc_F = gibbs_frac_curve(["cls_l3_r2_step2.code", "cls_l3_r2_step4.code"], 4.5)
gc1_N,gc1_F= gibbs_frac_curve(["cls_l2_r1_step2.code", "cls_l2_r1_step4.code"], 4.5)
gq_N, gq_F = gibbs_frac_curve(["sq_aniso_L2r1_s2_Zerr.code", "sq_aniso_L2r1_s4_Zerr.code"], 4.0)
gr_N, gr_F = gibbs_frac_curve(["nonloc_rp2_aniso_L2r1_s2_Xerr.code", "nonloc_rp2_aniso_L2r1_s3_Xerr.code"], 4.5)
gt2_N,gt2_F= gibbs_frac_curve(["loc3d_tor2_aniso_L2r1_s2_Xerr.code", "loc3d_tor2_aniso_L2r1_s3_Xerr.code"], 4.5)

s_fig11 = [
    {"xs": gl_N,  "ys": gl_F,  "color": "#92400e", "dash": "5,3", "marker": "square", "lw": 2.5,
     "label": "Class III: 3D Layer Codes (β=4.0: percolates 21% → 57%)"},
    {"xs": gt_N,  "ys": gt_F,  "color": "#4b5563", "dash": "3,3", "marker": "diam",   "lw": 2.4,
     "label": "Class I: 2D Toric Code (β=4.0: deconfines 2.4% → 49%)"},
    {"xs": gt2_N, "ys": gt2_F, "color": "#ea580c", "dash": "",    "marker": "tri",    "lw": 2.6,
     "label": "Class IV: 3D-Local T² BDL (s:2→3: collapses 25% → 0.58%, P_fail=0)"},
    {"xs": gr_N,  "ys": gr_F,  "color": "#e11d48", "dash": "",    "marker": "diam",   "lw": 2.6,
     "label": "Class IV: Non-Local RP² BDL (s:2→3: collapses 12% → 0.65%, P_fail=0)"},
    {"xs": gq_N,  "ys": gq_F,  "color": "#d97706", "dash": "6,3", "marker": "square", "lw": 2.4,
     "label": "Class IV: 3D Planar BDL Z (s:2→4, N=25148: 11% → 5.2%, P_fail=0)"},
    {"xs": gc_N,  "ys": gc_F,  "color": "#16a34a", "dash": "",    "marker": "circle", "lw": 2.6,
     "label": "Class IV: Proven 2-Rib BDL (k:1→2, N=16384: 0.51% → 0.32%, P_instab=0)"},
]

save_single_plot(
    os.path.join(FIG_DIR, "fig11_gibbs_confinement.svg"),
    "Chesi–Loss / BDL Sec. 5 Gibbs Defect Confinement: Max Cluster Fraction <|Q_max|/N>",
    "Physical System Size N (Qubits / Bits, Log Scale)", "Largest Thermal Cluster Fraction <|Q_max| / N>",
    (25.0, 32000.0), (1.5e-5, 1.2),
    [(30,"30"), (100,"10²"), (1000,"10³"), (10000,"10⁴"), (30000,"3×10⁴")],
    [(1e-4,"0.01%"), (1e-3,"0.1%"), (1e-2,"1%"), (1e-1,"10%"), (1.0,"100%")],
    s_fig11, xlog=True, ylog=True, legend_pos="bl",
    annotations=[
        (1600.0, 0.72, "Deconfined Percolation (P_fail = 48–77%)", "#78350f"),
        (1050.0, 0.0017, "Hierarchical UV Confinement (P_fail = 0, P_instab = 0)", "#15803d"),
    ]
)

# Convert all 4 SVGs to high-DPI PNGs via qlmanage + sips and copy to ART_DIR!
for nm in [
    "fig2_phase_crossing",
    "fig9_fixed_T_four_classes",
    "fig10_effective_barrier_scaling",
    "fig11_gibbs_confinement",
]:
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

print("Successfully rendered and exported all 4 V4 flagship PNG figures!")
