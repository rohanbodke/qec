#!/usr/bin/env python3
"""
render_v16_tasteful_figures.py
Renders 5 focused, visually crystal-clear figures (V16) serving a single unified physical argument:

  1. fig_v16_1_landscape.png: Free-energy competition across 2D & 3D quantum code families
                              (2D Toric, 2D BB, 3D Random Layer, 3D Haah Cubic, and Uniform 2x2 3D Welded)
  2. fig_v16_2_mechanism.png: Microscopic quantum welding (both X-face and Z-cylinder steps, why 4 arms &
                              2 steps are minimal, physical meaning of W/5 & 2/5 contraction, and how the
                              exact same 5-edge majority rule gives fast O(N) local readout)
  3. fig_v16_3_scaling.png:   Unified fixed thermal thresholds, sawtooth (X, Z) flow, 423,559-qubit
                              quantum scaling (16,623x), and fast O(N) local readout matching global UF
  4. fig_v16_4_geometry.png:  Bounded-degree graphs vs 3D Euclidean space (L >= 1+sqrt(3) volume-packing
                              cutoff, minimal 3x3 single-cross scaling, and uniform-J cubic-grid wire entropy)
  5. fig_v16_5_cooling.png:   Autonomous storage self-healing vs 4-scale polynomial hot-state cooling
"""

import os
import sys

WS = "/Users/harshilavlani/Downloads/6.S980_gemini"
FIG_DIR = os.path.join(WS, "figures")
ART_DIR = "/Users/harshilavlani/.gemini/antigravity/brain/35299b0d-7439-4f8e-8617-0b646151a931"

sys.path.insert(0, os.path.join(WS, "src"))
from render_v13_unified_figures import draw_panel, interp_tcool, load_all, ols_slope, save_svg_png
from render_v15_complete_figures import load_v12


def render_double(nm, p1, p2):
    W, H = 1180, 456
    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<style>text { font-family: -apple-system, Helvetica, Arial, sans-serif; fill: #1f2937; }</style>',
    ]
    draw_panel(elems, 86, 42, 472, 352, **p1)
    draw_panel(elems, 676, 42, 472, 352, **p2)
    save_svg_png(nm, elems, W, H)


def render_fig1_landscape(kmc4, kmc6, kmc9):
    """
    Figure 1: Complete 2D & 3D Code Family Landscape (including Random Layer Code turnover right in Fig. 1!)
    """
    def v4_norm_curve(cnames, b):
        pts = [(kmc4[cn][b][0], kmc4[cn][b][1]) for cn in cnames if b in kmc4.get(cn, {})]
        y0 = pts[0][1]
        return [x for x, _ in pts], [y / y0 for _, y in pts]

    hx, hy = v4_norm_curve([f"haah_cubic_L{L}_Xerr.code" for L in [3, 5, 7, 9, 11, 13]], 4.25)
    bx, by = v4_norm_curve([f"bb_gross_{l}x{m}_Xerr.code" for l, m in [(6,6),(12,6),(18,12),(24,18),(30,18),(36,18),(48,18),(60,18)]], 4.25)
    tx_n, ty_n = v4_norm_curve([f"toric2d_L{L}_Xerr.code" for L in [4, 6, 10, 16, 24, 34]], 4.50)

    # Gu-Caha 3D Random Layer Code ensemble across n = 11, 13, 15, 17, 19 (N = 2,194 .. 10,955)
    caha_groups = {
        11: ["caha_n11_code_100", "caha_n11_code_11", "caha_n11_code_113", "caha_n11_code_120", "caha_n11_code_128", "caha_n11_code_151"],
        13: ["caha_n13_code_102", "caha_n13_code_104", "caha_n13_code_106"],
        15: ["caha_n15_code_15",  "caha_n15_code_16",  "caha_n15_code_17"],
        17: ["caha_n17_code_1201","caha_n17_code_14",  "caha_n17_code_145"],
        19: ["caha_n19_code_129", "caha_n19_code_199", "caha_n19_code_236"],
    }
    lx, ly_raw, l_slopes = [], [], []
    for nv in [11, 13, 15, 17, 19]:
        codes = caha_groups[nv]
        ns = [kmc6[(c + "_X_err", "uniform", "uf", 7.0)][0] for c in codes if (c + "_X_err", "uniform", "uf", 7.0) in kmc6]
        v7 = [kmc6[(c + "_X_err", "uniform", "uf", 7.0)][1] for c in codes if (c + "_X_err", "uniform", "uf", 7.0) in kmc6]
        v6 = [kmc6[(c + "_X_err", "uniform", "uf", 6.0)][1] for c in codes if (c + "_X_err", "uniform", "uf", 6.0) in kmc6]
        lx.append(sum(ns) / len(ns))
        ly_raw.append(sum(v7) / len(v7))
        l_slopes.append(ols_slope([6.0, 7.0], [sum(v6) / len(v6), sum(v7) / len(v7)]))
    ly = [v / ly_raw[0] for v in ly_raw]

    q3_pts = [kmc9[(f"sq_iso_L2r1_s{s}_Z_err", "uf", 6.50)] for s in [1, 3, 5]]
    q3_x = [p[0] for p in q3_pts]
    q3_y = [p[1] / q3_pts[0][1] for p in q3_pts]

    w_pts = [kmc9[(f"sq_rx_iso_s{s}_X_err", "uf", 2.25)] for s in [1, 2, 3, 4, 5]]
    w_x = [p[0] for p in w_pts]
    w_y = [p[1] / w_pts[0][1] for p in w_pts]

    s_norm = [
        {"xs": q3_x, "ys": q3_y, "color": "#be123c", "dash": "",    "label": f"3D Welded Quantum (to 424k: {q3_y[-1]:,.0f}×)"},
        {"xs": w_x,  "ys": w_y,  "color": "#ea580c", "dash": "7,3", "label": f"3D Welded Single-Sector (5 scales: {w_y[-1]:,.0f}×)"},
        {"xs": hx,   "ys": hy,   "color": "#7c3aed", "dash": "",    "label": "Haah 3D Cubic Code (peaks at 25×, turns over)"},
        {"xs": lx,   "ys": ly,   "color": "#059669", "dash": "",    "label": "3D Random Layer Code (peaks at 1.3×, turns over)"},
        {"xs": bx,   "ys": by,   "color": "#2563eb", "dash": "5,4", "label": "2D Bivariate Bicycle & Toric (fall < 1×)"},
        {"xs": tx_n, "ys": ty_n, "color": "#64748b", "dash": "3,3", "label": ""},
    ]

    def v4_slope_curve(cnames, bmin=3.5, bmax=5.0):
        xs, ys = [], []
        for cn in cnames:
            bm = kmc4.get(cn, {})
            bs = sorted(b for b in bm if bmin <= b <= bmax)
            if len(bs) >= 2:
                sl = ols_slope(bs, [bm[b][1] for b in bs])
                if sl is not None:
                    xs.append(bm[bs[0]][0]); ys.append(sl)
        return xs, ys

    c_haah_N, c_haah_D = v4_slope_curve([f"haah_cubic_L{L}_Xerr.code" for L in [3, 5, 7, 9, 11, 13]])
    c_bb_N, c_bb_D = v4_slope_curve([f"bb_gross_{l}x{m}_Xerr.code" for l, m in [(6,6),(12,6),(18,12),(24,18),(30,18),(36,18),(48,18),(60,18)]])
    c_tor_N, c_tor_D = v4_slope_curve([f"toric2d_L{L}_Xerr.code" for L in [4, 6, 10, 16, 24, 34]])

    w_bN, w_bD = [], []
    for s in range(6):
        b_cands = [b for b in (2.00, 2.05, 2.10, 2.15, 2.20, 2.25) if (f"sq_rx_iso_s{s}_X_err", "uf", b) in kmc9]
        if s == 0:
            b_cands = [b for b in (1.75, 2.00, 2.25) if ("sq_rx_iso_s0_X_err", "uf", b) in kmc9]
        w_bN.append(kmc9[(f"sq_rx_iso_s{s}_X_err", "uf", b_cands[0])][0])
        w_bD.append(ols_slope(b_cands, [kmc9[(f"sq_rx_iso_s{s}_X_err", "uf", b)][1] for b in b_cands]))

    q3_bN, q3_bD = [], []
    for s in (1, 3, 5):
        b_cands = [b for b in (6.00, 6.10, 6.25, 6.35, 6.50) if (f"sq_iso_L2r1_s{s}_Z_err", "uf", b) in kmc9]
        q3_bN.append(kmc9[(f"sq_iso_L2r1_s{s}_Z_err", "uf", b_cands[0])][0])
        q3_bD.append(ols_slope(b_cands, [kmc9[(f"sq_iso_L2r1_s{s}_Z_err", "uf", b)][1] for b in b_cands]))

    s_bar = [
        {"xs": w_bN,     "ys": w_bD,     "color": "#ea580c", "dash": "7,3", "label": "3D Welded Single-Sector (curves up)"},
        {"xs": q3_bN,    "ys": q3_bD,    "color": "#be123c", "dash": "",    "label": "3D Welded Quantum (doubles each cycle)"},
        {"xs": c_haah_N, "ys": c_haah_D, "color": "#7c3aed", "dash": "",    "label": "Haah 3D Cubic (straight log N, bends down)"},
        {"xs": lx,       "ys": l_slopes, "color": "#059669", "dash": "",    "label": "3D Random Layer Code (slow log N)"},
        {"xs": c_bb_N,   "ys": c_bb_D,   "color": "#2563eb", "dash": "5,4", "label": "2D Bivariate Bicycle & Toric (flat)"},
        {"xs": c_tor_N,  "ys": c_tor_D,  "color": "#64748b", "dash": "3,3", "label": ""},
    ]

    render_double(
        "fig_v16_1_landscape",
        dict(xlim=(15.0, 580000.0), ylim=(0.02, 8.0e7),
             title="(a) Lifetime Growth Relative to Smallest Code",
             xlabel="Physical Qubit Count N (Log Scale)", ylabel="Relative Memory Lifetime",
             xticks=[(20,"20"), (200,"200"), (2000,"2k"), (20000,"20k"), (400000,"400k")],
             yticks=[(0.05,"0.05×"), (1.0,"1×"), (100.0,"10²×"), (10000.0,"10⁴×")],
             series=s_norm, xlog=True, ylog=True, leg_pos="tl",
             hlines=[(1.0, "#6b7280", "", "br")],
             notes=[(490000.0, 0.36, "1× = No Gain With Size", "#475569", "end")]),
        dict(xlim=(1.6, 580000.0), ylim=(0.0, max(23.5, max(w_bD) * 1.38)),
             title="(b) Additive vs. Multiplicative Energy Barriers",
             xlabel="Physical Qubit Count N (Log Scale)", ylabel="Effective Energy Barrier",
             xticks=[(2,"2"), (30,"30"), (500,"500"), (10000,"10k"), (400000,"400k")],
             yticks=[(0,"0"), (5,"5"), (10,"10"), (15,"15"), (20,"20")],
             series=s_bar, xlog=True, ylog=False, leg_pos="tl"),
    )
    return q3_bD


def render_fig2_mechanism():
    """
    Figure 2 (1180 x 720):
      Row 1 (a): Full Two-Step Quantum Welding Cycle — Seed (Z-face + X-star), Odd X-Face Bridge,
                 Even Z-Cylinder Vertex Split, and Why 4 Arms & 2 Steps Are Minimal.
      Row 2 Left (b):  Physical Meaning of Coarse Weight W / 5 (1 Parent Edge = 5 Sub-Edges) & 2/5 Contraction.
      Row 2 Right (c): How That Exact Same 5-Edge Rule Gives Fast O(N) Local Cell-by-Cell Readout!
    """
    W, H = 1180, 720
    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<style>text { font-family: -apple-system, Helvetica, Arial, sans-serif; fill: #1f2937; }</style>',
    ]

    # --- ROW 1 (a): Full-width 4-card visual explanation of BOTH steps of a Quantum Welding Cycle ---
    elems.append('<rect x="18" y="14" width="1144" height="298" rx="12" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.6"/>')
    elems.append('<text x="590" y="41" text-anchor="middle" font-size="16.0" font-weight="700">'
                 '(a) How One Two-Step Quantum Welding Cycle Confines Both X and Z Errors</text>')

    # Card 1: Seed 2D Surface Code (Z-face + X-vertex star) (x: 36..272)
    elems.append('<rect x="36" y="56" width="236" height="240" rx="9" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.4"/>')
    elems.append('<text x="154" y="78" text-anchor="middle" font-size="13.2" font-weight="700" fill="#1e293b">1. Seed 2D Surface Code</text>')
    # Draw 2x2 patch of squares so BOTH a Z-face check AND a 4-arm X-vertex star are clearly visible!
    sx, sy = 88, 96
    elems.append(f'<rect x="{sx}" y="{sy}" width="66" height="66" fill="#dbeafe" stroke="#2563eb" stroke-width="2.4"/>')
    elems.append(f'<rect x="{sx+66}" y="{sy}" width="66" height="66" fill="#f8fafc" stroke="#94a3b8" stroke-width="1.6"/>')
    elems.append(f'<rect x="{sx}" y="{sy+66}" width="66" height="66" fill="#f8fafc" stroke="#94a3b8" stroke-width="1.6"/>')
    elems.append(f'<rect x="{sx+66}" y="{sy+66}" width="66" height="66" fill="#f8fafc" stroke="#94a3b8" stroke-width="1.6"/>')
    # Highlight red X-vertex star at central vertex (sx+66, sy+66)
    vx, vy = sx + 66, sy + 66
    elems.append(f'<line x1="{vx-28}" y1="{vy}" x2="{vx+28}" y2="{vy}" stroke="#dc2626" stroke-width="4.5"/>')
    elems.append(f'<line x1="{vx}" y1="{vy-28}" x2="{vx}" y2="{vy+28}" stroke="#dc2626" stroke-width="4.5"/>')
    elems.append(f'<circle cx="{vx}" cy="{vy}" r="5.5" fill="#991b1b" stroke="#ffffff" stroke-width="1.5"/>')
    elems.append(f'<text x="{sx+33}" y="{sy+37}" text-anchor="middle" font-size="11.4" font-weight="700" fill="#1e40af">Z-Face</text>')
    elems.append(f'<text x="{vx+33}" y="{vy+22}" text-anchor="middle" font-size="11.4" font-weight="700" fill="#b91c1c">X-Star</text>')
    elems.append('<text x="154" y="252" text-anchor="middle" font-size="11.8" font-weight="600" fill="#475569">Z checks on faces, X at corners;</text>')
    elems.append('<text x="154" y="270" text-anchor="middle" font-size="11.8" font-weight="700" fill="#b91c1c">X &amp; Z defects walk free (ΔE = 0)</text>')

    # Arrow 1 -> 2
    elems.append('<line x1="278" y1="176" x2="302" y2="176" stroke="#475569" stroke-width="2.6"/>')
    elems.append('<polygon points="310,176 300,170 300,182" fill="#475569"/>')

    # Card 2: Odd Step — Face Bridge Cross (Confines X Errors) (x: 316..576)
    elems.append('<rect x="316" y="56" width="260" height="240" rx="9" fill="#eff6ff" stroke="#bfdbfe" stroke-width="1.4"/>')
    elems.append('<text x="446" y="78" text-anchor="middle" font-size="13.2" font-weight="700" fill="#1e40af">2. Odd Step: Face Weld (X)</text>')
    # Sheet 1 (bottom)
    elems.append('<polygon points="344,218 424,182 544,182 464,218" fill="#dbeafe" fill-opacity="0.78" stroke="#2563eb" stroke-width="1.8"/>')
    # Perpendicular 4-arm yellow face bridge cross
    elems.append('<polygon points="404,218 484,182 484,110 404,146" fill="#fde68a" fill-opacity="0.92" stroke="#d97706" stroke-width="2.0"/>')
    elems.append('<polygon points="384,200 504,200 504,128 384,128" fill="#fcd34d" fill-opacity="0.84" stroke="#d97706" stroke-width="2.0"/>')
    # Sheet 0 (top) + bold 3-way face seams
    elems.append('<polygon points="344,146 424,110 544,110 464,146" fill="#dbeafe" fill-opacity="0.82" stroke="#2563eb" stroke-width="1.8"/>')
    elems.append('<line x1="384" y1="128" x2="504" y2="128" stroke="#1d4ed8" stroke-width="4.0"/>')
    elems.append('<line x1="404" y1="146" x2="484" y2="110" stroke="#1d4ed8" stroke-width="4.0"/>')
    elems.append('<text x="446" y="246" text-anchor="middle" font-size="11.8" font-weight="700" fill="#1e3a8a">3 Z-faces share each seam edge</text>')
    elems.append('<text x="446" y="265" text-anchor="middle" font-size="11.8" font-weight="700" fill="#047857">X defect splits 1 → 2 (+ΔE)</text>')
    elems.append('<text x="446" y="283" text-anchor="middle" font-size="11.3" font-weight="600" fill="#475569">(X corner checks split smoothly)</text>')

    # Arrow 2 -> 3
    elems.append('<line x1="582" y1="176" x2="606" y2="176" stroke="#475569" stroke-width="2.6"/>')
    elems.append('<polygon points="614,176 604,170 604,182" fill="#475569"/>')

    # Card 3: Even Step — Dual Vertex/Cylinder Weld (Confines Z Errors) (x: 620..880)
    elems.append('<rect x="620" y="56" width="260" height="240" rx="9" fill="#fef2f2" stroke="#fecaca" stroke-width="1.4"/>')
    elems.append('<text x="750" y="78" text-anchor="middle" font-size="13.2" font-weight="700" fill="#991b1b">3. Even Step: Cylinder Weld (Z)</text>')
    # Sheet 1 (bottom) with red internal X-star lines
    elems.append('<polygon points="648,218 728,182 848,182 768,218" fill="#ffffff" fill-opacity="0.88" stroke="#94a3b8" stroke-width="1.7"/>')
    elems.append('<line x1="688" y1="200" x2="808" y2="200" stroke="#dc2626" stroke-width="3.0"/>')
    elems.append('<line x1="708" y1="218" x2="788" y2="182" stroke="#dc2626" stroke-width="3.0"/>')
    # Vertical cylinder qubit pillars connecting internal vertices between Sheet 0 & Sheet 1!
    for px, py_top, py_bot in [(748, 128, 200), (688, 128, 200), (808, 128, 200), (708, 146, 218), (788, 110, 182)]:
        elems.append(f'<line x1="{px}" y1="{py_top}" x2="{px}" y2="{py_bot}" stroke="#d97706" stroke-width="3.4" stroke-dasharray="4,2"/>')
        elems.append(f'<circle cx="{px}" cy="{(py_top+py_bot)/2:.1f}" r="4.2" fill="#d97706" stroke="#ffffff" stroke-width="1.2"/>')
    # Sheet 0 (top) with red X-star seams
    elems.append('<polygon points="648,146 728,110 848,110 768,146" fill="#ffffff" fill-opacity="0.82" stroke="#94a3b8" stroke-width="1.7"/>')
    elems.append('<line x1="688" y1="128" x2="808" y2="128" stroke="#dc2626" stroke-width="3.6"/>')
    elems.append('<line x1="708" y1="146" x2="788" y2="110" stroke="#dc2626" stroke-width="3.6"/>')
    elems.append('<text x="750" y="246" text-anchor="middle" font-size="11.8" font-weight="700" fill="#991b1b">Cylinder links + split X-stars</text>')
    elems.append('<text x="750" y="265" text-anchor="middle" font-size="11.8" font-weight="700" fill="#047857">Z defect splits 1 → 2 (+ΔE)</text>')
    elems.append('<text x="750" y="283" text-anchor="middle" font-size="11.3" font-weight="600" fill="#475569">(4-edge Z-collars keep commutation)</text>')

    # Divider / Card 4: Why 4 Arms & 2 Sequential Steps Are Minimal (x: 896..1144)
    elems.append('<rect x="896" y="56" width="248" height="240" rx="9" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5"/>')
    elems.append('<text x="1020" y="78" text-anchor="middle" font-size="13.0" font-weight="700" fill="#1e293b">4. Minimal Cell No-Gos</text>')
    # Top sub-box: Cut 1 arm -> Open hole
    elems.append('<rect x="910" y="90" width="220" height="92" rx="7" fill="#fff1f2" stroke="#fecdd3" stroke-width="1.3"/>')
    elems.append('<text x="1020" y="110" text-anchor="middle" font-size="12.0" font-weight="700" fill="#9f1239">Cut 1 Bridge Arm (3-Arm T):</text>')
    elems.append('<text x="1020" y="131" text-anchor="middle" font-size="11.5" font-weight="600" fill="#374151">Opens unsealed hole between</text>')
    elems.append('<text x="1020" y="148" text-anchor="middle" font-size="11.5" font-weight="600" fill="#374151">sheets → loops thread gap</text>')
    elems.append('<text x="1020" y="169" text-anchor="middle" font-size="11.8" font-weight="700" fill="#be123c">Torus qubits: 2 → 1,004 (τ → 0)</text>')
    # Bottom sub-box: Merge X & Z into 1 step -> Intersection shortcuts
    elems.append('<rect x="910" y="192" width="220" height="92" rx="7" fill="#fff1f2" stroke="#fecdd3" stroke-width="1.3"/>')
    elems.append('<text x="1020" y="212" text-anchor="middle" font-size="12.0" font-weight="700" fill="#9f1239">Cross X &amp; Z in One Step:</text>')
    elems.append('<text x="1020" y="233" text-anchor="middle" font-size="11.5" font-weight="600" fill="#374151">Incompatible corner checks</text>')
    elems.append('<text x="1020" y="250" text-anchor="middle" font-size="11.5" font-weight="600" fill="#374151">open weight-1 shortcuts</text>')
    elems.append('<text x="1020" y="271" text-anchor="middle" font-size="11.8" font-weight="700" fill="#be123c">Qubits jump: 1 → 3 → 49 → 737</text>')

    # --- ROW 2 LEFT (b): Physical Meaning of W / 5 & Why 2x2 Shrinks Errors by 2 / 5 ---
    elems.append('<rect x="18" y="330" width="560" height="374" rx="12" fill="#f9fafb" stroke="#cbd5e1" stroke-width="1.6"/>')
    elems.append('<text x="298" y="357" text-anchor="middle" font-size="15.2" font-weight="700">'
                 '(b) Why Coarse Wall Weight Is W / 5 &amp; Shrinks by 2 / 5</text>')

    bx0, by0, bsz = 54, 412, 198
    # Draw exploded 5-part top wall bar above the 2x2 cell to make "5 sub-edges = 1 parent edge" unmistakable!
    wy_bar = by0 - 26
    seg_cols = [("#2563eb", "S0"), ("#2563eb", "S0"), ("#d97706", "Br"), ("#94a3b8", "S1"), ("#94a3b8", "S1")]
    for i, (sc, slbl) in enumerate(seg_cols):
        sx_i = bx0 + i * (bsz / 5.0)
        elems.append(f'<rect x="{sx_i+1.5:.1f}" y="{wy_bar}" width="{bsz/5.0 - 3.0:.1f}" height="16" rx="3" fill="{sc}"/>')
        elems.append(f'<text x="{sx_i + bsz/10.0:.1f}" y="{wy_bar+12}" text-anchor="middle" font-size="9.8" font-weight="700" fill="#ffffff">{slbl}</text>')
    elems.append(f'<text x="{bx0+bsz/2:.1f}" y="{wy_bar-7}" text-anchor="middle" font-size="11.8" font-weight="700" fill="#1e3a8a">'
                 '1 Parent Wall = 5 Sub-Edges (2 flipped = 2/5)</text>')

    elems.append(f'<rect x="{bx0}" y="{by0}" width="{bsz}" height="{bsz}" fill="#dbeafe" stroke="#1d4ed8" stroke-width="2.8"/>')
    for qx, qy in [(bx0+49, by0+54), (bx0+148, by0+54), (bx0+49, by0+152), (bx0+148, by0+152)]:
        elems.append(f'<text x="{qx}" y="{qy}" text-anchor="middle" font-size="12.2" font-weight="700" fill="#1e40af">Sheet 0</text>')
        elems.append(f'<text x="{qx}" y="{qy+15}" text-anchor="middle" font-size="10.8" font-weight="600" fill="#2563eb">(flipped)</text>')
    elems.append(f'<line x1="{bx0+99}" y1="{by0}" x2="{bx0+99}" y2="{by0+bsz}" stroke="#dc2626" stroke-width="6.2"/>')
    elems.append(f'<line x1="{bx0}" y1="{by0+99}" x2="{bx0+bsz}" y2="{by0+99}" stroke="#dc2626" stroke-width="6.2"/>')
    elems.append(f'<circle cx="{bx0+99}" cy="{by0+99}" r="7.5" fill="#1e3a8a" stroke="#ffffff" stroke-width="1.5"/>')
    elems.append(f'<text x="{bx0+99}" y="{by0+bsz+22}" text-anchor="middle" font-size="12.0" font-weight="700" fill="#b91c1c">'
                 '4 Violated Interior Seams (V = 4)</text>')

    # Right explanation box inside (b)
    elems.append('<rect x="276" y="380" width="284" height="304" rx="9" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.4"/>')
    elems.append('<text x="418" y="408" text-anchor="middle" font-size="13.0" font-weight="700" fill="#1e293b">Physical Meaning of Coarse Weight:</text>')
    elems.append('<text x="418" y="432" text-anchor="middle" font-size="12.0" font-weight="600" fill="#374151">Flipping 1 coarse parent edge flips</text>')
    elems.append('<text x="418" y="450" text-anchor="middle" font-size="12.0" font-weight="700" fill="#1e40af">all 5 sub-edges on that outer wall</text>')
    elems.append('<text x="418" y="468" text-anchor="middle" font-size="11.6" font-weight="600" fill="#475569">(2 Sheet 0 + 2 Sheet 1 + 1 Bridge).</text>')
    elems.append('<line x1="296" y1="486" x2="540" y2="486" stroke="#e2e8f0" stroke-width="1.4"/>')
    elems.append('<text x="418" y="510" text-anchor="middle" font-size="12.2" font-weight="600" fill="#374151">W flipped wall sub-edges represent</text>')
    elems.append('<text x="418" y="530" text-anchor="middle" font-size="12.8" font-weight="700" fill="#1e293b">coarse error progress = W / 5.</text>')
    elems.append('<line x1="296" y1="548" x2="540" y2="548" stroke="#e2e8f0" stroke-width="1.4"/>')
    elems.append('<text x="418" y="574" text-anchor="middle" font-size="12.2" font-weight="600" fill="#374151">Worst case (flip entire Sheet 0):</text>')
    elems.append('<text x="418" y="596" text-anchor="middle" font-size="12.2" font-weight="600" fill="#374151">W = 8 sub-edges vs. V = 4 seams</text>')
    elems.append('<text x="418" y="626" text-anchor="middle" font-size="14.2" font-weight="700" fill="#047857">(W / 5) ÷ V = (8 / 5) ÷ 4 = 2 / 5</text>')
    elems.append('<text x="418" y="658" text-anchor="middle" font-size="12.2" font-weight="700" fill="#1e293b">174× smaller cell than scale-8</text>')

    # --- ROW 2 RIGHT (c): How That Exact Same Cell Rule Gives Fast O(N) Local Readout ---
    elems.append('<rect x="602" y="330" width="560" height="374" rx="12" fill="#f9fafb" stroke="#cbd5e1" stroke-width="1.6"/>')
    elems.append('<text x="882" y="357" text-anchor="middle" font-size="15.2" font-weight="700">'
                 '(c) Same 5-Edge Rule Defines Fast O(N) Local Readout</text>')

    # 3-Step Visual Pipeline inside Panel (c)
    # Step 1 box (x: 620..772)
    elems.append('<rect x="620" y="380" width="154" height="198" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.4"/>')
    elems.append('<text x="697" y="402" text-anchor="middle" font-size="12.2" font-weight="700" fill="#991b1b">1. Local Cell Error</text>')
    elems.append('<rect x="653" y="420" width="88" height="88" fill="#f8fafc" stroke="#475569" stroke-width="2.0"/>')
    elems.append('<rect x="653" y="420" width="44" height="44" fill="#fee2e2" stroke="#dc2626" stroke-width="1.8"/>')
    elems.append('<line x1="697" y1="420" x2="697" y2="464" stroke="#dc2626" stroke-width="4.5"/>')
    elems.append('<line x1="653" y1="464" x2="697" y2="464" stroke="#dc2626" stroke-width="4.5"/>')
    elems.append('<line x1="697" y1="464" x2="697" y2="508" stroke="#cbd5e1" stroke-width="1.6"/>')
    elems.append('<line x1="697" y1="464" x2="741" y2="464" stroke="#cbd5e1" stroke-width="1.6"/>')
    elems.append('<text x="697" y="534" text-anchor="middle" font-size="11.5" font-weight="600" fill="#374151">Violates interior</text>')
    elems.append('<text x="697" y="551" text-anchor="middle" font-size="11.5" font-weight="700" fill="#b91c1c">seams (V ≥ 1)</text>')

    # Arrow 1 -> 2
    elems.append('<line x1="778" y1="478" x2="798" y2="478" stroke="#475569" stroke-width="2.4"/>')
    elems.append('<polygon points="805,478 796,473 796,483" fill="#475569"/>')

    # Step 2 box (x: 808..960)
    elems.append('<rect x="808" y="380" width="152" height="198" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.4"/>')
    elems.append('<text x="884" y="402" text-anchor="middle" font-size="12.2" font-weight="700" fill="#1e40af">2. Push to Walls</text>')
    elems.append('<rect x="840" y="420" width="88" height="88" fill="#ecfdf5" stroke="#059669" stroke-width="2.2"/>')
    elems.append('<line x1="884" y1="420" x2="884" y2="508" stroke="#a7f3d0" stroke-width="2.0"/>')
    elems.append('<line x1="840" y1="464" x2="928" y2="464" stroke="#a7f3d0" stroke-width="2.0"/>')
    # Highlight <=2 sub-edges on top wall!
    elems.append('<line x1="840" y1="420" x2="875" y2="420" stroke="#2563eb" stroke-width="5.5"/>')
    elems.append('<text x="884" y="534" text-anchor="middle" font-size="11.5" font-weight="600" fill="#374151">Clear seams locally;</text>')
    elems.append('<text x="884" y="551" text-anchor="middle" font-size="11.5" font-weight="700" fill="#1e40af">at most W ≤ 2V on</text>')
    elems.append('<text x="884" y="567" text-anchor="middle" font-size="11.5" font-weight="700" fill="#1e40af">5-part outer walls</text>')

    # Arrow 2 -> 3
    elems.append('<line x1="964" y1="478" x2="984" y2="478" stroke="#475569" stroke-width="2.4"/>')
    elems.append('<polygon points="991,478 982,473 982,483" fill="#475569"/>')

    # Step 3 box (x: 994..1144)
    elems.append('<rect x="994" y="380" width="150" height="198" rx="8" fill="#ecfdf5" stroke="#a7f3d0" stroke-width="1.5"/>')
    elems.append('<text x="1069" y="402" text-anchor="middle" font-size="12.2" font-weight="700" fill="#065f46">3. 5-Edge Vote</text>')
    # Draw 5 voting boxes: [1, 0, 0, 0, 0] or [1, 1, 0, 0, 0] -> Majority = 0!
    for vi in range(5):
        vx_i = 1009 + vi * 24
        vcol = "#2563eb" if vi < 2 else "#cbd5e1"
        vtxt = "1" if vi < 2 else "0"
        elems.append(f'<rect x="{vx_i}" y="432" width="20" height="26" rx="3" fill="{vcol}"/>')
        elems.append(f'<text x="{vx_i+10}" y="450" text-anchor="middle" font-size="12.0" font-weight="700" fill="#ffffff">{vtxt}</text>')
    elems.append('<text x="1069" y="486" text-anchor="middle" font-size="13.2" font-weight="700" fill="#047857">Vote → 0 (Clean!)</text>')
    elems.append('<text x="1069" y="534" text-anchor="middle" font-size="11.5" font-weight="600" fill="#065f46">For V = 1: W ≤ 2 &lt; 5/2</text>')
    elems.append('<text x="1069" y="551" text-anchor="middle" font-size="11.5" font-weight="700" fill="#047857">erased in 1 step!</text>')

    # Bottom summary banner inside Panel (c)
    elems.append('<rect x="620" y="594" width="524" height="90" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.4"/>')
    elems.append('<text x="882" y="620" text-anchor="middle" font-size="13.0" font-weight="700" fill="#065f46">'
                 'Why Contraction Directly Powers Fast Linear-Time Readout:</text>')
    elems.append('<text x="882" y="643" text-anchor="middle" font-size="12.0" font-weight="600" fill="#374151">'
                 'Each cell decodes independently with no global cluster search; W ≤ 2V</text>')
    elems.append('<text x="882" y="663" text-anchor="middle" font-size="12.0" font-weight="600" fill="#374151">'
                 'ensures 5-edge wall voting shrinks errors × 2/5 per cycle in O(N) time.</text>')

    save_svg_png("fig_v16_2_mechanism", elems, W, H)


def render_fig3_scaling(rg, kmc, q3_bD):
    """
    Figure 3 (1180 x 810): Unified Fixed Thresholds, Quantum Scaling to 423,559 Qubits, and Fast Local Readout
      (a) Top-Left:  Live Defect Zoom-Out Inside One 322,159-Qubit Bath (Sharp Threshold at β ≈ 2.0)
      (b) Top-Right: Live Alternating (X, Z) Sawtooth Zoom-Out on 94,774 Torus Qubits (Threshold β ≈ 6.0)
      (c) Bot-Left:  Three Uniform 2x2 Quantum Levels (19 -> 2,911 -> 423,559 Qubits: 16,623x Gain)
      (d) Bot-Right: Boundaryless Torus (646 -> 94,774 Qubits): Fast Local Readout Matches Global Decoding
    """
    W, H = 1180, 824
    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<style>text { font-family: -apple-system, Helvetica, Arial, sans-serif; fill: #1f2937; }</style>',
    ]

    # (a) Live 322,159-Qubit Single-Sector Zoom-Out
    rg_series = []
    for b, col, dsh, lbl in [
        (1.35, "#dc2626", "5,4", "Warm Bath (β = 1.35: grows)"),
        (1.86, "#ea580c", "5,4", "Above Tc (β = 1.86: grows)"),
        (2.00, "#1f2937", "",    "At Threshold (β = 2.00: flat)"),
        (2.15, "#2563eb", "",    "Below Tc (β = 2.15 → 0%)"),
        (2.35, "#059669", "",    "Cold Bath (β = 2.35 → 0%)"),
    ]:
        rhos = rg[("sq_rx_iso_s5", b)]
        xs, ys = [], []
        for step_idx, s in enumerate([5, 4, 3, 2, 1]):
            if rhos[s] > 0:
                xs.append(float(step_idx)); ys.append(rhos[s])
            elif s == 1 and rhos[1] == 0.0:
                xs.append(float(step_idx)); ys.append(1.5e-4)
        rg_series.append({"xs": xs, "ys": ys, "color": col, "dash": dsh, "label": lbl})

    pa = dict(
        xlim=(-0.2, 4.25), ylim=(1.2e-4, 65.0),
        title="(a) Live Zoom-Out in One 322,159-Qubit Bath",
        xlabel="Zoom-Out Step: Fine (322k) → Coarse Cell (19q)",
        ylabel="Violated Check Fraction",
        xticks=[(0,"322k"), (1,"27k"), (2,"2.3k"), (3,"203q"), (4,"19q")],
        yticks=[(3e-4,"0.03%"), (3e-3,"0.3%"), (3e-2,"3%"), (3e-1,"30%")],
        series=rg_series, xlog=False, ylog=True, leg_pos="tl",
        ybands=[(0.048, 65.0, "#fff1f2"), (1.2e-4, 0.040, "#ecfdf5")],
        notes=[(2.65, 2.3e-4, "Drops to Exact 0% →", "#047857", "end")],
    )

    # (b) Live Alternating (X, Z) Sawtooth Zoom-Out on 94,774 Torus Qubits
    saw_series = []
    for b, col, dsh, lbl in [
        (4.50, "#dc2626", "5,4", "Warm Bath (β = 4.5: grows)"),
        (5.50, "#ea580c", "5,4", "Above Tc (β = 5.5: nonzero)"),
        (6.00, "#2563eb", "",    "At Threshold (β = 6.0 → 0%)"),
        (6.50, "#059669", "",    "Cold Bath (β = 6.5 → 0%)"),
    ]:
        rhos = rg[("rottor_iso_L2r1_s4", b)]
        xs, ys = [], []
        for step_idx, s in enumerate([4, 3, 2, 1, 0]):
            if rhos[s] > 0:
                xs.append(float(step_idx)); ys.append(rhos[s])
            elif s == 0 and rhos[0] == 0.0:
                xs.append(float(step_idx)); ys.append(2.8e-5)
        saw_series.append({"xs": xs, "ys": ys, "color": col, "dash": dsh, "label": lbl})

    pb = dict(
        xlim=(-0.2, 4.25), ylim=(2.5e-5, 25.0),
        title="(b) Alternating (X, Z) Sawtooth Flow (94,774q Torus)",
        xlabel="Zoom-Out Step (Z-Steps Wait; X-Steps Shrink × 2/5)",
        ylabel="Violated Check Fraction",
        xticks=[(0,"95k"), (1,"5.8k (Z)"), (2,"646q (X)"), (3,"38q (Z)"), (4,"Seed (X)")],
        yticks=[(1e-4,"0.01%"), (1e-3,"0.1%"), (1e-2,"1%"), (1e-1,"10%")],
        series=saw_series, xlog=False, ylog=True, leg_pos="tl",
        ybands=[(0.028, 25.0, "#fff1f2"), (2.5e-5, 0.022, "#ecfdf5")],
        notes=[(3.35, 5.2e-5, "Drops to Exact 0% →", "#047857", "end")],
    )

    # (c) Three Uniform 2x2 Quantum Levels (19 -> 2,911 -> 423,559 qubits)
    q_s1_bs = [b for b in (4.50, 5.00, 5.50, 5.75, 5.85, 6.00, 6.10, 6.25, 6.35, 6.50) if ("sq_iso_L2r1_s1_Z_err", "uf", b) in kmc]
    q_s3_bs = [b for b in (4.50, 5.00, 5.50, 5.75, 5.85, 6.00, 6.10, 6.25, 6.35, 6.50) if ("sq_iso_L2r1_s3_Z_err", "uf", b) in kmc]
    q_s5_bs = [b for b in (5.50, 5.75, 5.85, 6.00, 6.10, 6.25, 6.35, 6.50) if ("sq_iso_L2r1_s5_Z_err", "uf", b) in kmc]
    gain_q3 = kmc[("sq_iso_L2r1_s5_Z_err", "uf", 6.50)][1] / kmc[("sq_iso_L2r1_s1_Z_err", "uf", 6.50)][1]

    pc = dict(
        xlim=(4.40, 6.84), ylim=(6.0, 9.0e6),
        title="(c) Three Uniform Quantum Levels to 423,559 Qubits",
        xlabel="Inverse Bath Temperature β = 1 / T (Colder →)", ylabel="Quantum Memory Lifetime",
        xticks=[(4.5,"4.5"), (5.0,"5.0"), (5.5,"5.5"), (6.0,"6.0"), (6.5,"6.5")],
        yticks=[(1e1,"10¹"), (1e2,"10²"), (1e3,"10³"), (1e4,"10⁴"), (1e5,"10⁵"), (1e6,"10⁶")],
        series=[
            {"xs": q_s5_bs, "ys": [kmc[("sq_iso_L2r1_s5_Z_err", "uf", b)][1] for b in q_s5_bs],
             "color": "#be123c", "dash": "",    "label": f"Level 3 (423,559q: barrier = {q3_bD[2]:.1f})"},
            {"xs": q_s3_bs, "ys": [kmc[("sq_iso_L2r1_s3_Z_err", "uf", b)][1] for b in q_s3_bs],
             "color": "#059669", "dash": "",    "label": f"Level 2 (2,911q: barrier = {q3_bD[1]:.1f})"},
            {"xs": q_s1_bs, "ys": [kmc[("sq_iso_L2r1_s1_Z_err", "uf", b)][1] for b in q_s1_bs],
             "color": "#6b7280", "dash": "4,3", "label": f"Level 1 (19q: barrier = {q3_bD[0]:.1f})"},
        ],
        xlog=False, ylog=True, leg_pos="tl",
        varrows=[(6.50, kmc[("sq_iso_L2r1_s1_Z_err", "uf", 6.50)][1],
                  kmc[("sq_iso_L2r1_s5_Z_err", "uf", 6.50)][1],
                  f"{gain_q3:,.0f}×", "#be123c", "r", 3.5e3)],
    )

    # (d) Boundaryless Torus (94,774q): Fast Local Readout Matches Global Decoding
    tor_bs = [5.50, 6.00, 6.25, 6.50, 6.65]
    tor_gain = kmc[("rottor_iso_L2r1_s4_X_err", "uf", 6.65)][1] / kmc[("rottor_iso_L2r1_s2_X_err", "uf", 6.65)][1]

    pd = dict(
        xlim=(5.40, 6.92), ylim=(5.0e2, 4.0e7),
        title="(d) Torus (94,774q): Local Readout Matches Global",
        xlabel="Inverse Bath Temperature β = 1 / T (Colder →)", ylabel="Quantum Memory Lifetime",
        xticks=[(5.5,"5.5"), (6.0,"6.0"), (6.25,"6.25"), (6.5,"6.5"), (6.75,"6.75")],
        yticks=[(1e3,"10³"), (1e4,"10⁴"), (1e5,"10⁵"), (1e6,"10⁶"), (1e7,"10⁷")],
        series=[
            {"xs": tor_bs, "ys": [kmc[("rottor_iso_L2r1_s4_X_err", "uf", b)][1] for b in tor_bs],
             "color": "#be123c", "dash": "",    "label": f"Cycle 2 (94,774q): Global Decoder ({tor_gain:.0f}×)"},
            {"xs": tor_bs, "ys": [kmc[("rottor_iso_L2r1_s4_X_err", "rg_dec", b)][1] for b in tor_bs],
             "color": "#ea580c", "dash": "6,3", "hollow": True, "label": "Cycle 2 (94,774q): Fast Local Readout"},
            {"xs": tor_bs, "ys": [kmc[("rottor_iso_L2r1_s2_X_err", "uf", b)][1] for b in tor_bs],
             "color": "#2563eb", "dash": "",    "label": "Cycle 1 (646q): Global Decoder"},
            {"xs": tor_bs, "ys": [kmc[("rottor_iso_L2r1_s2_X_err", "rg_dec", b)][1] for b in tor_bs],
             "color": "#059669", "dash": "4,3", "hollow": True, "label": "Cycle 1 (646q): Fast Local Readout"},
        ],
        xlog=False, ylog=True, leg_pos="tl",
        varrows=[(6.65, kmc[("rottor_iso_L2r1_s2_X_err", "uf", 6.65)][1],
                  kmc[("rottor_iso_L2r1_s4_X_err", "uf", 6.65)][1],
                  f"{tor_gain:.0f}×", "#be123c", "r", 1.1e6)],
    )

    draw_panel(elems, 86,   42, 472, 322, **pa)
    draw_panel(elems, 676,  42, 472, 322, **pb)
    draw_panel(elems, 86,  456, 472, 316, **pc)
    draw_panel(elems, 676, 456, 472, 316, **pd)
    save_svg_png("fig_v16_3_scaling", elems, W, H)


def render_fig4_geometry(kmc12, v7_best):
    """
    Figure 4 (1180 x 824): Bounded-Degree Graphs vs. 3D Euclidean Space (100% Uniform J = 1!)
      (a) Top-Left (Visual):  3D Volume-Packing Cutoff (2x2 overflows 12 > 8 cubes; 3x3 fits 24 < 27 cubes)
      (b) Top-Right (Data):   Minimal 3D-Compatible 3x3 Single-Cross Scaling (41 -> 935 -> 22,097q: 68x)
      (c) Bot-Left (Visual):  Direct Bounded-Degree Link vs. Single-Grid Passive Wire (ΔE = 0 walk entropy)
      (d) Bot-Right (Data):   Uniform-Coupling Wire-Stretching Cost on Cubic Grids (1,480q vs 10,280q vs 56,312q)
    """
    W, H = 1180, 824
    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<style>text { font-family: -apple-system, Helvetica, Arial, sans-serif; fill: #1f2937; }</style>',
    ]

    # --- Panel (a): Visual 3D Euclidean Volume-Packing Cutoff ---
    x0, y0, pw, ph = 42, 42, 516, 322
    elems.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" rx="10" fill="#f8fafc" stroke="#374151" stroke-width="1.5"/>')
    elems.append(f'<text x="{x0+pw/2:.1f}" y="{y0-13:.1f}" text-anchor="middle" font-size="14.8" font-weight="700">'
                 '(a) Three-Dimensional Euclidean Volume-Packing Cutoff</text>')

    elems.append(f'<rect x="{x0+18}" y="{y0+18}" width="230" height="212" rx="9" fill="#fff1f2" stroke="#fecdd3" stroke-width="1.5"/>')
    elems.append(f'<text x="{x0+133}" y="{y0+40}" text-anchor="middle" font-size="13.0" font-weight="700" fill="#9f1239">2×2 Cross: Overflows 3D</text>')
    gx, gy, gs = x0 + 81, y0 + 56, 102
    elems.append(f'<rect x="{gx}" y="{gy}" width="{gs}" height="{gs}" fill="#ffffff" stroke="#be123c" stroke-width="2.4"/>')
    elems.append(f'<line x1="{gx+51}" y1="{gy}" x2="{gx+51}" y2="{gy+gs}" stroke="#dc2626" stroke-width="5.0"/>')
    elems.append(f'<line x1="{gx}" y1="{gy+51}" x2="{gx+gs}" y2="{gy+51}" stroke="#dc2626" stroke-width="5.0"/>')
    elems.append(f'<text x="{x0+133}" y="{y0+180}" text-anchor="middle" font-size="12.2" font-weight="700" fill="#9f1239">12 Squares &gt; 2³ = 8 Cubes</text>')
    elems.append(f'<text x="{x0+133}" y="{y0+199}" text-anchor="middle" font-size="11.6" font-weight="600" fill="#be123c">dim = log₂ 12 ≈ 3.58 &gt; 3</text>')
    elems.append(f'<text x="{x0+133}" y="{y0+218}" text-anchor="middle" font-size="11.5" font-weight="700" fill="#881337">Wires must stretch in R³</text>')

    elems.append(f'<rect x="{x0+268}" y="{y0+18}" width="230" height="212" rx="9" fill="#ecfdf5" stroke="#a7f3d0" stroke-width="1.5"/>')
    elems.append(f'<text x="{x0+383}" y="{y0+40}" text-anchor="middle" font-size="13.0" font-weight="700" fill="#065f46">3×3 Cross: Fits in 3D Cube</text>')
    hx, hy, hs = x0 + 331, y0 + 56, 102
    elems.append(f'<rect x="{hx}" y="{hy}" width="{hs}" height="{hs}" fill="#ffffff" stroke="#059669" stroke-width="2.4"/>')
    for k in (1, 2):
        lw = "4.8" if k == 1 else "1.4"
        lc = "#059669" if k == 1 else "#94a3b8"
        elems.append(f'<line x1="{hx+k*hs/3:.1f}" y1="{hy}" x2="{hx+k*hs/3:.1f}" y2="{hy+hs}" stroke="{lc}" stroke-width="{lw}"/>')
        elems.append(f'<line x1="{hx}" y1="{hy+k*hs/3:.1f}" x2="{hx+hs}" y2="{hy+k*hs/3:.1f}" stroke="{lc}" stroke-width="{lw}"/>')
    elems.append(f'<text x="{x0+383}" y="{y0+180}" text-anchor="middle" font-size="12.2" font-weight="700" fill="#065f46">24 Squares &lt; 3³ = 27 Cubes</text>')
    elems.append(f'<text x="{x0+383}" y="{y0+199}" text-anchor="middle" font-size="11.6" font-weight="600" fill="#047857">dim = log₃ 24 ≈ 2.89 &lt; 3</text>')
    elems.append(f'<text x="{x0+383}" y="{y0+218}" text-anchor="middle" font-size="11.5" font-weight="700" fill="#064e3b">7-part walls shrink × 2 / 7</text>')

    elems.append(f'<rect x="{x0+18}" y="{y0+242}" width="480" height="66" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.3"/>')
    elems.append(f'<text x="{x0+pw/2:.1f}" y="{y0+268}" text-anchor="middle" font-size="13.0" font-weight="700" fill="#1e293b">'
                 'Packing Cutoff:  2L² + 2L ≤ L³  ⇔  L ≥ 1 + √3 ≈ 2.73</text>')
    elems.append(f'<text x="{x0+pw/2:.1f}" y="{y0+292}" text-anchor="middle" font-size="12.0" font-weight="600" fill="#475569">'
                 '3×3 single-cross is the smallest integer cell whose volume fits in R³</text>')

    # --- Panel (b): Minimal 3D-Compatible 3x3 Single-Cross Scaling ---
    l3_series = []
    for s, col, dsh, lbl_base in [
        (3, "#059669", "",    "Scale 3 (22,097q"),
        (2, "#2563eb", "",    "Scale 2 (935q"),
        (1, "#6b7280", "4,3", "Scale 1 (41q"),
    ]:
        bs = [b for b in (1.40, 1.60, 1.75, 1.85, 1.95, 2.05, 2.15, 2.35, 2.55, 2.75, 2.95, 3.15, 3.35, 3.55)
              if (f"sq_rx_L3r1_s{s}_X_err", "uniform", "uf", b) in kmc12]
        ys = [kmc12[(f"sq_rx_L3r1_s{s}_X_err", "uniform", "uf", b)][1] for b in bs]
        sl = ols_slope(bs[-3:], ys[-3:])
        l3_series.append({"xs": bs, "ys": ys, "color": col, "dash": dsh, "label": f"{lbl_base}: barrier = {sl:.1f})"})

    y_s1_top = kmc12[("sq_rx_L3r1_s1_X_err", "uniform", "uf", 3.55)][1]
    y_s3_top = kmc12[("sq_rx_L3r1_s3_X_err", "uniform", "uf", 3.55)][1]
    l3_gain = y_s3_top / y_s1_top

    pb = dict(
        xlim=(1.32, 3.82), ylim=(1.2, 7.5e4),
        title="(b) 3D-Compatible 3×3 Single-Cross Scaling",
        xlabel="Inverse Bath Temperature β = 1 / T (Colder →)",
        ylabel="Mean Memory Lifetime",
        xticks=[(1.5,"1.5"), (2.0,"2.0"), (2.5,"2.5"), (3.0,"3.0"), (3.5,"3.5")],
        yticks=[(1e1,"10¹"), (1e2,"10²"), (1e3,"10³"), (1e4,"10⁴")],
        series=l3_series, xlog=False, ylog=True, leg_pos="tl",
        xbands=[(3.18, 3.82, "#ecfdf5")],
        varrows=[(3.55, y_s1_top, y_s3_top, f"{l3_gain:.0f}×", "#059669", "r", 1.8e3)],
    )
    draw_panel(elems, 676, 42, 472, 322, **pb)

    # --- Panel (c): Visual Direct Bounded-Degree Link vs Single-Grid Passive Wire (Uniform J = 1) ---
    cx0, cy0, cpw, cph = 42, 456, 516, 316
    elems.append(f'<rect x="{cx0}" y="{cy0}" width="{cpw}" height="{cph}" rx="10" fill="#f8fafc" stroke="#374151" stroke-width="1.5"/>')
    elems.append(f'<text x="{cx0+cpw/2:.1f}" y="{cy0-13:.1f}" text-anchor="middle" font-size="14.8" font-weight="700">'
                 '(c) Why Stretching Edges Onto One Grid Adds Wire Entropy</text>')

    # Top strip: Direct Bounded-Degree Connectivity
    elems.append(f'<rect x="{cx0+18}" y="{cy0+18}" width="480" height="128" rx="8" fill="#ecfdf5" stroke="#a7f3d0" stroke-width="1.4"/>')
    elems.append(f'<text x="{cx0+258}" y="{cy0+42}" text-anchor="middle" font-size="13.2" font-weight="700" fill="#065f46">'
                 'Direct Bounded-Degree Graph (1,480 Qubits)</text>')
    jx1, jx2, jy = cx0 + 168, cx0 + 348, cy0 + 88
    elems.append(f'<rect x="{jx1-34}" y="{jy-26}" width="34" height="52" rx="6" fill="#fde68a" stroke="#b45309" stroke-width="2.0"/>')
    elems.append(f'<rect x="{jx2}" y="{jy-26}" width="34" height="52" rx="6" fill="#fde68a" stroke="#b45309" stroke-width="2.0"/>')
    elems.append(f'<line x1="{jx1}" y1="{jy}" x2="{jx2}" y2="{jy}" stroke="#047857" stroke-width="4.2"/>')
    elems.append(f'<circle cx="{(jx1+jx2)/2:.1f}" cy="{jy}" r="8.0" fill="#059669" stroke="#ffffff" stroke-width="1.6"/>')
    elems.append(f'<text x="{cx0+258}" y="{cy0+132}" text-anchor="middle" font-size="12.0" font-weight="700" fill="#065f46">'
                 '1 qubit edge between 3-way junctions → Every flip costs +ΔE</text>')

    # Bottom strip: Scale-8 Single Cubic Grid (Passive Wire Walk at ΔE = 0)
    elems.append(f'<rect x="{cx0+18}" y="{cy0+162}" width="480" height="136" rx="8" fill="#fff1f2" stroke="#fecdd3" stroke-width="1.4"/>')
    elems.append(f'<text x="{cx0+258}" y="{cy0+186}" text-anchor="middle" font-size="13.2" font-weight="700" fill="#9f1239">'
                 'Scale-8 Single Cubic Grid (56,312 Qubits)</text>')
    wx1, wx2, wy = cx0 + 82, cx0 + 434, cy0 + 244
    elems.append(f'<rect x="{wx1-30}" y="{wy-26}" width="30" height="52" rx="6" fill="#fde68a" stroke="#b45309" stroke-width="2.0"/>')
    elems.append(f'<rect x="{wx2}" y="{wy-26}" width="30" height="52" rx="6" fill="#fde68a" stroke="#b45309" stroke-width="2.0"/>')
    elems.append(f'<line x1="{wx1}" y1="{wy}" x2="{wx2}" y2="{wy}" stroke="#374151" stroke-width="3.2"/>')
    for qi in range(7):
        qx = wx1 + 31 + qi * 48.0
        col = "#dc2626" if qi == 3 else "#2563eb"
        rad = 8.0 if qi == 3 else 5.5
        elems.append(f'<circle cx="{qx:.1f}" cy="{wy}" r="{rad}" fill="{col}" stroke="#ffffff" stroke-width="1.4"/>')
    elems.append(f'<line x1="{wx1+66}" y1="{wy-24}" x2="{wx2-66}" y2="{wy-24}" stroke="#dc2626" stroke-width="2.3"/>')
    elems.append(f'<polygon points="{wx1+56},{wy-24} {wx1+66},{wy-29} {wx1+66},{wy-19}" fill="#dc2626"/>')
    elems.append(f'<polygon points="{wx2-56},{wy-24} {wx2-66},{wy-29} {wx2-66},{wy-19}" fill="#dc2626"/>')
    elems.append(f'<text x="{cx0+258}" y="{wy-32}" text-anchor="middle" font-size="11.6" font-weight="700" fill="#b91c1c">'
                 'Defect slides freely along degree-2 wire (ΔE = 0)</text>')
    z3_bs = [5.50, 6.00, 6.50, 7.00]
    ys_comp = [v7_best("sq_L3r2_s2_X_err", "uniform", "uf", b) for b in z3_bs]
    ys_46   = [kmc12[("sq_z3_4_6_r2_X_err", "uniform", "uf", b)][1] for b in z3_bs]
    ys_88   = [kmc12[("sq_bdl_8_8_r2_X_err", "uniform", "uf", b)][1] for b in z3_bs]
    loss_65 = ys_comp[2] / ys_88[2]

    elems.append(f'<text x="{cx0+258}" y="{cy0+286}" text-anchor="middle" font-size="12.0" font-weight="700" fill="#9f1239">'
                 f'Wire entropy (+T·ΔS) halves barrier &amp; cuts lifetime {loss_65:.0f}×</text>')

    # --- Panel (d): Measured Wire-Stretching Cost Under Uniform Couplings (J = 1) ---

    pd = dict(
        xlim=(5.38, 7.22), ylim=(60.0, 4.5e6),
        title="(d) Measured Wire-Stretching Penalty (Uniform J = 1)",
        xlabel="Inverse Bath Temperature β = 1 / T (Colder →)",
        ylabel="Quantum Memory Lifetime",
        xticks=[(5.5,"5.5"), (6.0,"6.0"), (6.5,"6.5"), (7.0,"7.0")],
        yticks=[(1e2,"10²"), (1e3,"10³"), (1e4,"10⁴"), (1e5,"10⁵"), (1e6,"10⁶")],
        series=[
            {"xs": z3_bs, "ys": ys_comp, "color": "#059669", "dash": "", "label": "Direct Bounded-Degree Graph (1,480q)"},
            {"xs": z3_bs, "ys": ys_46,   "color": "#2563eb", "dash": "", "label": "Tighter (4,6) Cubic Grid (10,280q: 13×)"},
            {"xs": z3_bs, "ys": ys_88,   "color": "#dc2626", "dash": "", "label": "Scale-8 (8,8) Cubic Grid (56,312q)"},
        ],
        xlog=False, ylog=True, leg_pos="tl",
        varrows=[(6.50, ys_88[2], ys_comp[2], f"{loss_65:.0f}× loss", "#b91c1c", "r", 2.8e4)],
    )
    draw_panel(elems, 676, 456, 472, 316, **pd)
    save_svg_png("fig_v16_4_geometry", elems, W, H)


def render_fig5_cooling(cool7, cool11):
    """
    Figure 5: Autonomous Storage Self-Healing (Uniform 2x2 Torus Only!) vs 4-Scale Hot-State Cooling
    """
    ps = [0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.12]
    px_pct = [p * 100.0 for p in ps]
    s_heal = []
    for tag, col, dsh, lbl in [
        ("rottor_L2r1_s2_Z_err", "#2563eb", "6,4", "Torus Z-Errors (98% healed at 1% noise)"),
        ("rottor_L2r1_s2_X_err", "#059669", "",    "Torus X-Errors (76% healed at 1% noise)"),
    ]:
        ys = [100.0 * cool7[(tag, "quench", f"noisy_{p:.2f}", 1000.0)]["pheal"] for p in ps]
        s_heal.append({"xs": px_pct, "ys": ys, "color": col, "dash": dsh, "label": lbl})

    def family_tcool(fam, proto, scales=(1, 2, 3, 4), thresh=0.0030):
        ns, ts_out = [], []
        for s in scales:
            tag = f"{fam}_s{s}_X_err"
            t_list = sorted(tt for (tg, pr, im, tt) in cool11 if tg == tag and pr == proto and im == "hot")
            N = cool11[(tag, proto, "hot", t_list[0])]["N"]
            syns = [cool11[(tag, proto, "hot", tt)]["syn"] for tt in t_list]
            tc = interp_tcool(t_list, syns, thresh, decreasing=True)
            ns.append(N)
            ts_out.append(tc if tc is not None else 1.5e5)
        return ns, ts_out

    rx_an_n, rx_an_t = family_tcool("sq_rx_iso", "anneal", (1, 2, 3, 4))
    rx_qu_n, rx_qu_t = family_tcool("sq_rx_iso", "quench", (1, 2, 3, 4))
    sq_an_n, sq_an_t = family_tcool("sq_iso_L2r1", "anneal", (1, 2, 3, 4))
    sq_qu_n, sq_qu_t = family_tcool("sq_iso_L2r1", "quench", (1, 2, 3))

    s_cool_scale = [
        {"xs": rx_qu_n, "ys": rx_qu_t, "color": "#dc2626", "dash": "5,4", "hollow": True,
         "label": "Nested X-Welded: Sudden Quench (Pinned)"},
        {"xs": rx_an_n, "ys": rx_an_t, "color": "#be123c", "dash": "",
         "label": "Nested X-Welded: Gradual Cooling (~ √N)"},
        {"xs": sq_qu_n, "ys": sq_qu_t, "color": "#f97316", "dash": "4,3", "hollow": True,
         "label": "Full Quantum: Sudden Quench (10× slower)"},
        {"xs": sq_an_n, "ys": sq_an_t, "color": "#059669", "dash": "",
         "label": "Full Quantum: Gradual Cooling (to 47,387q)"},
    ]

    render_double(
        "fig_v16_5_cooling",
        dict(xlim=(0.5, 12.5), ylim=(-2.0, 110.0),
             title="(a) Small Storage Errors Heal Without a Decoder",
             xlabel="Initial Physical Error Rate (%)", ylabel="Exact Original Codeword Restored (%)",
             xticks=[(1,"1%"), (3,"3%"), (6,"6%"), (9,"9%"), (12,"12%")],
             yticks=[(0,"0%"), (25,"25%"), (50,"50%"), (75,"75%"), (100,"100%")],
             series=s_heal, xlog=False, ylog=False, leg_pos="tr",
             hlines=[(25.0, "#6b7280", "Random-Guess Floor (25%)", "bl")]),
        dict(xlim=(14.0, 90000.0), ylim=(110.0, 1.1e7),
             title="(b) Hot-State Cooling Time Across Four Scales",
             xlabel="Physical Qubit Count N (Log Scale)", ylabel="Time to Clear Defects (< 0.3%)",
             xticks=[(20,"20"), (200,"200"), (2000,"2k"), (20000,"20k"), (60000,"60k")],
             yticks=[(2e2,"200"), (1e3,"10³"), (1e4,"10⁴"), (1e5,"10⁵")],
             series=s_cool_scale, xlog=True, ylog=True, leg_pos="tl",
             ybands=[(1.02e5, 3.1e5, "#fff1f2")],
             notes=[(80000.0, 2.25e5, "Pinned on 3-Way Seams (> 10⁵)", "#b91c1c", "end")]),
    )


def main():
    kmc4, kmc6, kmc7, ffs7, cool7, kmc9, rg9, cool11 = load_all()
    kmc12 = load_v12()

    def v7_best(tag, jm, dec, b, min_reps=3):
        if (tag, jm, dec, b) in kmc7 and kmc7[(tag, jm, dec, b)][2] == 0:
            return kmc7[(tag, jm, dec, b)][1]
        if (tag, jm, dec, b) in ffs7 and ffs7[(tag, jm, dec, b)][2] >= min_reps:
            return ffs7[(tag, jm, dec, b)][1]
        return None

    q3_bD = render_fig1_landscape(kmc4, kmc6, kmc9)
    render_fig2_mechanism()
    render_fig3_scaling(rg9, kmc9, q3_bD)
    render_fig4_geometry(kmc12, v7_best)
    render_fig5_cooling(cool7, cool11)
    print("All 5 V16 figures rendered cleanly!")


if __name__ == "__main__":
    main()
