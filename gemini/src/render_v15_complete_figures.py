#!/usr/bin/env python3
"""
render_v15_complete_figures.py
Renders all 8 large, intuitive, visually self-explanatory publication figures (V15):

  1. fig_v15_1_classes.png:         Code family comparison across 5 decades of N (lifetime & barrier scaling)
  2. fig_v15_2_validation.png:      Thermal bath validation on published 3D Toric & Random Layer Code benchmarks
  3. fig_v15_3_architecture.png:    2-row visual welding construction + 2/5 contraction + 3-arm open hole
  4. fig_v15_4_threshold_rg.png:    Visual meaning of local zoom-out + 322k live zoom-out + 5-scale threshold
  5. fig_v15_5_quantum.png:         Alternating (X, Z) cycle + 95k sawtooth + 424k levels + fast local readout
  6. fig_v15_6_euclidean_wires.png: 2-row 3D Euclidean volume-packing cutoff (L=2 vs L=3) + L=3 scaling
                                    + passive wire walk vs graded wire hill + Z^3 cubic-grid 80x recovery
  7. fig_v15_7_compact_highrate.png:2-row mixed-step no-go & sector-balanced 586q/1,172q beating Layer Codes
                                    + hyperbolic {4,5}/{4,6} square tilings + K=6 & K=10 welded memories
  8. fig_v15_8_cooling.png:         Autonomous storage self-healing vs 4-scale polynomial hot-state cooling
"""

import math
import os
import re
import sys

WS = "/Users/harshilavlani/Downloads/6.S980_gemini"
FIG_DIR = os.path.join(WS, "figures")
ART_DIR = "/Users/harshilavlani/.gemini/antigravity/brain/35299b0d-7439-4f8e-8617-0b646151a931"

sys.path.insert(0, os.path.join(WS, "src"))
from render_v13_unified_figures import draw_panel, interp_tcool, load_all, ols_slope, save_svg_png
from render_v14_explanatory_figures import draw_mini_grid


def load_v12():
    """Loads uncapped, zero-bail KMC records from cluster_v12_results.txt."""
    p = os.path.join(WS, "cluster_v12_results.txt")
    kmc12 = {}
    pat = re.compile(
        r"KMC dec=(\S+) beta=([\d.]+) N=(\d+) K=(\d+) trials=(\d+) "
        r"mean_t=([\deE+.-]+) sem_t=([\deE+.-]+) median_t=([\deE+.-]+) "
        r"mean_flips=([\deE+.-]+) bails=(\d+) capped=(\d+)"
    )
    if not os.path.exists(p):
        return kmc12
    with open(p) as f:
        for raw in f:
            line = raw.strip()
            if "|" not in line:
                continue
            tid, rhs = [s.strip() for s in line.split("|", 1)]
            m = pat.search(rhs)
            if not m:
                continue
            dec, b, nq, k, tr, mt, st, med, mfl, bl, cap = m.groups()
            if int(bl) > 0 or int(cap) > 0:
                continue
            parts = tid.split("__")
            if len(parts) >= 5:
                _, code_sec, jsched, _, _ = parts
                kmc12[(code_sec, jsched, dec, round(float(b), 2))] = (int(nq), float(mt), float(st), int(k))
    return kmc12


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


def render_fig3_architecture():
    W, H = 1180, 650
    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<style>text { font-family: -apple-system, Helvetica, Arial, sans-serif; fill: #1f2937; }</style>',
    ]

    elems.append('<rect x="18" y="14" width="1144" height="278" rx="12" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.6"/>')
    elems.append('<text x="590" y="42" text-anchor="middle" font-size="16.0" font-weight="700">'
                 '(a) How One Three-Dimensional Welding Step Multiplies the Energy Barrier</text>')

    # Stage 1
    elems.append('<rect x="38" y="58" width="236" height="216" rx="9" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.3"/>')
    elems.append('<text x="156" y="80" text-anchor="middle" font-size="13.4" font-weight="700" fill="#1e3a8a">1. Parent 2D Square</text>')
    elems.append('<rect x="96" y="98" width="120" height="120" fill="#dbeafe" stroke="#2563eb" stroke-width="2.8"/>')
    for qx, qy in [(156, 98), (156, 218), (96, 158), (216, 158)]:
        elems.append(f'<circle cx="{qx}" cy="{qy}" r="5.5" fill="#1d4ed8" stroke="#ffffff" stroke-width="1.4"/>')
    elems.append('<text x="156" y="163" text-anchor="middle" font-size="12.5" font-weight="700" fill="#1e40af">Z-Check Face</text>')
    elems.append('<text x="156" y="250" text-anchor="middle" font-size="12.0" font-weight="600" fill="#475569">Qubits sit on edges;</text>')
    elems.append('<text x="156" y="266" text-anchor="middle" font-size="12.0" font-weight="600" fill="#b91c1c">1 defect walks free (ΔE = 0)</text>')

    elems.append('<line x1="280" y1="166" x2="306" y2="166" stroke="#475569" stroke-width="2.6"/>')
    elems.append('<polygon points="314,166 304,160 304,172" fill="#475569"/>')

    # Stage 2
    elems.append('<rect x="320" y="58" width="256" height="216" rx="9" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.3"/>')
    elems.append('<text x="448" y="80" text-anchor="middle" font-size="13.4" font-weight="700" fill="#1e3a8a">2. Subdivide 2×2 &amp; Duplicate</text>')
    elems.append('<polygon points="350,218 430,182 546,182 466,218" fill="#dbeafe" fill-opacity="0.88" stroke="#2563eb" stroke-width="2.0"/>')
    elems.append('<line x1="390" y1="200" x2="506" y2="200" stroke="#60a5fa" stroke-width="1.5"/>')
    elems.append('<line x1="408" y1="218" x2="488" y2="182" stroke="#60a5fa" stroke-width="1.5"/>')
    elems.append('<text x="538" y="214" text-anchor="end" font-size="11.6" font-weight="700" fill="#1e40af">Sheet 1</text>')
    elems.append('<polygon points="350,146 430,110 546,110 466,146" fill="#dbeafe" fill-opacity="0.90" stroke="#2563eb" stroke-width="2.0"/>')
    elems.append('<line x1="390" y1="128" x2="506" y2="128" stroke="#60a5fa" stroke-width="1.5"/>')
    elems.append('<line x1="408" y1="146" x2="488" y2="110" stroke="#60a5fa" stroke-width="1.5"/>')
    elems.append('<text x="358" y="106" font-size="11.6" font-weight="700" fill="#1e40af">Sheet 0</text>')
    elems.append('<text x="448" y="250" text-anchor="middle" font-size="12.0" font-weight="600" fill="#475569">Two parallel 2×2 copies</text>')
    elems.append('<text x="448" y="266" text-anchor="middle" font-size="12.0" font-weight="600" fill="#475569">ready to be linked</text>')

    elems.append('<line x1="582" y1="166" x2="608" y2="166" stroke="#475569" stroke-width="2.6"/>')
    elems.append('<polygon points="616,166 606,160 606,172" fill="#475569"/>')

    # Stage 3
    elems.append('<rect x="622" y="58" width="256" height="216" rx="9" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.3"/>')
    elems.append('<text x="750" y="80" text-anchor="middle" font-size="13.4" font-weight="700" fill="#b45309">3. Weld 4-Arm Bridge Cross</text>')
    elems.append('<polygon points="652,218 732,182 848,182 768,218" fill="#dbeafe" fill-opacity="0.72" stroke="#2563eb" stroke-width="1.8"/>')
    elems.append('<polygon points="710,218 790,182 790,110 710,146" fill="#fde68a" fill-opacity="0.92" stroke="#d97706" stroke-width="2.2"/>')
    elems.append('<polygon points="692,200 808,200 808,128 692,128" fill="#fcd34d" fill-opacity="0.82" stroke="#d97706" stroke-width="2.2"/>')
    elems.append('<polygon points="652,146 732,110 848,110 768,146" fill="#dbeafe" fill-opacity="0.78" stroke="#2563eb" stroke-width="1.8"/>')
    elems.append('<line x1="692" y1="128" x2="808" y2="128" stroke="#dc2626" stroke-width="3.8"/>')
    elems.append('<line x1="710" y1="146" x2="790" y2="110" stroke="#dc2626" stroke-width="3.8"/>')
    elems.append('<text x="750" y="250" text-anchor="middle" font-size="12.0" font-weight="700" fill="#b91c1c">Red lines = 3-way seams</text>')
    elems.append('<text x="750" y="266" text-anchor="middle" font-size="12.0" font-weight="600" fill="#475569">(3 faces meet at 1 edge)</text>')

    elems.append('<line x1="884" y1="166" x2="910" y2="166" stroke="#475569" stroke-width="2.6"/>')
    elems.append('<polygon points="918,166 908,160 908,172" fill="#475569"/>')

    # Stage 4
    elems.append('<rect x="924" y="58" width="220" height="216" rx="9" fill="#ecfdf5" stroke="#a7f3d0" stroke-width="1.4"/>')
    elems.append('<text x="1034" y="80" text-anchor="middle" font-size="13.4" font-weight="700" fill="#065f46">4. Seam Splits 1 → 2</text>')
    elems.append('<line x1="952" y1="138" x2="1034" y2="138" stroke="#2563eb" stroke-width="5.0"/>')
    elems.append('<line x1="1034" y1="138" x2="1116" y2="138" stroke="#2563eb" stroke-width="5.0"/>')
    elems.append('<line x1="1034" y1="138" x2="1034" y2="216" stroke="#d97706" stroke-width="5.0"/>')
    elems.append('<circle cx="1034" cy="138" r="8.5" fill="#1e293b" stroke="#ffffff" stroke-width="2.0"/>')
    elems.append('<circle cx="978" cy="122" r="9.0" fill="#dc2626" stroke="#ffffff" stroke-width="1.6"/>')
    elems.append('<circle cx="1090" cy="122" r="9.0" fill="#dc2626" stroke="#ffffff" stroke-width="1.6"/>')
    elems.append('<circle cx="1054" cy="192" r="9.0" fill="#dc2626" stroke="#ffffff" stroke-width="1.6"/>')
    elems.append('<path d="M 990,120 Q 1034,102 1078,120" fill="none" stroke="#991b1b" stroke-width="2.2" stroke-dasharray="4,3"/>')
    elems.append('<path d="M 988,128 Q 1020,164 1044,186" fill="none" stroke="#991b1b" stroke-width="2.2" stroke-dasharray="4,3"/>')
    elems.append('<text x="1034" y="250" text-anchor="middle" font-size="12.2" font-weight="700" fill="#065f46">Crossing costs +ΔE energy</text>')
    elems.append('<text x="1034" y="266" text-anchor="middle" font-size="12.0" font-weight="600" fill="#047857">→ Multiplies barrier each scale</text>')

    # Row 2 Left (b)
    elems.append('<rect x="18" y="308" width="560" height="326" rx="12" fill="#f9fafb" stroke="#cbd5e1" stroke-width="1.6"/>')
    elems.append('<text x="298" y="336" text-anchor="middle" font-size="15.5" font-weight="700">'
                 '(b) Why a 2×2 Four-Arm Cross Shrinks Errors by 2 / 5</text>')
    bx0, by0, bsz = 64, 368, 214
    elems.append(f'<rect x="{bx0}" y="{by0}" width="{bsz}" height="{bsz}" fill="#dbeafe" stroke="#1d4ed8" stroke-width="3.0"/>')
    for qx, qy in [(bx0+53, by0+58), (bx0+160, by0+58), (bx0+53, by0+165), (bx0+160, by0+165)]:
        elems.append(f'<text x="{qx}" y="{qy}" text-anchor="middle" font-size="12.8" font-weight="700" fill="#1e40af">Sheet 0</text>')
        elems.append(f'<text x="{qx}" y="{qy+16}" text-anchor="middle" font-size="11.2" font-weight="600" fill="#2563eb">(flipped)</text>')
    elems.append(f'<line x1="{bx0+107}" y1="{by0}" x2="{bx0+107}" y2="{by0+bsz}" stroke="#dc2626" stroke-width="6.5"/>')
    elems.append(f'<line x1="{bx0}" y1="{by0+107}" x2="{bx0+bsz}" y2="{by0+107}" stroke="#dc2626" stroke-width="6.5"/>')
    for wx, wy in [(bx0+107, by0), (bx0+107, by0+bsz), (bx0, by0+107), (bx0+bsz, by0+107)]:
        elems.append(f'<circle cx="{wx}" cy="{wy}" r="7.5" fill="#d97706" stroke="#ffffff" stroke-width="1.6"/>')
    elems.append(f'<circle cx="{bx0+107}" cy="{by0+107}" r="8.0" fill="#1e3a8a" stroke="#ffffff" stroke-width="1.6"/>')
    elems.append(f'<text x="{bx0+107}" y="{by0-10}" text-anchor="middle" font-size="12.2" font-weight="700" fill="#1e3a8a">5 Sub-Edges per Outer Wall</text>')
    elems.append(f'<text x="{bx0+107}" y="{by0+bsz+22}" text-anchor="middle" font-size="12.4" font-weight="700" fill="#b91c1c">4 Violated Interior Seams (Red)</text>')

    elems.append('<rect x="304" y="368" width="254" height="214" rx="9" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.4"/>')
    elems.append('<text x="431" y="398" text-anchor="middle" font-size="13.2" font-weight="700" fill="#1e293b">Each Outer Wall Has 5 Parts:</text>')
    elems.append('<text x="431" y="420" text-anchor="middle" font-size="12.5" font-weight="600" fill="#2563eb">2 sub-edges on Sheet 0</text>')
    elems.append('<text x="431" y="440" text-anchor="middle" font-size="12.5" font-weight="600" fill="#475569">2 sub-edges on Sheet 1</text>')
    elems.append('<text x="431" y="460" text-anchor="middle" font-size="12.5" font-weight="600" fill="#b45309">1 sub-edge on Bridge</text>')
    elems.append('<line x1="324" y1="478" x2="538" y2="478" stroke="#e2e8f0" stroke-width="1.4"/>')
    elems.append('<text x="431" y="502" text-anchor="middle" font-size="12.6" font-weight="600" fill="#374151">Flipping Sheet 0 = 2/5 on 4 walls</text>')
    elems.append('<text x="431" y="528" text-anchor="middle" font-size="14.0" font-weight="700" fill="#047857">Coarse / Fine = (8/5) ÷ 4 = 2/5</text>')
    elems.append('<text x="431" y="558" text-anchor="middle" font-size="12.5" font-weight="700" fill="#1e293b">174× smaller cell than scale-8</text>')

    # Row 2 Right (c)
    elems.append('<rect x="602" y="308" width="560" height="326" rx="12" fill="#f9fafb" stroke="#cbd5e1" stroke-width="1.6"/>')
    elems.append('<text x="882" y="336" text-anchor="middle" font-size="15.5" font-weight="700">'
                 '(c) Why Cutting One Bridge Arm Destroys Protection</text>')
    cx0, cy0, csz = 648, 368, 214
    elems.append(f'<rect x="{cx0}" y="{cy0}" width="{csz}" height="{csz}" fill="#fef2f2" stroke="#b91c1c" stroke-width="3.0"/>')
    elems.append(f'<line x1="{cx0}" y1="{cy0+107}" x2="{cx0+csz}" y2="{cy0+107}" stroke="#2563eb" stroke-width="6.5"/>')
    elems.append(f'<line x1="{cx0+107}" y1="{cy0}" x2="{cx0+107}" y2="{cy0+107}" stroke="#2563eb" stroke-width="6.5"/>')
    elems.append(f'<line x1="{cx0+107}" y1="{cy0+107}" x2="{cx0+107}" y2="{cy0+csz}" stroke="#9ca3af" stroke-width="3.0" stroke-dasharray="6,6"/>')
    elems.append(f'<circle cx="{cx0+107}" cy="{cy0+154}" r="30" fill="#fee2e2" stroke="#dc2626" stroke-width="3.0" stroke-dasharray="5,4"/>')
    elems.append(f'<text x="{cx0+107}" y="{cy0+150}" text-anchor="middle" font-size="12.8" font-weight="700" fill="#b91c1c">Open</text>')
    elems.append(f'<text x="{cx0+107}" y="{cy0+167}" text-anchor="middle" font-size="12.8" font-weight="700" fill="#b91c1c">Hole</text>')
    elems.append(f'<text x="{cx0+107}" y="{cy0-10}" text-anchor="middle" font-size="12.2" font-weight="700" fill="#b91c1c">3-Arm T-Bridge (4th Arm Cut)</text>')
    elems.append(f'<text x="{cx0+107}" y="{cy0+csz+22}" text-anchor="middle" font-size="12.4" font-weight="700" fill="#991b1b">Unchecked Loop Threads Gap</text>')

    elems.append('<rect x="888" y="368" width="254" height="214" rx="9" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.4"/>')
    elems.append('<text x="1015" y="400" text-anchor="middle" font-size="13.2" font-weight="700" fill="#991b1b">Unsealed Sheet Gap:</text>')
    elems.append('<text x="1015" y="428" text-anchor="middle" font-size="12.4" font-weight="600" fill="#374151">A 3-arm bridge ends mid-cell,</text>')
    elems.append('<text x="1015" y="448" text-anchor="middle" font-size="12.4" font-weight="600" fill="#374151">leaving a gap between Sheet 0</text>')
    elems.append('<text x="1015" y="468" text-anchor="middle" font-size="12.4" font-weight="600" fill="#374151">and Sheet 1 with no check.</text>')
    elems.append('<line x1="908" y1="486" x2="1122" y2="486" stroke="#e2e8f0" stroke-width="1.4"/>')
    elems.append('<text x="1015" y="514" text-anchor="middle" font-size="13.4" font-weight="700" fill="#b91c1c">Spurious Qubits: 2 → 1,004</text>')
    elems.append('<text x="1015" y="540" text-anchor="middle" font-size="12.4" font-weight="600" fill="#374151">Lifetime collapses to ~ 0;</text>')
    elems.append('<text x="1015" y="560" text-anchor="middle" font-size="12.5" font-weight="700" fill="#1e293b">4-arm cross is minimal</text>')

    save_svg_png("fig_v15_3_architecture", elems, W, H)


def render_fig4_threshold(rg, kmc):
    W, H = 1180, 660
    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<style>text { font-family: -apple-system, Helvetica, Arial, sans-serif; fill: #1f2937; }</style>',
    ]

    elems.append('<rect x="18" y="12" width="1144" height="202" rx="12" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.6"/>')
    elems.append('<text x="590" y="38" text-anchor="middle" font-size="15.6" font-weight="700">'
                 '(a) Local Zoom-Out Across 2×2 Cells (No Global Decoder)</text>')

    elems.append('<rect x="36" y="50" width="536" height="150" rx="9" fill="#ecfdf5" stroke="#a7f3d0" stroke-width="1.4"/>')
    elems.append('<text x="304" y="72" text-anchor="middle" font-size="13.6" font-weight="700" fill="#065f46">'
                 'Cold Bath (Below Tc): Small Pairs Cancel Inside 2×2 Cells</text>')
    draw_mini_grid(elems, 76, 86, 4, 24, [(0, 0), (0, 1), (2, 2), (3, 2)], sub2=True)
    elems.append('<text x="124" y="194" text-anchor="middle" font-size="11.5" font-weight="600" fill="#065f46">Fine Lattice</text>')
    elems.append('<line x1="192" y1="134" x2="258" y2="134" stroke="#047857" stroke-width="2.8"/>')
    elems.append('<polygon points="266,134 256,128 256,140" fill="#047857"/>')
    elems.append('<text x="228" y="124" text-anchor="middle" font-size="11.4" font-weight="700" fill="#047857">Group 2×2</text>')
    draw_mini_grid(elems, 284, 98, 2, 36, [], sub2=False)
    elems.append('<text x="320" y="194" text-anchor="middle" font-size="11.5" font-weight="700" fill="#065f46">Clean Coarse Grid</text>')
    elems.append('<text x="462" y="124" text-anchor="middle" font-size="12.4" font-weight="700" fill="#065f46">Defects shrink</text>')
    elems.append('<text x="462" y="144" text-anchor="middle" font-size="12.4" font-weight="700" fill="#065f46">to 0% as we</text>')
    elems.append('<text x="462" y="164" text-anchor="middle" font-size="12.4" font-weight="700" fill="#065f46">zoom out ↓</text>')

    elems.append('<rect x="608" y="50" width="536" height="150" rx="9" fill="#fff1f2" stroke="#fecdd3" stroke-width="1.4"/>')
    elems.append('<text x="876" y="72" text-anchor="middle" font-size="13.6" font-weight="700" fill="#9f1239">'
                 'Warm Bath (Above Tc): Error Chains Cross Cell Walls &amp; Multiply</text>')
    draw_mini_grid(elems, 648, 86, 4, 24, [(0, 1), (1, 2), (2, 0), (2, 3), (3, 1)], sub2=True)
    elems.append('<text x="696" y="194" text-anchor="middle" font-size="11.5" font-weight="600" fill="#9f1239">Fine Lattice</text>')
    elems.append('<line x1="764" y1="134" x2="830" y2="134" stroke="#be123c" stroke-width="2.8"/>')
    elems.append('<polygon points="838,134 828,128 828,140" fill="#be123c"/>')
    elems.append('<text x="800" y="124" text-anchor="middle" font-size="11.4" font-weight="700" fill="#be123c">Group 2×2</text>')
    draw_mini_grid(elems, 856, 98, 2, 36, [(0, 0), (0, 1), (1, 0), (1, 1)], sub2=False)
    elems.append('<text x="892" y="194" text-anchor="middle" font-size="11.5" font-weight="700" fill="#9f1239">Scrambled Grid</text>')
    elems.append('<text x="1036" y="124" text-anchor="middle" font-size="12.4" font-weight="700" fill="#9f1239">Defects grow</text>')
    elems.append('<text x="1036" y="144" text-anchor="middle" font-size="12.4" font-weight="700" fill="#9f1239">toward 50%</text>')
    elems.append('<text x="1036" y="164" text-anchor="middle" font-size="12.4" font-weight="700" fill="#9f1239">random noise ↑</text>')

    rg_series = []
    for b, col, dsh, lbl in [
        (1.35, "#dc2626", "5,4", "Warm Bath (β = 1.35)"),
        (1.86, "#ea580c", "5,4", "Just Above Tc (β = 1.86)"),
        (2.00, "#1f2937", "",    "At Threshold (β = 2.00)"),
        (2.15, "#2563eb", "",    "Below Tc (β = 2.15)"),
        (2.35, "#059669", "",    "Cold Bath (β = 2.35)"),
    ]:
        rhos = rg[("sq_rx_iso_s5", b)]
        xs, ys = [], []
        for step_idx, s in enumerate([5, 4, 3, 2, 1]):
            if rhos[s] > 0:
                xs.append(float(step_idx)); ys.append(rhos[s])
        rg_series.append({"xs": xs, "ys": ys, "color": col, "dash": dsh, "label": lbl})

    p1 = dict(
        xlim=(-0.2, 4.25), ylim=(1.2e-4, 65.0),
        title="(b) Live Defect Zoom-Out Inside One 322k-Qubit Bath",
        xlabel="Zoom-Out Step: Fine Lattice (322k) → Coarse Cell (19q)",
        ylabel="Violated Check Fraction",
        xticks=[(0,"322k"), (1,"27k"), (2,"2.3k"), (3,"203q"), (4,"19q")],
        yticks=[(3e-4,"0.03%"), (3e-3,"0.3%"), (3e-2,"3%"), (3e-1,"30%")],
        series=rg_series, xlog=False, ylog=True, leg_pos="tl",
        ybands=[(0.048, 65.0, "#fff1f2"), (1.2e-4, 0.040, "#ecfdf5")],
        notes=[
            (4.10, 0.42,   "Too Warm: Multiplies ↑", "#b91c1c", "end"),
            (4.12, 8.2e-4, "Hits Exact 0% ↓", "#047857", "end"),
        ],
    )

    ms_series = []
    for s, col, dsh, nlbl in [
        (5, "#7f1d1d", "",    "Scale 5 (322,159 qubits)"),
        (4, "#be123c", "",    "Scale 4 (27,107 qubits)"),
        (3, "#ea580c", "",    "Scale 3 (2,311 qubits)"),
        (2, "#2563eb", "6,3", "Scale 2 (203 qubits)"),
        (1, "#6b7280", "3,3", "Scale 1 (19 qubits)"),
    ]:
        bs_s = [b for b in (1.25, 1.50, 1.75, 1.90, 2.00, 2.05, 2.10, 2.15, 2.20, 2.25) if (f"sq_rx_iso_s{s}_X_err", "uf", b) in kmc]
        ms_series.append({"xs": bs_s, "ys": [kmc[(f"sq_rx_iso_s{s}_X_err", "uf", b)][1] for b in bs_s], "color": col, "dash": dsh, "label": nlbl})

    top_y = kmc[("sq_rx_iso_s5_X_err", "uf", 2.25)][1]
    gain_5 = top_y / kmc[("sq_rx_iso_s1_X_err", "uf", 2.25)][1]

    p2 = dict(
        xlim=(1.20, 2.48), ylim=(2.5, max(1.5e5, top_y * 4.5)),
        title="(c) Five Code Sizes Share the Same Threshold",
        xlabel="Inverse Bath Temperature β = 1 / T (Colder →)",
        ylabel="Mean Memory Lifetime",
        xticks=[(1.25,"1.25"), (1.5,"1.5"), (1.75,"1.75"), (2.0,"2.0"), (2.25,"2.25")],
        yticks=[(1e1,"10¹"), (1e2,"10²"), (1e3,"10³"), (1e4,"10⁴"), (1e5,"10⁵")],
        series=ms_series, xlog=False, ylog=True, leg_pos="tl",
        xbands=[(2.00, 2.48, "#ecfdf5")],
        vlines=[(2.00, "#374151", "Fixed Threshold β ≈ 2.0", "rb")],
        varrows=[(2.25, kmc[("sq_rx_iso_s1_X_err", "uf", 2.25)][1], top_y, f"{gain_5:,.0f}×", "#7f1d1d", "r", 9.0e2)],
    )

    draw_panel(elems, 86,  250, 472, 348, **p1)
    draw_panel(elems, 676, 250, 472, 348, **p2)
    save_svg_png("fig_v15_4_threshold_rg", elems, W, H)


def render_fig5_quantum(rg, kmc, q3_bD):
    W, H = 1180, 836
    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<style>text { font-family: -apple-system, Helvetica, Arial, sans-serif; fill: #1f2937; }</style>',
    ]

    x0, y0, pw, ph = 42, 42, 516, 326
    elems.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" rx="10" fill="#f8fafc" stroke="#374151" stroke-width="1.5"/>')
    elems.append(f'<text x="{x0+pw/2:.1f}" y="{y0-13:.1f}" text-anchor="middle" font-size="14.8" font-weight="700">'
                 '(a) Alternating X and Z Bridges Protect Both Sectors</text>')

    elems.append(f'<rect x="{x0+20}" y="{y0+22}" width="214" height="196" rx="9" fill="#eff6ff" stroke="#93c5fd" stroke-width="1.5"/>')
    elems.append(f'<text x="{x0+127}" y="{y0+45}" text-anchor="middle" font-size="13.2" font-weight="700" fill="#1e40af">Odd Step: X-Bridges</text>')
    elems.append(f'<rect x="{x0+75}" y="{y0+62}" width="104" height="104" fill="#ffffff" stroke="#2563eb" stroke-width="2.4"/>')
    elems.append(f'<line x1="{x0+127}" y1="{y0+62}" x2="{x0+127}" y2="{y0+166}" stroke="#2563eb" stroke-width="5.2"/>')
    elems.append(f'<line x1="{x0+75}" y1="{y0+114}" x2="{x0+179}" y2="{y0+114}" stroke="#2563eb" stroke-width="5.2"/>')
    elems.append(f'<text x="{x0+127}" y="{y0+188}" text-anchor="middle" font-size="11.8" font-weight="700" fill="#047857">X errors shrink × 2/5</text>')
    elems.append(f'<text x="{x0+127}" y="{y0+206}" text-anchor="middle" font-size="11.5" font-weight="600" fill="#64748b">(Z sector waits: × 1)</text>')

    elems.append(f'<line x1="{x0+240}" y1="{y0+120}" x2="{x0+270}" y2="{y0+120}" stroke="#475569" stroke-width="2.6"/>')
    elems.append(f'<polygon points="{x0+278},{y0+120} {x0+268},{y0+114} {x0+268},{y0+126}" fill="#475569"/>')

    elems.append(f'<rect x="{x0+282}" y="{y0+22}" width="214" height="196" rx="9" fill="#fef2f2" stroke="#fca5a5" stroke-width="1.5"/>')
    elems.append(f'<text x="{x0+389}" y="{y0+45}" text-anchor="middle" font-size="13.2" font-weight="700" fill="#991b1b">Even Step: Z-Bridges</text>')
    elems.append(f'<rect x="{x0+337}" y="{y0+62}" width="104" height="104" fill="#ffffff" stroke="#dc2626" stroke-width="2.4"/>')
    elems.append(f'<line x1="{x0+389}" y1="{y0+62}" x2="{x0+389}" y2="{y0+166}" stroke="#dc2626" stroke-width="5.2"/>')
    elems.append(f'<line x1="{x0+337}" y1="{y0+114}" x2="{x0+441}" y2="{y0+114}" stroke="#dc2626" stroke-width="5.2"/>')
    elems.append(f'<text x="{x0+389}" y="{y0+188}" text-anchor="middle" font-size="11.8" font-weight="700" fill="#047857">Z errors shrink × 2/5</text>')
    elems.append(f'<text x="{x0+389}" y="{y0+206}" text-anchor="middle" font-size="11.5" font-weight="600" fill="#64748b">(X sector waits: × 1)</text>')

    elems.append(f'<rect x="{x0+20}" y="{y0+234}" width="476" height="76" rx="8" fill="#ecfdf5" stroke="#a7f3d0" stroke-width="1.4"/>')
    elems.append(f'<text x="{x0+pw/2:.1f}" y="{y0+260}" text-anchor="middle" font-size="13.2" font-weight="700" fill="#065f46">'
                 'Net Effect Over Every 2-Step Quantum Cycle:</text>')
    elems.append(f'<text x="{x0+pw/2:.1f}" y="{y0+283}" text-anchor="middle" font-size="12.4" font-weight="600" fill="#1f2937">'
                 'Both X and Z syndromes contract by 2/5; waiting steps</text>')
    elems.append(f'<text x="{x0+pw/2:.1f}" y="{y0+301}" text-anchor="middle" font-size="12.4" font-weight="600" fill="#1f2937">'
                 'create the sawtooth flow in (b) and shift Tc to β ≈ 6.0</text>')

    saw_series = []
    for b, col, dsh, lbl in [
        (4.50, "#dc2626", "5,4", "Too Warm (β = 4.5: grows)"),
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
        title="(b) Alternating (X, Z) Zoom-Out (94,774q Torus)",
        xlabel="Zoom-Out Step (Z-Steps Tick Up; X-Steps Drop to 0%)",
        ylabel="Violated Check Fraction",
        xticks=[(0,"95k"), (1,"5.8k (Z)"), (2,"646q (X)"), (3,"38q (Z)"), (4,"Seed (X)")],
        yticks=[(1e-4,"0.01%"), (1e-3,"0.1%"), (1e-2,"1%"), (1e-1,"10%")],
        series=saw_series, xlog=False, ylog=True, leg_pos="tl",
        ybands=[(0.028, 25.0, "#fff1f2"), (2.5e-5, 0.022, "#ecfdf5")],
        notes=[(3.35, 5.2e-5, "Drops to Exact 0% →", "#047857", "end")],
    )
    draw_panel(elems, 676, 42, 472, 326, **pb)

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

    draw_panel(elems, 86,  464, 472, 316, **pc)
    draw_panel(elems, 676, 464, 472, 316, **pd)
    save_svg_png("fig_v15_5_quantum", elems, W, H)


def render_fig6_euclidean_and_wires(kmc12, v7_best):
    """
    2-Row, 4-Panel Visual + Data Figure (1180 x 836):
      (a) Visual 3D Euclidean Volume-Packing Cutoff: 2x2 (12 > 8 cubes, dim=3.58) vs 3x3 (24 < 27 cubes, dim=2.89)
      (b) Data: Minimal 3D-Compatible 3x3 Single-Cross Scaling (41 -> 935 -> 22,097 qubits: 68x gain; 685x quantum)
      (c) Visual: Passive Flat Grid Wire (ΔE = 0 walk) vs Bounded Wire-Energy Hill (J in [1.0, 2.2])
      (d) Data: Collision-Free Z^3 Cubic Grids (56,312q & 10,280q): Wire Hills Double Barrier & Recover 80x Lifetime
    """
    W, H = 1180, 836
    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<style>text { font-family: -apple-system, Helvetica, Arial, sans-serif; fill: #1f2937; }</style>',
    ]

    # --- Panel (a): Visual 3D Euclidean Volume-Packing Cutoff ---
    x0, y0, pw, ph = 42, 42, 516, 326
    elems.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" rx="10" fill="#f8fafc" stroke="#374151" stroke-width="1.5"/>')
    elems.append(f'<text x="{x0+pw/2:.1f}" y="{y0-13:.1f}" text-anchor="middle" font-size="14.8" font-weight="700">'
                 '(a) Three-Dimensional Euclidean Volume-Packing Cutoff</text>')

    # Left sub-card: 2x2 Cross Overflows 2^3 = 8 Sub-Cubes!
    elems.append(f'<rect x="{x0+18}" y="{y0+20}" width="230" height="214" rx="9" fill="#fff1f2" stroke="#fecdd3" stroke-width="1.5"/>')
    elems.append(f'<text x="{x0+133}" y="{y0+42}" text-anchor="middle" font-size="13.0" font-weight="700" fill="#9f1239">2×2 Cross: Overflows 3D</text>')
    # Draw 2x2 square with central cross + count
    gx, gy, gs = x0 + 81, y0 + 58, 104
    elems.append(f'<rect x="{gx}" y="{gy}" width="{gs}" height="{gs}" fill="#ffffff" stroke="#be123c" stroke-width="2.4"/>')
    elems.append(f'<line x1="{gx+52}" y1="{gy}" x2="{gx+52}" y2="{gy+gs}" stroke="#dc2626" stroke-width="5.0"/>')
    elems.append(f'<line x1="{gx}" y1="{gy+52}" x2="{gx+gs}" y2="{gy+52}" stroke="#dc2626" stroke-width="5.0"/>')
    elems.append(f'<text x="{x0+133}" y="{y0+184}" text-anchor="middle" font-size="12.2" font-weight="700" fill="#9f1239">12 Squares &gt; 2³ = 8 Cubes</text>')
    elems.append(f'<text x="{x0+133}" y="{y0+203}" text-anchor="middle" font-size="11.6" font-weight="600" fill="#be123c">dim = log₂ 12 ≈ 3.58 &gt; 3</text>')
    elems.append(f'<text x="{x0+133}" y="{y0+221}" text-anchor="middle" font-size="11.5" font-weight="700" fill="#881337">Wires must stretch in R³</text>')

    # Right sub-card: 3x3 Single-Cross Fits Inside 3^3 = 27 Sub-Cubes!
    elems.append(f'<rect x="{x0+268}" y="{y0+20}" width="230" height="214" rx="9" fill="#ecfdf5" stroke="#a7f3d0" stroke-width="1.5"/>')
    elems.append(f'<text x="{x0+383}" y="{y0+42}" text-anchor="middle" font-size="13.0" font-weight="700" fill="#065f46">3×3 Cross: Fits in 3D Cube</text>')
    hx, hy, hs = x0 + 331, y0 + 58, 104
    elems.append(f'<rect x="{hx}" y="{hy}" width="{hs}" height="{hs}" fill="#ffffff" stroke="#059669" stroke-width="2.4"/>')
    for k in (1, 2):
        lw = "4.8" if k == 1 else "1.4"
        lc = "#059669" if k == 1 else "#94a3b8"
        elems.append(f'<line x1="{hx+k*hs/3:.1f}" y1="{hy}" x2="{hx+k*hs/3:.1f}" y2="{hy+hs}" stroke="{lc}" stroke-width="{lw}"/>')
        elems.append(f'<line x1="{hx}" y1="{hy+k*hs/3:.1f}" x2="{hx+hs}" y2="{hy+k*hs/3:.1f}" stroke="{lc}" stroke-width="{lw}"/>')
    elems.append(f'<text x="{x0+383}" y="{y0+184}" text-anchor="middle" font-size="12.2" font-weight="700" fill="#065f46">24 Squares &lt; 3³ = 27 Cubes</text>')
    elems.append(f'<text x="{x0+383}" y="{y0+203}" text-anchor="middle" font-size="11.6" font-weight="600" fill="#047857">dim = log₃ 24 ≈ 2.89 &lt; 3</text>')
    elems.append(f'<text x="{x0+383}" y="{y0+221}" text-anchor="middle" font-size="11.5" font-weight="700" fill="#064e3b">Shrinks errors × 10 / 63</text>')

    # Bottom banner inside Panel (a)
    elems.append(f'<rect x="{x0+18}" y="{y0+246}" width="480" height="66" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.3"/>')
    elems.append(f'<text x="{x0+pw/2:.1f}" y="{y0+272}" text-anchor="middle" font-size="13.0" font-weight="700" fill="#1e293b">'
                 'Packing Condition:  2L² + 2L ≤ L³  ⇔  L ≥ 1 + √3 ≈ 2.73</text>')
    elems.append(f'<text x="{x0+pw/2:.1f}" y="{y0+296}" text-anchor="middle" font-size="12.0" font-weight="600" fill="#475569">'
                 '3×3 single-cross is the smallest integer cell whose volume fits in R³</text>')

    # --- Panel (b): Minimal 3D-Compatible 3x3 Single-Cross Multi-Scale Lifetimes ---
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
    draw_panel(elems, 676, 42, 472, 326, **pb)

    # --- Panel (c): Visual Flat Grid Wire vs Bounded Wire-Energy Hill ---
    cx0, cy0, cpw, cph = 42, 464, 516, 316
    elems.append(f'<rect x="{cx0}" y="{cy0}" width="{cpw}" height="{cph}" rx="10" fill="#f8fafc" stroke="#374151" stroke-width="1.5"/>')
    elems.append(f'<text x="{cx0+cpw/2:.1f}" y="{cy0-13:.1f}" text-anchor="middle" font-size="14.8" font-weight="700">'
                 '(c) Suppressing Wire Entropy on One Cubic Grid</text>')

    # Sub-box 1: Uniform Grid Wire (ΔE = 0 walk)
    elems.append(f'<rect x="{cx0+18}" y="{cy0+18}" width="480" height="128" rx="8" fill="#fff1f2" stroke="#fecdd3" stroke-width="1.4"/>')
    elems.append(f'<text x="{cx0+258}" y="{cy0+40}" text-anchor="middle" font-size="13.0" font-weight="700" fill="#9f1239">'
                 'Uniform Grid Wire (J = 1): Defect Walks Freely at ΔE = 0</text>')
    wy1 = cy0 + 82
    elems.append(f'<line x1="{cx0+68}" y1="{wy1}" x2="{cx0+448}" y2="{wy1}" stroke="#dc2626" stroke-width="3.6"/>')
    for k in range(8):
        qx = cx0 + 68 + k * 54.28
        if k in (0, 7):
            elems.append(f'<rect x="{qx-9:.1f}" y="{wy1-9:.1f}" width="18" height="18" rx="3" fill="#059669" stroke="#ffffff" stroke-width="1.5"/>')
        else:
            elems.append(f'<circle cx="{qx:.1f}" cy="{wy1}" r="6.0" fill="#ffffff" stroke="#dc2626" stroke-width="2.2"/>')
    elems.append(f'<circle cx="{cx0+231:.1f}" cy="{wy1-18}" r="7.5" fill="#dc2626"/>')
    elems.append(f'<line x1="{cx0+145}" y1="{wy1-18}" x2="{cx0+318}" y2="{wy1-18}" stroke="#991b1b" stroke-width="2.0" stroke-dasharray="4,3"/>')
    elems.append(f'<text x="{cx0+258}" y="{cy0+128}" text-anchor="middle" font-size="11.8" font-weight="600" fill="#881337">'
                 '16-step passive wires on scale-8 grid inflate entropy T·ΔS → 321× loss</text>')

    # Sub-box 2: Bounded Wire-Energy Hill (J in [1.0, 2.2])
    elems.append(f'<rect x="{cx0+18}" y="{cy0+164}" width="480" height="134" rx="8" fill="#ecfdf5" stroke="#a7f3d0" stroke-width="1.4"/>')
    elems.append(f'<text x="{cx0+258}" y="{cy0+186}" text-anchor="middle" font-size="13.0" font-weight="700" fill="#065f46">'
                 'Graded Wire Hill (J ∈ [1.0, 2.2]): Pushes Defects Back to Junctions</text>')
    # Smooth hill profile curve + balls rolling downhill to green junctions!
    elems.append(f'<path d="M {cx0+68},{cy0+254} Q {cx0+258},{cy0+196} {cx0+448},{cy0+254}" fill="#a7f3d0" fill-opacity="0.55" stroke="#059669" stroke-width="3.2"/>')
    elems.append(f'<rect x="{cx0+59}" y="{cy0+245}" width="18" height="18" rx="3" fill="#059669" stroke="#ffffff" stroke-width="1.5"/>')
    elems.append(f'<rect x="{cx0+439}" y="{cy0+245}" width="18" height="18" rx="3" fill="#059669" stroke="#ffffff" stroke-width="1.5"/>')
    elems.append(f'<circle cx="{cx0+195}" cy="{cy0+222}" r="7.5" fill="#dc2626"/>')
    elems.append(f'<line x1="{cx0+182}" y1="{cy0+226}" x2="{cx0+112}" y2="{cy0+246}" stroke="#047857" stroke-width="2.6"/>')
    elems.append(f'<polygon points="{cx0+104},{cy0+248} {cx0+115},{cy0+240} {cx0+117},{cy0+250}" fill="#047857"/>')
    elems.append(f'<text x="{cx0+258}" y="{cy0+284}" text-anchor="middle" font-size="11.8" font-weight="700" fill="#065f46">'
                 'Bounded coupling hill doubles scale-8 grid barrier (1.6 → 3.1) &amp; recovers 80×</text>')

    # --- Panel (d): Collision-Free Z^3 Cubic Grids: Uniform vs Wire Hills ---
    bs_comp = [5.50, 6.00, 6.50, 7.00]
    ys_comp = [v7_best("sq_L3r2_s2_X_err", "uniform", "uf", b) for b in bs_comp]

    bs_z3 = [5.00, 5.50, 6.00, 6.50, 7.00]
    ys_46_hill = [kmc12[("sq_z3_4_6_r2_X_err", "wire_hill", "uf", b)][1] for b in bs_z3]
    ys_46_unif = [kmc12[("sq_z3_4_6_r2_X_err", "uniform",   "uf", b)][1] for b in bs_z3]
    ys_88_hill = [kmc12[("sq_bdl_8_8_r2_X_err", "wire_hill", "uf", b)][1] for b in bs_z3]
    ys_88_unif = [kmc12[("sq_bdl_8_8_r2_X_err", "uniform",   "uf", b)][1] for b in bs_z3]
    gain_88 = ys_88_hill[-1] / ys_88_unif[-1]

    pd = dict(
        xlim=(4.85, 7.34), ylim=(35.0, 3.0e7),
        title="(d) Cubic-Grid Wire Hills Restore Barrier & Lifetime",
        xlabel="Inverse Bath Temperature β = 1 / T (Colder →)",
        ylabel="Quantum Memory Lifetime",
        xticks=[(5.0,"5.0"), (5.5,"5.5"), (6.0,"6.0"), (6.5,"6.5"), (7.0,"7.0")],
        yticks=[(1e2,"10²"), (1e3,"10³"), (1e4,"10⁴"), (1e5,"10⁵"), (1e6,"10⁶")],
        series=[
            {"xs": bs_comp, "ys": ys_comp,    "color": "#1e3a8a", "dash": "3,3", "label": "Unstretched Graph (1,480q)"},
            {"xs": bs_z3,   "ys": ys_46_hill, "color": "#059669", "dash": "",    "label": "Grid (4,6) + Wire Hill (10,280q: 7×)"},
            {"xs": bs_z3,   "ys": ys_88_hill, "color": "#2563eb", "dash": "",    "label": f"Scale-8 Grid + Wire Hill (56,312q: {gain_88:.0f}×)"},
            {"xs": bs_z3,   "ys": ys_46_unif, "color": "#ea580c", "dash": "5,3", "hollow": True, "label": "Grid (4,6) Uniform Wire (10,280q)"},
            {"xs": bs_z3,   "ys": ys_88_unif, "color": "#dc2626", "dash": "5,3", "hollow": True, "label": "Scale-8 Grid Uniform Wire (56,312q)"},
        ],
        xlog=False, ylog=True, leg_pos="tl",
        varrows=[(7.00, ys_88_unif[-1], ys_88_hill[-1], f"{gain_88:.0f}×", "#2563eb", "r", 8.5e3)],
    )
    draw_panel(elems, 676, 464, 472, 316, **pd)
    save_svg_png("fig_v15_6_euclidean_wires", elems, W, H)


def render_fig7_compact_and_highrate(kmc12, kmc6):
    """
    2-Row, 4-Panel Visual + Data Figure (1180 x 836):
      (a) Visual: Mixed 1-Step Seam Failure (K=1->737) vs Sector-Balanced (3x3 X, 2x2 Z) at 586 Qubits
      (b) Data: Sub-2,500-Qubit Balanced Codes (586q & 1,172q) Beat Random Layer Codes & Scale to 129,148q
      (c) Visual: Flat {4,4} Torus (4 squares/corner, K=2) vs Hyperbolic {4,5}/{4,6} Square Tilings (K=6..10)
      (d) Data: Multi-Qubit Hyperbolic Square-Welded Memories (K = 6 and K = 10 Logical Qubits)
    """
    W, H = 1180, 836
    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<style>text { font-family: -apple-system, Helvetica, Arial, sans-serif; fill: #1f2937; }</style>',
    ]

    # --- Panel (a): Visual Mixed 1-Step Failure vs Sector-Balanced 586q Cycle ---
    x0, y0, pw, ph = 42, 42, 516, 326
    elems.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" rx="10" fill="#f8fafc" stroke="#374151" stroke-width="1.5"/>')
    elems.append(f'<text x="{x0+pw/2:.1f}" y="{y0-13:.1f}" text-anchor="middle" font-size="14.8" font-weight="700">'
                 '(a) Why Two Steps Are Forced &amp; Sector-Balanced 586q Cell</text>')

    # Left sub-card: Single-Step Mixed (X + Z) Cross Fails
    elems.append(f'<rect x="{x0+18}" y="{y0+20}" width="228" height="214" rx="9" fill="#fff1f2" stroke="#fecdd3" stroke-width="1.5"/>')
    elems.append(f'<text x="{x0+132}" y="{y0+42}" text-anchor="middle" font-size="12.8" font-weight="700" fill="#9f1239">1-Step Mixed Cross Fails</text>')
    mx, my, ms = x0 + 80, y0 + 58, 104
    elems.append(f'<rect x="{mx}" y="{my}" width="{ms}" height="{ms}" fill="#ffffff" stroke="#be123c" stroke-width="2.2"/>')
    elems.append(f'<line x1="{mx+52}" y1="{my}" x2="{mx+52}" y2="{my+ms}" stroke="#2563eb" stroke-width="5.2"/>')
    elems.append(f'<line x1="{mx}" y1="{my+52}" x2="{mx+ms}" y2="{my+52}" stroke="#dc2626" stroke-width="5.2"/>')
    elems.append(f'<circle cx="{mx+52}" cy="{my+52}" r="12" fill="#fee2e2" stroke="#b91c1c" stroke-width="2.6"/>')
    elems.append(f'<text x="{mx+52}" y="{my+56}" text-anchor="middle" font-size="13.0" font-weight="700" fill="#991b1c">!</text>')
    elems.append(f'<text x="{x0+132}" y="{y0+184}" text-anchor="middle" font-size="11.8" font-weight="700" fill="#9f1239">X/Z Crossing Breaks Checks</text>')
    elems.append(f'<text x="{x0+132}" y="{y0+203}" text-anchor="middle" font-size="11.6" font-weight="600" fill="#be123c">Spurious K: 1 → 3 → 49 → 737</text>')
    elems.append(f'<text x="{x0+132}" y="{y0+221}" text-anchor="middle" font-size="11.4" font-weight="700" fill="#881337">Weight-1 logical shortcuts</text>')

    # Right sub-card: Sector-Balanced (3x3 2-bridge X -> 2x2 1-bridge Z)
    elems.append(f'<rect x="{x0+268}" y="{y0+20}" width="230" height="214" rx="9" fill="#ecfdf5" stroke="#a7f3d0" stroke-width="1.5"/>')
    elems.append(f'<text x="{x0+383}" y="{y0+42}" text-anchor="middle" font-size="12.8" font-weight="700" fill="#065f46">Balanced 2-Step Cycle</text>')
    # Mini 3x3 X-cell -> 2x2 Z-cell
    b1x, b1y, b1s = x0 + 286, y0 + 68, 78
    elems.append(f'<rect x="{b1x}" y="{b1y}" width="{b1s}" height="{b1s}" fill="#ffffff" stroke="#2563eb" stroke-width="2.0"/>')
    for k in (1, 2):
        elems.append(f'<line x1="{b1x+k*26}" y1="{b1y}" x2="{b1x+k*26}" y2="{b1y+b1s}" stroke="#2563eb" stroke-width="3.2"/>')
        elems.append(f'<line x1="{b1x}" y1="{b1y+k*26}" x2="{b1x+b1s}" y2="{b1y+k*26}" stroke="#2563eb" stroke-width="3.2"/>')
    elems.append(f'<line x1="{x0+370}" y1="{y0+107}" x2="{x0+394}" y2="{y0+107}" stroke="#047857" stroke-width="2.4"/>')
    elems.append(f'<polygon points="{x0+400},{y0+107} {x0+392},{y0+102} {x0+392},{y0+112}" fill="#047857"/>')
    b2x, b2y, b2s = x0 + 404, y0 + 68, 78
    elems.append(f'<rect x="{b2x}" y="{b2y}" width="{b2s}" height="{b2s}" fill="#ffffff" stroke="#dc2626" stroke-width="2.0"/>')
    elems.append(f'<line x1="{b2x+39}" y1="{b2y}" x2="{b2x+39}" y2="{b2y+b2s}" stroke="#dc2626" stroke-width="3.8"/>')
    elems.append(f'<line x1="{b2x}" y1="{b2y+39}" x2="{b2x+b2s}" y2="{b2y+39}" stroke="#dc2626" stroke-width="3.8"/>')
    elems.append(f'<text x="{x0+383}" y="{y0+184}" text-anchor="middle" font-size="11.8" font-weight="700" fill="#065f46">3×3 X-Step + 2×2 Z-Step</text>')
    elems.append(f'<text x="{x0+383}" y="{y0+203}" text-anchor="middle" font-size="11.6" font-weight="600" fill="#047857">586q Planar (K=1) / 1,172q Torus</text>')
    elems.append(f'<text x="{x0+383}" y="{y0+221}" text-anchor="middle" font-size="11.4" font-weight="700" fill="#064e3b">Matches X &amp; Z barriers early</text>')

    elems.append(f'<rect x="{x0+18}" y="{y0+246}" width="480" height="66" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.3"/>')
    elems.append(f'<text x="{x0+pw/2:.1f}" y="{y0+272}" text-anchor="middle" font-size="12.8" font-weight="700" fill="#1e293b">'
                 'Eliminates Waiting-Step Dilution Below 2,500 Physical Qubits:</text>')
    elems.append(f'<text x="{x0+pw/2:.1f}" y="{y0+296}" text-anchor="middle" font-size="12.0" font-weight="600" fill="#475569">'
                 '586q per logical qubit beats 3,645q Random Layer Codes &amp; scales to 129k</text>')

    # --- Panel (b): Sub-2,500-Qubit Balanced Codes Beat Random Layer Codes & Scale to 129,148q ---
    bs_rot = [b for b in (5.00, 5.50, 6.00, 6.50, 7.00) if ("rottor_hyb21_s2_X_err", "uniform", "uf", b) in kmc12]
    ys_rot = [kmc12[("rottor_hyb21_s2_X_err", "uniform", "uf", b)][1] for b in bs_rot]

    bs_h2 = [b for b in (5.00, 5.50, 6.00, 6.50, 7.00) if ("sq_hyb21_s2_X_err", "uniform", "uf", b) in kmc12]
    ys_h2 = [kmc12[("sq_hyb21_s2_X_err", "uniform", "uf", b)][1] for b in bs_h2]

    bs_h4 = [b for b in (4.50, 5.00, 5.25, 5.50, 5.75, 6.00, 6.25, 6.40) if ("sq_hyb21_s4_X_err", "uniform", "uf", b) in kmc12]
    ys_h4 = [kmc12[("sq_hyb21_s4_X_err", "uniform", "uf", b)][1] for b in bs_h4]
    sl_h4 = ols_slope(bs_h4[-3:], ys_h4[-3:])

    # Gu-Caha peak n=13 (N=3,645) ensemble mean at beta = 5.0, 6.0, 7.0
    caha13 = ["caha_n13_code_102", "caha_n13_code_104", "caha_n13_code_106"]
    bs_c13, ys_c13 = [], []
    for b in (5.00, 6.00, 7.00):
        vs = [kmc6[(c + "_X_err", "uniform", "uf", b)][1] for c in caha13 if (c + "_X_err", "uniform", "uf", b) in kmc6]
        if vs:
            bs_c13.append(b); ys_c13.append(sum(vs) / len(vs))

    gain_plan = ys_h2[-1] / ys_c13[-1]

    pb = dict(
        xlim=(4.42, 7.36), ylim=(80.0, 2.5e10),
        title="(b) Compact Balanced Codes Beat Layer Codes",
        xlabel="Inverse Bath Temperature β = 1 / T (Colder →)",
        ylabel="Quantum Memory Lifetime",
        xticks=[(4.5,"4.5"), (5.0,"5.0"), (5.5,"5.5"), (6.0,"6.0"), (6.5,"6.5"), (7.0,"7.0")],
        yticks=[(1e2,"10²"), (1e4,"10⁴"), (1e6,"10⁶"), (1e8,"10⁸")],
        series=[
            {"xs": bs_rot, "ys": ys_rot, "color": "#059669", "dash": "",    "label": "Balanced Torus (1,172q, K=2: 4,074×)"},
            {"xs": bs_h4,  "ys": ys_h4,  "color": "#be123c", "dash": "",    "label": f"Balanced Cycle 2 (129,148q: barrier {sl_h4:.1f})"},
            {"xs": bs_h2,  "ys": ys_h2,  "color": "#2563eb", "dash": "",    "label": f"Balanced Cycle 1 (586q, K=1: {gain_plan:.0f}×)"},
            {"xs": bs_c13, "ys": ys_c13, "color": "#6b7280", "dash": "5,4", "hollow": True, "label": "Peak Random Layer Code (3,645q, K=1)"},
        ],
        xlog=False, ylog=True, leg_pos="tl",
        varrows=[(7.00, ys_c13[-1], ys_h2[-1], f"{gain_plan:.0f}×", "#2563eb", "r", 3.5e5)],
    )
    draw_panel(elems, 676, 42, 472, 326, **pb)

    # --- Panel (c): Visual Flat {4,4} vs Hyperbolic {4,5} & {4,6} Square Tilings ---
    cx0, cy0, cpw, cph = 42, 464, 516, 316
    elems.append(f'<rect x="{cx0}" y="{cy0}" width="{cpw}" height="{cph}" rx="10" fill="#f8fafc" stroke="#374151" stroke-width="1.5"/>')
    elems.append(f'<text x="{cx0+cpw/2:.1f}" y="{cy0-13:.1f}" text-anchor="middle" font-size="14.8" font-weight="700">'
                 '(c) Multi-Qubit Hyperbolic Square Tilings ({4, 5} &amp; {4, 6})</text>')

    # Left card: Flat {4,4} corner (4 squares meet -> K = 2)
    elems.append(f'<rect x="{cx0+18}" y="{cy0+20}" width="224" height="206" rx="9" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5"/>')
    elems.append(f'<text x="{cx0+130}" y="{cy0+42}" text-anchor="middle" font-size="12.8" font-weight="700" fill="#334155">Flat {{4, 4}} Torus Seed</text>')
    fx, fy, fs = cx0 + 82, cy0 + 60, 96
    elems.append(f'<rect x="{fx}" y="{fy}" width="{fs}" height="{fs}" fill="#dbeafe" stroke="#2563eb" stroke-width="2.2"/>')
    elems.append(f'<line x1="{fx+48}" y1="{fy}" x2="{fx+48}" y2="{fy+fs}" stroke="#2563eb" stroke-width="2.2"/>')
    elems.append(f'<line x1="{fx}" y1="{fy+48}" x2="{fx+fs}" y2="{fy+48}" stroke="#2563eb" stroke-width="2.2"/>')
    elems.append(f'<circle cx="{fx+48}" cy="{fy+48}" r="6.5" fill="#1e3a8a" stroke="#ffffff" stroke-width="1.5"/>')
    elems.append(f'<text x="{cx0+130}" y="{cy0+180}" text-anchor="middle" font-size="11.8" font-weight="700" fill="#334155">4 Squares Meet per Corner</text>')
    elems.append(f'<text x="{cx0+130}" y="{cy0+202}" text-anchor="middle" font-size="11.6" font-weight="600" fill="#64748b">Zero curvature → only K = 2</text>')

    # Right card: Hyperbolic {4, 6} corner (6 four-sided squares meet at 1 vertex -> K = 6..10!)
    elems.append(f'<rect x="{cx0+262}" y="{cy0+20}" width="236" height="206" rx="9" fill="#ecfdf5" stroke="#a7f3d0" stroke-width="1.5"/>')
    elems.append(f'<text x="{cx0+380}" y="{cy0+42}" text-anchor="middle" font-size="12.8" font-weight="700" fill="#065f46">Hyperbolic {{4, 5}} &amp; {{4, 6}} Seeds</text>')
    vx, vy = cx0 + 380, cy0 + 108
    for idx in range(6):
        ang1 = math.radians(idx * 60.0)
        ang2 = math.radians((idx + 1) * 60.0)
        angm = math.radians(idx * 60.0 + 30.0)
        r1, rm = 44.0, 56.0
        p1 = f"{vx + r1*math.cos(ang1):.1f},{vy + r1*math.sin(ang1):.1f}"
        pm = f"{vx + rm*math.cos(angm):.1f},{vy + rm*math.sin(angm):.1f}"
        p2 = f"{vx + r1*math.cos(ang2):.1f},{vy + r1*math.sin(ang2):.1f}"
        fc = "#bfdbfe" if idx % 2 == 0 else "#a7f3d0"
        elems.append(f'<polygon points="{vx},{vy} {p1} {pm} {p2}" fill="{fc}" fill-opacity="0.88" stroke="#047857" stroke-width="1.8"/>')
    elems.append(f'<circle cx="{vx}" cy="{vy}" r="6.5" fill="#065f46" stroke="#ffffff" stroke-width="1.5"/>')
    elems.append(f'<text x="{cx0+380}" y="{cy0+180}" text-anchor="middle" font-size="11.8" font-weight="700" fill="#065f46">5 or 6 Squares Meet per Corner</text>')
    elems.append(f'<text x="{cx0+380}" y="{cy0+202}" text-anchor="middle" font-size="11.6" font-weight="700" fill="#047857">Encodes K = 6 to 10 Qubits</text>')

    elems.append(f'<rect x="{cx0+18}" y="{cy0+238}" width="480" height="62" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.3"/>')
    elems.append(f'<text x="{cx0+cpw/2:.1f}" y="{cy0+262}" text-anchor="middle" font-size="12.6" font-weight="700" fill="#1e293b">'
                 'Every Face Remains a 4-Sided Square:</text>')
    elems.append(f'<text x="{cx0+cpw/2:.1f}" y="{cy0+285}" text-anchor="middle" font-size="12.0" font-weight="600" fill="#475569">'
                 'Our exact 2/5 contraction proof holds face-by-face at 646–1,292q / logical</text>')

    # --- Panel (d): Multi-Qubit Hyperbolic Square-Welded Memories (K = 6 and K = 10) ---
    hyp_series = []
    for tag, col, dsh, lbl in [
        ("hyp46_hyb21_s2",    "#059669", "",    "{4,6} Balanced (7,032q, K=6: 1,172q/log)"),
        ("hyp45_unif_s2",     "#2563eb", "",    "{4,5} Uniform (6,460q, K=6: 1,077q/log)"),
        ("hyp45_10q_unif_s2", "#be123c", "",    "{4,5} 10-Qubit (12,920q, K=10: 1,292q/log)"),
        ("hyp46_unif_s2",     "#7c3aed", "5,3", "{4,6} Compact (3,876q, K=6: 646q/log)"),
    ]:
        bs = [b for b in (5.00, 5.50, 6.00, 6.50) if (f"{tag}_X_err", "uniform", "uf", b) in kmc12]
        ys = [kmc12[(f"{tag}_X_err", "uniform", "uf", b)][1] for b in bs]
        hyp_series.append({"xs": bs, "ys": ys, "color": col, "dash": dsh, "label": lbl})

    pd = dict(
        xlim=(4.90, 6.68), ylim=(120.0, 2.5e7),
        title="(d) High-Rate Multi-Qubit Welded Memories (K = 6, 10)",
        xlabel="Inverse Bath Temperature β = 1 / T (Colder →)",
        ylabel="Quantum Memory Lifetime (All K Qubits)",
        xticks=[(5.0,"5.0"), (5.5,"5.5"), (6.0,"6.0"), (6.5,"6.5")],
        yticks=[(1e2,"10²"), (1e3,"10³"), (1e4,"10⁴"), (1e5,"10⁵"), (1e6,"10⁶")],
        series=hyp_series, xlog=False, ylog=True, leg_pos="tl",
    )
    draw_panel(elems, 676, 464, 472, 316, **pd)
    save_svg_png("fig_v15_7_compact_highrate", elems, W, H)


def main():
    kmc4, kmc6, kmc7, ffs7, cool7, kmc9, rg9, cool11 = load_all()
    kmc12 = load_v12()

    def v7_best(tag, jm, dec, b, min_reps=3):
        if (tag, jm, dec, b) in kmc7 and kmc7[(tag, jm, dec, b)][2] == 0:
            return kmc7[(tag, jm, dec, b)][1]
        if (tag, jm, dec, b) in ffs7 and ffs7[(tag, jm, dec, b)][2] >= min_reps:
            return ffs7[(tag, jm, dec, b)][1]
        return None

    # --- 1. `fig_v15_1_classes.png` ---
    def v4_norm_curve(cnames, b):
        pts = [(kmc4[cn][b][0], kmc4[cn][b][1]) for cn in cnames if b in kmc4.get(cn, {})]
        y0 = pts[0][1]
        return [x for x, _ in pts], [y / y0 for _, y in pts]

    hx, hy = v4_norm_curve([f"haah_cubic_L{L}_Xerr.code" for L in [3, 5, 7, 9, 11, 13]], 4.25)
    bx, by = v4_norm_curve([f"bb_gross_{l}x{m}_Xerr.code" for l, m in [(6,6),(12,6),(18,12),(24,18),(30,18),(36,18),(48,18),(60,18)]], 4.25)
    tx_n, ty_n = v4_norm_curve([f"toric2d_L{L}_Xerr.code" for L in [4, 6, 10, 16, 24, 34]], 4.50)

    q3_pts = [kmc9[(f"sq_iso_L2r1_s{s}_Z_err", "uf", 6.50)] for s in [1, 3, 5]]
    q3_x = [p[0] for p in q3_pts]
    q3_y = [p[1] / q3_pts[0][1] for p in q3_pts]

    w_pts = [kmc9[(f"sq_rx_iso_s{s}_X_err", "uf", 2.25)] for s in [1, 2, 3, 4, 5]]
    w_x = [p[0] for p in w_pts]
    w_y = [p[1] / w_pts[0][1] for p in w_pts]

    s_norm = [
        {"xs": q3_x, "ys": q3_y, "color": "#ea580c", "dash": "7,3", "label": f"3D Welded Quantum (to 424k: {q3_y[-1]:,.0f}×)"},
        {"xs": w_x,  "ys": w_y,  "color": "#be123c", "dash": "",    "label": f"3D Welded Single-Sector (5 scales: {w_y[-1]:,.0f}×)"},
        {"xs": hx,   "ys": hy,   "color": "#7c3aed", "dash": "",    "label": "Haah 3D Cubic Code (peaks at 25×, turns over)"},
        {"xs": bx,   "ys": by,   "color": "#2563eb", "dash": "5,4", "label": "Scaled 2D Bivariate Bicycle (falls < 1×)"},
        {"xs": tx_n, "ys": ty_n, "color": "#4b5563", "dash": "3,3", "label": "2D Toric Code (falls as 1 / N)"},
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
            b_cands = [b for b in (1.75, 2.00, 2.25) if (f"sq_rx_iso_s0_X_err", "uf", b) in kmc9]
        w_bN.append(kmc9[(f"sq_rx_iso_s{s}_X_err", "uf", b_cands[0])][0])
        w_bD.append(ols_slope(b_cands, [kmc9[(f"sq_rx_iso_s{s}_X_err", "uf", b)][1] for b in b_cands]))

    q3_bN, q3_bD = [], []
    for s in (1, 3, 5):
        b_cands = [b for b in (6.00, 6.10, 6.25, 6.35, 6.50) if (f"sq_iso_L2r1_s{s}_Z_err", "uf", b) in kmc9]
        q3_bN.append(kmc9[(f"sq_iso_L2r1_s{s}_Z_err", "uf", b_cands[0])][0])
        q3_bD.append(ols_slope(b_cands, [kmc9[(f"sq_iso_L2r1_s{s}_Z_err", "uf", b)][1] for b in b_cands]))

    s_bar = [
        {"xs": w_bN,     "ys": w_bD,     "color": "#be123c", "dash": "",    "label": "3D Welded Single-Sector (6 levels: curves up)"},
        {"xs": q3_bN,    "ys": q3_bD,    "color": "#ea580c", "dash": "7,3", "label": "3D Welded Quantum (424k: doubles each cycle)"},
        {"xs": c_haah_N, "ys": c_haah_D, "color": "#7c3aed", "dash": "",    "label": "Haah 3D Cubic (straight log N, bends down)"},
        {"xs": c_bb_N,   "ys": c_bb_D,   "color": "#2563eb", "dash": "5,4", "label": "Scaled 2D Bivariate Bicycle (flat ≈ 5.4)"},
        {"xs": c_tor_N,  "ys": c_tor_D,  "color": "#4b5563", "dash": "3,3", "label": "2D Toric Code (flat ≤ 2.3)"},
    ]

    render_double(
        "fig_v15_1_classes",
        dict(xlim=(15.0, 580000.0), ylim=(0.02, 4.0e7),
             title="(a) Lifetime Growth Relative to Smallest Code",
             xlabel="Physical Qubit Count N (Log Scale)", ylabel="Relative Memory Lifetime",
             xticks=[(20,"20"), (200,"200"), (2000,"2k"), (20000,"20k"), (400000,"400k")],
             yticks=[(0.05,"0.05×"), (1.0,"1×"), (100.0,"10²×"), (10000.0,"10⁴×")],
             series=s_norm, xlog=True, ylog=True, leg_pos="tl",
             hlines=[(1.0, "#6b7280", "1× = No Gain From Growing Code Size", "br")]),
        dict(xlim=(1.6, 580000.0), ylim=(0.0, max(21.0, max(w_bD) * 1.32)),
             title="(b) Additive vs. Multiplicative Energy Barriers",
             xlabel="Physical Qubit Count N (Log Scale)", ylabel="Effective Energy Barrier",
             xticks=[(2,"2"), (30,"30"), (500,"500"), (10000,"10k"), (400000,"400k")],
             yticks=[(0,"0"), (5,"5"), (10,"10"), (15,"15"), (20,"20")],
             series=s_bar, xlog=True, ylog=False, leg_pos="tl"),
    )

    # --- 2. `fig_v15_2_validation.png` ---
    s_tor = []
    for b, col, is_z, lbl in [
        (2.00, "#dc2626", False, "Loop X-Errors (β = 2.0: grows exponentially)"),
        (1.75, "#f97316", False, "Loop X-Errors (β = 1.75: grows exponentially)"),
        (2.00, "#2563eb", True,  "Point Z-Errors (β = 2.0: shrinks as 1 / N)"),
    ]:
        xs, ys = [], []
        for L in [3, 4, 5, 6, 7]:
            rec = kmc6.get((f"toric3d_L{L}_Z_err", "uniform", "uf", b)) if is_z else (
                  kmc7.get((f"toric3d_L{L}_X_err", "uniform", "maj_uf", b)) or kmc6.get((f"toric3d_L{L}_X_err", "uniform", "sweep_uf", b)))
            if rec and rec[2] == 0:
                xs.append(L); ys.append(rec[1])
        s_tor.append({"xs": xs, "ys": ys, "color": col, "dash": "5,4" if is_z else "", "label": lbl})

    caha_groups = {
        11: ["caha_n11_code_100", "caha_n11_code_11", "caha_n11_code_113", "caha_n11_code_120", "caha_n11_code_128", "caha_n11_code_151"],
        13: ["caha_n13_code_102", "caha_n13_code_104", "caha_n13_code_106"],
        15: ["caha_n15_code_15",  "caha_n15_code_16",  "caha_n15_code_17"],
        17: ["caha_n17_code_1201","caha_n17_code_14",  "caha_n17_code_145"],
        19: ["caha_n19_code_129", "caha_n19_code_199", "caha_n19_code_236"],
    }
    s_caha = []
    for b, col, dsh, lbl in [
        (7.00, "#059669", "",    "Cooler Bath (β = 7.0: peaks at n = 13, turns over)"),
        (6.00, "#2563eb", "5,4", "Warmer Bath (β = 6.0: peaks at n = 11, declines)"),
    ]:
        xs, ys = [], []
        for n_val in [11, 13, 15, 17, 19]:
            vals = [kmc6[(cn + "_X_err", "uniform", "uf", b)][1]
                    for cn in caha_groups[n_val]
                    if (cn + "_X_err", "uniform", "uf", b) in kmc6 and kmc6[(cn + "_X_err", "uniform", "uf", b)][2] == 0]
            if vals:
                xs.append(n_val); ys.append(sum(vals) / len(vals))
        s_caha.append({"xs": xs, "ys": ys, "color": col, "dash": dsh, "label": lbl})

    render_double(
        "fig_v15_2_validation",
        dict(xlim=(2.8, 7.2), ylim=(0.3, 4e9),
             title="(a) 3D Toric Code Benchmark",
             xlabel="Lattice Size L (N = 3L³ Qubits)", ylabel="Mean Sector Lifetime",
             xticks=[(3,"3"), (4,"4"), (5,"5"), (6,"6"), (7,"7")],
             yticks=[(1e0,"1"), (1e2,"10²"), (1e4,"10⁴"), (1e6,"10⁶"), (1e8,"10⁸")],
             series=s_tor, xlog=False, ylog=True, leg_pos="tl"),
        dict(xlim=(10.6, 19.4), ylim=(6e2, 4e5),
             title="(b) Published Random Layer Code Turnover",
             xlabel="Layer Size n (2,200 to 11,000 Qubits)", ylabel="Ensemble Mean Lifetime",
             xticks=[(11,"11"), (13,"13"), (15,"15"), (17,"17"), (19,"19")],
             yticks=[(1e3,"10³"), (1e4,"10⁴"), (1e5,"10⁵")],
             series=s_caha, xlog=False, ylog=True, leg_pos="tl"),
    )

    # --- 3..7 ---
    render_fig3_architecture()
    render_fig4_threshold(rg9, kmc9)
    render_fig5_quantum(rg9, kmc9, q3_bD)
    render_fig6_euclidean_and_wires(kmc12, v7_best)
    render_fig7_compact_and_highrate(kmc12, kmc6)

    # --- 8. `fig_v15_8_cooling.png` ---
    ps = [0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.12]
    px_pct = [p * 100.0 for p in ps]
    s_heal = []
    for tag, col, dsh, lbl in [
        ("rottor_L2r1_s2_Z_err",  "#2563eb", "6,4", "Torus Z-Errors (98% healed at 1% noise)"),
        ("rottor_hyb21_s2_X_err", "#059669", "",    "Balanced Torus (89% healed at 1% noise)"),
        ("rottor_L2r1_s2_X_err",  "#dc2626", "",    "Torus X-Errors (76% healed at 1% noise)"),
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
        "fig_v15_8_cooling",
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
    print("All 8 V15 figures rendered cleanly!")


if __name__ == "__main__":
    main()
