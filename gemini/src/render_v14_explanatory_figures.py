#!/usr/bin/env python3
"""
render_v14_explanatory_figures.py
Renders 7 large, intuitive, visually self-explanatory figures (V14) designed for effortless reading:

  1. fig_v14_1_classes.png:      Code family comparison across 5 decades of N (lifetime & barrier scaling)
  2. fig_v14_2_validation.png:   Thermal bath validation on published 3D Toric & Random Layer Code benchmarks
  3. fig_v14_3_architecture.png: LARGE 2-row visual guide (1180x640):
                                 Top (a): 4-stage visual construction of a 2x2 welded step & 3-way seam splitting
                                 Bottom (b): Why a 2x2 cross shrinks coarse syndromes by 2/5
                                 Bottom (c): Why cutting one bridge arm opens a hole between sheets
  4. fig_v14_4_threshold_rg.png: LARGE contextualized 2-row zoom-out & threshold figure (1180x650):
                                 Top (a): Visual lattice illustration of cold pair cancellation vs warm chain growth
                                 Bottom (b): Live defect zoom-out inside one 322,159-qubit bath
                                 Bottom (c): Five separate code sizes (19 to 322,159q) crossing at fixed threshold
  5. fig_v14_5_quantum.png:      LARGE 2-row full quantum memory figure (1180x760):
                                 Top (a): Visual diagram of alternating odd X-bridges + even Z-bridges (1 cycle)
                                 Top (b): Live alternating (X, Z) sawtooth zoom-out on a 94,774-qubit torus
                                 Bottom (c): Three uniform 2x2 quantum levels (19 -> 2,911 -> 423,559q: 16,623x)
                                 Bottom (d): Boundaryless torus (94,774q): Fast local readout matches global decoding
  6. fig_v14_6_grid_wires.png:   Visual compact-vs-wire diagram + cubic-grid wire-stretching penalty (321x loss)
  7. fig_v14_7_cooling.png:      Autonomous storage self-healing vs 4-scale hot-state cooling time scaling
"""

import html
import math
import os
import subprocess
import sys

WS = "/Users/harshilavlani/Downloads/6.S980_gemini"
FIG_DIR = os.path.join(WS, "figures")
ART_DIR = "/Users/harshilavlani/.gemini/antigravity/brain/35299b0d-7439-4f8e-8617-0b646151a931"

sys.path.insert(0, os.path.join(WS, "src"))
from render_v13_unified_figures import draw_panel, interp_tcool, load_all, ols_slope, save_svg_png


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


def render_fig3_large_architecture():
    """
    LARGE 2-Row Visual Architecture Figure (1180 x 650):
      Row 1 (full width): Step-by-step visual construction of 3D welding (1 Square -> Subdivide & Double -> Weld Cross -> Seam Split)
      Row 2 Left:  Why a 2x2 four-arm cross shrinks syndromes by 2/5
      Row 2 Right: Why removing one bridge arm opens an unsealed hole between sheets
    """
    W, H = 1180, 650
    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<style>text { font-family: -apple-system, Helvetica, Arial, sans-serif; fill: #1f2937; }</style>',
    ]

    # --- ROW 1: Full-width card showing the 4-stage visual construction of one welding step ---
    elems.append('<rect x="18" y="14" width="1144" height="278" rx="12" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.6"/>')
    elems.append('<text x="590" y="42" text-anchor="middle" font-size="16.0" font-weight="700">'
                 '(a) How One Three-Dimensional Welding Step Multiplies the Energy Barrier</text>')

    # Stage 1: Seed 2D Surface-Code Face (x: 44..244)
    elems.append('<rect x="38" y="58" width="236" height="216" rx="9" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.3"/>')
    elems.append('<text x="156" y="80" text-anchor="middle" font-size="13.4" font-weight="700" fill="#1e3a8a">1. Parent 2D Square</text>')
    elems.append('<rect x="96" y="98" width="120" height="120" fill="#dbeafe" stroke="#2563eb" stroke-width="2.8"/>')
    for qx, qy in [(156, 98), (156, 218), (96, 158), (216, 158)]:
        elems.append(f'<circle cx="{qx}" cy="{qy}" r="5.5" fill="#1d4ed8" stroke="#ffffff" stroke-width="1.4"/>')
    elems.append('<text x="156" y="163" text-anchor="middle" font-size="12.5" font-weight="700" fill="#1e40af">Z-Check Face</text>')
    elems.append('<text x="156" y="250" text-anchor="middle" font-size="12.0" font-weight="600" fill="#475569">Qubits sit on edges;</text>')
    elems.append('<text x="156" y="266" text-anchor="middle" font-size="12.0" font-weight="600" fill="#b91c1c">1 defect walks free (ΔE = 0)</text>')

    # Arrow 1 -> 2
    elems.append('<line x1="280" y1="166" x2="306" y2="166" stroke="#475569" stroke-width="2.6"/>')
    elems.append('<polygon points="314,166 304,160 304,172" fill="#475569"/>')

    # Stage 2: Subdivide 2x2 & Duplicate Into Parallel Sheets (x: 320..576)
    elems.append('<rect x="320" y="58" width="256" height="216" rx="9" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.3"/>')
    elems.append('<text x="448" y="80" text-anchor="middle" font-size="13.4" font-weight="700" fill="#1e3a8a">2. Subdivide 2×2 &amp; Duplicate</text>')
    # Bottom sheet (Sheet 1)
    elems.append('<polygon points="350,218 430,182 546,182 466,218" fill="#dbeafe" fill-opacity="0.88" stroke="#2563eb" stroke-width="2.0"/>')
    elems.append('<line x1="390" y1="200" x2="506" y2="200" stroke="#60a5fa" stroke-width="1.5"/>')
    elems.append('<line x1="408" y1="218" x2="488" y2="182" stroke="#60a5fa" stroke-width="1.5"/>')
    elems.append('<text x="538" y="214" text-anchor="end" font-size="11.6" font-weight="700" fill="#1e40af">Sheet 1</text>')
    # Top sheet (Sheet 0)
    elems.append('<polygon points="350,146 430,110 546,110 466,146" fill="#dbeafe" fill-opacity="0.90" stroke="#2563eb" stroke-width="2.0"/>')
    elems.append('<line x1="390" y1="128" x2="506" y2="128" stroke="#60a5fa" stroke-width="1.5"/>')
    elems.append('<line x1="408" y1="146" x2="488" y2="110" stroke="#60a5fa" stroke-width="1.5"/>')
    elems.append('<text x="358" y="106" font-size="11.6" font-weight="700" fill="#1e40af">Sheet 0</text>')
    elems.append('<text x="448" y="250" text-anchor="middle" font-size="12.0" font-weight="600" fill="#475569">Two parallel 2×2 copies</text>')
    elems.append('<text x="448" y="266" text-anchor="middle" font-size="12.0" font-weight="600" fill="#475569">ready to be linked</text>')

    # Arrow 2 -> 3
    elems.append('<line x1="582" y1="166" x2="608" y2="166" stroke="#475569" stroke-width="2.6"/>')
    elems.append('<polygon points="616,166 606,160 606,172" fill="#475569"/>')

    # Stage 3: Weld Perpendicular 4-Arm Bridge Cross (x: 622..878)
    elems.append('<rect x="622" y="58" width="256" height="216" rx="9" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.3"/>')
    elems.append('<text x="750" y="80" text-anchor="middle" font-size="13.4" font-weight="700" fill="#b45309">3. Weld 4-Arm Bridge Cross</text>')
    # Sheet 1
    elems.append('<polygon points="652,218 732,182 848,182 768,218" fill="#dbeafe" fill-opacity="0.72" stroke="#2563eb" stroke-width="1.8"/>')
    # Two perpendicular yellow bridge planes intersecting at center!
    elems.append('<polygon points="710,218 790,182 790,110 710,146" fill="#fde68a" fill-opacity="0.92" stroke="#d97706" stroke-width="2.2"/>')
    elems.append('<polygon points="692,200 808,200 808,128 692,128" fill="#fcd34d" fill-opacity="0.82" stroke="#d97706" stroke-width="2.2"/>')
    # Sheet 0 on top with bold crimson 3-way cross seams
    elems.append('<polygon points="652,146 732,110 848,110 768,146" fill="#dbeafe" fill-opacity="0.78" stroke="#2563eb" stroke-width="1.8"/>')
    elems.append('<line x1="692" y1="128" x2="808" y2="128" stroke="#dc2626" stroke-width="3.8"/>')
    elems.append('<line x1="710" y1="146" x2="790" y2="110" stroke="#dc2626" stroke-width="3.8"/>')
    elems.append('<text x="750" y="250" text-anchor="middle" font-size="12.0" font-weight="700" fill="#b91c1c">Red lines = 3-way seams</text>')
    elems.append('<text x="750" y="266" text-anchor="middle" font-size="12.0" font-weight="600" fill="#475569">(3 faces meet at 1 edge)</text>')

    # Arrow 3 -> 4
    elems.append('<line x1="884" y1="166" x2="910" y2="166" stroke="#475569" stroke-width="2.6"/>')
    elems.append('<polygon points="918,166 908,160 908,172" fill="#475569"/>')

    # Stage 4: Cross-Section of a 3-Way Seam Splitting 1 Defect -> 2 Defects (x: 924..1144)
    elems.append('<rect x="924" y="58" width="220" height="216" rx="9" fill="#ecfdf5" stroke="#a7f3d0" stroke-width="1.4"/>')
    elems.append('<text x="1034" y="80" text-anchor="middle" font-size="13.4" font-weight="700" fill="#065f46">4. Seam Splits 1 → 2</text>')
    # Y-junction / T-junction cross-section showing Left Sheet, Right Sheet, Downward Bridge meeting at central seam qubit!
    elems.append('<line x1="952" y1="138" x2="1034" y2="138" stroke="#2563eb" stroke-width="5.0"/>')
    elems.append('<line x1="1034" y1="138" x2="1116" y2="138" stroke="#2563eb" stroke-width="5.0"/>')
    elems.append('<line x1="1034" y1="138" x2="1034" y2="216" stroke="#d97706" stroke-width="5.0"/>')
    elems.append('<circle cx="1034" cy="138" r="8.5" fill="#1e293b" stroke="#ffffff" stroke-width="2.0"/>')
    # Incoming 1 defect on left -> splits into 2 defects (right + bridge)!
    elems.append('<circle cx="978" cy="122" r="9.0" fill="#dc2626" stroke="#ffffff" stroke-width="1.6"/>')
    elems.append('<circle cx="1090" cy="122" r="9.0" fill="#dc2626" stroke="#ffffff" stroke-width="1.6"/>')
    elems.append('<circle cx="1054" cy="192" r="9.0" fill="#dc2626" stroke="#ffffff" stroke-width="1.6"/>')
    elems.append('<path d="M 990,120 Q 1034,102 1078,120" fill="none" stroke="#991b1b" stroke-width="2.2" stroke-dasharray="4,3"/>')
    elems.append('<path d="M 988,128 Q 1020,164 1044,186" fill="none" stroke="#991b1b" stroke-width="2.2" stroke-dasharray="4,3"/>')
    elems.append('<text x="1034" y="250" text-anchor="middle" font-size="12.2" font-weight="700" fill="#065f46">Crossing costs +ΔE energy</text>')
    elems.append('<text x="1034" y="266" text-anchor="middle" font-size="12.0" font-weight="600" fill="#047857">→ Multiplies barrier each scale</text>')

    # --- ROW 2 LEFT: Large Panel (b) Why a 2x2 Cross Shrinks Syndromes by 2/5 ---
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

    # Right-side explanation inside Panel (b)
    elems.append('<rect x="304" y="368" width="254" height="214" rx="9" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.4"/>')
    elems.append('<text x="431" y="398" text-anchor="middle" font-size="13.2" font-weight="700" fill="#1e293b">Each Outer Wall Has 5 Parts:</text>')
    elems.append('<text x="431" y="420" text-anchor="middle" font-size="12.5" font-weight="600" fill="#2563eb">• 2 sub-edges on Sheet 0</text>')
    elems.append('<text x="431" y="440" text-anchor="middle" font-size="12.5" font-weight="600" fill="#475569">• 2 sub-edges on Sheet 1</text>')
    elems.append('<text x="431" y="460" text-anchor="middle" font-size="12.5" font-weight="600" fill="#b45309">• 1 sub-edge on Bridge</text>')
    elems.append('<line x1="324" y1="478" x2="538" y2="478" stroke="#e2e8f0" stroke-width="1.4"/>')
    elems.append('<text x="431" y="502" text-anchor="middle" font-size="12.6" font-weight="600" fill="#374151">Flipping Sheet 0 = 2/5 on 4 walls</text>')
    elems.append('<text x="431" y="528" text-anchor="middle" font-size="14.0" font-weight="700" fill="#047857">Coarse / Fine = (8/5) ÷ 4 = 2/5</text>')
    elems.append('<text x="431" y="558" text-anchor="middle" font-size="12.5" font-weight="700" fill="#1e293b">174× smaller cell than scale-8</text>')

    # --- ROW 2 RIGHT: Large Panel (c) Why Cutting One Bridge Arm Fails ---
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
    elems.append('<text x="1015" y="400" text-anchor="middle" font-size="13.2" font-weight="700" fill="#991b1b">What Goes Wrong:</text>')
    elems.append('<text x="1015" y="428" text-anchor="middle" font-size="12.4" font-weight="600" fill="#374151">A 3-arm bridge ends mid-cell,</text>')
    elems.append('<text x="1015" y="448" text-anchor="middle" font-size="12.4" font-weight="600" fill="#374151">leaving a gap between Sheet 0</text>')
    elems.append('<text x="1015" y="468" text-anchor="middle" font-size="12.4" font-weight="600" fill="#374151">and Sheet 1 with no check.</text>')
    elems.append('<line x1="908" y1="486" x2="1122" y2="486" stroke="#e2e8f0" stroke-width="1.4"/>')
    elems.append('<text x="1015" y="514" text-anchor="middle" font-size="13.4" font-weight="700" fill="#b91c1c">Spurious Qubits: 2 → 1,004</text>')
    elems.append('<text x="1015" y="540" text-anchor="middle" font-size="12.4" font-weight="600" fill="#374151">Lifetime collapses to ~ 0;</text>')
    elems.append('<text x="1015" y="560" text-anchor="middle" font-size="12.5" font-weight="700" fill="#1e293b">4-arm cross is minimal</text>')

    save_svg_png("fig_v14_3_architecture", elems, W, H)


def draw_mini_grid(elems, gx, gy, n_cells, cell_px, red_faces, stroke_c="#64748b", sub2=False):
    tot = n_cells * cell_px
    elems.append(f'<rect x="{gx}" y="{gy}" width="{tot}" height="{tot}" fill="#ffffff" stroke="{stroke_c}" stroke-width="2.0"/>')
    for (r, c) in red_faces:
        elems.append(f'<rect x="{gx + c*cell_px + 1.5:.1f}" y="{gy + r*cell_px + 1.5:.1f}" '
                     f'width="{cell_px - 3.0:.1f}" height="{cell_px - 3.0:.1f}" fill="#ef4444" fill-opacity="0.86"/>')
    for i in range(1, n_cells):
        lw = "2.2" if (sub2 and i % 2 == 0) else "1.0"
        lc = "#1e293b" if (sub2 and i % 2 == 0) else "#cbd5e1"
        elems.append(f'<line x1="{gx + i*cell_px}" y1="{gy}" x2="{gx + i*cell_px}" y2="{gy+tot}" stroke="{lc}" stroke-width="{lw}"/>')
        elems.append(f'<line x1="{gx}" y1="{gy + i*cell_px}" x2="{gx+tot}" y2="{gy + i*cell_px}" stroke="{lc}" stroke-width="{lw}"/>')
    elems.append(f'<rect x="{gx}" y="{gy}" width="{tot}" height="{tot}" fill="none" stroke="#1e293b" stroke-width="2.2"/>')


def render_fig4_intuitive_zoomout(rg, kmc):
    """
    LARGE Self-Explanatory 2-Row Figure (1180 x 660):
      Top Row (a): Real visual lattice grids showing WHY zooming out cancels local pairs below Tc vs scrambles above Tc!
      Bottom-Left (b): Live zoom-out defect curves inside one 322,159-qubit bath
      Bottom-Right (c): Five separate code sizes (19 to 322,159 qubits) crossing at the same fixed threshold
    """
    W, H = 1180, 660
    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<style>text { font-family: -apple-system, Helvetica, Arial, sans-serif; fill: #1f2937; }</style>',
    ]

    # --- TOP VISUAL GUIDE (a): Cold Pair Cancellation vs Warm Chain Growth ---
    elems.append('<rect x="18" y="12" width="1144" height="202" rx="12" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.6"/>')
    elems.append('<text x="590" y="38" text-anchor="middle" font-size="15.6" font-weight="700">'
                 '(a) What "Zooming Out" Across 2×2 Cells Means Visually (No Global Decoder)</text>')

    # Left sub-box: Cold Bath (Below Tc) -> Pairs inside 2x2 cells cancel to Clean Coarse Grid!
    elems.append('<rect x="36" y="50" width="536" height="150" rx="9" fill="#ecfdf5" stroke="#a7f3d0" stroke-width="1.4"/>')
    elems.append('<text x="304" y="72" text-anchor="middle" font-size="13.6" font-weight="700" fill="#065f46">'
                 'Cold Bath (Below Tc): Small Pairs Cancel Inside 2×2 Cells</text>')
    # Fine 4x4 grid (with bold 2x2 cell borders) showing 2 even pairs inside cells:
    draw_mini_grid(elems, 76, 86, 4, 24, [(0, 0), (0, 1), (2, 2), (3, 2)], sub2=True)
    elems.append('<text x="124" y="194" text-anchor="middle" font-size="11.5" font-weight="600" fill="#065f46">Fine Lattice</text>')
    elems.append('<line x1="192" y1="134" x2="258" y2="134" stroke="#047857" stroke-width="2.8"/>')
    elems.append('<polygon points="266,134 256,128 256,140" fill="#047857"/>')
    elems.append('<text x="228" y="124" text-anchor="middle" font-size="11.4" font-weight="700" fill="#047857">Group 2×2</text>')
    # Coarse 2x2 grid: 0 red faces (100% clean!)
    draw_mini_grid(elems, 284, 98, 2, 36, [], sub2=False)
    elems.append('<text x="320" y="194" text-anchor="middle" font-size="11.5" font-weight="700" fill="#065f46">Clean Coarse Grid!</text>')
    elems.append('<text x="462" y="124" text-anchor="middle" font-size="12.4" font-weight="700" fill="#065f46">Defects shrink</text>')
    elems.append('<text x="462" y="144" text-anchor="middle" font-size="12.4" font-weight="700" fill="#065f46">to 0% as we</text>')
    elems.append('<text x="462" y="164" text-anchor="middle" font-size="12.4" font-weight="700" fill="#065f46">zoom out ↓</text>')

    # Right sub-box: Warm Bath (Above Tc) -> Chains cross 2x2 walls and multiply!
    elems.append('<rect x="608" y="50" width="536" height="150" rx="9" fill="#fff1f2" stroke="#fecdd3" stroke-width="1.4"/>')
    elems.append('<text x="876" y="72" text-anchor="middle" font-size="13.6" font-weight="700" fill="#9f1239">'
                 'Warm Bath (Above Tc): Error Chains Cross Cell Walls &amp; Multiply</text>')
    draw_mini_grid(elems, 648, 86, 4, 24, [(0, 1), (1, 2), (2, 0), (2, 3), (3, 1)], sub2=True)
    elems.append('<text x="696" y="194" text-anchor="middle" font-size="11.5" font-weight="600" fill="#9f1239">Fine Lattice</text>')
    elems.append('<line x1="764" y1="134" x2="830" y2="134" stroke="#be123c" stroke-width="2.8"/>')
    elems.append('<polygon points="838,134 828,128 828,140" fill="#be123c"/>')
    elems.append('<text x="800" y="124" text-anchor="middle" font-size="11.4" font-weight="700" fill="#be123c">Group 2×2</text>')
    draw_mini_grid(elems, 856, 98, 2, 36, [(0, 0), (0, 1), (1, 0), (1, 1)], sub2=False)
    elems.append('<text x="892" y="194" text-anchor="middle" font-size="11.5" font-weight="700" fill="#9f1239">Scrambled Grid!</text>')
    elems.append('<text x="1036" y="124" text-anchor="middle" font-size="12.4" font-weight="700" fill="#9f1239">Defects grow</text>')
    elems.append('<text x="1036" y="144" text-anchor="middle" font-size="12.4" font-weight="700" fill="#9f1239">toward 50%</text>')
    elems.append('<text x="1036" y="164" text-anchor="middle" font-size="12.4" font-weight="700" fill="#9f1239">random noise ↑</text>')

    # --- BOTTOM LEFT (b): Live 322,159-Qubit Syndrome Zoom-Out ---
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

    # --- BOTTOM RIGHT (c): 5 Separate Code Sizes Converging Onto Same Fixed Threshold ---
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
    save_svg_png("fig_v14_4_threshold_rg", elems, W, H)


def render_fig5_full_quantum(rg, kmc, q3_bD):
    """
    LARGE 2-Row Full Quantum Self-Correction Figure (1180 x 790):
      Top-Left (a):  Visual schematic of Alternating Odd X-Bridges & Even Z-Bridges (Why a full cycle protects both X and Z)
      Top-Right (b): Live Alternating (X, Z) Sawtooth Zoom-Out on 94,774 Torus Qubits (explaining β ≈ 6.0 threshold!)
      Bot-Left (c):  Three Uniform 2x2 Quantum Levels (19 -> 2,911 -> 423,559 qubits: 16,623x gain)
      Bot-Right (d): Boundaryless Torus (94,774q): Fast Linear-Time Local Readout Matches Global Decoding
    """
    W, H = 1180, 836
    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<style>text { font-family: -apple-system, Helvetica, Arial, sans-serif; fill: #1f2937; }</style>',
    ]

    # --- Panel (a): Visual schematic of One Full Two-Step (X, Z) Quantum Cycle ---
    x0, y0, pw, ph = 42, 42, 516, 326
    elems.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" rx="10" fill="#f8fafc" stroke="#374151" stroke-width="1.5"/>')
    elems.append(f'<text x="{x0+pw/2:.1f}" y="{y0-13:.1f}" text-anchor="middle" font-size="14.8" font-weight="700">'
                 '(a) Alternating X and Z Bridges Protect Both Sectors</text>')

    # Odd step box (Blue X-confining cross)
    elems.append(f'<rect x="{x0+20}" y="{y0+22}" width="214" height="196" rx="9" fill="#eff6ff" stroke="#93c5fd" stroke-width="1.5"/>')
    elems.append(f'<text x="{x0+127}" y="{y0+45}" text-anchor="middle" font-size="13.2" font-weight="700" fill="#1e40af">Odd Step: X-Bridges</text>')
    elems.append(f'<rect x="{x0+75}" y="{y0+62}" width="104" height="104" fill="#ffffff" stroke="#2563eb" stroke-width="2.4"/>')
    elems.append(f'<line x1="{x0+127}" y1="{y0+62}" x2="{x0+127}" y2="{y0+166}" stroke="#2563eb" stroke-width="5.2"/>')
    elems.append(f'<line x1="{x0+75}" y1="{y0+114}" x2="{x0+179}" y2="{y0+114}" stroke="#2563eb" stroke-width="5.2"/>')
    elems.append(f'<text x="{x0+127}" y="{y0+188}" text-anchor="middle" font-size="11.8" font-weight="700" fill="#047857">X errors shrink × 2/5</text>')
    elems.append(f'<text x="{x0+127}" y="{y0+206}" text-anchor="middle" font-size="11.5" font-weight="600" fill="#64748b">(Z sector waits: × 1)</text>')

    # Arrow Odd -> Even
    elems.append(f'<line x1="{x0+240}" y1="{y0+120}" x2="{x0+270}" y2="{y0+120}" stroke="#475569" stroke-width="2.6"/>')
    elems.append(f'<polygon points="{x0+278},{y0+120} {x0+268},{y0+114} {x0+268},{y0+126}" fill="#475569"/>')

    # Even step box (Red Z-confining cross)
    elems.append(f'<rect x="{x0+282}" y="{y0+22}" width="214" height="196" rx="9" fill="#fef2f2" stroke="#fca5a5" stroke-width="1.5"/>')
    elems.append(f'<text x="{x0+389}" y="{y0+45}" text-anchor="middle" font-size="13.2" font-weight="700" fill="#991b1b">Even Step: Z-Bridges</text>')
    elems.append(f'<rect x="{x0+337}" y="{y0+62}" width="104" height="104" fill="#ffffff" stroke="#dc2626" stroke-width="2.4"/>')
    elems.append(f'<line x1="{x0+389}" y1="{y0+62}" x2="{x0+389}" y2="{y0+166}" stroke="#dc2626" stroke-width="5.2"/>')
    elems.append(f'<line x1="{x0+337}" y1="{y0+114}" x2="{x0+441}" y2="{y0+114}" stroke="#dc2626" stroke-width="5.2"/>')
    elems.append(f'<text x="{x0+389}" y="{y0+188}" text-anchor="middle" font-size="11.8" font-weight="700" fill="#047857">Z errors shrink × 2/5</text>')
    elems.append(f'<text x="{x0+389}" y="{y0+206}" text-anchor="middle" font-size="11.5" font-weight="600" fill="#64748b">(X sector waits: × 1)</text>')

    # Summary banner at bottom of Panel (a)
    elems.append(f'<rect x="{x0+20}" y="{y0+234}" width="476" height="76" rx="8" fill="#ecfdf5" stroke="#a7f3d0" stroke-width="1.4"/>')
    elems.append(f'<text x="{x0+pw/2:.1f}" y="{y0+260}" text-anchor="middle" font-size="13.2" font-weight="700" fill="#065f46">'
                 'Net Effect Over Every 2-Step Quantum Cycle:</text>')
    elems.append(f'<text x="{x0+pw/2:.1f}" y="{y0+283}" text-anchor="middle" font-size="12.4" font-weight="600" fill="#1f2937">'
                 'Both X and Z syndromes contract by 2/5; waiting steps</text>')
    elems.append(f'<text x="{x0+pw/2:.1f}" y="{y0+301}" text-anchor="middle" font-size="12.4" font-weight="600" fill="#1f2937">'
                 'create the sawtooth flow in (b) and shift Tc to β ≈ 6.0</text>')

    # --- Panel (b): Live Alternating (X, Z) Sawtooth Zoom-Out on 94,774 Torus Qubits ---
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
                # Plot drop to bottom floor (2.5e-5) to show extinction at Seed!
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

    # --- Panel (c): Three Uniform 2x2 Quantum Levels (19 -> 2,911 -> 423,559 qubits) ---
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

    # --- Panel (d): Boundaryless Torus (94,774q): Local Linear-Time Readout vs Global Decoding ---
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
    save_svg_png("fig_v14_5_quantum", elems, W, H)


def main():
    kmc4, kmc6, kmc7, ffs7, cool7, kmc9, rg9, cool11 = load_all()

    def v7_best(tag, jm, dec, b, min_reps=3):
        if (tag, jm, dec, b) in kmc7 and kmc7[(tag, jm, dec, b)][2] == 0:
            return kmc7[(tag, jm, dec, b)][1]
        if (tag, jm, dec, b) in ffs7 and ffs7[(tag, jm, dec, b)][2] >= min_reps:
            return ffs7[(tag, jm, dec, b)][1]
        return None

    # --- 1. `fig_v14_1_classes.png`: Code-Family Comparison ---
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
        "fig_v14_1_classes",
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

    # --- 2. `fig_v14_2_validation.png`: Thermal Bath Validation (NOW FIGURE 2!) ---
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
        "fig_v14_2_validation",
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

    # --- 3. `fig_v14_3_architecture.png`: Large 2-Row Welding Construction & Minimal Cell ---
    render_fig3_large_architecture()

    # --- 4. `fig_v14_4_threshold_rg.png`: Intuitive Visual Zoom-Out + 5-Scale Threshold ---
    render_fig4_intuitive_zoomout(rg9, kmc9)

    # --- 5. `fig_v14_5_quantum.png`: 2-Row Alternating Cycle, Sawtooth, 424k Levels, & Local Readout ---
    render_fig5_full_quantum(rg9, kmc9, q3_bD)

    # --- 6. `fig_v14_6_grid_wires.png`: Cubic-Grid Wire Penalty ---
    from render_v13_unified_figures import render_fig6_wires_visual_and_plot
    render_fig6_wires_visual_and_plot(v7_best)
    # Copy v13_6 to v14_6 cleanly:
    for ext in ("png", "svg"):
        with open(os.path.join(FIG_DIR, f"fig_v13_6_grid_wires.{ext}"), "rb") as fi:
            raw_b = fi.read()
        with open(os.path.join(FIG_DIR, f"fig_v14_6_grid_wires.{ext}"), "wb") as fo:
            fo.write(raw_b)
    with open(os.path.join(FIG_DIR, "fig_v14_6_grid_wires.png"), "rb") as fi, open(os.path.join(ART_DIR, "fig_v14_6_grid_wires.png"), "wb") as fo:
        fo.write(fi.read())

    # --- 7. `fig_v14_7_cooling.png`: Storage Self-Healing vs Multi-Scale Hot-State Cooling ---
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
        "fig_v14_7_cooling",
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


if __name__ == "__main__":
    main()
