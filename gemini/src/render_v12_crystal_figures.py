#!/usr/bin/env python3
"""
render_v12_crystal_figures.py
Renders 7 large, crystal-clear, self-explanatory figures (V12) unified around the
provably contractive 2x2 four-arm cross cell (zero variant clutter, zero jargon):

  1. fig_v12_1_classes.png:         Code family comparison across 5 decades of N (lifetime & barrier scaling)
  2. fig_v12_2_architecture.png:    LARGE 3-panel visual explanation of 3D welding, analytic 2/5 contraction, & holes
  3. fig_v12_3_validation.png:      Simulator validation on published 3D Toric & Random Layer Code benchmarks
  4. fig_v12_4_threshold_rg.png:    Contextualized 2-panel + top-schematic figure:
                                    (a) Zooming out on syndromes in one 322,159-qubit bath (shaded phases)
                                    (b) Memory lifetime crossings across 4 scales settling at fixed beta ≈ 1.9
  5. fig_v12_5_quantum_scaling.png: Full (X, Z) quantum memory scaling on the uniform 2x2 cell:
                                    (a) 3-level quantum ladder (up to 190k / 424k qubits: >10,000x gain)
                                    (b) Two full cycles on planar (47,387q) & boundaryless torus (94,774q)
  6. fig_v12_6_grid_wires.png:      Visual compact-vs-wire diagram + cubic-grid wire-stretching penalty (321x loss)
  7. fig_v12_7_cooling.png:         Zero-measurement storage self-healing vs gradual cooling from hot states
"""

import html
import math
import os
import subprocess

WS = "/Users/harshilavlani/Downloads/6.S980_gemini"
FIG_DIR = os.path.join(WS, "figures")
ART_DIR = "/Users/harshilavlani/.gemini/antigravity/brain/35299b0d-7439-4f8e-8617-0b646151a931"


def parse_kv(text):
    d = {}
    for tok in text.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            d[k] = v
    return d


def ols_slope(xs, ys):
    pts = [(x, math.log(y)) for x, y in zip(xs, ys) if y is not None and y > 0]
    if len(pts) < 2:
        return None
    mx = sum(x for x, _ in pts) / len(pts)
    my = sum(y for _, y in pts) / len(pts)
    den = sum((x - mx) ** 2 for x, _ in pts)
    return sum((x - mx) * (y - my) for x, y in pts) / den if den > 1e-12 else None


def save_svg_png(nm, elems, W, H):
    svg_p = os.path.join(FIG_DIR, f"{nm}.svg")
    with open(svg_p, "w") as f:
        f.write("\n".join(elems) + "\n</svg>\n")
    out_png = os.path.join(FIG_DIR, f"{nm}.png")
    subprocess.run(["/opt/homebrew/bin/rsvg-convert", "-z", "2", "-f", "png", "-o", out_png, svg_p], check=True)
    with open(out_png, "rb") as fi, open(os.path.join(ART_DIR, f"{nm}.png"), "wb") as fo:
        fo.write(fi.read())
    print(f"Rendered {nm}.png ({2*W}x{2*H})")


def draw_panel(elems, x0, y0, pw, ph, xlim, ylim, title, xlabel, ylabel,
               xticks, yticks, series, xlog=False, ylog=True, leg_pos="tl",
               hlines=None, vlines=None, varrows=None, notes=None,
               ybands=None, xbands=None, y_off=58):
    elems.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="#fafafa" stroke="#374151" stroke-width="1.5"/>')

    def tx(v):
        if xlog:
            return x0 + (math.log10(v) - math.log10(xlim[0])) / (math.log10(xlim[1]) - math.log10(xlim[0])) * pw
        return x0 + (v - xlim[0]) / (xlim[1] - xlim[0]) * pw

    def ty(v):
        vc = max(ylim[0], min(ylim[1], v))
        if ylog:
            return y0 + ph - (math.log10(vc) - math.log10(ylim[0])) / (math.log10(ylim[1]) - math.log10(ylim[0])) * ph
        return y0 + ph - (vc - ylim[0]) / (ylim[1] - ylim[0]) * ph

    if ybands:
        for y_lo, y_hi, fill_c in ybands:
            py_top = ty(y_hi)
            py_bot = ty(y_lo)
            elems.append(f'<rect x="{x0+0.8:.1f}" y="{py_top:.1f}" width="{pw-1.6:.1f}" height="{max(0.0, py_bot-py_top):.1f}" fill="{fill_c}"/>')

    if xbands:
        for x_lo, x_hi, fill_c in xbands:
            px_l = tx(x_lo)
            px_r = tx(x_hi)
            elems.append(f'<rect x="{px_l:.1f}" y="{y0+0.8:.1f}" width="{max(0.0, px_r-px_l):.1f}" height="{ph-1.6:.1f}" fill="{fill_c}"/>')

    elems.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="none" stroke="#374151" stroke-width="1.5"/>')
    elems.append(f'<text x="{x0+pw/2:.1f}" y="{y0-13:.1f}" text-anchor="middle" font-size="14.8" font-weight="700">{html.escape(title)}</text>')
    elems.append(f'<text x="{x0+pw/2:.1f}" y="{y0+ph+41:.1f}" text-anchor="middle" font-size="13.5" font-weight="600">{html.escape(xlabel)}</text>')
    elems.append(f'<text x="{x0-y_off:.1f}" y="{y0+ph/2:.1f}" text-anchor="middle" font-size="13.5" font-weight="600" '
                 f'transform="rotate(-90 {x0-y_off:.1f} {y0+ph/2:.1f})">{html.escape(ylabel)}</text>')

    for xv, xlbl in xticks:
        px = tx(xv)
        elems.append(f'<line x1="{px:.1f}" y1="{y0}" x2="{px:.1f}" y2="{y0+ph}" stroke="#e5e7eb" stroke-width="1"/>')
        elems.append(f'<line x1="{px:.1f}" y1="{y0+ph}" x2="{px:.1f}" y2="{y0+ph+5}" stroke="#374151" stroke-width="1.2"/>')
        elems.append(f'<text x="{px:.1f}" y="{y0+ph+20}" text-anchor="middle" font-size="12.1">{html.escape(xlbl)}</text>')

    for yv, ylbl in yticks:
        py = ty(yv)
        elems.append(f'<line x1="{x0}" y1="{py:.1f}" x2="{x0+pw}" y2="{py:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        elems.append(f'<line x1="{x0-5}" y1="{py:.1f}" x2="{x0}" y2="{py:.1f}" stroke="#374151" stroke-width="1.2"/>')
        elems.append(f'<text x="{x0-8}" y="{py+4.5:.1f}" text-anchor="end" font-size="12.1">{html.escape(ylbl)}</text>')

    if hlines:
        for yv, col, lbl, halign in hlines:
            py = ty(yv)
            elems.append(f'<line x1="{x0}" y1="{py:.1f}" x2="{x0+pw}" y2="{py:.1f}" stroke="{col}" stroke-width="1.6" stroke-dasharray="6,4"/>')
            hx = x0 + 10 if "l" in halign else x0 + pw - 10
            anch = "start" if "l" in halign else "end"
            hy = py + 15.5 if "b" in halign else py - 6.0
            elems.append(f'<text x="{hx:.1f}" y="{hy:.1f}" text-anchor="{anch}" font-size="11.6" font-weight="600" fill="{col}">{html.escape(lbl)}</text>')

    if vlines:
        for xv, col, lbl, valign in vlines:
            px = tx(xv)
            elems.append(f'<line x1="{px:.1f}" y1="{y0}" x2="{px:.1f}" y2="{y0+ph}" stroke="{col}" stroke-width="1.7" stroke-dasharray="5,4"/>')
            dx = 7.5 if "r" in valign else -7.5
            anch = "start" if "r" in valign else "end"
            vy = y0 + ph - 14.0 if "b" in valign else y0 + 22.0
            elems.append(f'<text x="{px+dx:.1f}" y="{vy:.1f}" text-anchor="{anch}" font-size="11.7" font-weight="700" fill="{col}">{html.escape(lbl)}</text>')

    for s in series:
        pts = [(tx(x), ty(y)) for x, y in zip(s["xs"], s["ys"]) if y is not None and xlim[0] <= x <= xlim[1]]
        if not pts:
            continue
        d_attr = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
        sw = s.get("lw", 3.0)
        pstr = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        if len(pts) >= 2:
            elems.append(f'<polyline fill="none" stroke="{s["color"]}" stroke-width="{sw}"{d_attr} points="{pstr}"/>')
        for px, py in pts:
            elems.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.8" fill="{s["color"]}" stroke="#ffffff" stroke-width="1.2"/>')

    if varrows:
        for item in varrows:
            xv, y_lo, y_hi, lbl, col, side = item[:6]
            y_txt = item[6] if len(item) > 6 else None
            px = tx(xv)
            py1, py2 = ty(y_hi), ty(y_lo)
            ty_lbl = ty(y_txt) + 4.0 if y_txt is not None else (py1 + py2) / 2.0 + 4.0
            elems.append(f'<line x1="{px:.1f}" y1="{py1+8:.1f}" x2="{px:.1f}" y2="{py2-8:.1f}" stroke="{col}" stroke-width="2.1" stroke-dasharray="3,3"/>')
            elems.append(f'<polygon points="{px:.1f},{py1+2:.1f} {px-4.5:.1f},{py1+10:.1f} {px+4.5:.1f},{py1+10:.1f}" fill="{col}"/>')
            elems.append(f'<polygon points="{px:.1f},{py2-2:.1f} {px-4.5:.1f},{py2-10:.1f} {px+4.5:.1f},{py2-10:.1f}" fill="{col}"/>')
            dx = -8.5 if side == "l" else 8.5
            anch = "end" if side == "l" else "start"
            elems.append(f'<text x="{px+dx:.1f}" y="{ty_lbl:.1f}" text-anchor="{anch}" font-size="12.4" font-weight="700" fill="{col}">{html.escape(lbl)}</text>')

    if notes:
        for xv, yv, txt, col, anch in notes:
            elems.append(f'<text x="{tx(xv):.1f}" y="{ty(yv):.1f}" text-anchor="{anch}" font-size="11.8" font-weight="700" fill="{col}">{html.escape(txt)}</text>')

    leg_series = [s for s in series if s.get("label")]
    row_h = 19.8
    lh = row_h * len(leg_series) + 10
    lw = min(pw - 18.0, max(195.0, max(len(s["label"]) for s in leg_series) * 6.55 + 46.0))
    lx = x0 + 10 if "l" in leg_pos else x0 + pw - lw - 10
    ly = y0 + 10 if "t" in leg_pos else y0 + ph - lh - 10
    elems.append(f'<rect x="{lx:.1f}" y="{ly:.1f}" width="{lw:.1f}" height="{lh:.1f}" rx="6" fill="#ffffff" fill-opacity="0.96" stroke="#d1d5db" stroke-width="1.1"/>')
    for i, s in enumerate(leg_series):
        iy = ly + 15.8 + i * row_h
        d_attr = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
        elems.append(f'<line x1="{lx+8:.1f}" y1="{iy-4.0:.1f}" x2="{lx+29:.1f}" y2="{iy-4.0:.1f}" stroke="{s["color"]}" stroke-width="2.7"{d_attr}/>')
        elems.append(f'<circle cx="{lx+18.5:.1f}" cy="{iy-4.0:.1f}" r="3.9" fill="{s["color"]}"/>')
        elems.append(f'<text x="{lx+36:.1f}" y="{iy:.1f}" font-size="11.6">{html.escape(s["label"])}</text>')


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


def render_fig2_large_architecture():
    """
    LARGE 3-panel architectural diagram (1180 x 510):
      (a) Welding Splits Errors at 3-Way Seams (flat sheet ΔE=0 walk vs 3-way seam 1->2 split)
      (b) Analytic 2/5 Contraction on a 2x2 Cross (5 wall sub-edges vs 4 interior seam checks)
      (c) Snipping One Arm Opens a Sheet Hole (buttonhole gap spawning unwanted logical qubits)
    """
    W, H = 1180, 505
    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<style>text { font-family: -apple-system, Helvetica, Arial, sans-serif; fill: #1f2937; }</style>',
    ]
    cards = [
        (16,  14, 368, 476, "(a) Welding Splits Errors at 3-Way Seams"),
        (406, 14, 368, 476, "(b) Why a 2×2 Cross Contracts by 2 / 5"),
        (796, 14, 368, 476, "(c) Cutting One Arm Opens a Sheet Hole"),
    ]
    for cx, cy, cw, ch, ctitle in cards:
        elems.append(f'<rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" rx="12" fill="#f9fafb" stroke="#cbd5e1" stroke-width="1.6"/>')
        elems.append(f'<text x="{cx+cw/2:.1f}" y="{cy+32}" text-anchor="middle" font-size="15.2" font-weight="700">{html.escape(ctitle)}</text>')

    # --- Card (a): Large 3D Isometric Doubled Sheets + Perpendicular Bridge ---
    # Bottom sheet (Sheet 1)
    elems.append('<polygon points="44,322 174,264 356,264 226,322" fill="#dbeafe" fill-opacity="0.88" stroke="#2563eb" stroke-width="2.2"/>')
    elems.append('<line x1="109" y1="293" x2="291" y2="293" stroke="#93c5fd" stroke-width="1.4"/>')
    elems.append('<line x1="135" y1="322" x2="265" y2="264" stroke="#93c5fd" stroke-width="1.4"/>')
    elems.append('<text x="348" y="284" text-anchor="end" font-size="12.2" font-weight="700" fill="#1e40af">Sheet 1</text>')

    # Bridge sheet (vertical yellow plane connecting Sheet 0 and Sheet 1 along central seam)
    elems.append('<polygon points="135,293 265,235 265,115 135,173" fill="#fde68a" fill-opacity="0.93" stroke="#d97706" stroke-width="2.4"/>')
    elems.append('<line x1="200" y1="264" x2="200" y2="144" stroke="#f59e0b" stroke-width="1.5"/>')
    elems.append('<text x="206" y="224" font-size="12.4" font-weight="700" fill="#b45309">Bridge</text>')

    # Top sheet (Sheet 0)
    elems.append('<polygon points="44,202 174,144 356,144 226,202" fill="#dbeafe" fill-opacity="0.86" stroke="#2563eb" stroke-width="2.2"/>')
    elems.append('<line x1="109" y1="173" x2="291" y2="173" stroke="#93c5fd" stroke-width="1.4"/>')
    # Highlight 3-way seam in crimson
    elems.append('<line x1="135" y1="173" x2="265" y2="115" stroke="#b91c1c" stroke-width="4.5"/>')
    # Wait: top sheet seam from (135,202) to (265,144)! Let's align exact midpoint of (44->226) = 135 and (174->356) = 265!
    # Re-draw clean top seam right on top sheet:
    elems.pop()
    elems.append('<line x1="135" y1="202" x2="265" y2="144" stroke="#b91c1c" stroke-width="4.6"/>')
    elems.append('<text x="72" y="194" font-size="12.2" font-weight="700" fill="#1e40af">Sheet 0</text>')

    # Show 1 incoming defect splitting into 2 defects across the 3-way seam:
    elems.append('<circle cx="118" cy="166" r="9.5" fill="#dc2626" stroke="#ffffff" stroke-width="1.8"/>')
    elems.append('<circle cx="234" cy="164" r="9.5" fill="#dc2626" stroke="#ffffff" stroke-width="1.8"/>')
    elems.append('<circle cx="188" cy="212" r="9.5" fill="#dc2626" stroke="#ffffff" stroke-width="1.8"/>')
    elems.append('<path d="M 130,166 Q 178,152 220,162" fill="none" stroke="#991b1b" stroke-width="2.3" stroke-dasharray="4,3"/>')
    elems.append('<path d="M 130,170 Q 158,195 176,208" fill="none" stroke="#991b1b" stroke-width="2.3" stroke-dasharray="4,3"/>')
    elems.append('<text x="200" y="96" text-anchor="middle" font-size="13.2" font-weight="700" fill="#b91c1c">3-Way Seam Splits 1 Defect → 2</text>')

    # Bottom explanation box in Card (a)
    elems.append('<rect x="34" y="362" width="332" height="108" rx="8" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.3"/>')
    elems.append('<text x="200" y="388" text-anchor="middle" font-size="13.2" font-weight="600" fill="#374151">Flat sheet move: 1 defect → 1 (ΔE = 0)</text>')
    elems.append('<text x="200" y="416" text-anchor="middle" font-size="13.4" font-weight="700" fill="#047857">3-way seam move: 1 defect → 2 (ΔE &gt; 0)</text>')
    elems.append('<text x="200" y="446" text-anchor="middle" font-size="12.8" font-weight="600" fill="#1f2937">Every scale multiplies the energy barrier</text>')

    # --- Card (b): Large 2x2 Cell Showing 5 Sub-Edges/Wall & 4 Interior Seams -> 2/5 Contraction ---
    bx0, by0, bsz = 480, 96, 220
    # Shaded Sheet 0 flipped (4 quadrants)
    elems.append(f'<rect x="{bx0}" y="{by0}" width="{bsz}" height="{bsz}" fill="#dbeafe" stroke="#1d4ed8" stroke-width="2.8"/>')
    # 4 quadrant labels
    for qx, qy in [(bx0+55, by0+60), (bx0+165, by0+60), (bx0+55, by0+170), (bx0+165, by0+170)]:
        elems.append(f'<text x="{qx}" y="{qy}" text-anchor="middle" font-size="12.4" font-weight="700" fill="#1e40af">Sheet 0</text>')
    # 4 interior three-way bridge seams (highlighted crimson/blue)
    elems.append(f'<line x1="{bx0+110}" y1="{by0}" x2="{bx0+110}" y2="{by0+bsz}" stroke="#dc2626" stroke-width="6.5"/>')
    elems.append(f'<line x1="{bx0}" y1="{by0+110}" x2="{bx0+bsz}" y2="{by0+110}" stroke="#dc2626" stroke-width="6.5"/>')
    for wx, wy in [(bx0+110, by0), (bx0+110, by0+bsz), (bx0, by0+110), (bx0+bsz, by0+110)]:
        elems.append(f'<circle cx="{wx}" cy="{wy}" r="7.5" fill="#d97706" stroke="#ffffff" stroke-width="1.6"/>')
    elems.append(f'<circle cx="{bx0+110}" cy="{by0+110}" r="8.0" fill="#1e3a8a" stroke="#ffffff" stroke-width="1.6"/>')
    elems.append(f'<text x="{bx0+110}" y="{by0-12}" text-anchor="middle" font-size="12.8" font-weight="700" fill="#1e3a8a">Each Wall = 5 Sub-Edges (2 + 2 + 1)</text>')
    elems.append(f'<text x="{bx0+110}" y="{by0+bsz+24}" text-anchor="middle" font-size="12.8" font-weight="700" fill="#b91c1c">4 Interior Seam Checks Violated</text>')

    elems.append('<rect x="424" y="362" width="332" height="108" rx="8" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.3"/>')
    elems.append('<text x="590" y="388" text-anchor="middle" font-size="13.1" font-weight="600" fill="#374151">Flipping 1 sheet = 2/5 on each of 4 walls (8/5)</text>')
    elems.append('<text x="590" y="416" text-anchor="middle" font-size="13.5" font-weight="700" fill="#047857">Coarse syndrome shrinks by (8/5) ÷ 4 = 2/5</text>')
    elems.append('<text x="590" y="446" text-anchor="middle" font-size="12.8" font-weight="600" fill="#1f2937">174× smaller cell per qubit than scale-8</text>')

    # --- Card (c): Large 3-Arm T-Bridge Buttonhole Failure ---
    cx0, cy0, csz = 870, 96, 220
    elems.append(f'<rect x="{cx0}" y="{cy0}" width="{csz}" height="{csz}" fill="#fef2f2" stroke="#b91c1c" stroke-width="2.8"/>')
    elems.append(f'<line x1="{cx0}" y1="{cy0+110}" x2="{cx0+csz}" y2="{cy0+110}" stroke="#2563eb" stroke-width="6.5"/>')
    elems.append(f'<line x1="{cx0+110}" y1="{cy0}" x2="{cx0+110}" y2="{cy0+110}" stroke="#2563eb" stroke-width="6.5"/>')
    elems.append(f'<line x1="{cx0+110}" y1="{cy0+110}" x2="{cx0+110}" y2="{cy0+csz}" stroke="#9ca3af" stroke-width="3.0" stroke-dasharray="6,6"/>')
    elems.append(f'<circle cx="{cx0+110}" cy="{cy0+155}" r="28" fill="#fee2e2" stroke="#dc2626" stroke-width="3.0" stroke-dasharray="5,4"/>')
    elems.append(f'<text x="{cx0+110}" y="{cy0+152}" text-anchor="middle" font-size="12.8" font-weight="700" fill="#b91c1c">Open</text>')
    elems.append(f'<text x="{cx0+110}" y="{cy0+168}" text-anchor="middle" font-size="12.8" font-weight="700" fill="#b91c1c">Hole</text>')
    elems.append(f'<text x="{cx0+110}" y="{cy0-12}" text-anchor="middle" font-size="12.8" font-weight="700" fill="#b91c1c">Missing 4th Bridge Arm</text>')
    elems.append(f'<text x="{cx0+110}" y="{cy0+csz+24}" text-anchor="middle" font-size="12.8" font-weight="700" fill="#991b1b">Unchecked Logical Loop Threads Gap</text>')

    elems.append('<rect x="814" y="362" width="332" height="108" rx="8" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.3"/>')
    elems.append('<text x="980" y="388" text-anchor="middle" font-size="13.1" font-weight="600" fill="#374151">Dangling bridge end opens a sheet gap</text>')
    elems.append('<text x="980" y="416" text-anchor="middle" font-size="13.5" font-weight="700" fill="#b91c1c">Encoded torus qubits jump: 2 → 1,004</text>')
    elems.append('<text x="980" y="446" text-anchor="middle" font-size="12.8" font-weight="600" fill="#1f2937">Proves the 4-arm cross is minimal</text>')

    save_svg_png("fig_v12_2_architecture", elems, W, H)


def render_fig4_contextualized_rg_and_threshold(rg9, kmc9):
    """
    Contextualized 2-panel figure with Top Visual Zoom-Out Banner (1180 x 545):
      Top Banner: Visual explanation of zooming out by grouping 2x2 cells (322,159 -> 27,107 -> 2,311 -> 203 -> 19q)
      Panel (a): Defect density while zooming out inside ONE 322,159-qubit bath (shaded Too Warm vs Self-Correcting)
      Panel (b): Memory lifetime across 4 code sizes crossing at the SAME fixed threshold beta ≈ 1.9
    """
    W, H = 1180, 545
    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<style>text { font-family: -apple-system, Helvetica, Arial, sans-serif; fill: #1f2937; }</style>',
    ]

    # Top Contextual Banner explaining what "Zooming Out Across Scales" means:
    elems.append('<rect x="40" y="10" width="1108" height="68" rx="10" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.4"/>')
    elems.append('<text x="594" y="31" text-anchor="middle" font-size="13.6" font-weight="700" fill="#1e293b">'
                 'How Local Zoom-Out Tests Self-Correction Inside One 322,159-Qubit Thermal Bath (No Global Decoder)</text>')

    boxes = [
        (68,  "Scale 5: 322,159q", "#eff6ff", "#2563eb"),
        (300, "Scale 4: 27,107q",  "#eff6ff", "#2563eb"),
        (524, "Scale 3: 2,311q",   "#eff6ff", "#2563eb"),
        (740, "Scale 2: 203q",     "#eff6ff", "#2563eb"),
        (948, "Scale 1: 19q",      "#ecfdf5", "#059669"),
    ]
    for idx, (bx, blbl, bfill, bstr) in enumerate(boxes):
        bw = 172 if idx < 2 else (156 if idx == 2 else 148)
        elems.append(f'<rect x="{bx}" y="40" width="{bw}" height="28" rx="6" fill="{bfill}" stroke="{bstr}" stroke-width="1.4"/>')
        elems.append(f'<text x="{bx+bw/2:.1f}" y="58.5" text-anchor="middle" font-size="12.0" font-weight="700" fill="#1e293b">{html.escape(blbl)}</text>')
        if idx < 4:
            ax1 = bx + bw + 4
            ax2 = boxes[idx + 1][0] - 6
            elems.append(f'<line x1="{ax1}" y1="54" x2="{ax2}" y2="54" stroke="#475569" stroke-width="2.0"/>')
            elems.append(f'<polygon points="{ax2+4},54 {ax2-4},49.5 {ax2-4},58.5" fill="#475569"/>')
            elems.append(f'<text x="{(ax1+ax2)/2:.1f}" y="48" text-anchor="middle" font-size="10.2" font-weight="600" fill="#475569">2×2</text>')

    # Panel (a): Live 322k-qubit syndrome zoom-out with shaded physical regimes!
    rg_series = []
    for b, col, dsh, lbl in [
        (1.35, "#dc2626", "5,4", "Warm Bath (β = 1.35)"),
        (1.86, "#ea580c", "5,4", "Just Above Tc (β = 1.86)"),
        (2.00, "#1f2937", "",    "Critical Threshold (β = 2.00)"),
        (2.15, "#2563eb", "",    "Below Tc (β = 2.15)"),
        (2.35, "#059669", "",    "Cold Bath (β = 2.35)"),
    ]:
        rhos = rg9[("sq_rx_iso_s5", b)]
        xs, ys = [], []
        for step_idx, s in enumerate([5, 4, 3, 2, 1]):
            if rhos[s] > 0:
                xs.append(float(step_idx)); ys.append(rhos[s])
        rg_series.append({"xs": xs, "ys": ys, "color": col, "dash": dsh, "label": lbl})

    p1 = dict(
        xlim=(-0.2, 4.25), ylim=(1.2e-4, 65.0),
        title="(a) Defect Density While Zooming Out (322k Qubits)",
        xlabel="Coarse-Graining Step (Fine Lattice → Coarse Cell)",
        ylabel="Fraction of Violated Checks",
        xticks=[(0,"Scale 5"), (1,"Scale 4"), (2,"Scale 3"), (3,"Scale 2"), (4,"Scale 1")],
        yticks=[(3e-4,"0.03%"), (3e-3,"0.3%"), (3e-2,"3%"), (3e-1,"30%")],
        series=rg_series, xlog=False, ylog=True, leg_pos="tl",
        ybands=[(0.048, 65.0, "#fff1f2"), (1.2e-4, 0.040, "#ecfdf5")],
        notes=[
            (4.10, 0.42,   "Too Warm: Errors Multiply ↑", "#b91c1c", "end"),
            (4.12, 8.2e-4, "Shrinks to 0 ↓", "#047857", "end"),
        ],
    )

    # Panel (b): 4-Scale Lifetime Crossings settling at fixed beta ≈ 1.86
    ms_series = []
    for s, col, dsh, nlbl in [
        (4, "#be123c", "",    "Scale 4 (27,107 qubits)"),
        (3, "#ea580c", "",    "Scale 3 (2,311 qubits)"),
        (2, "#2563eb", "6,3", "Scale 2 (203 qubits)"),
        (1, "#6b7280", "3,3", "Scale 1 (19 qubits)"),
    ]:
        bs_s = [b for b in (1.25, 1.50, 1.75, 1.90, 2.00, 2.10, 2.15, 2.25) if (f"sq_rx_iso_s{s}_X_err", "uf", b) in kmc9]
        ms_series.append({"xs": bs_s, "ys": [kmc9[(f"sq_rx_iso_s{s}_X_err", "uf", b)][1] for b in bs_s], "color": col, "dash": dsh, "label": nlbl})

    p2 = dict(
        xlim=(1.20, 2.48), ylim=(2.5, 9.5e4),
        title="(b) Lifetime Crossings Settle at Same Threshold",
        xlabel="Inverse Bath Temperature β = 1 / T (Colder →)",
        ylabel="Mean Memory Lifetime",
        xticks=[(1.25,"1.25"), (1.5,"1.5"), (1.75,"1.75"), (2.0,"2.0"), (2.25,"2.25")],
        yticks=[(1e1,"10¹"), (1e2,"10²"), (1e3,"10³"), (1e4,"10⁴")],
        series=ms_series, xlog=False, ylog=True, leg_pos="tl",
        xbands=[(1.86, 2.48, "#ecfdf5")],
        vlines=[(1.86, "#374151", "Fixed Threshold β ≈ 1.9", "rb")],
        varrows=[(2.25, kmc9[("sq_rx_iso_s1_X_err", "uf", 2.25)][1], kmc9[("sq_rx_iso_s4_X_err", "uf", 2.25)][1], "396×", "#be123c", "r", 7.0e2)],
    )

    draw_panel(elems, 86,  122, 472, 360, **p1)
    draw_panel(elems, 676, 122, 472, 360, **p2)
    save_svg_png("fig_v12_4_threshold_rg", elems, W, H)


def render_fig6_wires_visual_and_plot(v7_best):
    """
    Combines Visual Compact-vs-Wire Diagram (Panel a) with Cubic-Grid Lifetime Plot (Panel b)!
    """
    W, H = 1180, 456
    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<style>text { font-family: -apple-system, Helvetica, Arial, sans-serif; fill: #1f2937; }</style>',
    ]

    # Panel (a): Visual schematic comparing Compact Graph vs Scale-8 Cubic Grid Wire
    x0, y0, pw, ph = 42, 42, 516, 352
    elems.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" rx="10" fill="#f9fafb" stroke="#374151" stroke-width="1.5"/>')
    elems.append(f'<text x="{x0+pw/2:.1f}" y="{y0-13:.1f}" text-anchor="middle" font-size="14.8" font-weight="700">(a) Why Single-Grid Routing Adds Wire Entropy</text>')

    # Top sub-card: Compact Graph (1 edge between junctions)
    elems.append(f'<rect x="{x0+18}" y="{y0+18}" width="{pw-36}" height="146" rx="8" fill="#ecfdf5" stroke="#a7f3d0" stroke-width="1.4"/>')
    elems.append(f'<text x="{x0+pw/2:.1f}" y="{y0+42}" text-anchor="middle" font-size="13.6" font-weight="700" fill="#065f46">Compact Graph Connectivity (1,480 Qubits)</text>')
    # Two junctions directly linked by 1 edge
    jx1, jx2, jy = x0 + 156, x0 + 324, y0 + 94
    elems.append(f'<rect x="{jx1-36}" y="{jy-30}" width="36" height="60" rx="6" fill="#fde68a" stroke="#b45309" stroke-width="2.2"/>')
    elems.append(f'<rect x="{jx2}" y="{jy-30}" width="36" height="60" rx="6" fill="#fde68a" stroke="#b45309" stroke-width="2.2"/>')
    elems.append(f'<line x1="{jx1}" y1="{jy}" x2="{jx2}" y2="{jy}" stroke="#047857" stroke-width="4.0"/>')
    elems.append(f'<circle cx="{(jx1+jx2)/2:.1f}" cy="{jy}" r="8.5" fill="#059669" stroke="#ffffff" stroke-width="1.6"/>')
    elems.append(f'<text x="{x0+pw/2:.1f}" y="{y0+148}" text-anchor="middle" font-size="12.6" font-weight="600" fill="#065f46">'
                 '1 qubit between 3-way junctions → Every flip costs +ΔE</text>')

    # Bottom sub-card: Scale-8 Cubic Grid (7-qubit wire between junctions)
    elems.append(f'<rect x="{x0+18}" y="{y0+184}" width="{pw-36}" height="150" rx="8" fill="#fef2f2" stroke="#fecaca" stroke-width="1.4"/>')
    elems.append(f'<text x="{x0+pw/2:.1f}" y="{y0+208}" text-anchor="middle" font-size="13.6" font-weight="700" fill="#991b1b">Scale-8 Single Cubic Grid (56,312 Qubits)</text>')
    wx1, wx2, wy = x0 + 82, x0 + 434, y0 + 272
    elems.append(f'<rect x="{wx1-30}" y="{wy-30}" width="30" height="60" rx="6" fill="#fde68a" stroke="#b45309" stroke-width="2.2"/>')
    elems.append(f'<rect x="{wx2}" y="{wy-30}" width="30" height="60" rx="6" fill="#fde68a" stroke="#b45309" stroke-width="2.2"/>')
    elems.append(f'<line x1="{wx1}" y1="{wy}" x2="{wx2}" y2="{wy}" stroke="#374151" stroke-width="3.2"/>')
    for qi in range(7):
        qx = wx1 + 31 + qi * 48.0
        col = "#dc2626" if qi == 3 else "#2563eb"
        rad = 8.2 if qi == 3 else 5.8
        elems.append(f'<circle cx="{qx:.1f}" cy="{wy}" r="{rad}" fill="{col}" stroke="#ffffff" stroke-width="1.4"/>')
    elems.append(f'<line x1="{wx1+65}" y1="{wy-26}" x2="{wx2-65}" y2="{wy-26}" stroke="#dc2626" stroke-width="2.4"/>')
    elems.append(f'<polygon points="{wx1+55},{wy-26} {wx1+66},{wy-31} {wx1+66},{wy-21}" fill="#dc2626"/>')
    elems.append(f'<polygon points="{wx2-55},{wy-26} {wx2-66},{wy-31} {wx2-66},{wy-21}" fill="#dc2626"/>')
    elems.append(f'<text x="{x0+pw/2:.1f}" y="{wy-34}" text-anchor="middle" font-size="12.0" font-weight="700" fill="#dc2626">Defect walks freely along 7-qubit wire (ΔE = 0)</text>')
    elems.append(f'<text x="{x0+pw/2:.1f}" y="{y0+320}" text-anchor="middle" font-size="12.6" font-weight="600" fill="#991b1b">'
                 'Wire entropy (+T ΔS) cuts memory lifetime 321×</text>')

    # Panel (b): Cubic-Grid Wire Stretching Plot
    z3_bs = [5.50, 6.00, 6.50, 7.00]
    s_z3 = [
        {"xs": z3_bs, "ys": [v7_best("sq_L3r2_s2_X_err", "uniform", "uf", b) for b in z3_bs],
         "color": "#059669", "dash": "", "label": "Compact Graph (1,480 qubits)"},
        {"xs": z3_bs, "ys": [v7_best("sq_z3_4_6_r2_X_err", "uniform", "uf", b) for b in z3_bs],
         "color": "#2563eb", "dash": "", "label": "Tighter Cubic Grid (10,280 qubits)"},
        {"xs": [5.50, 6.00, 6.50], "ys": [v7_best("sq_bdl_8_8_r2_X_err", "uniform", "uf", b) for b in [5.50, 6.00, 6.50]],
         "color": "#dc2626", "dash": "", "label": "Scale-8 Cubic Grid (56,312 qubits)"},
    ]
    p2 = dict(
        xlim=(5.4, 7.1), ylim=(70.0, 2.5e6),
        title="(b) Measured Lifetime Across Grid Embeddings",
        xlabel="Inverse Bath Temperature β = 1 / T (Colder →)", ylabel="Quantum Memory Lifetime",
        xticks=[(5.5,"5.5"), (6.0,"6.0"), (6.5,"6.5"), (7.0,"7.0")],
        yticks=[(1e2,"10²"), (1e3,"10³"), (1e4,"10⁴"), (1e5,"10⁵"), (1e6,"10⁶")],
        series=s_z3, xlog=False, ylog=True, leg_pos="tl",
        varrows=[(6.50, v7_best("sq_bdl_8_8_r2_X_err", "uniform", "uf", 6.50),
                  v7_best("sq_L3r2_s2_X_err", "uniform", "uf", 6.50), "321× loss", "#b91c1c", "r", 3.2e4)],
    )
    draw_panel(elems, 676, 42, 472, 352, **p2)
    save_svg_png("fig_v12_6_grid_wires", elems, W, H)


def load_all():
    kmc4 = {}
    with open(os.path.join(WS, "cluster_v4_results.txt")) as f:
        for raw in f:
            ln = raw.strip()
            if not ln or ln.startswith(("ERR|", "TIMEOUT|")):
                continue
            parts = ln.split("|", 3)
            if len(parts) == 4 and parts[3].startswith("KMC "):
                kv = parse_kv(parts[3])
                if int(kv["capped"]) == 0:
                    kmc4.setdefault(parts[1], {})[float(kv["beta"])] = (int(kv["N"]), float(kv["mean_t"]))

    kmc6 = {}
    with open(os.path.join(WS, "cluster_v6_results.txt")) as f:
        for raw in f:
            ln = raw.strip()
            if not ln or "|" not in ln:
                continue
            tid, payload = [x.strip() for x in ln.split("|", 1)]
            if payload.startswith(("TIMEOUT", "ERROR")) or "VALID_REPS=0" in payload:
                continue
            kv = parse_kv(payload)
            parts = tid.split("__")
            if parts[0] == "kmc":
                _, tag, jm, dec, bs = parts
                kmc6[(tag, jm, dec, float(bs[1:]))] = (int(kv["N"]), float(kv["mean_t"]), int(kv["capped"]))

    kmc7, ffs7, cool7 = {}, {}, {}
    with open(os.path.join(WS, "cluster_v7_results.txt")) as f:
        for raw in f:
            ln = raw.strip()
            if not ln or "|" not in ln:
                continue
            tid, payload = [x.strip() for x in ln.split("|", 1)]
            if payload.startswith(("TIMEOUT", "ERROR")) or "VALID_REPS=0" in payload:
                continue
            kv = parse_kv(payload)
            parts = tid.split("__")
            if parts[0] == "kmc":
                _, tag, jm, dec, bs = parts
                kmc7[(tag, jm, dec, float(bs[1:]))] = (int(kv["N"]), float(kv["mean_t"]), int(kv["capped"]))
            elif parts[0] == "ffs":
                _, tag, jm, dec, bs = parts
                vr, nr = [int(x) for x in kv["valid_reps"].split("/")]
                ffs7[(tag, jm, dec, float(bs[1:]))] = (int(kv["N"]), float(kv["mean_t"]), vr, nr)
            elif parts[0] == "cool":
                _, tag, proto, init_m, ts = parts
                cool7[(tag, proto, init_m, float(ts[1:]))] = {
                    "N": int(kv["N"]),
                    "K": int(kv["K"]),
                    "p0": float(kv["p_zero_syn"]),
                    "pheal": float(kv["p_exact_heal"]),
                    "psurv": float(kv["p_log_survive"]),
                }

    kmc9, rg9 = {}, {}
    for fn in ("cluster_v8_results.txt", "cluster_v9_results.txt", "cluster_v9b_results.txt", "cluster_v10u_results.txt"):
        fp = os.path.join(WS, fn)
        if not os.path.exists(fp):
            continue
        with open(fp) as f:
            for raw in f:
                ln = raw.strip()
                if not ln or "|" not in ln:
                    continue
                tid, payload = [x.strip() for x in ln.split("|", 1)]
                if payload.startswith(("TIMEOUT", "ERROR")) or "VALID_REPS=0" in payload:
                    continue
                kv = parse_kv(payload)
                parts = tid.split("__")
                if parts[0] == "kmc" and int(kv["capped"]) == 0 and int(kv.get("bails", "0")) == 0:
                    _, tag, jm, dec, bs = parts
                    kmc9[(tag, dec, float(bs[1:]))] = (int(kv["N"]), float(kv["mean_t"]))
                elif parts[0] == "rg":
                    _, rg_nm, bs = parts
                    S_max = int(kv["S_max"])
                    rhos = [float(kv[f"rho_{s}"]) for s in range(S_max + 1)]
                    rg9[(rg_nm, float(bs[1:]))] = rhos

    return kmc4, kmc6, kmc7, ffs7, cool7, kmc9, rg9


def main():
    render_fig2_large_architecture()
    kmc4, kmc6, kmc7, ffs7, cool7, kmc9, rg9 = load_all()

    def v7_best(tag, jm, dec, b, min_reps=3):
        if (tag, jm, dec, b) in kmc7 and kmc7[(tag, jm, dec, b)][2] == 0:
            return kmc7[(tag, jm, dec, b)][1]
        if (tag, jm, dec, b) in ffs7 and ffs7[(tag, jm, dec, b)][2] >= min_reps:
            return ffs7[(tag, jm, dec, b)][1]
        return None

    # Check if uniform 2x2 `sq_iso_L2r1_s5_Z_err` (423,559q) finished uncapped at 6.50, else use `sq_L2r1` (189,988q)
    use_iso_s5 = all((f"sq_iso_L2r1_s{s}_Z_err", "uf", b) in kmc9 for s in (1, 3, 5) for b in (6.25, 6.50))
    q_fam = "sq_iso_L2r1" if use_iso_s5 else "sq_L2r1"

    # Check if uniform 2x2 torus `rottor_iso_L2r1_s4_X_err` (94,774q) has >= 4 points, else `rottor_L2r1`
    iso_tor_bs = [b for b in (5.50, 6.00, 6.25, 6.50, 6.65, 6.80) if ("rottor_iso_L2r1_s4_X_err", "uf", b) in kmc9]
    use_iso_tor = len(iso_tor_bs) >= 3

    # --- 1. `fig_v12_1_classes.png`: Clean Code-Family Comparison ---
    def v4_norm_curve(cnames, b):
        pts = [(kmc4[cn][b][0], kmc4[cn][b][1]) for cn in cnames if b in kmc4.get(cn, {})]
        y0 = pts[0][1]
        return [x for x, _ in pts], [y / y0 for _, y in pts]

    hx, hy = v4_norm_curve([f"haah_cubic_L{L}_Xerr.code" for L in [3, 5, 7, 9, 11, 13]], 4.25)
    bx, by = v4_norm_curve([f"bb_gross_{l}x{m}_Xerr.code" for l, m in [(6,6),(12,6),(18,12),(24,18),(30,18),(36,18),(48,18),(60,18)]], 4.25)
    tx_n, ty_n = v4_norm_curve([f"toric2d_L{L}_Xerr.code" for L in [4, 6, 10, 16, 24, 34]], 4.50)

    w_pts = [kmc9[(f"sq_rx_iso_s{s}_X_err", "uf", 2.25)] for s in [1, 2, 3, 4]]
    w_x = [p[0] for p in w_pts]
    w_y = [p[1] / w_pts[0][1] for p in w_pts]

    q3_pts = [kmc9[(f"{q_fam}_s{s}_Z_err", "uf", 6.50)] for s in [1, 3, 5]]
    q3_x = [p[0] for p in q3_pts]
    q3_y = [p[1] / q3_pts[0][1] for p in q3_pts]
    q_top_k = f"{q3_x[-1]//1000}k"

    s_norm = [
        {"xs": q3_x, "ys": q3_y, "color": "#ea580c", "dash": "7,3", "label": f"3D Welded Quantum (to {q_top_k}: {q3_y[-1]:,.0f}×)"},
        {"xs": w_x,  "ys": w_y,  "color": "#be123c", "dash": "",    "label": "3D Welded Single-Sector (4 scales: 396×)"},
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

    w_bN = [kmc9[(f"sq_rx_iso_s{s}_X_err", "uf", 1.75)][0] for s in range(5)]
    w_bD = [ols_slope([1.75, 2.00, 2.25], [kmc9[(f"sq_rx_iso_s{s}_X_err", "uf", b)][1] for b in [1.75, 2.00, 2.25]]) for s in range(5)]

    q3_bwin = [b for b in (6.25, 6.35, 6.50) if (f"{q_fam}_s5_Z_err", "uf", b) in kmc9]
    q3_bN = [kmc9[(f"{q_fam}_s{s}_Z_err", "uf", 6.50)][0] for s in (1, 3, 5)]
    q3_bD = [
        ols_slope([6.00, 6.25, 6.50], [kmc9[(f"{q_fam}_s{s}_Z_err", "uf", b)][1] for b in (6.00, 6.25, 6.50)])
        if s < 5 else ols_slope(q3_bwin, [kmc9[(f"{q_fam}_s5_Z_err", "uf", b)][1] for b in q3_bwin])
        for s in (1, 3, 5)
    ]

    s_bar = [
        {"xs": w_bN,     "ys": w_bD,     "color": "#be123c", "dash": "",    "label": "3D Welded Single-Sector (curves upward)"},
        {"xs": q3_bN,    "ys": q3_bD,    "color": "#ea580c", "dash": "7,3", "label": f"3D Welded Quantum (doubles each cycle)"},
        {"xs": c_haah_N, "ys": c_haah_D, "color": "#7c3aed", "dash": "",    "label": "Haah 3D Cubic (straight log N, bends down)"},
        {"xs": c_bb_N,   "ys": c_bb_D,   "color": "#2563eb", "dash": "5,4", "label": "Scaled 2D Bivariate Bicycle (flat ≈ 5.4)"},
        {"xs": c_tor_N,  "ys": c_tor_D,  "color": "#4b5563", "dash": "3,3", "label": "2D Toric Code (flat ≤ 2.3)"},
    ]

    render_double(
        "fig_v12_1_classes",
        dict(xlim=(15.0, 520000.0), ylim=(0.02, 2.5e7),
             title="(a) Lifetime Growth Relative to Smallest Code",
             xlabel="Physical Qubit Count N (Log Scale)", ylabel="Relative Memory Lifetime",
             xticks=[(20,"20"), (200,"200"), (2000,"2k"), (20000,"20k"), (300000,"300k")],
             yticks=[(0.05,"0.05×"), (1.0,"1×"), (100.0,"10²×"), (10000.0,"10⁴×")],
             series=s_norm, xlog=True, ylog=True, leg_pos="tl",
             hlines=[(1.0, "#6b7280", "1× = No Gain From Growing Code Size", "br")]),
        dict(xlim=(1.6, 520000.0), ylim=(0.0, 17.5),
             title="(b) Additive vs. Multiplicative Energy Barriers",
             xlabel="Physical Qubit Count N (Log Scale)", ylabel="Effective Energy Barrier",
             xticks=[(2,"2"), (30,"30"), (500,"500"), (10000,"10k"), (300000,"300k")],
             yticks=[(0,"0"), (4,"4"), (8,"8"), (12,"12"), (16,"16")],
             series=s_bar, xlog=True, ylog=False, leg_pos="tl"),
    )

    # --- 3. `fig_v12_3_validation.png`: Simulator Validation ---
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
        "fig_v12_3_validation",
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

    # --- 4. `fig_v12_4_threshold_rg.png`: Contextualized Zoom-Out & Threshold Crossings ---
    render_fig4_contextualized_rg_and_threshold(rg9, kmc9)

    # --- 5. `fig_v12_5_quantum_scaling.png`: Dedicated Full (X, Z) Quantum Memory Scaling ---
    q_s1_bs = [4.50, 5.00, 5.50, 5.75, 6.00, 6.25, 6.50]
    q_s3_bs = [4.50, 5.00, 5.50, 5.75, 6.00, 6.25, 6.50]
    q_s5_bs = [b for b in (6.25, 6.35, 6.50) if (f"{q_fam}_s5_Z_err", "uf", b) in kmc9]
    n_s3 = kmc9[(f"{q_fam}_s3_Z_err", "uf", 6.50)][0]
    n_s5 = kmc9[(f"{q_fam}_s5_Z_err", "uf", 6.50)][0]
    gain_q3 = kmc9[(f"{q_fam}_s5_Z_err", "uf", 6.50)][1] / kmc9[(f"{q_fam}_s1_Z_err", "uf", 6.50)][1]

    iso_bs = [5.00, 5.50, 6.00, 6.25, 6.50, 6.65]
    iso_s2 = [kmc9[("sq_iso_L2r1_s2_X_err", "uf", b)][1] for b in iso_bs]
    iso_s4 = [kmc9[("sq_iso_L2r1_s4_X_err", "uf", b)][1] for b in iso_bs]

    if use_iso_tor:
        tor_bs = iso_tor_bs
        tor_s2 = [kmc9[("rottor_iso_L2r1_s2_X_err", "uf", b)][1] for b in tor_bs]
        tor_s4 = [kmc9[("rottor_iso_L2r1_s4_X_err", "uf", b)][1] for b in tor_bs]
        tor_gain = tor_s4[-1] / tor_s2[-1]
        tor_lbl4 = f"Torus Cycle 2 (94,774q: {tor_gain:.1f}×)"
        tor_lbl2 = "Torus Cycle 1 (646q)"
    else:
        tor_bs = [5.50, 6.00, 6.25, 6.50, 6.65]
        tor_s2 = [v7_best("rottor_L2r1_s2_X_err", "uniform", "uf", b) if b <= 6.50 else kmc9[("rottor_L2r1_s2_X_err", "uf", b)][1] for b in tor_bs]
        tor_s4 = [v7_best("rottor_L2r1_s4_X_err", "uniform", "uf", b) if b <= 6.50 else kmc9[("rottor_L2r1_s4_X_err", "uf", b)][1] for b in tor_bs]
        tor_gain = tor_s4[-1] / tor_s2[-1]
        tor_lbl4 = f"Torus Cycle 2 (50,296q: {tor_gain:.1f}×)"
        tor_lbl2 = "Torus Cycle 1 (508q)"

    print(f"Quantum Fig 5: q_fam={q_fam} (N_s5={n_s5}, gain={gain_q3:.0f}x), use_iso_tor={use_iso_tor} (tor_bs={tor_bs}, gain={tor_gain:.1f}x)")

    render_double(
        "fig_v12_5_quantum_scaling",
        dict(xlim=(4.40, 6.86), ylim=(6.0, max(5.0e6, kmc9[(f"{q_fam}_s5_Z_err", "uf", 6.50)][1] * 4.5)),
             title=f"(a) Three Quantum Levels Up to {n_s5:,} Qubits",
             xlabel="Inverse Bath Temperature β = 1 / T (Colder →)", ylabel="Quantum Memory Lifetime",
             xticks=[(4.5,"4.5"), (5.0,"5.0"), (5.5,"5.5"), (6.0,"6.0"), (6.5,"6.5")],
             yticks=[(1e1,"10¹"), (1e2,"10²"), (1e3,"10³"), (1e4,"10⁴"), (1e5,"10⁵"), (1e6,"10⁶")],
             series=[
                 {"xs": q_s5_bs, "ys": [kmc9[(f"{q_fam}_s5_Z_err", "uf", b)][1] for b in q_s5_bs],
                  "color": "#be123c", "dash": "",    "label": f"Level 3 ({n_s5:,}q: barrier = {q3_bD[2]:.1f})"},
                 {"xs": q_s3_bs, "ys": [kmc9[(f"{q_fam}_s3_Z_err", "uf", b)][1] for b in q_s3_bs],
                  "color": "#059669", "dash": "",    "label": f"Level 2 ({n_s3:,}q: barrier = {q3_bD[1]:.1f})"},
                 {"xs": q_s1_bs, "ys": [kmc9[(f"{q_fam}_s1_Z_err", "uf", b)][1] for b in q_s1_bs],
                  "color": "#6b7280", "dash": "4,3", "label": f"Level 1 (19q: barrier = {q3_bD[0]:.1f})"},
             ],
             xlog=False, ylog=True, leg_pos="tl",
             varrows=[(6.50, kmc9[(f"{q_fam}_s1_Z_err", "uf", 6.50)][1], kmc9[(f"{q_fam}_s5_Z_err", "uf", 6.50)][1],
                       f"{gain_q3:,.0f}×", "#be123c", "r", 3.5e3)]),
        dict(xlim=(4.92, max(6.88, tor_bs[-1] + 0.22)), ylim=(3.0e2, max(6.0e7, max(tor_s4) * 3.5)),
             title="(b) Full Quantum Cycles (Planar & Torus)",
             xlabel="Inverse Bath Temperature β = 1 / T (Colder →)", ylabel="Quantum Memory Lifetime",
             xticks=[(5.0,"5.0"), (5.5,"5.5"), (6.0,"6.0"), (6.5,"6.5")],
             yticks=[(1e3,"10³"), (1e4,"10⁴"), (1e5,"10⁵"), (1e6,"10⁶"), (1e7,"10⁷")],
             series=[
                 {"xs": tor_bs, "ys": tor_s4, "color": "#be123c", "dash": "",    "label": tor_lbl4},
                 {"xs": tor_bs, "ys": tor_s2, "color": "#f97316", "dash": "5,4", "label": tor_lbl2},
                 {"xs": iso_bs, "ys": iso_s4, "color": "#059669", "dash": "",    "label": "Planar Cycle 2 (47,387q: 71×)"},
                 {"xs": iso_bs, "ys": iso_s2, "color": "#2563eb", "dash": "5,4", "label": "Planar Cycle 1 (323q)"},
             ],
             xlog=False, ylog=True, leg_pos="tl",
             varrows=[(6.65, iso_s2[-1], iso_s4[-1], "71×", "#047857", "r", 1.5e5)]),
    )

    # --- 6. `fig_v12_6_grid_wires.png`: Visual Wire Mechanism + Grid Data ---
    render_fig6_wires_visual_and_plot(v7_best)

    # --- 7. `fig_v12_7_cooling.png`: Storage Self-Healing vs Hot-State Cooling ---
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

    ts = [100.0, 400.0, 1500.0]
    s_cool = [
        {"xs": ts, "ys": [100.0 * cool7[("rottor_L2r1_s2_Z_err", "anneal", "hot", t)]["p0"] for t in ts],
         "color": "#059669", "dash": "",    "label": "Two-Step Torus: Gradual Cooling (100%)"},
        {"xs": ts, "ys": [100.0 * cool7[("sq_L2r1_s3_X_err", "anneal", "hot", t)]["p0"] for t in ts],
         "color": "#2563eb", "dash": "",    "label": "Three-Step Code: Gradual Cooling (63%)"},
        {"xs": ts, "ys": [100.0 * cool7[("sq_L2r1_s3_X_err", "quench", "hot", t)]["p0"] for t in ts],
         "color": "#dc2626", "dash": "5,4", "label": "Three-Step Code: Sudden Quench (14%)"},
    ]
    render_double(
        "fig_v12_7_cooling",
        dict(xlim=(0.5, 12.5), ylim=(-2.0, 110.0),
             title="(a) Small Storage Errors Heal Without a Decoder",
             xlabel="Initial Physical Error Rate (%)", ylabel="Exact Original Codeword Restored (%)",
             xticks=[(1,"1%"), (3,"3%"), (6,"6%"), (9,"9%"), (12,"12%")],
             yticks=[(0,"0%"), (25,"25%"), (50,"50%"), (75,"75%"), (100,"100%")],
             series=s_heal, xlog=False, ylog=False, leg_pos="tr",
             hlines=[(25.0, "#6b7280", "Random-Guess Floor (25%)", "bl")]),
        dict(xlim=(80.0, 1800.0), ylim=(-2.0, 134.0),
             title="(b) Cooling From an Uninitialized Hot State",
             xlabel="Total Bath Cooling Time (Log Scale)", ylabel="Clean Defect-Free Ground States (%)",
             xticks=[(100,"100"), (400,"400"), (1500,"1,500")],
             yticks=[(0,"0%"), (25,"25%"), (50,"50%"), (75,"75%"), (100,"100%")],
             series=s_cool, xlog=True, ylog=False, leg_pos="tl",
             varrows=[(1500.0, 14.0, 63.0, "4.5× more clean states", "#1d4ed8", "l", 26.0)]),
    )


if __name__ == "__main__":
    main()
