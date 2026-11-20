#!/usr/bin/env python3
"""
render_v13_unified_figures.py
Renders 7 unified, crystal-clear figures (V13) built 100% around the provably contractive
uniform 2x2 four-arm cross cell up to 423,559 physical qubits:

  1. fig_v13_1_classes.png:         Code family comparison across 5 decades of N (to 423,559q)
  2. fig_v13_2_architecture.png:    Large 3-panel visual explanation of 3D welding, 2/5 contraction, & holes
  3. fig_v13_3_validation.png:      Simulator validation on published 3D Toric & Random Layer Code benchmarks
  4. fig_v13_4_threshold_rg.png:    Live 322,159q syndrome zoom-out + 5-scale lifetime crossings (19 to 322,159q)
  5. fig_v13_5_quantum_scaling.png: (a) 3-level uniform 2x2 quantum hierarchy (19 -> 2,911 -> 423,559q)
                                    (b) Boundaryless torus (646 -> 94,774q): Local O(N) RG vs Global UF
  6. fig_v13_6_grid_wires.png:      Visual compact-vs-wire diagram + cubic-grid wire-stretching penalty (321x loss)
  7. fig_v13_7_cooling.png:         (a) Zero-measurement storage self-healing
                                    (b) Multi-scale cooling-time scaling law t_cool(N) across 4 scales (to 94,774q)
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


def interp_tcool(t_pts, y_pts, target, decreasing=True):
    for i in range(len(t_pts) - 1):
        t0, t1 = t_pts[i], t_pts[i + 1]
        y0, y1 = y_pts[i], y_pts[i + 1]
        if decreasing:
            if y0 >= target >= y1 and y0 > y1:
                f = (y0 - target) / (y0 - y1)
                return 10.0 ** (math.log10(t0) + f * (math.log10(t1) - math.log10(t0)))
        else:
            if y0 <= target <= y1 and y1 > y0:
                f = (target - y0) / (y1 - y0)
                return 10.0 ** (math.log10(t0) + f * (math.log10(t1) - math.log10(t0)))
    if decreasing and y_pts and y_pts[0] < target:
        return t_pts[0]
    if not decreasing and y_pts and y_pts[0] > target:
        return t_pts[0]
    return None


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
        sw = s.get("lw", 2.9)
        hollow = s.get("hollow", False)
        pstr = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        if len(pts) >= 2:
            elems.append(f'<polyline fill="none" stroke="{s["color"]}" stroke-width="{sw}"{d_attr} points="{pstr}"/>')
        for px, py in pts:
            if hollow:
                elems.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.9" fill="#ffffff" stroke="{s["color"]}" stroke-width="2.2"/>')
            else:
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
            elems.append(f'<text x="{px+dx:.1f}" y="{ty_lbl:.1f}" text-anchor="{anch}" font-size="12.3" font-weight="700" fill="{col}">{html.escape(lbl)}</text>')

    if notes:
        for xv, yv, txt, col, anch in notes:
            elems.append(f'<text x="{tx(xv):.1f}" y="{ty(yv):.1f}" text-anchor="{anch}" font-size="11.8" font-weight="700" fill="{col}">{html.escape(txt)}</text>')

    leg_series = [s for s in series if s.get("label")]
    row_h = 19.4
    lh = row_h * len(leg_series) + 10
    lw = min(pw - 18.0, max(195.0, max(len(s["label"]) for s in leg_series) * 6.50 + 46.0))
    lx = x0 + 10 if "l" in leg_pos else x0 + pw - lw - 10
    ly = y0 + 10 if "t" in leg_pos else y0 + ph - lh - 10
    elems.append(f'<rect x="{lx:.1f}" y="{ly:.1f}" width="{lw:.1f}" height="{lh:.1f}" rx="6" fill="#ffffff" fill-opacity="0.96" stroke="#d1d5db" stroke-width="1.1"/>')
    for i, s in enumerate(leg_series):
        iy = ly + 15.6 + i * row_h
        d_attr = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
        elems.append(f'<line x1="{lx+8:.1f}" y1="{iy-4.0:.1f}" x2="{lx+29:.1f}" y2="{iy-4.0:.1f}" stroke="{s["color"]}" stroke-width="2.6"{d_attr}/>')
        if s.get("hollow", False):
            elems.append(f'<circle cx="{lx+18.5:.1f}" cy="{iy-4.0:.1f}" r="3.9" fill="#ffffff" stroke="{s["color"]}" stroke-width="2.0"/>')
        else:
            elems.append(f'<circle cx="{lx+18.5:.1f}" cy="{iy-4.0:.1f}" r="3.9" fill="{s["color"]}"/>')
        elems.append(f'<text x="{lx+36:.1f}" y="{iy:.1f}" font-size="11.5">{html.escape(s["label"])}</text>')


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

    elems.append('<polygon points="44,322 174,264 356,264 226,322" fill="#dbeafe" fill-opacity="0.88" stroke="#2563eb" stroke-width="2.2"/>')
    elems.append('<line x1="109" y1="293" x2="291" y2="293" stroke="#93c5fd" stroke-width="1.4"/>')
    elems.append('<line x1="135" y1="322" x2="265" y2="264" stroke="#93c5fd" stroke-width="1.4"/>')
    elems.append('<text x="348" y="284" text-anchor="end" font-size="12.2" font-weight="700" fill="#1e40af">Sheet 1</text>')

    elems.append('<polygon points="135,293 265,235 265,115 135,173" fill="#fde68a" fill-opacity="0.93" stroke="#d97706" stroke-width="2.4"/>')
    elems.append('<line x1="200" y1="264" x2="200" y2="144" stroke="#f59e0b" stroke-width="1.5"/>')
    elems.append('<text x="206" y="224" font-size="12.4" font-weight="700" fill="#b45309">Bridge</text>')

    elems.append('<polygon points="44,202 174,144 356,144 226,202" fill="#dbeafe" fill-opacity="0.86" stroke="#2563eb" stroke-width="2.2"/>')
    elems.append('<line x1="109" y1="173" x2="291" y2="173" stroke="#93c5fd" stroke-width="1.4"/>')
    elems.append('<line x1="135" y1="202" x2="265" y2="144" stroke="#b91c1c" stroke-width="4.6"/>')
    elems.append('<text x="72" y="194" font-size="12.2" font-weight="700" fill="#1e40af">Sheet 0</text>')

    elems.append('<circle cx="118" cy="166" r="9.5" fill="#dc2626" stroke="#ffffff" stroke-width="1.8"/>')
    elems.append('<circle cx="234" cy="164" r="9.5" fill="#dc2626" stroke="#ffffff" stroke-width="1.8"/>')
    elems.append('<circle cx="188" cy="212" r="9.5" fill="#dc2626" stroke="#ffffff" stroke-width="1.8"/>')
    elems.append('<path d="M 130,166 Q 178,152 220,162" fill="none" stroke="#991b1b" stroke-width="2.3" stroke-dasharray="4,3"/>')
    elems.append('<path d="M 130,170 Q 158,195 176,208" fill="none" stroke="#991b1b" stroke-width="2.3" stroke-dasharray="4,3"/>')
    elems.append('<text x="200" y="96" text-anchor="middle" font-size="13.2" font-weight="700" fill="#b91c1c">3-Way Seam Splits 1 Defect → 2</text>')

    elems.append('<rect x="34" y="362" width="332" height="108" rx="8" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.3"/>')
    elems.append('<text x="200" y="388" text-anchor="middle" font-size="13.2" font-weight="600" fill="#374151">Flat sheet move: 1 defect → 1 (ΔE = 0)</text>')
    elems.append('<text x="200" y="416" text-anchor="middle" font-size="13.4" font-weight="700" fill="#047857">3-way seam move: 1 defect → 2 (ΔE &gt; 0)</text>')
    elems.append('<text x="200" y="446" text-anchor="middle" font-size="12.8" font-weight="600" fill="#1f2937">Every scale multiplies the energy barrier</text>')

    bx0, by0, bsz = 480, 96, 220
    elems.append(f'<rect x="{bx0}" y="{by0}" width="{bsz}" height="{bsz}" fill="#dbeafe" stroke="#1d4ed8" stroke-width="2.8"/>')
    for qx, qy in [(bx0+55, by0+60), (bx0+165, by0+60), (bx0+55, by0+170), (bx0+165, by0+170)]:
        elems.append(f'<text x="{qx}" y="{qy}" text-anchor="middle" font-size="12.4" font-weight="700" fill="#1e40af">Sheet 0</text>')
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

    save_svg_png("fig_v13_2_architecture", elems, W, H)


def render_fig4_contextualized_rg_and_5scales(rg, kmc):
    W, H = 1180, 545
    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<style>text { font-family: -apple-system, Helvetica, Arial, sans-serif; fill: #1f2937; }</style>',
    ]
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

    rg_series = []
    for b, col, dsh, lbl in [
        (1.35, "#dc2626", "5,4", "Warm Bath (β = 1.35)"),
        (1.86, "#ea580c", "5,4", "Just Above Tc (β = 1.86)"),
        (2.00, "#1f2937", "",    "Critical Threshold (β = 2.00)"),
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

    # Panel (b): All 5 Consecutive Scales (19 -> 203 -> 2,311 -> 27,107 -> 322,159 qubits)!
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

    top_b = 2.25 if ("sq_rx_iso_s5_X_err", "uf", 2.25) in kmc else 2.20
    top_y = max(kmc[("sq_rx_iso_s4_X_err", "uf", 2.25)][1], kmc.get(("sq_rx_iso_s5_X_err", "uf", 2.25), (0, 1e4))[1])
    gain_5 = top_y / kmc[("sq_rx_iso_s1_X_err", "uf", 2.25)][1]

    p2 = dict(
        xlim=(1.20, 2.48), ylim=(2.5, max(1.5e5, top_y * 4.5)),
        title="(b) Five Scales Converge Onto Fixed Threshold",
        xlabel="Inverse Bath Temperature β = 1 / T (Colder →)",
        ylabel="Mean Memory Lifetime",
        xticks=[(1.25,"1.25"), (1.5,"1.5"), (1.75,"1.75"), (2.0,"2.0"), (2.25,"2.25")],
        yticks=[(1e1,"10¹"), (1e2,"10²"), (1e3,"10³"), (1e4,"10⁴"), (1e5,"10⁵")],
        series=ms_series, xlog=False, ylog=True, leg_pos="tl",
        xbands=[(2.00, 2.48, "#ecfdf5")],
        vlines=[(2.00, "#374151", "Fixed Threshold β ≈ 2.0", "rb")],
        varrows=[(2.25, kmc[("sq_rx_iso_s1_X_err", "uf", 2.25)][1], top_y, f"{gain_5:,.0f}×", "#7f1d1d", "r", 9.0e2)],
    )

    draw_panel(elems, 86,  122, 472, 360, **p1)
    draw_panel(elems, 676, 122, 472, 360, **p2)
    save_svg_png("fig_v13_4_threshold_rg", elems, W, H)


def render_fig6_wires_visual_and_plot(v7_best):
    W, H = 1180, 456
    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<style>text { font-family: -apple-system, Helvetica, Arial, sans-serif; fill: #1f2937; }</style>',
    ]
    x0, y0, pw, ph = 42, 42, 516, 352
    elems.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" rx="10" fill="#f9fafb" stroke="#374151" stroke-width="1.5"/>')
    elems.append(f'<text x="{x0+pw/2:.1f}" y="{y0-13:.1f}" text-anchor="middle" font-size="14.8" font-weight="700">(a) Why Single-Grid Routing Adds Wire Entropy</text>')

    elems.append(f'<rect x="{x0+18}" y="{y0+18}" width="{pw-36}" height="146" rx="8" fill="#ecfdf5" stroke="#a7f3d0" stroke-width="1.4"/>')
    elems.append(f'<text x="{x0+pw/2:.1f}" y="{y0+42}" text-anchor="middle" font-size="13.6" font-weight="700" fill="#065f46">Compact Graph Connectivity (1,480 Qubits)</text>')
    jx1, jx2, jy = x0 + 156, x0 + 324, y0 + 94
    elems.append(f'<rect x="{jx1-36}" y="{jy-30}" width="36" height="60" rx="6" fill="#fde68a" stroke="#b45309" stroke-width="2.2"/>')
    elems.append(f'<rect x="{jx2}" y="{jy-30}" width="36" height="60" rx="6" fill="#fde68a" stroke="#b45309" stroke-width="2.2"/>')
    elems.append(f'<line x1="{jx1}" y1="{jy}" x2="{jx2}" y2="{jy}" stroke="#047857" stroke-width="4.0"/>')
    elems.append(f'<circle cx="{(jx1+jx2)/2:.1f}" cy="{jy}" r="8.5" fill="#059669" stroke="#ffffff" stroke-width="1.6"/>')
    elems.append(f'<text x="{x0+pw/2:.1f}" y="{y0+148}" text-anchor="middle" font-size="12.6" font-weight="600" fill="#065f46">'
                 '1 qubit between 3-way junctions → Every flip costs +ΔE</text>')

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
    save_svg_png("fig_v13_6_grid_wires", elems, W, H)


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

    kmc9, rg9, cool11 = {}, {}, {}
    for fn in (
        "cluster_v8_results.txt",
        "cluster_v9_results.txt",
        "cluster_v9b_results.txt",
        "cluster_v10u_results.txt",
        "cluster_v11_results.txt",
    ):
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
                if parts[0] == "kmc" and int(kv.get("capped", "0")) == 0 and int(kv.get("bails", "0")) == 0:
                    _, tag, jm, dec, bs = parts
                    kmc9[(tag, dec, float(bs[1:]))] = (int(kv["N"]), float(kv["mean_t"]))
                elif parts[0] in ("rg", "rgflow"):
                    _, rg_nm, bs = parts
                    S_max = int(kv["S_max"])
                    rhos = [float(kv[f"rho_{s}"]) for s in range(S_max + 1)]
                    rg9[(rg_nm, float(bs[1:]))] = rhos
                elif parts[0] == "cool":
                    _, tag, proto, init_m, ts = parts
                    cool11[(tag, proto, init_m, float(ts[1:]))] = {
                        "N": int(kv["N"]),
                        "K": int(kv["K"]),
                        "syn": float(kv["syn_dens"]),
                        "p0": float(kv["p_zero_syn"]),
                    }

    return kmc4, kmc6, kmc7, ffs7, cool7, kmc9, rg9, cool11


def main():
    render_fig2_large_architecture()
    kmc4, kmc6, kmc7, ffs7, cool7, kmc9, rg9, cool11 = load_all()

    def v7_best(tag, jm, dec, b, min_reps=3):
        if (tag, jm, dec, b) in kmc7 and kmc7[(tag, jm, dec, b)][2] == 0:
            return kmc7[(tag, jm, dec, b)][1]
        if (tag, jm, dec, b) in ffs7 and ffs7[(tag, jm, dec, b)][2] >= min_reps:
            return ffs7[(tag, jm, dec, b)][1]
        return None

    # 100% Unified Uniform 2x2 3-Level Quantum Hierarchy (`sq_iso_L2r1` s=1,3,5: 19 -> 2,911 -> 423,559q):
    q_top_b = 6.50 if ("sq_iso_L2r1_s5_Z_err", "uf", 6.50) in kmc9 else 6.35
    q3_pts = [kmc9[(f"sq_iso_L2r1_s{s}_Z_err", "uf", q_top_b)] for s in [1, 3, 5]]
    q3_x = [p[0] for p in q3_pts]
    q3_y = [p[1] / q3_pts[0][1] for p in q3_pts]

    w_top_b = 2.25 if ("sq_rx_iso_s5_X_err", "uf", 2.25) in kmc9 else 2.20
    w_scales = [1, 2, 3, 4, 5] if ("sq_rx_iso_s5_X_err", "uf", w_top_b) in kmc9 else [1, 2, 3, 4]
    w_pts = [kmc9[(f"sq_rx_iso_s{s}_X_err", "uf", w_top_b)] for s in w_scales]
    w_x = [p[0] for p in w_pts]
    w_y = [p[1] / w_pts[0][1] for p in w_pts]

    # --- 1. `fig_v13_1_classes.png`: Clean Code-Family Comparison up to 423,559 Qubits ---
    def v4_norm_curve(cnames, b):
        pts = [(kmc4[cn][b][0], kmc4[cn][b][1]) for cn in cnames if b in kmc4.get(cn, {})]
        y0 = pts[0][1]
        return [x for x, _ in pts], [y / y0 for _, y in pts]

    hx, hy = v4_norm_curve([f"haah_cubic_L{L}_Xerr.code" for L in [3, 5, 7, 9, 11, 13]], 4.25)
    bx, by = v4_norm_curve([f"bb_gross_{l}x{m}_Xerr.code" for l, m in [(6,6),(12,6),(18,12),(24,18),(30,18),(36,18),(48,18),(60,18)]], 4.25)
    tx_n, ty_n = v4_norm_curve([f"toric2d_L{L}_Xerr.code" for L in [4, 6, 10, 16, 24, 34]], 4.50)

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
        if len(b_cands) >= 2:
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
        "fig_v13_1_classes",
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

    # --- 3. `fig_v13_3_validation.png`: Simulator Validation ---
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
        "fig_v13_3_validation",
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

    # --- 4. `fig_v13_4_threshold_rg.png`: Contextualized Zoom-Out & 5-Scale Crossings ---
    render_fig4_contextualized_rg_and_5scales(rg9, kmc9)

    # --- 5. `fig_v13_5_quantum_scaling.png`: 100% Unified 423,559q Quantum + Local RG vs UF ---
    q_s1_bs = [b for b in (4.50, 5.00, 5.50, 5.75, 5.85, 6.00, 6.10, 6.25, 6.35, 6.50) if ("sq_iso_L2r1_s1_Z_err", "uf", b) in kmc9]
    q_s3_bs = [b for b in (4.50, 5.00, 5.50, 5.75, 5.85, 6.00, 6.10, 6.25, 6.35, 6.50) if ("sq_iso_L2r1_s3_Z_err", "uf", b) in kmc9]
    q_s5_bs = [b for b in (5.50, 5.75, 5.85, 6.00, 6.10, 6.25, 6.35, 6.50) if ("sq_iso_L2r1_s5_Z_err", "uf", b) in kmc9]
    gain_q3 = kmc9[("sq_iso_L2r1_s5_Z_err", "uf", q_s5_bs[-1])][1] / kmc9[("sq_iso_L2r1_s1_Z_err", "uf", q_s5_bs[-1])][1]

    tor_uf_bs = [b for b in (5.50, 6.00, 6.25, 6.50, 6.65) if ("rottor_iso_L2r1_s4_X_err", "uf", b) in kmc9]
    tor_rg_bs = [b for b in (5.50, 6.00, 6.25, 6.50, 6.65) if ("rottor_iso_L2r1_s4_X_err", "rg_dec", b) in kmc9]
    tor_gain = kmc9[("rottor_iso_L2r1_s4_X_err", "uf", tor_uf_bs[-1])][1] / kmc9[("rottor_iso_L2r1_s2_X_err", "uf", tor_uf_bs[-1])][1]

    print(f"Unified Quantum Fig 5: 423,559q top_b={q_s5_bs[-1]} (gain={gain_q3:,.0f}x, barriers={[round(x,2) for x in q3_bD]}), torus gain={tor_gain:.1f}x")

    render_double(
        "fig_v13_5_quantum_scaling",
        dict(xlim=(4.40, 6.84), ylim=(6.0, max(8.0e6, kmc9[("sq_iso_L2r1_s5_Z_err", "uf", q_s5_bs[-1])][1] * 4.5)),
             title="(a) Uniform 2×2 Quantum Levels to 423,559 Qubits",
             xlabel="Inverse Bath Temperature β = 1 / T (Colder →)", ylabel="Quantum Memory Lifetime",
             xticks=[(4.5,"4.5"), (5.0,"5.0"), (5.5,"5.5"), (6.0,"6.0"), (6.5,"6.5")],
             yticks=[(1e1,"10¹"), (1e2,"10²"), (1e3,"10³"), (1e4,"10⁴"), (1e5,"10⁵"), (1e6,"10⁶")],
             series=[
                 {"xs": q_s5_bs, "ys": [kmc9[("sq_iso_L2r1_s5_Z_err", "uf", b)][1] for b in q_s5_bs],
                  "color": "#be123c", "dash": "",    "label": f"Level 3 (423,559q: slope = {q3_bD[2]:.1f})"},
                 {"xs": q_s3_bs, "ys": [kmc9[("sq_iso_L2r1_s3_Z_err", "uf", b)][1] for b in q_s3_bs],
                  "color": "#059669", "dash": "",    "label": f"Level 2 (2,911q: slope = {q3_bD[1]:.1f})"},
                 {"xs": q_s1_bs, "ys": [kmc9[("sq_iso_L2r1_s1_Z_err", "uf", b)][1] for b in q_s1_bs],
                  "color": "#6b7280", "dash": "4,3", "label": f"Level 1 (19q: slope = {q3_bD[0]:.1f})"},
             ],
             xlog=False, ylog=True, leg_pos="tl",
             varrows=[(q_s5_bs[-1], kmc9[("sq_iso_L2r1_s1_Z_err", "uf", q_s5_bs[-1])][1],
                       kmc9[("sq_iso_L2r1_s5_Z_err", "uf", q_s5_bs[-1])][1],
                       f"{gain_q3:,.0f}×", "#be123c", "r", 3.5e3)]),
        dict(xlim=(5.40, 6.92), ylim=(5.0e2, 4.0e7),
             title="(b) Torus (94,774q): Local O(N) RG vs. Global UF",
             xlabel="Inverse Bath Temperature β = 1 / T (Colder →)", ylabel="Quantum Memory Lifetime",
             xticks=[(5.5,"5.5"), (6.0,"6.0"), (6.25,"6.25"), (6.5,"6.5"), (6.75,"6.75")],
             yticks=[(1e3,"10³"), (1e4,"10⁴"), (1e5,"10⁵"), (1e6,"10⁶"), (1e7,"10⁷")],
             series=[
                 {"xs": tor_uf_bs, "ys": [kmc9[("rottor_iso_L2r1_s4_X_err", "uf", b)][1] for b in tor_uf_bs],
                  "color": "#be123c", "dash": "",    "label": f"Cycle 2 (94,774q): Global UF ({tor_gain:.0f}×)"},
                 {"xs": tor_rg_bs, "ys": [kmc9[("rottor_iso_L2r1_s4_X_err", "rg_dec", b)][1] for b in tor_rg_bs],
                  "color": "#ea580c", "dash": "6,3", "hollow": True, "label": "Cycle 2 (94,774q): Local O(N) RG"},
                 {"xs": tor_uf_bs, "ys": [kmc9[("rottor_iso_L2r1_s2_X_err", "uf", b)][1] for b in tor_uf_bs],
                  "color": "#2563eb", "dash": "",    "label": "Cycle 1 (646q): Global UF"},
                 {"xs": tor_rg_bs, "ys": [kmc9[("rottor_iso_L2r1_s2_X_err", "rg_dec", b)][1] for b in tor_rg_bs],
                  "color": "#059669", "dash": "4,3", "hollow": True, "label": "Cycle 1 (646q): Local O(N) RG"},
             ],
             xlog=False, ylog=True, leg_pos="tl",
             varrows=[(tor_uf_bs[-1], kmc9[("rottor_iso_L2r1_s2_X_err", "uf", tor_uf_bs[-1])][1],
                       kmc9[("rottor_iso_L2r1_s4_X_err", "uf", tor_uf_bs[-1])][1],
                       f"{tor_gain:.0f}×", "#be123c", "r", 1.1e6)]),
    )

    # --- 6. `fig_v13_6_grid_wires.png`: Visual Wire Mechanism + Grid Data ---
    render_fig6_wires_visual_and_plot(v7_best)

    # --- 7. `fig_v13_7_cooling.png`: Storage Self-Healing vs Multi-Scale Cooling Law t_cool(N) ---
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

    def family_tcool(fam, proto, scales=(1, 2, 3, 4), thresh=0.002):
        ns, ts_out = [], []
        for s in scales:
            tag = f"{fam}_s{s}_X_err"
            t_list = sorted(tt for (tg, pr, im, tt) in cool11 if tg == tag and pr == proto and im == "hot")
            if not t_list:
                continue
            N = cool11[(tag, proto, "hot", t_list[0])]["N"]
            syns = [cool11[(tag, proto, "hot", tt)]["syn"] for tt in t_list]
            tc = interp_tcool(t_list, syns, thresh, decreasing=True)
            ns.append(N)
            ts_out.append(tc if tc is not None else 1.5e5)
        return ns, ts_out

    rx_an_n, rx_an_t = family_tcool("sq_rx_iso", "anneal", (1, 2, 3, 4), thresh=0.0030)
    rx_qu_n, rx_qu_t = family_tcool("sq_rx_iso", "quench", (1, 2, 3, 4), thresh=0.0030)
    sq_an_n, sq_an_t = family_tcool("sq_iso_L2r1", "anneal", (1, 2, 3, 4), thresh=0.0030)
    sq_qu_n, sq_qu_t = family_tcool("sq_iso_L2r1", "quench", (1, 2, 3), thresh=0.0030)

    gam_rx = ols_slope([math.log(n) for n in rx_an_n[1:]], rx_an_t[1:])
    print(f"Cooling Fig 7b (<0.3%): sq_rx_iso anneal N={rx_an_n} t={rx_an_t} (gamma={gam_rx:.2f}); sq anneal N={sq_an_n} t={sq_an_t}")

    s_cool_scale = [
        {"xs": rx_qu_n,  "ys": rx_qu_t,  "color": "#dc2626", "dash": "5,4", "hollow": True,
         "label": "Nested X-Welded: Sudden Quench (Pinned)"},
        {"xs": rx_an_n,  "ys": rx_an_t,  "color": "#be123c", "dash": "",
         "label": f"Nested X-Welded: Annealing (~ N^{gam_rx:.2f})"},
        {"xs": sq_qu_n,  "ys": sq_qu_t,  "color": "#f97316", "dash": "4,3", "hollow": True,
         "label": "Planar Quantum: Sudden Quench (10× slower)"},
        {"xs": sq_an_n,  "ys": sq_an_t,  "color": "#059669", "dash": "",
         "label": "Planar Quantum: Annealing (to 47,387q)"},
    ]

    render_double(
        "fig_v13_7_cooling",
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
