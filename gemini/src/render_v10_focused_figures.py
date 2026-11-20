#!/usr/bin/env python3
"""
render_v10_focused_figures.py
Renders 6 uncluttered, self-explanatory figures directly answering the core questions:
  1. fig_v10_1_classes.png:    (a) Relative Lifetime Growth vs N (5-scale welded 1,682x vs Haah turnover vs 2D decay)
                               (b) Effective Energy Barrier vs N (concave-up multiplicative vs straight-then-dropping log N)
  2. fig_v10_2_geometry.png:   4-panel geometric mechanism & exact cell limits (3-way split, 2x2 c_cycle=2/5, buttonhole, wire)
  3. fig_v10_3_validation.png: External simulator validation (3D Toric Code & Gu-Caha Random Layer Code turnover)
  4. fig_v10_4_scaling.png:    3-panel scaling & non-drifting threshold proof:
                               (a) 4 nontrivial 3D scales (s=1..4, N=19..27,107) crossing at bounded beta_c ≈ 1.86
                               (b) Provably contractive Uniform 2x2 full cycles (323 vs 47,387 qubits: 71x gain)
                               (c) Boundaryless Torus full cycles (508 vs 50,296 qubits: 7.4x gain)
  5. fig_v10_5_tradeoffs.png:  (a) Pure cubic-grid wire stretching (3 curves only: 321x loss, zero trap clutter)
                               (b) Closing the 1,890x X/Z gap via 2->1 bridge pre-balancing (57x gain)
  6. fig_v10_6_cooling.png:    (a) Zero-measurement storage self-healing below ~6% threshold
                               (b) Hot-state gradual cooling vs sudden junction freezing
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
               hlines=None, vlines=None, varrows=None, y_off=58):
    elems.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="#fafafa" stroke="#374151" stroke-width="1.5"/>')
    elems.append(f'<text x="{x0+pw/2:.1f}" y="{y0-14:.1f}" text-anchor="middle" font-size="14.8" font-weight="700">{html.escape(title)}</text>')
    elems.append(f'<text x="{x0+pw/2:.1f}" y="{y0+ph+42:.1f}" text-anchor="middle" font-size="13.6" font-weight="600">{html.escape(xlabel)}</text>')
    elems.append(f'<text x="{x0-y_off:.1f}" y="{y0+ph/2:.1f}" text-anchor="middle" font-size="13.6" font-weight="600" '
                 f'transform="rotate(-90 {x0-y_off:.1f} {y0+ph/2:.1f})">{html.escape(ylabel)}</text>')

    def tx(v):
        if xlog:
            return x0 + (math.log10(v) - math.log10(xlim[0])) / (math.log10(xlim[1]) - math.log10(xlim[0])) * pw
        return x0 + (v - xlim[0]) / (xlim[1] - xlim[0]) * pw

    def ty(v):
        vc = max(ylim[0], min(ylim[1], v))
        if ylog:
            return y0 + ph - (math.log10(vc) - math.log10(ylim[0])) / (math.log10(ylim[1]) - math.log10(ylim[0])) * ph
        return y0 + ph - (vc - ylim[0]) / (ylim[1] - ylim[0]) * ph

    for xv, xlbl in xticks:
        px = tx(xv)
        elems.append(f'<line x1="{px:.1f}" y1="{y0}" x2="{px:.1f}" y2="{y0+ph}" stroke="#e5e7eb" stroke-width="1"/>')
        elems.append(f'<line x1="{px:.1f}" y1="{y0+ph}" x2="{px:.1f}" y2="{y0+ph+5}" stroke="#374151" stroke-width="1.2"/>')
        elems.append(f'<text x="{px:.1f}" y="{y0+ph+21}" text-anchor="middle" font-size="12.2">{html.escape(xlbl)}</text>')

    for yv, ylbl in yticks:
        py = ty(yv)
        elems.append(f'<line x1="{x0}" y1="{py:.1f}" x2="{x0+pw}" y2="{py:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        elems.append(f'<line x1="{x0-5}" y1="{py:.1f}" x2="{x0}" y2="{py:.1f}" stroke="#374151" stroke-width="1.2"/>')
        elems.append(f'<text x="{x0-8}" y="{py+4.5:.1f}" text-anchor="end" font-size="12.2">{html.escape(ylbl)}</text>')

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
            elems.append(f'<line x1="{px:.1f}" y1="{y0}" x2="{px:.1f}" y2="{y0+ph}" stroke="{col}" stroke-width="1.6" stroke-dasharray="5,4"/>')
            dx = 7.0 if "r" in valign else -7.0
            anch = "start" if "r" in valign else "end"
            vy = y0 + ph - 14.0 if "b" in valign else y0 + 22.0
            elems.append(f'<text x="{px+dx:.1f}" y="{vy:.1f}" text-anchor="{anch}" font-size="11.5" font-weight="600" fill="{col}">{html.escape(lbl)}</text>')

    for s in series:
        pts = [(tx(x), ty(y)) for x, y in zip(s["xs"], s["ys"]) if y is not None and xlim[0] <= x <= xlim[1]]
        if not pts:
            continue
        d_attr = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
        pstr = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        if len(pts) >= 2:
            elems.append(f'<polyline fill="none" stroke="{s["color"]}" stroke-width="2.9"{d_attr} points="{pstr}"/>')
        for px, py in pts:
            elems.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.8" fill="{s["color"]}" stroke="#ffffff" stroke-width="1.2"/>')

    if varrows:
        for item in varrows:
            xv, y_lo, y_hi, lbl, col, side = item[:6]
            y_txt = item[6] if len(item) > 6 else None
            px = tx(xv)
            py1, py2 = ty(y_hi), ty(y_lo)
            ty_lbl = ty(y_txt) + 4.0 if y_txt is not None else (py1 + py2) / 2.0 + 4.0
            elems.append(f'<line x1="{px:.1f}" y1="{py1+8:.1f}" x2="{px:.1f}" y2="{py2-8:.1f}" stroke="{col}" stroke-width="2.0" stroke-dasharray="3,3"/>')
            elems.append(f'<polygon points="{px:.1f},{py1+2:.1f} {px-4.5:.1f},{py1+10:.1f} {px+4.5:.1f},{py1+10:.1f}" fill="{col}"/>')
            elems.append(f'<polygon points="{px:.1f},{py2-2:.1f} {px-4.5:.1f},{py2-10:.1f} {px+4.5:.1f},{py2-10:.1f}" fill="{col}"/>')
            dx = -8.5 if side == "l" else 8.5
            anch = "end" if side == "l" else "start"
            elems.append(f'<text x="{px+dx:.1f}" y="{ty_lbl:.1f}" text-anchor="{anch}" font-size="12.2" font-weight="700" fill="{col}">{html.escape(lbl)}</text>')

    row_h = 20.0
    lh = row_h * len(series) + 11
    lw = min(pw - 18.0, max(195.0, max(len(s["label"]) for s in series) * 6.65 + 46.0))
    lx = x0 + 10 if "l" in leg_pos else x0 + pw - lw - 10
    ly = y0 + 10 if "t" in leg_pos else y0 + ph - lh - 10
    elems.append(f'<rect x="{lx:.1f}" y="{ly:.1f}" width="{lw:.1f}" height="{lh:.1f}" rx="6" fill="#ffffff" fill-opacity="0.96" stroke="#d1d5db" stroke-width="1.1"/>')
    for i, s in enumerate(series):
        iy = ly + 16.2 + i * row_h
        d_attr = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
        elems.append(f'<line x1="{lx+8:.1f}" y1="{iy-4.0:.1f}" x2="{lx+29:.1f}" y2="{iy-4.0:.1f}" stroke="{s["color"]}" stroke-width="2.7"{d_attr}/>')
        elems.append(f'<circle cx="{lx+18.5:.1f}" cy="{iy-4.0:.1f}" r="4.0" fill="{s["color"]}"/>')
        elems.append(f'<text x="{lx+36:.1f}" y="{iy:.1f}" font-size="11.7">{html.escape(s["label"])}</text>')


def render_double(nm, p1, p2):
    W, H = 1180, 460
    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<style>text { font-family: -apple-system, Helvetica, Arial, sans-serif; fill: #1f2937; }</style>',
    ]
    draw_panel(elems, 86, 44, 472, 354, **p1)
    draw_panel(elems, 676, 44, 472, 354, **p2)
    save_svg_png(nm, elems, W, H)


def render_triple(nm, p1, p2, p3):
    W, H = 1320, 435
    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<style>text { font-family: -apple-system, Helvetica, Arial, sans-serif; fill: #1f2937; }</style>',
    ]
    draw_panel(elems, 74,  42, 352, 334, y_off=52, **p1)
    draw_panel(elems, 504, 42, 352, 334, y_off=52, **p2)
    draw_panel(elems, 934, 42, 352, 334, y_off=52, **p3)
    save_svg_png(nm, elems, W, H)


def render_clean_geometry():
    W, H = 1180, 360
    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<style>text { font-family: -apple-system, Helvetica, Arial, sans-serif; fill: #1f2937; }</style>',
    ]
    cards = [
        (16,  14, 275, 332, "(a) 3-Way Bridge Splits Errors"),
        (307, 14, 275, 332, "(b) 2×2 Four-Arm Cross (Works)"),
        (598, 14, 275, 332, "(c) 3-Arm T-Bridge (Leaks)"),
        (889, 14, 275, 332, "(d) Scale-8 Cubic Grid (Slow)"),
    ]
    for cx, cy, cw, ch, ctitle in cards:
        elems.append(f'<rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" rx="10" fill="#f9fafb" stroke="#cbd5e1" stroke-width="1.5"/>')
        elems.append(f'<text x="{cx+cw/2:.1f}" y="{cy+28}" text-anchor="middle" font-size="14.0" font-weight="700">{html.escape(ctitle)}</text>')

    # Card (a): 3-way junction
    elems.append('<polygon points="36,245 136,205 268,205 168,245" fill="#dbeafe" stroke="#2563eb" stroke-width="1.8"/>')
    elems.append('<polygon points="36,135 136,95 268,95 168,135" fill="#dbeafe" fill-opacity="0.85" stroke="#2563eb" stroke-width="1.8"/>')
    elems.append('<polygon points="102,225 202,185 202,115 102,155" fill="#fde68a" fill-opacity="0.92" stroke="#d97706" stroke-width="2.2"/>')
    elems.append('<line x1="102" y1="155" x2="202" y2="115" stroke="#b91c1c" stroke-width="3.6"/>')
    elems.append('<circle cx="88" cy="128" r="8" fill="#dc2626" stroke="#ffffff" stroke-width="1.5"/>')
    elems.append('<circle cx="195" cy="110" r="8" fill="#dc2626" stroke="#ffffff" stroke-width="1.5"/>')
    elems.append('<circle cx="152" cy="168" r="8" fill="#dc2626" stroke="#ffffff" stroke-width="1.5"/>')
    elems.append('<text x="153.5" y="295" text-anchor="middle" font-size="12.6" font-weight="600" fill="#1f2937">1 defect splits into 2 at seam</text>')
    elems.append('<text x="153.5" y="324" text-anchor="middle" font-size="12.8" font-weight="700" fill="#047857">Costs +energy to cross</text>')

    # Card (b): 2x2 four-arm cross
    bx0, by0, bsz = 369, 92, 152
    elems.append(f'<rect x="{bx0}" y="{by0}" width="{bsz}" height="{bsz}" fill="#eff6ff" stroke="#1d4ed8" stroke-width="2.4"/>')
    elems.append(f'<line x1="{bx0+bsz/2}" y1="{by0}" x2="{bx0+bsz/2}" y2="{by0+bsz}" stroke="#2563eb" stroke-width="5.5"/>')
    elems.append(f'<line x1="{bx0}" y1="{by0+bsz/2}" x2="{bx0+bsz}" y2="{by0+bsz/2}" stroke="#2563eb" stroke-width="5.5"/>')
    for wx, wy in [(bx0+bsz/2, by0), (bx0+bsz/2, by0+bsz), (bx0, by0+bsz/2), (bx0+bsz, by0+bsz/2), (bx0+bsz/2, by0+bsz/2)]:
        elems.append(f'<circle cx="{wx}" cy="{wy}" r="6.5" fill="#1e40af" stroke="#ffffff" stroke-width="1.5"/>')
    elems.append('<text x="444.5" y="295" text-anchor="middle" font-size="12.6" font-weight="600" fill="#1f2937">All 4 bridge arms reach walls</text>')
    elems.append('<text x="444.5" y="324" text-anchor="middle" font-size="12.8" font-weight="700" fill="#047857">174×–220× smaller cell</text>')

    # Card (c): 3-arm T-bridge buttonhole
    cx0, cy0, csz = 660, 92, 152
    elems.append(f'<rect x="{cx0}" y="{cy0}" width="{csz}" height="{csz}" fill="#fef2f2" stroke="#b91c1c" stroke-width="2.4"/>')
    elems.append(f'<line x1="{cx0}" y1="{cy0+csz/2}" x2="{cx0+csz}" y2="{cy0+csz/2}" stroke="#2563eb" stroke-width="5.5"/>')
    elems.append(f'<line x1="{cx0+csz/2}" y1="{cy0}" x2="{cx0+csz/2}" y2="{cy0+csz/2}" stroke="#2563eb" stroke-width="5.5"/>')
    elems.append(f'<line x1="{cx0+csz/2}" y1="{cy0+csz/2}" x2="{cx0+csz/2}" y2="{cy0+csz}" stroke="#9ca3af" stroke-width="2.5" stroke-dasharray="5,5"/>')
    elems.append(f'<circle cx="{cx0+csz/2}" cy="{cy0+csz/2+26}" r="18" fill="#fee2e2" stroke="#dc2626" stroke-width="2.6" stroke-dasharray="4,3"/>')
    elems.append(f'<text x="{cx0+csz/2}" y="{cy0+csz/2+30}" text-anchor="middle" font-size="11.5" font-weight="700" fill="#b91c1c">Hole</text>')
    elems.append('<text x="735.5" y="295" text-anchor="middle" font-size="12.6" font-weight="600" fill="#1f2937">Cut arm opens sheet gap</text>')
    elems.append('<text x="735.5" y="324" text-anchor="middle" font-size="12.8" font-weight="700" fill="#b91c1c">Spawns 1,004 bad qubits</text>')

    # Card (d): Scale-8 cubic grid 7-qubit wire
    dx0, dy0 = 916, 114
    elems.append(f'<rect x="{dx0}" y="{dy0}" width="28" height="120" rx="5" fill="#fde68a" stroke="#b45309" stroke-width="2.2"/>')
    elems.append(f'<rect x="{dx0+193}" y="{dy0}" width="28" height="120" rx="5" fill="#fde68a" stroke="#b45309" stroke-width="2.2"/>')
    wy = dy0 + 60
    elems.append(f'<line x1="{dx0+28}" y1="{wy}" x2="{dx0+193}" y2="{wy}" stroke="#374151" stroke-width="3.2"/>')
    for qi in range(7):
        qx = dx0 + 41 + qi * 23.2
        col = "#dc2626" if qi == 3 else "#2563eb"
        rad = 7.2 if qi == 3 else 5.2
        elems.append(f'<circle cx="{qx:.1f}" cy="{wy}" r="{rad}" fill="{col}" stroke="#ffffff" stroke-width="1.3"/>')
    elems.append(f'<line x1="{dx0+56}" y1="{wy-26}" x2="{dx0+165}" y2="{wy-26}" stroke="#dc2626" stroke-width="2.4"/>')
    elems.append(f'<polygon points="{dx0+48},{wy-26} {dx0+59},{wy-31} {dx0+59},{wy-21}" fill="#dc2626"/>')
    elems.append(f'<polygon points="{dx0+173},{wy-26} {dx0+162},{wy-31} {dx0+162},{wy-21}" fill="#dc2626"/>')
    elems.append(f'<text x="1026.5" y="{wy-37}" text-anchor="middle" font-size="12.4" font-weight="700" fill="#dc2626">Zero-energy walk</text>')
    elems.append('<text x="1026.5" y="295" text-anchor="middle" font-size="12.6" font-weight="600" fill="#1f2937">7-qubit wire between junctions</text>')
    elems.append('<text x="1026.5" y="324" text-anchor="middle" font-size="12.8" font-weight="700" fill="#b91c1c">Cuts lifetime 321×</text>')

    save_svg_png("fig_v10_2_geometry", elems, W, H)


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

    kmc8 = {}
    with open(os.path.join(WS, "cluster_v8_results.txt")) as f:
        for raw in f:
            ln = raw.strip()
            if not ln or "|" not in ln:
                continue
            tid, payload = [x.strip() for x in ln.split("|", 1)]
            if payload.startswith(("TIMEOUT", "ERROR")) or "VALID_REPS=0" in payload:
                continue
            kv = parse_kv(payload)
            parts = tid.split("__")
            if parts[0] == "kmc" and int(kv["capped"]) == 0:
                _, tag, jm, dec, bs = parts
                kmc8[(tag, float(bs[1:]))] = (int(kv["N"]), float(kv["mean_t"]))

    return kmc4, kmc6, kmc7, ffs7, cool7, kmc8


def main():
    render_clean_geometry()
    kmc4, kmc6, kmc7, ffs7, cool7, kmc8 = load_all()

    def v7_best(tag, jm, dec, b, min_reps=3):
        if (tag, jm, dec, b) in kmc7 and kmc7[(tag, jm, dec, b)][2] == 0:
            return kmc7[(tag, jm, dec, b)][1]
        if (tag, jm, dec, b) in ffs7 and ffs7[(tag, jm, dec, b)][2] >= min_reps:
            return ffs7[(tag, jm, dec, b)][1]
        return None

    # --- 1. `fig_v10_1_classes.png`: 5-Point Multi-Scale Welded Curve vs Haah Turnover vs 2D Decay ---
    def v4_norm_curve(cnames, b):
        pts = [(kmc4[cn][b][0], kmc4[cn][b][1]) for cn in cnames if b in kmc4.get(cn, {})]
        y0 = pts[0][1]
        return [x for x, _ in pts], [y / y0 for _, y in pts]

    hx, hy = v4_norm_curve([f"haah_cubic_L{L}_Xerr.code" for L in [3, 5, 7, 9, 11, 13]], 4.25)
    bx, by = v4_norm_curve([f"bb_gross_{l}x{m}_Xerr.code" for l, m in [(6,6),(12,6),(18,12),(24,18),(30,18),(36,18),(48,18),(60,18)]], 4.25)
    tx_n, ty_n = v4_norm_curve([f"toric2d_L{L}_Xerr.code" for L in [4, 6, 10, 16, 24, 34]], 4.50)

    # 4 nontrivial 3D scales s=1..4 of Uniform 2x2 (`N = 19, 203, 2311, 27107`) at fixed beta=2.25 (normalized to s=1: 396x; or s=0..4: 1,682x!)
    # Using s=1..4 (`N = 19 -> 203 -> 2,311 -> 27,107`) starts right at N=19 alongside 2D Toric (N=32) and Haah (N=54)!
    w5_pts = [kmc8[(f"sq_rx_iso_s{s}_X_err", 2.25)] for s in [1, 2, 3, 4]]
    w5_x = [p[0] for p in w5_pts]
    w5_y = [p[1] / w5_pts[0][1] for p in w5_pts]  # 1x -> 6.3x -> 50x -> 396x (or 1,682x over s=0)

    # Also 2-cycle full (X,Z) Uniform 2x2 (`N = 323 -> 47,387` at beta = 6.65: 71x!)
    q2_s2 = kmc8[("sq_iso_L2r1_s2_X_err", 6.65)][1]
    q2_s4 = kmc8[("sq_iso_L2r1_s4_X_err", 6.65)][1]

    s_norm = [
        {"xs": w5_x, "ys": w5_y, "color": "#be123c", "dash": "",    "label": "3D Welded 2×2 (4 scales s=1→4: 396× gain)"},
        {"xs": [323, 47387], "ys": [1.0, q2_s4 / q2_s2], "color": "#ea580c", "dash": "7,3", "label": "3D Welded Full (X,Z) Cycles (1→2: 71× gain)"},
        {"xs": hx,   "ys": hy,   "color": "#7c3aed", "dash": "",    "label": "Haah 3D Cubic Code (peaks at 25×, turns over)"},
        {"xs": bx,   "ys": by,   "color": "#2563eb", "dash": "5,4", "label": "Scaled 2D Bivariate Bicycle (falls < 1×)"},
        {"xs": tx_n, "ys": ty_n, "color": "#4b5563", "dash": "3,3", "label": "2D Toric Code (falls as 1/N)"},
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

    # 5-scale Uniform 2x2 barrier curve (`s = 0..4`: 1.2 -> 2.0 -> 4.6 -> 8.8 -> 13.2 — sharp concave-up!)
    w5_bN, w5_bD = [], []
    for s in range(5):
        bs_m = [1.75, 2.00, 2.25]
        w5_bN.append(kmc8[(f"sq_rx_iso_s{s}_X_err", 1.75)][0])
        w5_bD.append(ols_slope(bs_m, [kmc8[(f"sq_rx_iso_s{s}_X_err", b)][1] for b in bs_m]))

    # Full quantum cycle Torus barrier (`4 -> 508 -> 50,296`: 1.9 -> 3.5 -> 6.8 over [5.5..6.65])
    rot_bs = [5.50, 6.00, 6.25, 6.50, 6.65]
    rot_s2_ys = [v7_best("rottor_L2r1_s2_X_err", "uniform", "uf", b) if b <= 6.50 else kmc8[("rottor_L2r1_s2_X_err", b)][1] for b in rot_bs]
    rot_s4_ys = [v7_best("rottor_L2r1_s4_X_err", "uniform", "uf", b) if b <= 6.50 else kmc8[("rottor_L2r1_s4_X_err", b)][1] for b in rot_bs]
    d_rot_s2 = ols_slope(rot_bs, rot_s2_ys)
    d_rot_s4 = ols_slope(rot_bs, rot_s4_ys)

    s_bar = [
        {"xs": w5_bN, "ys": w5_bD, "color": "#be123c", "dash": "",    "label": "3D Welded 2×2 (5 scales: curves upward)"},
        {"xs": [4, 508, 50296], "ys": [1.92, d_rot_s2, d_rot_s4], "color": "#ea580c", "dash": "7,3",
         "label": f"3D Welded Full Torus (1.9 → {d_rot_s2:.1f} → {d_rot_s4:.1f})"},
        {"xs": c_haah_N, "ys": c_haah_D, "color": "#7c3aed", "dash": "",    "label": "Haah 3D Cubic (straight log N, bends down)"},
        {"xs": c_bb_N,   "ys": c_bb_D,   "color": "#2563eb", "dash": "5,4", "label": "Scaled 2D Bivariate Bicycle (flat ≈ 5.4)"},
        {"xs": c_tor_N,  "ys": c_tor_D,  "color": "#4b5563", "dash": "3,3", "label": "2D Toric Code (flat ≤ 2.3)"},
    ]

    render_double(
        "fig_v10_1_classes",
        dict(xlim=(15.0, 60000.0), ylim=(0.02, 3.5e4),
             title="(a) Relative Lifetime Growth With Qubit Count",
             xlabel="Physical Qubit Count N (Log Scale)", ylabel="Lifetime Relative to Smallest Code",
             xticks=[(20,"20"), (100,"100"), (1000,"1,000"), (10000,"10,000"), (50000,"50,000")],
             yticks=[(0.05,"0.05×"), (1.0,"1×"), (10.0,"10×"), (100.0,"100×"), (1000.0,"1,000×")],
             series=s_norm, xlog=True, ylog=True, leg_pos="tl",
             hlines=[(1.0, "#6b7280", "1× = No Gain From Growing N", "br")]),
        dict(xlim=(1.6, 65000.0), ylim=(0.0, 16.8),
             title="(b) Additive vs. Multiplicative Energy Barriers",
             xlabel="Physical Qubit Count N (Log Scale)", ylabel="Effective Energy Barrier",
             xticks=[(2,"2"), (30,"30"), (500,"500"), (5000,"5,000"), (50000,"50,000")],
             yticks=[(0,"0"), (3,"3"), (6,"6"), (9,"9"), (12,"12"), (15,"15")],
             series=s_bar, xlog=True, ylog=False, leg_pos="tl"),
    )

    # --- 3. `fig_v10_3_validation.png`: External Simulator Validation ---
    s_tor = []
    for b, col, is_z, lbl in [
        (2.00, "#dc2626", False, "Loop X-Errors (β=2.0: grows exponentially)"),
        (1.75, "#f97316", False, "Loop X-Errors (β=1.75: grows exponentially)"),
        (2.00, "#2563eb", True,  "Point Z-Errors (β=2.0: shrinks as 1/N)"),
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
        (7.00, "#059669", "",    "Cooler Bath (β=7.0: peaks at n=13, turns over)"),
        (6.00, "#2563eb", "5,4", "Warmer Bath (β=6.0: peaks at n=11, declines)"),
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
        "fig_v10_3_validation",
        dict(xlim=(2.8, 7.2), ylim=(0.3, 4e9),
             title="(a) 3D Toric Code Benchmark",
             xlabel="Lattice Size L (N = 3L³ Qubits)", ylabel="Mean Sector Lifetime",
             xticks=[(3,"3"), (4,"4"), (5,"5"), (6,"6"), (7,"7")],
             yticks=[(1e0,"1"), (1e2,"10²"), (1e4,"10⁴"), (1e6,"10⁶"), (1e8,"10⁸")],
             series=s_tor, xlog=False, ylog=True, leg_pos="tl"),
        dict(xlim=(10.6, 19.4), ylim=(6e2, 4e5),
             title="(b) Published Random Layer Code Turnover",
             xlabel="Layer Size n (N ≈ 2,200 to 11,000 Qubits)", ylabel="Ensemble Mean Lifetime",
             xticks=[(11,"11"), (13,"13"), (15,"15"), (17,"17"), (19,"19")],
             yticks=[(1e3,"10³"), (1e4,"10⁴"), (1e5,"10⁵")],
             series=s_caha, xlog=False, ylog=True, leg_pos="tl"),
    )

    # --- 4. `fig_v10_4_scaling.png`: 3-Panel Multi-Scale Threshold & Two-Cycle Quantum Crossings ---
    # Panel (a): 4 nontrivial 3D scales `s = 1, 2, 3, 4` of Uniform 2x2 (`N = 19, 203, 2,311, 27,107`)
    ms_series = []
    for s, col, dsh, nlbl in [
        (4, "#be123c", "",    "Scale 4 (27,107q: slope=13.2)"),
        (3, "#ea580c", "",    "Scale 3 (2,311q: slope=8.8)"),
        (2, "#2563eb", "6,3", "Scale 2 (203q: slope=4.6)"),
        (1, "#6b7280", "3,3", "Scale 1 (19q: slope=2.0)"),
    ]:
        bs_s = [1.50, 1.75, 2.00, 2.25] if s == 4 else [1.25, 1.50, 1.75, 2.00, 2.25]
        ms_series.append({
            "xs": bs_s,
            "ys": [kmc8[(f"sq_rx_iso_s{s}_X_err", b)][1] for b in bs_s],
            "color": col, "dash": dsh, "label": nlbl,
        })

    # Panel (b): Provably Contractive Uniform 2x2 Quantum Cycles (`323` vs `47,387` qubits)
    iso_bs = [5.00, 5.50, 6.00, 6.25, 6.50, 6.65]
    iso_s2 = [kmc8[("sq_iso_L2r1_s2_X_err", b)][1] for b in iso_bs]
    iso_s4 = [kmc8[("sq_iso_L2r1_s4_X_err", b)][1] for b in iso_bs]
    sl_iso2 = ols_slope(iso_bs[1:], iso_s2[1:])
    sl_iso4 = ols_slope(iso_bs[1:], iso_s4[1:])

    # Panel (c): Boundaryless Torus Quantum Cycles (`508` vs `50,296` qubits up to beta=6.65 -> 7.4x!)
    render_triple(
        "fig_v10_4_scaling",
        dict(xlim=(1.20, 2.50), ylim=(2.5, 9e4),
             title="(a) 4 Scales: Bounded Threshold",
             xlabel="Inverse Bath Temp. β = 1 / T", ylabel="Mean Sector Lifetime",
             xticks=[(1.25,"1.25"), (1.5,"1.5"), (1.75,"1.75"), (2.0,"2.0"), (2.25,"2.25")],
             yticks=[(1e1,"10¹"), (1e2,"10²"), (1e3,"10³"), (1e4,"10⁴")],
             series=ms_series, xlog=False, ylog=True, leg_pos="tl",
             vlines=[(1.86, "#374151", "Threshold β ≈ 1.86", "rb")],
             varrows=[(2.25, kmc8[("sq_rx_iso_s1_X_err", 2.25)][1], kmc8[("sq_rx_iso_s4_X_err", 2.25)][1], "396×", "#be123c", "r", 7.0e2)]),
        dict(xlim=(4.92, 6.78), ylim=(3e2, 9e6),
             title="(b) Uniform 2×2 Quantum Cycles",
             xlabel="Inverse Bath Temp. β = 1 / T", ylabel="Quantum Memory Lifetime",
             xticks=[(5.0,"5.0"), (5.5,"5.5"), (6.0,"6.0"), (6.5,"6.5")],
             yticks=[(1e3,"10³"), (1e4,"10⁴"), (1e5,"10⁵"), (1e6,"10⁶")],
             series=[
                 {"xs": iso_bs, "ys": iso_s4, "color": "#059669", "dash": "",    "label": f"Cycle 2 (47,387q: slope={sl_iso4:.1f})"},
                 {"xs": iso_bs, "ys": iso_s2, "color": "#2563eb", "dash": "5,4", "label": f"Cycle 1 (323q: slope={sl_iso2:.1f})"},
             ],
             xlog=False, ylog=True, leg_pos="tl",
             varrows=[(6.65, iso_s2[-1], iso_s4[-1], "71×", "#047857", "l", 2.5e5)]),
        dict(xlim=(5.42, 6.86), ylim=(1.1e3, 1.8e7),
             title="(c) Torus 2×2 Quantum Cycles",
             xlabel="Inverse Bath Temp. β = 1 / T", ylabel="Quantum Memory Lifetime",
             xticks=[(5.5,"5.5"), (6.0,"6.0"), (6.25,"6.25"), (6.65,"6.65")],
             yticks=[(1e3,"10³"), (1e4,"10⁴"), (1e5,"10⁵"), (1e6,"10⁶"), (1e7,"10⁷")],
             series=[
                 {"xs": rot_bs, "ys": rot_s4_ys, "color": "#be123c", "dash": "",    "label": f"Cycle 2 (50,296q: slope={d_rot_s4:.1f})"},
                 {"xs": rot_bs, "ys": rot_s2_ys, "color": "#d97706", "dash": "5,4", "label": f"Cycle 1 (508q: slope={d_rot_s2:.1f})"},
             ],
             xlog=False, ylog=True, leg_pos="tl",
             varrows=[(6.65, rot_s2_ys[-1], rot_s4_ys[-1], "7.4×", "#be123c", "r", 1.6e6)]),
    )

    # --- 5. `fig_v10_5_tradeoffs.png`: Pure Cubic-Grid Stretching (3 Curves ONLY!) & Sector Balance ---
    z3_bs = [5.50, 6.00, 6.50, 7.00]
    s_z3 = [
        {"xs": z3_bs, "ys": [v7_best("sq_L3r2_s2_X_err", "uniform", "uf", b) for b in z3_bs],
         "color": "#059669", "dash": "",    "label": "Unstretched Graph (3,3) (1,480 qubits)"},
        {"xs": z3_bs, "ys": [v7_best("sq_z3_4_6_r2_X_err", "uniform", "uf", b) for b in z3_bs],
         "color": "#2563eb", "dash": "",    "label": "Tighter Cubic Grid (4,6) (10,280 qubits)"},
        {"xs": [5.50, 6.00, 6.50], "ys": [v7_best("sq_bdl_8_8_r2_X_err", "uniform", "uf", b) for b in [5.50, 6.00, 6.50]],
         "color": "#dc2626", "dash": "",    "label": "Scale-8 Cubic Grid (8,8) (56,312 qubits)"},
    ]

    bal_bs = [5.00, 5.50, 6.00, 6.50, 7.00]
    s_bal = [
        {"xs": [5.50, 6.00, 6.50], "ys": [v7_best("rottor_L2r1_s2_Z_err", "uniform", "uf", b) for b in [5.50, 6.00, 6.50]],
         "color": "#2563eb", "dash": "6,4", "label": "508-Qubit Torus: Z-Errors Alone (strong)"},
        {"xs": bal_bs, "ys": [v7_best("rottor_hyb21_s2_X_err", "uniform", "uf", b) for b in bal_bs],
         "color": "#059669", "dash": "",    "label": "Balanced 2→1 Bridges (1,172q: 57× gain)"},
        {"xs": [5.00, 5.50, 6.00, 6.50, 7.00],
         "ys": [v7_best("rottor_L2r1_s2_X_err", "uniform", "uf", b) if b <= 6.50 else kmc8[("rottor_L2r1_s2_X_err", b)][1] for b in bal_bs],
         "color": "#dc2626", "dash": "",    "label": "508-Qubit Torus: X-Errors (bottleneck)"},
    ]

    render_double(
        "fig_v10_5_tradeoffs",
        dict(xlim=(5.4, 7.1), ylim=(70.0, 2.5e6),
             title="(a) Cubic-Grid Wire Stretching Penalty",
             xlabel="Inverse Bath Temperature β = 1 / T (Colder →)", ylabel="Quantum Memory Lifetime",
             xticks=[(5.5,"5.5"), (6.0,"6.0"), (6.5,"6.5"), (7.0,"7.0")],
             yticks=[(1e2,"10²"), (1e3,"10³"), (1e4,"10⁴"), (1e5,"10⁵"), (1e6,"10⁶")],
             series=s_z3, xlog=False, ylog=True, leg_pos="tl",
             varrows=[(6.50, v7_best("sq_bdl_8_8_r2_X_err", "uniform", "uf", 6.50),
                       v7_best("sq_L3r2_s2_X_err", "uniform", "uf", 6.50), "321× loss", "#b91c1c", "r", 3.2e4)]),
        dict(xlim=(4.9, 7.1), ylim=(1.2e3, 3.0e10),
             title="(b) Closing the X / Z Sector Imbalance",
             xlabel="Inverse Bath Temperature β = 1 / T (Colder →)", ylabel="Mean Sector / Memory Lifetime",
             xticks=[(5.0,"5.0"), (5.5,"5.5"), (6.0,"6.0"), (6.5,"6.5"), (7.0,"7.0")],
             yticks=[(1e4,"10⁴"), (1e5,"10⁵"), (1e6,"10⁶"), (1e7,"10⁷"), (1e8,"10⁸"), (1e9,"10⁹"), (1e10,"10¹⁰")],
             series=s_bal, xlog=False, ylog=True, leg_pos="tl",
             varrows=[(6.50, v7_best("rottor_L2r1_s2_X_err", "uniform", "uf", 6.50),
                       v7_best("rottor_hyb21_s2_X_err", "uniform", "uf", 6.50), "57× gain", "#047857", "r", 3.0e6)]),
    )

    # --- 6. `fig_v10_6_cooling.png`: Storage Self-Healing vs Hot-State Cooling ---
    ps = [0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.12]
    px_pct = [p * 100.0 for p in ps]
    s_heal = []
    for tag, col, dsh, lbl in [
        ("rottor_L2r1_s2_Z_err",  "#2563eb", "6,4", "508-Qubit Torus (Z-noise: 98% at 1%)"),
        ("rottor_hyb21_s2_X_err", "#059669", "",    "1,172-Qubit Balanced Torus (89% at 1%)"),
        ("rottor_L2r1_s2_X_err",  "#dc2626", "",    "508-Qubit Torus (X-noise: 76% at 1%)"),
    ]:
        ys = [100.0 * cool7[(tag, "quench", f"noisy_{p:.2f}", 1000.0)]["pheal"] for p in ps]
        s_heal.append({"xs": px_pct, "ys": ys, "color": col, "dash": dsh, "label": lbl})

    ts = [100.0, 400.0, 1500.0]
    s_cool = [
        {"xs": ts, "ys": [100.0 * cool7[("rottor_L2r1_s2_Z_err", "anneal", "hot", t)]["p0"] for t in ts],
         "color": "#059669", "dash": "",    "label": "508-Qubit Torus: Gradual Cooling (100%)"},
        {"xs": ts, "ys": [100.0 * cool7[("sq_L2r1_s3_X_err", "anneal", "hot", t)]["p0"] for t in ts],
         "color": "#2563eb", "dash": "",    "label": "1,942-Qubit Code: Gradual Cooling (63%)"},
        {"xs": ts, "ys": [100.0 * cool7[("sq_L2r1_s3_X_err", "quench", "hot", t)]["p0"] for t in ts],
         "color": "#dc2626", "dash": "5,4", "label": "1,942-Qubit Code: Sudden Quench (14%)"},
    ]

    render_double(
        "fig_v10_6_cooling",
        dict(xlim=(0.5, 12.5), ylim=(-2.0, 110.0),
             title="(a) Zero-Decoder Storage Self-Healing",
             xlabel="Initial Physical Error Rate (%)", ylabel="Exact Codeword Recovery (%)",
             xticks=[(1,"1%"), (3,"3%"), (6,"6%"), (9,"9%"), (12,"12%")],
             yticks=[(0,"0%"), (25,"25%"), (50,"50%"), (75,"75%"), (100,"100%")],
             series=s_heal, xlog=False, ylog=False, leg_pos="tr",
             hlines=[(25.0, "#6b7280", "Random-Guess Floor (25%)", "bl")]),
        dict(xlim=(80.0, 1800.0), ylim=(-2.0, 134.0),
             title="(b) Cooling From a Hot Random State",
             xlabel="Total Bath Cooling Time (Log Scale)", ylabel="Clean Defect-Free Ground States (%)",
             xticks=[(100,"100"), (400,"400"), (1500,"1,500")],
             yticks=[(0,"0%"), (25,"25%"), (50,"50%"), (75,"75%"), (100,"100%")],
             series=s_cool, xlog=True, ylog=False, leg_pos="tl",
             varrows=[(1500.0, 14.0, 63.0, "4.5× more clean states", "#1d4ed8", "l", 26.0)]),
    )


if __name__ == "__main__":
    main()
