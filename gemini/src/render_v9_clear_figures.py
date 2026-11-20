#!/usr/bin/env python3
"""
render_v9_clear_figures.py
Renders 6 uncluttered, immediately understandable figures for the V9 REVTeX paper:
  1. fig_v9_1_classes.png:      (a) Normalized Lifetime Growth <t(N)>/<t(N0)> vs N  |  (b) Energy Barrier vs N
  2. fig_v9_2_geometry.png:     Clean 4-panel visual diagram (no text walls inside cards!)
  3. fig_v9_3_validation.png:   (a) 3D Toric Code (3 curves)  |  (b) Gu-Caha Random Layer Codes (2 curves)
  4. fig_v9_4_crossings.png:    (a) Planar Cycle 1 vs Cycle 2 (106x!)  |  (b) Torus Cycle 1 vs Cycle 2 (2 curves each!)
  5. fig_v9_5_tradeoffs.png:    (a) Cubic-Grid Wire Stretching (321x)  |  (b) X/Z Sector Gap & 2->1 Balancing (3 curves!)
  6. fig_v9_6_cooling.png:      (a) Zero-Decoder Self-Healing (matched K=2)  |  (b) Hot-State Gradual Cooling vs Quench
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
               xticks, yticks, series, xlog=False, ylog=True, leg_pos="tl", hlines=None, varrows=None):
    elems.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="#fafafa" stroke="#374151" stroke-width="1.5"/>')
    elems.append(f'<text x="{x0+pw/2:.1f}" y="{y0-15:.1f}" text-anchor="middle" font-size="15.4" font-weight="700">{html.escape(title)}</text>')
    elems.append(f'<text x="{x0+pw/2:.1f}" y="{y0+ph+44:.1f}" text-anchor="middle" font-size="14.0" font-weight="600">{html.escape(xlabel)}</text>')
    elems.append(f'<text x="{x0-60:.1f}" y="{y0+ph/2:.1f}" text-anchor="middle" font-size="14.0" font-weight="600" '
                 f'transform="rotate(-90 {x0-60:.1f} {y0+ph/2:.1f})">{html.escape(ylabel)}</text>')

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
        elems.append(f'<text x="{px:.1f}" y="{y0+ph+22}" text-anchor="middle" font-size="12.6">{html.escape(xlbl)}</text>')

    for yv, ylbl in yticks:
        py = ty(yv)
        elems.append(f'<line x1="{x0}" y1="{py:.1f}" x2="{x0+pw}" y2="{py:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        elems.append(f'<line x1="{x0-5}" y1="{py:.1f}" x2="{x0}" y2="{py:.1f}" stroke="#374151" stroke-width="1.2"/>')
        elems.append(f'<text x="{x0-9}" y="{py+4.5:.1f}" text-anchor="end" font-size="12.6">{html.escape(ylbl)}</text>')

    if hlines:
        for yv, col, lbl, halign in hlines:
            py = ty(yv)
            elems.append(f'<line x1="{x0}" y1="{py:.1f}" x2="{x0+pw}" y2="{py:.1f}" stroke="{col}" stroke-width="1.6" stroke-dasharray="6,4"/>')
            hx = x0 + 12 if "l" in halign else x0 + pw - 12
            anch = "start" if "l" in halign else "end"
            hy = py + 16.5 if "b" in halign else py - 6.5
            elems.append(f'<text x="{hx:.1f}" y="{hy:.1f}" text-anchor="{anch}" font-size="12.0" font-weight="600" fill="{col}">{html.escape(lbl)}</text>')

    for s in series:
        pts = [(tx(x), ty(y)) for x, y in zip(s["xs"], s["ys"]) if y is not None and xlim[0] <= x <= xlim[1]]
        if not pts:
            continue
        d_attr = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
        pstr = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        if len(pts) >= 2:
            elems.append(f'<polyline fill="none" stroke="{s["color"]}" stroke-width="3.0"{d_attr} points="{pstr}"/>')
        for px, py in pts:
            elems.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5.0" fill="{s["color"]}" stroke="#ffffff" stroke-width="1.2"/>')

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
            dx = -9.0 if side == "l" else 9.0
            anch = "end" if side == "l" else "start"
            elems.append(f'<text x="{px+dx:.1f}" y="{ty_lbl:.1f}" text-anchor="{anch}" font-size="12.4" font-weight="700" fill="{col}">{html.escape(lbl)}</text>')

    row_h = 21.0
    lh = row_h * len(series) + 12
    lw = min(pw - 20.0, max(210.0, max(len(s["label"]) for s in series) * 6.95 + 48.0))
    lx = x0 + 11 if "l" in leg_pos else x0 + pw - lw - 11
    ly = y0 + 11 if "t" in leg_pos else y0 + ph - lh - 11
    elems.append(f'<rect x="{lx:.1f}" y="{ly:.1f}" width="{lw:.1f}" height="{lh:.1f}" rx="6" fill="#ffffff" fill-opacity="0.96" stroke="#d1d5db" stroke-width="1.1"/>')
    for i, s in enumerate(series):
        iy = ly + 17.0 + i * row_h
        d_attr = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
        elems.append(f'<line x1="{lx+9:.1f}" y1="{iy-4.2:.1f}" x2="{lx+31:.1f}" y2="{iy-4.2:.1f}" stroke="{s["color"]}" stroke-width="2.8"{d_attr}/>')
        elems.append(f'<circle cx="{lx+20.0:.1f}" cy="{iy-4.2:.1f}" r="4.2" fill="{s["color"]}"/>')
        elems.append(f'<text x="{lx+38:.1f}" y="{iy:.1f}" font-size="12.2">{html.escape(s["label"])}</text>')


def render_double(nm, p1, p2):
    W, H = 1180, 465
    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<style>text { font-family: -apple-system, Helvetica, Arial, sans-serif; fill: #1f2937; }</style>',
    ]
    draw_panel(elems, 88, 46, 470, 358, **p1)
    draw_panel(elems, 678, 46, 470, 358, **p2)
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
        elems.append(f'<text x="{cx+cw/2:.1f}" y="{cy+28}" text-anchor="middle" font-size="14.2" font-weight="700">{html.escape(ctitle)}</text>')

    # (a) Spacious 3D parallel sheets + vertical bridge + 1 -> 2 split
    elems.append('<polygon points="36,274 202,274 266,222 100,222" fill="#dbeafe" fill-opacity="0.78" stroke="#2563eb" stroke-width="2.0"/>')
    elems.append('<polygon points="36,154 202,154 266,102 100,102" fill="#dbeafe" fill-opacity="0.78" stroke="#2563eb" stroke-width="2.0"/>')
    elems.append('<polygon points="122,274 122,154 186,102 186,222" fill="#fde68a" fill-opacity="0.85" stroke="#d97706" stroke-width="2.2"/>')
    elems.append('<line x1="122" y1="154" x2="186" y2="102" stroke="#b45309" stroke-width="3.6"/>')
    # 1 defect on left -> splits into 2 defects across junction
    elems.append('<circle cx="84" cy="132" r="8.5" fill="#dc2626" stroke="#ffffff" stroke-width="1.6"/>')
    elems.append('<line x1="95" y1="132" x2="142" y2="130" stroke="#dc2626" stroke-width="2.8"/>')
    elems.append('<line x1="150" y1="128" x2="205" y2="124" stroke="#dc2626" stroke-width="2.8"/>')
    elems.append('<line x1="150" y1="132" x2="156" y2="184" stroke="#dc2626" stroke-width="2.8"/>')
    elems.append('<circle cx="213" cy="123" r="8.5" fill="#dc2626" stroke="#ffffff" stroke-width="1.6"/>')
    elems.append('<circle cx="157" cy="192" r="8.5" fill="#dc2626" stroke="#ffffff" stroke-width="1.6"/>')
    elems.append('<text x="76" y="92" text-anchor="middle" font-size="12.4" font-weight="600" fill="#1e40af">Top Sheet</text>')
    elems.append('<text x="82" y="296" text-anchor="middle" font-size="12.4" font-weight="600" fill="#1e40af">Bottom Sheet</text>')
    elems.append('<text x="226" y="190" text-anchor="middle" font-size="12.4" font-weight="700" fill="#b45309">Bridge</text>')
    elems.append('<text x="153" y="326" text-anchor="middle" font-size="12.8" font-weight="700" fill="#991b1b">1 defect → 2 defects (+energy)</text>')

    # (b) Compact 2x2 Four-Arm Cross
    bx, by, bs = 352, 82, 185
    elems.append(f'<rect x="{bx}" y="{by}" width="{bs}" height="{bs}" fill="#eff6ff" stroke="#1e3a8a" stroke-width="2.6"/>')
    elems.append(f'<line x1="{bx+bs/2}" y1="{by}" x2="{bx+bs/2}" y2="{by+bs}" stroke="#dc2626" stroke-width="5.6"/>')
    elems.append(f'<line x1="{bx}" y1="{by+bs/2}" x2="{bx+bs}" y2="{by+bs/2}" stroke="#dc2626" stroke-width="5.6"/>')
    for px, py in [(bx+bs/2, by), (bx+bs/2, by+bs), (bx, by+bs/2), (bx+bs, by+bs/2), (bx+bs/2, by+bs/2)]:
        elems.append(f'<circle cx="{px}" cy="{py}" r="6.2" fill="#991b1b" stroke="#ffffff" stroke-width="1.4"/>')
    elems.append(f'<text x="444.5" y="298" text-anchor="middle" font-size="12.6" font-weight="600" fill="#1f2937">Top view of one 2×2 cell</text>')
    elems.append(f'<text x="444.5" y="326" text-anchor="middle" font-size="12.8" font-weight="700" fill="#047857">All 4 bridge arms reach walls</text>')

    # (c) Pruned 3-Arm T-Bridge ("Buttonhole" Leak)
    cx0, cy0, cs = 643, 82, 185
    elems.append(f'<rect x="{cx0}" y="{cy0}" width="{cs}" height="{cs}" fill="#eff6ff" stroke="#1e3a8a" stroke-width="2.6"/>')
    elems.append(f'<line x1="{cx0+cs/2}" y1="{cy0}" x2="{cx0+cs/2}" y2="{cy0+cs/2}" stroke="#dc2626" stroke-width="5.6"/>')
    elems.append(f'<line x1="{cx0}" y1="{cy0+cs/2}" x2="{cx0+cs}" y2="{cy0+cs/2}" stroke="#dc2626" stroke-width="5.6"/>')
    elems.append(f'<line x1="{cx0+cs/2}" y1="{cy0+cs/2}" x2="{cx0+cs/2}" y2="{cy0+cs}" stroke="#9ca3af" stroke-width="2.8" stroke-dasharray="6,5"/>')
    elems.append(f'<circle cx="{cx0+cs/2}" cy="{cy0+cs/2+16}" r="25" fill="#fee2e2" fill-opacity="0.78" stroke="#b91c1c" stroke-width="2.4" stroke-dasharray="4,3"/>')
    elems.append(f'<circle cx="{cx0+cs/2}" cy="{cy0+cs/2}" r="7.0" fill="#b91c1c" stroke="#ffffff" stroke-width="1.5"/>')
    elems.append(f'<text x="735.5" y="298" text-anchor="middle" font-size="12.6" font-weight="600" fill="#1f2937">Missing 4th arm leaves a hole</text>')
    elems.append(f'<text x="735.5" y="326" text-anchor="middle" font-size="12.8" font-weight="700" fill="#b91c1c">Spawns unprotected qubits</text>')

    # (d) Scale-8 Cubic Grid (7-qubit free wire)
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
    elems.append(f'<text x="1026.5" y="298" text-anchor="middle" font-size="12.6" font-weight="600" fill="#1f2937">7-qubit wire between junctions</text>')
    elems.append(f'<text x="1026.5" y="326" text-anchor="middle" font-size="12.8" font-weight="700" fill="#b91c1c">Cuts lifetime 321×</text>')

    save_svg_png("fig_v9_2_geometry", elems, W, H)


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

    kmc5 = {}
    with open(os.path.join(WS, "cluster_v5_results.txt")) as f:
        for raw in f:
            ln = raw.strip()
            if not ln or "|KMC " not in ln:
                continue
            parts = ln.split("|", 3)
            kv = parse_kv(parts[3])
            if int(kv["capped"]) == 0:
                kmc5.setdefault(parts[1], {})[float(kv["beta"])] = (int(kv["N"]), float(kv["mean_t"]))

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
    return kmc4, kmc5, kmc6, kmc7, ffs7, cool7


def main():
    render_clean_geometry()
    kmc4, kmc5, kmc6, kmc7, ffs7, cool7 = load_all()

    def v7_best(tag, jm, dec, b, min_reps=3):
        if (tag, jm, dec, b) in kmc7 and kmc7[(tag, jm, dec, b)][2] == 0:
            return kmc7[(tag, jm, dec, b)][1]
        if (tag, jm, dec, b) in ffs7 and ffs7[(tag, jm, dec, b)][2] >= min_reps:
            return ffs7[(tag, jm, dec, b)][1]
        return None

    def v7_min2(pref_s, b, min_reps=3):
        vx = v7_best(f"{pref_s}_X_err", "uniform", "uf", b, min_reps)
        vz = v7_best(f"{pref_s}_Z_err", "uniform", "uf", b, min_reps)
        if vx is not None and vz is not None:
            return min(vx, vz)
        if vx is not None and (pref_s.endswith("_s2") or pref_s.endswith("_s4")):
            return vx
        return None

    # --- 1. `fig_v9_1_classes.png`: (a) Normalized Relative Lifetime <t(N)>/<t(N_small)> | (b) Energy Barrier ---
    def v4_norm_curve(cnames, b):
        pts = [(kmc4[cn][b][0], kmc4[cn][b][1]) for cn in cnames if b in kmc4.get(cn, {})]
        y0 = pts[0][1]
        return [x for x, _ in pts], [y / y0 for _, y in pts]

    hx, hy = v4_norm_curve([f"haah_cubic_L{L}_Xerr.code" for L in [3, 5, 7, 9, 11, 13]], 4.25)
    bx, by = v4_norm_curve([f"bb_gross_{l}x{m}_Xerr.code" for l, m in [(6,6),(12,6),(18,12),(24,18),(30,18),(36,18),(48,18),(60,18)]], 4.25)
    tx_n, ty_n = v4_norm_curve([f"toric2d_L{L}_Xerr.code" for L in [4, 6, 10, 16, 24, 34]], 4.50)
    # Compact 3D welded planar Cycle 1 (N=254) -> Cycle 2 (N=25,148) at fixed beta=7.0: 8.85e4 -> 9.34e6 (106x!)
    t_s2_70 = kmc5["sq_aniso_L2r1_s2_Xerr.code"][7.0][1]
    t_s4_70 = kmc5["sq_aniso_L2r1_s4_Xerr.code"][7.0][1]
    wx_n, wy_n = [254, 25148], [1.0, t_s4_70 / t_s2_70]

    s_norm = [
        {"xs": wx_n, "ys": wy_n, "color": "#be123c", "dash": "",    "label": "Compact 3D Welded Code (Cycle 1→2: 106× gain)"},
        {"xs": hx,   "ys": hy,   "color": "#7c3aed", "dash": "",    "label": "Haah 3D Cubic Code (peaks at 25×, turns over)"},
        {"xs": bx,   "ys": by,   "color": "#2563eb", "dash": "6,4", "label": "Scaled 2D Bivariate Bicycle Code (falls < 1×)"},
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
    rot_win = [5.50, 6.00, 6.25, 6.50]
    d_rot_s2 = ols_slope(rot_win, [v7_min2("rottor_L2r1_s2", b) for b in rot_win])
    d_rot_s4 = ols_slope(rot_win, [v7_min2("rottor_L2r1_s4", b) for b in rot_win])

    s_bar = [
        {"xs": c_haah_N, "ys": c_haah_D, "color": "#7c3aed", "dash": "",    "label": "Haah 3D Cubic Code (logarithmic, saturates)"},
        {"xs": [4, 508, 50296], "ys": [1.92, d_rot_s2, d_rot_s4], "color": "#be123c", "dash": "",
         "label": f"Compact 3D Welded Torus (1.9 → {d_rot_s2:.1f} → {d_rot_s4:.1f})"},
        {"xs": c_bb_N,   "ys": c_bb_D,   "color": "#2563eb", "dash": "6,4", "label": "Scaled 2D Bivariate Bicycle Code (flat ≈ 5.4)"},
        {"xs": c_tor_N,  "ys": c_tor_D,  "color": "#4b5563", "dash": "3,3", "label": "2D Toric Code (flat ≤ 2.3)"},
    ]

    render_double(
        "fig_v9_1_classes",
        dict(xlim=(25.0, 35000.0), ylim=(0.02, 2.5e3),
             title="(a) Relative Lifetime Growth With Qubit Count",
             xlabel="Physical Qubit Count N (Log Scale)", ylabel="Lifetime Relative to Smallest Code",
             xticks=[(30,"30"), (100,"100"), (1000,"1,000"), (10000,"10,000"), (30000,"30,000")],
             yticks=[(0.05,"0.05×"), (1.0,"1×"), (10.0,"10×"), (100.0,"100×")],
             series=s_norm, xlog=True, ylog=True, leg_pos="tl",
             hlines=[(1.0, "#6b7280", "1× = No Gain From Growing N", "br")]),
        dict(xlim=(3.0, 65000.0), ylim=(0.0, 14.2),
             title="(b) Effective Energy Barrier vs. Qubit Count",
             xlabel="Physical Qubit Count N (Log Scale)", ylabel="Effective Energy Barrier",
             xticks=[(4,"4"), (100,"100"), (1000,"1,000"), (10000,"10,000"), (50000,"50,000")],
             yticks=[(0,"0"), (3,"3"), (6,"6"), (9,"9"), (12,"12")],
             series=s_bar, xlog=True, ylog=False, leg_pos="tl"),
    )

    # --- 3. `fig_v9_3_validation.png`: Clean 2-Panel External Validation (Minimal Curves!) ---
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
        (7.00, "#059669", "",    "Cooler Bath (β=7.0: rises to n=13, turns over)"),
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
        "fig_v9_3_validation",
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

    # --- 4. `fig_v9_4_crossings.png`: WIDE-OPEN Two-Cycle Crossings (ONLY 2 Curves Per Panel!!) ---
    # Panel (a): Planar 2x2 Cross (N=254 vs N=25,148 up to beta=7.0 -> 106x gap!)
    pl_bs = [5.50, 6.00, 6.25, 6.50, 6.75, 7.00]
    pl_s2 = [v7_min2("sq_L2r1_s2", b) if b <= 6.5 else kmc5["sq_aniso_L2r1_s2_Xerr.code"][b][1] for b in pl_bs]
    pl_s4 = [v7_min2("sq_L2r1_s4", b) if b <= 6.5 else kmc5["sq_aniso_L2r1_s4_Xerr.code"][b][1] for b in pl_bs]

    # Panel (b): Boundaryless Torus 2x2 Cross (N=508 vs N=50,296 across beta=5.5..6.5 -> clean crossing!)
    tor_bs = [5.50, 6.00, 6.25, 6.50]
    tor_s2 = [v7_min2("rottor_L2r1_s2", b) for b in tor_bs]
    tor_s4 = [v7_min2("rottor_L2r1_s4", b) for b in tor_bs]

    render_double(
        "fig_v9_4_crossings",
        dict(xlim=(5.40, 7.12), ylim=(1.5e3, 8e7),
             title="(a) Planar 2×2 Cross: Cycle 1 vs. Cycle 2",
             xlabel="Inverse Bath Temperature β = 1 / T  (Colder →)", ylabel="Quantum Memory Lifetime",
             xticks=[(5.5,"5.5"), (6.0,"6.0"), (6.5,"6.5"), (7.0,"7.0")],
             yticks=[(1e4,"10⁴"), (1e5,"10⁵"), (1e6,"10⁶"), (1e7,"10⁷")],
             series=[
                 {"xs": pl_bs, "ys": pl_s4, "color": "#059669", "dash": "",    "label": "Cycle 2 (25,148 qubits: slope = 5.3)"},
                 {"xs": pl_bs, "ys": pl_s2, "color": "#2563eb", "dash": "5,4", "label": "Cycle 1 (254 qubits: slope = 2.4)"},
             ], xlog=False, ylog=True, leg_pos="tl",
             varrows=[(7.00, pl_s2[-1], pl_s4[-1], "106× gain", "#047857", "l")]),
        dict(xlim=(5.42, 6.72), ylim=(8e2, 8e6),
             title="(b) Boundaryless Torus: Cycle 1 vs. Cycle 2",
             xlabel="Inverse Bath Temperature β = 1 / T  (Colder →)", ylabel="Quantum Memory Lifetime",
             xticks=[(5.5,"5.5"), (6.0,"6.0"), (6.25,"6.25"), (6.5,"6.5")],
             yticks=[(1e3,"10³"), (1e4,"10⁴"), (1e5,"10⁵"), (1e6,"10⁶")],
             series=[
                 {"xs": tor_bs, "ys": tor_s4, "color": "#be123c", "dash": "",    "label": "Cycle 2 (50,296 qubits: slope = 6.3)"},
                 {"xs": tor_bs, "ys": tor_s2, "color": "#f97316", "dash": "5,4", "label": "Cycle 1 (508 qubits: slope = 3.3)"},
             ], xlog=False, ylog=True, leg_pos="tl",
             varrows=[(6.50, tor_s2[-1], tor_s4[-1], "2.45×", "#be123c", "r")]),
    )

    # --- 5. `fig_v9_5_tradeoffs.png`: (a) Cubic-Grid Stretching (321x) | (b) Closing X/Z Gap (3 Curves!) ---
    s_grid = []
    xs_u = [5.50, 6.00, 6.50, 7.00]
    ys_u = [v7_min2("sq_L3r2_s2", b) for b in xs_u]
    s_grid.append({"xs": xs_u, "ys": ys_u, "color": "#2563eb", "dash": "", "label": "Compact Unstretched (1,480 qubits)"})
    for zc, jm, col, dsh, lbl in [
        ("sq_z3_4_6_r2",  "uniform",  "#059669", "",    "Tighter (4,6) Cubic Grid (10,280 qubits)"),
        ("sq_bdl_8_8_r2", "pin_norm", "#dc2626", "",    "Scale-8 Grid + Wire Traps (56,312 qubits)"),
        ("sq_bdl_8_8_r2", "uniform",  "#4b5563", "5,4", "Scale-8 Cubic Grid (56,312 qubits)"),
    ]:
        xs, ys = [], []
        for b in [5.50, 6.00, 6.50, 7.00]:
            rx = kmc7.get((f"{zc}_X_err", jm, "uf", b))
            rz = kmc7.get((f"{zc}_Z_err", jm, "uf", b))
            if rx and rx[2] == 0:
                xs.append(b); ys.append(min(rx[1], rz[1]) if (rz and rz[2] == 0) else rx[1])
        s_grid.append({"xs": xs, "ys": ys, "color": col, "dash": dsh, "label": lbl})

    s_bal = []
    for tag, col, dsh, lbl in [
        ("rottor_L2r1_s2_Z_err",  "#059669", "5,4", "Symmetric 1→1: Z-Errors Only"),
        ("rottor_hyb21_s2_X_err", "#7c3aed", "",    "Balanced 2→1 Overall Memory (1,172 qubits)"),
        ("rottor_L2r1_s2_X_err",  "#dc2626", "4,3", "Symmetric 1→1: X-Error Bottleneck (508 qubits)"),
    ]:
        xs, ys = [], []
        for b in [5.50, 6.00, 6.25, 6.50, 6.75, 7.00, 7.50]:
            v = v7_best(tag, "uniform", "uf", b, min_reps=3)
            if v is not None:
                xs.append(b); ys.append(v)
        s_bal.append({"xs": xs, "ys": ys, "color": col, "dash": dsh, "label": lbl})

    render_double(
        "fig_v9_5_tradeoffs",
        dict(xlim=(5.40, 7.12), ylim=(50.0, 2e8),
             title="(a) Cubic-Grid Wire Stretching Cost",
             xlabel="Inverse Bath Temperature β = 1 / T", ylabel="Quantum Memory Lifetime",
             xticks=[(5.5,"5.5"), (6.0,"6.0"), (6.5,"6.5"), (7.0,"7.0")],
             yticks=[(1e2,"10²"), (1e4,"10⁴"), (1e6,"10⁶"), (1e8,"10⁸")],
             series=s_grid, xlog=False, ylog=True, leg_pos="tl",
             varrows=[(6.50, 440.9, 1.413e5, "321× loss", "#1e40af", "l", 2.8e4)]),
        dict(xlim=(5.40, 7.62), ylim=(4e3, 8e13),
             title="(b) Closing the X / Z Sector Asymmetry",
             xlabel="Inverse Bath Temperature β = 1 / T", ylabel="Sector / Memory Lifetime",
             xticks=[(5.5,"5.5"), (6.0,"6.0"), (6.5,"6.5"), (7.0,"7.0"), (7.5,"7.5")],
             yticks=[(1e4,"10⁴"), (1e7,"10⁷"), (1e10,"10¹⁰"), (1e13,"10¹³")],
             series=s_bal, xlog=False, ylog=True, leg_pos="tl",
             varrows=[(6.50, 4.026e5, 7.615e8, "1,890× gap", "#374151", "r", 2.5e6)]),
    )

    # --- 6. `fig_v9_6_cooling.png`: (a) Matched K=2 Self-Healing | (b) Hot-State Cooling ---
    p_vals = [0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.12]
    s_heal = []
    for cname, sec, col, dsh, lbl in [
        ("rottor_L2r1_s2",  "Z_err", "#2563eb", "",    "1→1 Torus Z-Errors (98% healed at 1%)"),
        ("rottor_hyb21_s2", "X_err", "#059669", "",    "Balanced 2→1 X-Errors (89% healed at 1%)"),
        ("rottor_L2r1_s2",  "X_err", "#dc2626", "5,4", "1→1 Torus X-Errors (76% healed at 1%)"),
    ]:
        xs, ys = [], []
        for p0 in p_vals:
            cr = cool7.get((f"{cname}_{sec}", "quench", f"noisy_{p0:.2f}", 1000.0))
            if cr:
                xs.append(100.0 * p0); ys.append(100.0 * cr["pheal"])
        s_heal.append({"xs": xs, "ys": ys, "color": col, "dash": dsh, "label": lbl})

    s_hot = []
    for tag, proto, col, dsh, lbl in [
        ("rottor_L2r1_s2_Z_err", "anneal", "#059669", "",    "508-Qubit Torus: Gradual Cooling (100%)"),
        ("sq_L2r1_s3_X_err",     "anneal", "#2563eb", "",    "1,942-Qubit 3-Step: Gradual Cooling (63%)"),
        ("sq_L2r1_s3_X_err",     "quench", "#dc2626", "5,4", "1,942-Qubit 3-Step: Sudden Quench (14%)"),
    ]:
        xs, ys = [], []
        for t_val in [100.0, 400.0, 1500.0]:
            cr = cool7.get((tag, proto, "hot", t_val))
            if cr:
                xs.append(t_val); ys.append(100.0 * cr["p0"])
        s_hot.append({"xs": xs, "ys": ys, "color": col, "dash": dsh, "label": lbl})

    render_double(
        "fig_v9_6_cooling",
        dict(xlim=(0.5, 12.5), ylim=(-2.0, 110.0),
             title="(a) Self-Healing During Storage (Two-Qubit Torus)",
             xlabel="Initial Physical Error Rate (%)", ylabel="Zero-Decoder Exact Recovery (%)",
             xticks=[(1,"1%"), (2,"2%"), (4,"4%"), (6,"6%"), (8,"8%"), (10,"10%"), (12,"12%")],
             yticks=[(0,"0%"), (25,"25%"), (50,"50%"), (75,"75%"), (100,"100%")],
             series=s_heal, xlog=False, ylog=False, leg_pos="tr",
             hlines=[(25.0, "#6b7280", "Random Guess Floor (25%)", "bl")]),
        dict(xlim=(80.0, 1800.0), ylim=(-2.0, 134.0),
             title="(b) Cooling From a Hot Random State (50% Noise)",
             xlabel="Bath Cooling Time (Log Scale)", ylabel="Defect-Free Ground States (%)",
             xticks=[(100,"100"), (400,"400"), (1500,"1,500")],
             yticks=[(0,"0%"), (25,"25%"), (50,"50%"), (75,"75%"), (100,"100%")],
             series=s_hot, xlog=True, ylog=False, leg_pos="tl",
             varrows=[(1500.0, 14.0, 63.0, "4.5× more clean states", "#1e40af", "l", 26.0)]),
    )


if __name__ == "__main__":
    main()
