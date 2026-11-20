#!/usr/bin/env python3
"""
render_v8_clean_figures.py
Renders 7 spacious, plain-English, large-font figures for the simplified V8 paper:
  1. fig_v8_1_geometry.png:         4-panel visual schematic of 3D welded codes, 2x2 cross, buttonholes, & wire stretching
  2. fig_v8_2_four_classes.png:     Fixed-temperature lifetime vs system size N across 5 quantum code families
  3. fig_v8_3_barriers.png:         Effective energy barrier vs system size N across quantum code families
  4. fig_v8_4_validation.png:       2-panel external validation: (a) 3D Toric Code, (b) Gu-Caha Random Layer Codes
  5. fig_v8_5_multicycle_floor.png: Multi-cycle 3D self-correction crossings vs pruned 3-arm T-bridge collapse
  6. fig_v8_6_grid_and_balance.png: 2-panel design trade-offs: (a) Cubic-grid wire stretching, (b) X/Z sector balancing
  7. fig_v8_7_healing_cooling.png:  2-panel thermal dynamics: (a) Noisy codeword self-healing, (b) Hot-state cooling
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
    subprocess.run(["/opt/homebrew/bin/rsvg-convert", "-z", "2", "-f", "png", "-o", out_png, svg_p],
                   check=True)
    with open(out_png, "rb") as fi, open(os.path.join(ART_DIR, f"{nm}.png"), "wb") as fo:
        fo.write(fi.read())
    print(f"Rendered {nm}.png ({2*W}x{2*H})")


def draw_axes_panel(elems, x0, y0, pw, ph, xlim, ylim, title, xlabel, ylabel,
                    xticks, yticks, series, xlog=False, ylog=True, leg_pos="tl", hlines=None, annots=None, big=False):
    fs_title = 15.6 if big else 14.8
    fs_lbl   = 14.4 if big else 13.4
    fs_tick  = 13.0 if big else 12.2
    fs_leg   = 12.2 if big else 11.6
    row_h    = 21.0 if big else 19.8
    char_w   = 6.85 if big else 6.55

    elems.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="#fafafa" stroke="#374151" stroke-width="1.5"/>')
    elems.append(f'<text x="{x0+pw/2:.1f}" y="{y0-15:.1f}" text-anchor="middle" font-size="{fs_title}" font-weight="700">{html.escape(title)}</text>')
    elems.append(f'<text x="{x0+pw/2:.1f}" y="{y0+ph+44:.1f}" text-anchor="middle" font-size="{fs_lbl}" font-weight="600">{html.escape(xlabel)}</text>')
    elems.append(f'<text x="{x0-60:.1f}" y="{y0+ph/2:.1f}" text-anchor="middle" font-size="{fs_lbl}" font-weight="600" '
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
        elems.append(f'<text x="{px:.1f}" y="{y0+ph+22}" text-anchor="middle" font-size="{fs_tick}">{html.escape(xlbl)}</text>')

    for yv, ylbl in yticks:
        py = ty(yv)
        elems.append(f'<line x1="{x0}" y1="{py:.1f}" x2="{x0+pw}" y2="{py:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        elems.append(f'<line x1="{x0-5}" y1="{py:.1f}" x2="{x0}" y2="{py:.1f}" stroke="#374151" stroke-width="1.2"/>')
        elems.append(f'<text x="{x0-9}" y="{py+4.5:.1f}" text-anchor="end" font-size="{fs_tick}">{html.escape(ylbl)}</text>')

    if hlines:
        for item in hlines:
            yv, col, lbl = item[:3]
            halign = item[3] if len(item) > 3 else "tr"
            py = ty(yv)
            elems.append(f'<line x1="{x0}" y1="{py:.1f}" x2="{x0+pw}" y2="{py:.1f}" stroke="{col}" stroke-width="1.5" stroke-dasharray="5,4"/>')
            hx = x0 + 12 if "l" in halign else x0 + pw - 10
            anch = "start" if "l" in halign else "end"
            hy = py + 15.5 if "b" in halign else py - 6.0
            elems.append(f'<text x="{hx:.1f}" y="{hy:.1f}" text-anchor="{anch}" font-size="11.6" font-weight="600" fill="{col}">{html.escape(lbl)}</text>')

    for s in series:
        pts = [(tx(x), ty(y)) for x, y in zip(s["xs"], s["ys"]) if y is not None and xlim[0] <= x <= xlim[1]]
        if not pts:
            continue
        d_attr = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
        pstr = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        if len(pts) >= 2 and not s.get("pts_only"):
            elems.append(f'<polyline fill="none" stroke="{s["color"]}" stroke-width="3.0"{d_attr} points="{pstr}"/>')
        for px, py in pts:
            r = s.get("r", 5.0 if big else 4.6)
            if r > 0:
                elems.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{r}" fill="{s["color"]}" stroke="#ffffff" stroke-width="1.1"/>')

    if annots:
        for xv, yv, txt, col, anch in annots:
            elems.append(f'<text x="{tx(xv):.1f}" y="{ty(yv):.1f}" text-anchor="{anch}" font-size="{fs_leg}" font-weight="700" fill="{col}">{html.escape(txt)}</text>')

    lh = row_h * len(series) + 12
    lw = min(pw - 16.0, max(230.0, max(len(s["label"]) for s in series) * char_w + 48.0))
    lx = x0 + 10 if "l" in leg_pos else x0 + pw - lw - 10
    ly = y0 + 10 if "t" in leg_pos else y0 + ph - lh - 10
    elems.append(f'<rect x="{lx:.1f}" y="{ly:.1f}" width="{lw:.1f}" height="{lh:.1f}" rx="6" fill="#ffffff" fill-opacity="0.95" stroke="#d1d5db" stroke-width="1.1"/>')
    for i, s in enumerate(series):
        iy = ly + 17.0 + i * row_h
        d_attr = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
        if not s.get("pts_only"):
            elems.append(f'<line x1="{lx+9:.1f}" y1="{iy-4.2:.1f}" x2="{lx+30:.1f}" y2="{iy-4.2:.1f}" stroke="{s["color"]}" stroke-width="2.6"{d_attr}/>')
        elems.append(f'<circle cx="{lx+19.5:.1f}" cy="{iy-4.2:.1f}" r="4.0" fill="{s["color"]}"/>')
        elems.append(f'<text x="{lx+37:.1f}" y="{iy:.1f}" font-size="{fs_leg}">{html.escape(s["label"])}</text>')


def render_single(nm, xlim, ylim, title, xlabel, ylabel, xticks, yticks, series,
                  xlog=False, ylog=True, leg_pos="tl", hlines=None, annots=None):
    W, H = 740, 480
    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<style>text { font-family: -apple-system, Helvetica, Arial, sans-serif; fill: #1f2937; }</style>',
    ]
    draw_axes_panel(elems, 88, 48, 626, 372, xlim, ylim, title, xlabel, ylabel,
                    xticks, yticks, series, xlog, ylog, leg_pos, hlines, annots, big=True)
    save_svg_png(nm, elems, W, H)


def render_double(nm, p1, p2):
    W, H = 1180, 480
    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<style>text { font-family: -apple-system, Helvetica, Arial, sans-serif; fill: #1f2937; }</style>',
    ]
    draw_axes_panel(elems, 88, 48, 468, 372, **p1)
    draw_axes_panel(elems, 678, 48, 468, 372, **p2)
    save_svg_png(nm, elems, W, H)


def render_geometry_schematic():
    W, H = 1180, 500
    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<style>text { font-family: -apple-system, Helvetica, Arial, sans-serif; fill: #1f2937; }</style>',
        '<text x="590" y="28" text-anchor="middle" font-size="16.5" font-weight="700">'
        'How 3D Welded Quantum Memories Work — and Three Geometric Design Limits</text>',
    ]

    # 4 card boxes across 1180px
    cards = [
        (20,  52, 272, 426, "(a) 3-Way Bridge Stops Free Walks"),
        (308, 52, 272, 426, "(b) Compact 2×2 Cross (Works)"),
        (596, 52, 272, 426, "(c) Pruned 3-Arm Bridge (Leaks)"),
        (884, 52, 272, 426, "(d) Stretched Cubic Grid (Slow)"),
    ]
    for cx, cy, cw, ch, ctitle in cards:
        elems.append(f'<rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" rx="10" fill="#f9fafb" stroke="#cbd5e1" stroke-width="1.4"/>')
        elems.append(f'<text x="{cx+cw/2:.1f}" y="{cy+26}" text-anchor="middle" font-size="13.5" font-weight="700">{html.escape(ctitle)}</text>')

    # --- Panel (a): Two parallel sheets + vertical bridge + 1->2 defect splitting ---
    # Bottom sheet parallelogram
    elems.append('<polygon points="42,290 198,290 256,242 100,242" fill="#dbeafe" fill-opacity="0.72" stroke="#2563eb" stroke-width="1.8"/>')
    # Top sheet parallelogram
    elems.append('<polygon points="42,182 198,182 256,134 100,134" fill="#dbeafe" fill-opacity="0.72" stroke="#2563eb" stroke-width="1.8"/>')
    # Vertical perpendicular bridge sheet joining center seam
    elems.append('<polygon points="122,290 122,182 180,134 180,242" fill="#fde68a" fill-opacity="0.80" stroke="#d97706" stroke-width="2.0"/>')
    # Highlight 3-way junction seam lines
    elems.append('<line x1="122" y1="182" x2="180" y2="134" stroke="#b45309" stroke-width="3.2"/>')
    elems.append('<line x1="122" y1="290" x2="180" y2="242" stroke="#b45309" stroke-width="2.4" stroke-dasharray="4,3"/>')
    # Defect walking on top sheet (left) -> hitting seam -> splitting into TWO defects (top right + bridge)
    elems.append('<circle cx="88" cy="162" r="7.5" fill="#dc2626" stroke="#ffffff" stroke-width="1.5"/>')
    elems.append('<line x1="98" y1="162" x2="138" y2="160" stroke="#dc2626" stroke-width="2.4"/>')
    # Split arrows across junction
    elems.append('<line x1="146" y1="158" x2="196" y2="154" stroke="#dc2626" stroke-width="2.4"/>')
    elems.append('<line x1="146" y1="162" x2="152" y2="205" stroke="#dc2626" stroke-width="2.4"/>')
    elems.append('<circle cx="203" cy="153" r="7.5" fill="#dc2626" stroke="#ffffff" stroke-width="1.5"/>')
    elems.append('<circle cx="153" cy="212" r="7.5" fill="#dc2626" stroke="#ffffff" stroke-width="1.5"/>')
    elems.append('<text x="78" y="122" text-anchor="middle" font-size="11.5" font-weight="600" fill="#1e40af">2D Sheet 1</text>')
    elems.append('<text x="74" y="310" text-anchor="middle" font-size="11.5" font-weight="600" fill="#1e40af">2D Sheet 2</text>')
    elems.append('<text x="216" y="212" text-anchor="middle" font-size="11.5" font-weight="700" fill="#b45309">Bridge</text>')
    # Caption inside card (a)
    for idx, txt in enumerate([
        "Parallel 2D sheets are welded by",
        "perpendicular bridges at all scales.",
        "Crossing any 3-way seam forces",
        "1 defect to split into 2 (costs +ΔE),",
        "blocking free 2D error walks.",
    ]):
        elems.append(f'<text x="156" y="{352 + idx*19}" text-anchor="middle" font-size="11.8">{html.escape(txt)}</text>')

    # --- Panel (b): Top view of Compact 2x2 Four-Arm Cross (L=2, r=1) ---
    bx, by, bs = 354, 114, 180
    elems.append(f'<rect x="{bx}" y="{by}" width="{bs}" height="{bs}" fill="#eff6ff" stroke="#1e3a8a" stroke-width="2.4"/>')
    # 2x2 grid lines
    elems.append(f'<line x1="{bx+bs/2}" y1="{by}" x2="{bx+bs/2}" y2="{by+bs}" stroke="#dc2626" stroke-width="5.0"/>')
    elems.append(f'<line x1="{bx}" y1="{by+bs/2}" x2="{bx+bs}" y2="{by+bs/2}" stroke="#dc2626" stroke-width="5.0"/>')
    # Wall anchor dots on all 4 sides + center
    for px, py in [(bx+bs/2, by), (bx+bs/2, by+bs), (bx, by+bs/2), (bx+bs, by+bs/2), (bx+bs/2, by+bs/2)]:
        elems.append(f'<circle cx="{px}" cy="{py}" r="5.5" fill="#991b1b" stroke="#ffffff" stroke-width="1.3"/>')
    elems.append(f'<text x="{bx+bs/2}" y="{by+bs+24}" text-anchor="middle" font-size="12.0" font-weight="700" fill="#047857">'
                 'All 4 bridge arms reach cell walls</text>')
    for idx, txt in enumerate([
        "Smallest uniform confining cell:",
        "254 qubits (planar) or 508 (torus),",
        "220× smaller than 8×8 layout.",
        "Preserves encoded qubits across",
        "cycles (106× gain at Cycle 2).",
    ]):
        elems.append(f'<text x="444" y="{352 + idx*19}" text-anchor="middle" font-size="11.8">{html.escape(txt)}</text>')

    # --- Panel (c): Top view of Pruned 3-Arm T-Bridge ("Buttonhole" Leak) ---
    cx0, cy0, cs = 642, 114, 180
    elems.append(f'<rect x="{cx0}" y="{cy0}" width="{cs}" height="{cs}" fill="#eff6ff" stroke="#1e3a8a" stroke-width="2.4"/>')
    # 3 arms present (left, right, top), bottom arm missing!
    elems.append(f'<line x1="{cx0+cs/2}" y1="{cy0}" x2="{cx0+cs/2}" y2="{cy0+cs/2}" stroke="#dc2626" stroke-width="5.0"/>')
    elems.append(f'<line x1="{cx0}" y1="{cy0+cs/2}" x2="{cx0+cs}" y2="{cy0+cs/2}" stroke="#dc2626" stroke-width="5.0"/>')
    # Dashed missing 4th arm
    elems.append(f'<line x1="{cx0+cs/2}" y1="{cy0+cs/2}" x2="{cx0+cs/2}" y2="{cy0+cs}" stroke="#9ca3af" stroke-width="2.5" stroke-dasharray="5,5"/>')
    # Buttonhole loop around dangling center endpoint!
    elems.append(f'<circle cx="{cx0+cs/2}" cy="{cy0+cs/2+14}" r="22" fill="#fee2e2" fill-opacity="0.7" stroke="#b91c1c" stroke-width="2.2" stroke-dasharray="4,3"/>')
    elems.append(f'<circle cx="{cx0+cs/2}" cy="{cy0+cs/2}" r="6.5" fill="#b91c1c" stroke="#ffffff" stroke-width="1.5"/>')
    elems.append(f'<text x="{cx0+cs/2}" y="{cy0+cs+24}" text-anchor="middle" font-size="12.0" font-weight="700" fill="#b91c1c">'
                 'Dangling end tears "buttonhole"</text>')
    for idx, txt in enumerate([
        "Stopping a bridge mid-cell opens",
        "a hole between the two sheets.",
        "Spawns O(N) unprotected qubits",
        "(K jumps from 2 to 1,004) whose",
        "lifetime collapses as 1/N.",
    ]):
        elems.append(f'<text x="732" y="{352 + idx*19}" text-anchor="middle" font-size="11.8">{html.escape(txt)}</text>')

    # --- Panel (d): Cubic-Grid Wire Stretching (l=8 -> 7-qubit free wire) ---
    dx0, dy0 = 910, 145
    # Left & right 3-way junction blocks
    elems.append(f'<rect x="{dx0}" y="{dy0}" width="26" height="110" rx="4" fill="#fde68a" stroke="#b45309" stroke-width="2.0"/>')
    elems.append(f'<rect x="{dx0+192}" y="{dy0}" width="26" height="110" rx="4" fill="#fde68a" stroke="#b45309" stroke-width="2.0"/>')
    # 7-qubit horizontal wire between junctions
    wy = dy0 + 55
    elems.append(f'<line x1="{dx0+26}" y1="{wy}" x2="{dx0+192}" y2="{wy}" stroke="#374151" stroke-width="3.0"/>')
    for qi in range(7):
        qx = dx0 + 40 + qi * 23
        col = "#dc2626" if qi == 3 else "#2563eb"
        rad = 6.5 if qi == 3 else 4.8
        elems.append(f'<circle cx="{qx}" cy="{wy}" r="{rad}" fill="{col}" stroke="#ffffff" stroke-width="1.2"/>')
    # Double-headed free-hop arrow above wire
    elems.append(f'<line x1="{dx0+56}" y1="{wy-24}" x2="{dx0+162}" y2="{wy-24}" stroke="#dc2626" stroke-width="2.2"/>')
    elems.append(f'<polygon points="{dx0+50},{wy-24} {dx0+60},{wy-29} {dx0+60},{wy-19}" fill="#dc2626"/>')
    elems.append(f'<polygon points="{dx0+168},{wy-24} {dx0+158},{wy-29} {dx0+158},{wy-19}" fill="#dc2626"/>')
    elems.append(f'<text x="{dx0+109}" y="{wy-34}" text-anchor="middle" font-size="11.8" font-weight="700" fill="#dc2626">ΔE = 0 free walk</text>')
    elems.append(f'<text x="{dx0+109}" y="{dy0+145}" text-anchor="middle" font-size="12.0" font-weight="700" fill="#374151">'
                 '7-qubit wire on 3D cubic grid</text>')
    for idx, txt in enumerate([
        "Routing on one 3D cubic grid (ℓ=8)",
        "turns each edge into a 7-qubit wire.",
        "Defects hop along wires for free,",
        "cutting lifetime 321× (56,312 qubits",
        "vs. 1,480 unstretched).",
    ]):
        elems.append(f'<text x="1020" y="{352 + idx*19}" text-anchor="middle" font-size="11.8">{html.escape(txt)}</text>')

    save_svg_png("fig_v8_1_geometry", elems, W, H)


def load_all():
    kmc4 = {}
    v4_p = os.path.join(WS, "cluster_v4_results.txt")
    if os.path.exists(v4_p):
        with open(v4_p) as f:
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
    v6_p = os.path.join(WS, "cluster_v6_results.txt")
    if os.path.exists(v6_p):
        with open(v6_p) as f:
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
    v7_p = os.path.join(WS, "cluster_v7_results.txt")
    with open(v7_p) as f:
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
                    "syn": float(kv["syn_dens"]),
                }
    return kmc4, kmc6, kmc7, ffs7, cool7


def main():
    render_geometry_schematic()
    kmc4, kmc6, kmc7, ffs7, cool7 = load_all()

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
        if vx is not None and "cb" not in pref_s and (pref_s.endswith("_s2") or pref_s.endswith("_s4")):
            return vx
        return None

    # --- 2. `fig_v8_2_four_classes.png`: Clean 5-Curve Pure-Quantum Fixed-Temperature Lifetime vs N ---
    def v4_fixed_b(cnames, b):
        xs, ys = [], []
        for cn in cnames:
            bm = kmc4.get(cn, {})
            if b in bm:
                xs.append(bm[b][0])
                ys.append(bm[b][1])
        return xs, ys

    haah_x, haah_y = v4_fixed_b([f"haah_cubic_L{L}_Xerr.code" for L in [3, 5, 7, 9, 11, 13]], 4.25)
    bb_x, bb_y = v4_fixed_b([f"bb_gross_{l}x{m}_Xerr.code" for l, m in [(6,6),(12,6),(18,12),(24,18),(30,18),(36,18),(48,18),(60,18)]], 4.25)
    lay_x, lay_y = v4_fixed_b(["layer_512_Xerr.code","layer_steane_Xerr.code","layer_913_Xerr.code","layer_13_1_3_Xerr.code","layer_913_ksep2_Xerr.code","layer_15_35_Xerr.code"], 4.50)
    tor_x, tor_y = v4_fixed_b([f"toric2d_L{L}_Xerr.code" for L in [4, 6, 10, 16, 24, 34]], 4.50)
    weld_x = [2, 254, 25148]
    weld_y = [kmc4.get("q_l2_r1_step0_X.code", {}).get(6.0, (2, 3.0))[1],
              v7_min2("sq_L2r1_s2", 6.50),
              v7_min2("sq_L2r1_s4", 6.50)]

    s_f2 = [
        {"xs": haah_x, "ys": haah_y, "color": "#7c3aed", "dash": "",    "label": "Haah 3D Cubic Code (β=4.25: rises, peaks at L=11, turns over)"},
        {"xs": weld_x, "ys": weld_y, "color": "#be123c", "dash": "",    "label": "3D Multi-Scale Welded Code (β=6.5: rises across cycles)"},
        {"xs": bb_x,   "ys": bb_y,   "color": "#2563eb", "dash": "",    "label": "Scaled 2D Bivariate Bicycle Code (β=4.25: decays ~1/N)"},
        {"xs": lay_x,  "ys": lay_y,  "color": "#92400e", "dash": "6,4", "label": "3D Sparse-Seam Surface Layer Code (β=4.5: flat-sheet decay)"},
        {"xs": tor_x,  "ys": tor_y,  "color": "#4b5563", "dash": "3,3", "label": "2D Toric Code (β=4.5: constant barrier, decays ~1/N)"},
    ]
    render_single(
        "fig_v8_2_four_classes", (1.6, 36000.0), (0.8, 1e12),
        "Fixed-Temperature Memory Lifetime vs. System Size Across Quantum Code Families",
        "Physical Qubit Count N (Log Scale)", "Mean Memory Lifetime at Fixed Temperature",
        [(2,"2"), (10,"10"), (100,"100"), (1000,"1,000"), (10000,"10,000"), (30000,"30,000")],
        [(1e0,"1"), (1e2,"10²"), (1e4,"10⁴"), (1e6,"10⁶"), (1e8,"10⁸"), (1e10,"10¹⁰")],
        s_f2, xlog=True, ylog=True, leg_pos="tl",
        annots=[(9500, 4.2e7, "Cubic Code Ceiling ↓", "#6d28d9", "middle"),
                (12500, 1.8e6, "Welded Code Rises ↑", "#be123c", "middle")],
    )

    # --- 3. `fig_v8_3_barriers.png`: Clean Effective Energy Barrier vs System Size N ---
    def v4_slope_curve(cnames, bmin=3.5, bmax=5.0):
        xs, ys = [], []
        for cn in cnames:
            bm = kmc4.get(cn, {})
            bs = sorted(b for b in bm if bmin <= b <= bmax)
            if len(bs) >= 2:
                N = bm[bs[0]][0]
                sl = ols_slope(bs, [bm[b][1] for b in bs])
                if sl is not None:
                    xs.append(N); ys.append(sl)
        return xs, ys

    c_haah_N, c_haah_D = v4_slope_curve([f"haah_cubic_L{L}_Xerr.code" for L in [3, 5, 7, 9, 11, 13]], 3.5, 5.0)
    c_bb_N, c_bb_D = v4_slope_curve([f"bb_gross_{l}x{m}_Xerr.code" for l, m in [(6,6),(12,6),(18,12),(24,18),(30,18),(36,18),(48,18),(60,18)]], 3.5, 5.0)
    c_lay_N, c_lay_D = v4_slope_curve(["layer_512_Xerr.code","layer_steane_Xerr.code","layer_913_Xerr.code","layer_13_1_3_Xerr.code","layer_913_ksep2_Xerr.code","layer_15_35_Xerr.code"], 3.5, 5.0)
    c_tor_N, c_tor_D = v4_slope_curve([f"toric2d_L{L}_Xerr.code" for L in [4, 6, 10, 16, 24, 34]], 3.5, 5.0)

    rot_win = [5.50, 6.00, 6.25, 6.50]
    d_rot_s2 = ols_slope(rot_win, [v7_min2("rottor_L2r1_s2", b) for b in rot_win])
    d_rot_s4 = ols_slope(rot_win, [v7_min2("rottor_L2r1_s4", b) for b in rot_win])
    d_sq_s2 = ols_slope(rot_win, [v7_min2("sq_L2r1_s2", b) for b in rot_win])
    d_sq_s4 = ols_slope(rot_win, [v7_min2("sq_L2r1_s4", b) for b in rot_win])

    s_f3 = [
        {"xs": c_haah_N, "ys": c_haah_D, "color": "#7c3aed", "dash": "", "label": "Haah 3D Cubic Code (logarithmic barrier: 4.3 → 9.3 → 8.8)"},
        {"xs": [4, 508, 50296], "ys": [1.92, d_rot_s2, d_rot_s4], "color": "#be123c", "dash": "",
         "label": f"3D Welded Torus Code (Cycles 0, 1, 2: 1.9 → {d_rot_s2:.1f} → {d_rot_s4:.1f})"},
        {"xs": [254, 25148], "ys": [d_sq_s2, d_sq_s4], "color": "#ea580c", "dash": "6,4",
         "label": f"3D Welded Planar Code (Cycles 1, 2: {d_sq_s2:.1f} → {d_sq_s4:.1f})"},
        {"xs": c_bb_N, "ys": c_bb_D, "color": "#2563eb", "dash": "", "label": "Scaled 2D Bivariate Bicycle Code (flat ceiling ≈ 5.4)"},
        {"xs": c_lay_N, "ys": c_lay_D, "color": "#92400e", "dash": "6,4", "label": "3D Sparse-Seam Surface Layer Code (flat ≤ 2.1)"},
        {"xs": c_tor_N, "ys": c_tor_D, "color": "#4b5563", "dash": "3,3", "label": "2D Toric Code (flat ≤ 2.3)"},
    ]
    render_single(
        "fig_v8_3_barriers", (2.5, 65000.0), (0.0, 15.2),
        "Effective Energy Barrier vs. System Size Across Quantum Code Families",
        "Physical Qubit Count N (Log Scale)", "Effective Energy Barrier (Log-Lifetime Slope)",
        [(4,"4"), (10,"10"), (100,"100"), (1000,"1,000"), (10000,"10,000"), (50000,"50,000")],
        [(0,"0"), (2,"2"), (4,"4"), (6,"6"), (8,"8"), (10,"10"), (12,"12"), (14,"14")],
        s_f3, xlog=True, ylog=False, leg_pos="tl",
    )

    # --- 4. `fig_v8_4_validation.png`: Clean 2-Panel External Validation ---
    s_tor = []
    for b, col, lbl in [
        (2.25, "#be123c", "Loop X-Errors (β=2.25: exponential)"),
        (2.00, "#dc2626", "Loop X-Errors (β=2.00: exponential)"),
        (1.75, "#f97316", "Loop X-Errors (β=1.75: exponential)"),
        (4.00, "#2563eb", "Point Z-Errors (β=4.00: ~1/N decay)"),
        (2.00, "#6b7280", "Point Z-Errors (β=2.00: ~1/N decay)"),
    ]:
        xs, ys = [], []
        is_z = "Point" in lbl
        for L in [3, 4, 5, 6, 7]:
            if is_z:
                rec = kmc6.get((f"toric3d_L{L}_Z_err", "uniform", "uf", b))
            else:
                rec = kmc7.get((f"toric3d_L{L}_X_err", "uniform", "maj_uf", b)) or kmc6.get((f"toric3d_L{L}_X_err", "uniform", "sweep_uf", b))
            if rec and rec[2] == 0:
                xs.append(L); ys.append(rec[1])
        s_tor.append({"xs": xs, "ys": ys, "color": col, "dash": "5,4" if is_z else "", "label": lbl})

    # Panel (b): Gu-Caha Random Layer Code Ensemble vs layer count n = 11..19 + published code_100 match!
    caha_groups = {
        11: ["caha_n11_code_100", "caha_n11_code_11", "caha_n11_code_113", "caha_n11_code_120", "caha_n11_code_128", "caha_n11_code_151"],
        13: ["caha_n13_code_102", "caha_n13_code_104", "caha_n13_code_106"],
        15: ["caha_n15_code_15",  "caha_n15_code_16",  "caha_n15_code_17"],
        17: ["caha_n17_code_1201","caha_n17_code_14",  "caha_n17_code_145"],
        19: ["caha_n19_code_129", "caha_n19_code_199", "caha_n19_code_236"],
    }
    s_caha = []
    for b, col, lbl in [
        (8.00, "#7c3aed", "Layer Ensemble Mean (β=8.0: peaks n=13)"),
        (7.00, "#059669", "Layer Ensemble Mean (β=7.0: peaks n=13)"),
        (6.00, "#2563eb", "Layer Ensemble Mean (β=6.0: peaks n=11)"),
    ]:
        xs, ys = [], []
        for n_val in [11, 13, 15, 17, 19]:
            vals = [kmc6[(cn + "_X_err", "uniform", "uf", b)][1]
                    for cn in caha_groups[n_val]
                    if (cn + "_X_err", "uniform", "uf", b) in kmc6 and kmc6[(cn + "_X_err", "uniform", "uf", b)][2] == 0]
            if vals:
                xs.append(n_val); ys.append(sum(vals) / len(vals))
        s_caha.append({"xs": xs, "ys": ys, "color": col, "dash": "", "label": lbl})
    c100_our = kmc6[("caha_n11_code_100_X_err", "uniform", "uf", 6.00)][1]
    s_caha.append({"xs": [11], "ys": [c100_our], "color": "#dc2626", "pts_only": True, "r": 6.2,
                   "label": f"Our Simulator on code_100 (β=6: {c100_our:,.0f})"})
    s_caha.append({"xs": [11.22], "ys": [1513.9], "color": "#111827", "pts_only": True, "r": 6.2,
                   "label": "Published PRL Value on code_100 (1,514)"})

    render_double(
        "fig_v8_4_validation",
        dict(xlim=(2.75, 7.25), ylim=(0.3, 6e11),
             title="(a) 3D Toric Code Benchmark (Gu, Caha et al.)",
             xlabel="Lattice Size L (N = 3L³ Qubits)", ylabel="Mean Sector Lifetime",
             xticks=[(3,"3"), (4,"4"), (5,"5"), (6,"6"), (7,"7")],
             yticks=[(1e0,"1"), (1e2,"10²"), (1e4,"10⁴"), (1e6,"10⁶"), (1e8,"10⁸"), (1e10,"10¹⁰")],
             series=s_tor, xlog=False, ylog=True, leg_pos="tl"),
        dict(xlim=(10.5, 19.5), ylim=(5e2, 8e8),
             title="(b) Random Layer Code Validation & Turnover",
             xlabel="Layer Count n (N ≈ 2,200 to 11,000 Qubits)", ylabel="Mean Memory Lifetime",
             xticks=[(11,"11"), (13,"13"), (15,"15"), (17,"17"), (19,"19")],
             yticks=[(1e3,"10³"), (1e5,"10⁵"), (1e7,"10⁷")],
             series=s_caha, xlog=False, ylog=True, leg_pos="tl"),
    )

    # --- 5. `fig_v8_5_multicycle_floor.png`: Clean Multi-Cycle Crossings vs Pruned Collapse ---
    s_f5 = []
    for pref_s, col, dsh, lbl in [
        ("rottor_L2r1_s4",   "#be123c", "",    "Torus Cycle 2 (50,296 qubits: overtakes Cycle 1 at β≈6.3)"),
        ("rottor_L2r1_s2",   "#f97316", "5,4", "Torus Cycle 1 (508 qubits, 4-arm cross)"),
        ("sq_L2r1_s4",       "#059669", "",    "Planar Cycle 2 (25,148 qubits: overtakes Cycle 1 at β≈5.5)"),
        ("sq_L2r1_s2",       "#2563eb", "5,4", "Planar Cycle 1 (254 qubits, 4-arm cross)"),
        ("rottor_cb3rib_s4", "#4b5563", "2,3", "Pruned 3-Arm T-Bridge (40,264 qubits: buttonhole collapse)"),
    ]:
        xs, ys = [], []
        for b in [5.00, 5.50, 6.00, 6.25, 6.50, 6.75, 7.00, 7.50, 8.00]:
            v = v7_min2(pref_s, b, min_reps=3)
            if v is not None:
                xs.append(b); ys.append(v)
        s_f5.append({"xs": xs, "ys": ys, "color": col, "dash": dsh, "label": lbl})
    render_single(
        "fig_v8_5_multicycle_floor", (4.85, 8.15), (4e-4, 2e12),
        "Two-Cycle 3D Self-Correction Crossings vs. Pruned-Bridge Collapse",
        "Inverse Bath Temperature β = 1 / T  (Colder →)", "Quantum Memory Lifetime",
        [(5.0,"5.0"), (5.5,"5.5"), (6.0,"6.0"), (6.5,"6.5"), (7.0,"7.0"), (7.5,"7.5"), (8.0,"8.0")],
        [(1e-3,"10⁻³"), (1e0,"1"), (1e3,"10³"), (1e6,"10⁶"), (1e9,"10⁹")],
        s_f5, xlog=False, ylog=True, leg_pos="tl",
        annots=[(6.8, 1.6e-2, "3-Arm Pruned Bridge Collapses (~0.002)", "#4b5563", "middle")],
    )

    # --- 6. `fig_v8_6_grid_and_balance.png`: 2-Panel Cubic-Grid Cost & Sector Balancing ---
    s_grid = []
    xs_u, ys_u = [], []
    for b in [5.50, 6.00, 6.50, 7.00]:
        v = v7_min2("sq_L3r2_s2", b, min_reps=3)
        if v is not None:
            xs_u.append(b); ys_u.append(v)
    s_grid.append({"xs": xs_u, "ys": ys_u, "color": "#2563eb", "dash": "", "label": "Unstretched 2-Bridge (1,480 qubits: 321× longer)"})
    for zc, jm, col, dsh, lbl in [
        ("sq_z3_4_6_r2",  "uniform",  "#059669", "",    "Tighter Cubic Grid (4,6) (10,280 qubits: 13× longer)"),
        ("sq_bdl_8_8_r2", "pin_norm", "#dc2626", "",    "Cubic Grid (8,8) + Wire Traps (56,312 qubits: 1.4×–1.9×)"),
        ("sq_bdl_8_8_r2", "uniform",  "#4b5563", "5,4", "Cubic Grid (8,8) Uniform (56,312 qubits: baseline)"),
    ]:
        xs, ys = [], []
        for b in [5.50, 6.00, 6.50, 7.00]:
            rx = kmc7.get((f"{zc}_X_err", jm, "uf", b))
            rz = kmc7.get((f"{zc}_Z_err", jm, "uf", b))
            if rx and rx[2] == 0:
                v = min(rx[1], rz[1]) if (rz and rz[2] == 0) else rx[1]
                xs.append(b); ys.append(v)
        s_grid.append({"xs": xs, "ys": ys, "color": col, "dash": dsh, "label": lbl})

    s_bal = []
    for tag, col, dsh, lbl in [
        ("rottor_L2r1_s2_Z_err",  "#059669", "5,4", "1→1 Torus Z-Errors Only (508 qubits)"),
        ("rottor_hyb21_s2_X_err", "#7c3aed", "",    "Pre-Balanced 2→1 Lifetime (1,172 qubits: 336× > 1→2)"),
        ("rottor_L2r1_s2_X_err",  "#dc2626", "",    "1→1 Torus X-Errors / Overall (508 qubits)"),
        ("rottor_hyb12_s2_X_err", "#f97316", "4,3", "Reversed 1→2 Lifetime (1,292 qubits)"),
    ]:
        xs, ys = [], []
        for b in [5.50, 6.00, 6.25, 6.50, 6.75, 7.00, 7.50]:
            v = v7_best(tag, "uniform", "uf", b, min_reps=3)
            if v is not None:
                xs.append(b); ys.append(v)
        s_bal.append({"xs": xs, "ys": ys, "color": col, "dash": dsh, "label": lbl})

    render_double(
        "fig_v8_6_grid_and_balance",
        dict(xlim=(5.38, 7.15), ylim=(40.0, 2e8),
             title="(a) Cubic-Grid Wire Stretching Cost",
             xlabel="Inverse Bath Temperature β = 1 / T", ylabel="Quantum Memory Lifetime",
             xticks=[(5.5,"5.5"), (6.0,"6.0"), (6.5,"6.5"), (7.0,"7.0")],
             yticks=[(1e2,"10²"), (1e4,"10⁴"), (1e6,"10⁶"), (1e8,"10⁸")],
             series=s_grid, xlog=False, ylog=True, leg_pos="tl"),
        dict(xlim=(5.38, 7.65), ylim=(1e3, 2e14),
             title="(b) X/Z Asymmetry & 2→1 Pre-Balancing",
             xlabel="Inverse Bath Temperature β = 1 / T", ylabel="Sector / Memory Lifetime",
             xticks=[(5.5,"5.5"), (6.0,"6.0"), (6.5,"6.5"), (7.0,"7.0"), (7.5,"7.5")],
             yticks=[(1e3,"10³"), (1e5,"10⁵"), (1e7,"10⁷"), (1e9,"10⁹"), (1e11,"10¹¹"), (1e13,"10¹³")],
             series=s_bal, xlog=False, ylog=True, leg_pos="tl"),
    )

    # --- 7. `fig_v8_7_healing_cooling.png`: 2-Panel Self-Healing & Hot-State Cooling ---
    p_vals = [0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.12]
    s_heal = []
    for cname, sec, col, dsh, lbl in [
        ("rottor_L2r1_s2", "Z_err", "#2563eb", "",    "Torus Z-Errors (508 qubits, K=2: 98% at 1%)"),
        ("sq_L2r1_s2",     "X_err", "#dc2626", "",    "Planar X-Errors (254 qubits, K=1: 81% at 1%)"),
        ("rottor_L2r1_s2", "X_err", "#7c3aed", "5,4", "Torus X-Errors (508 qubits, K=2: 76% at 1%)"),
    ]:
        xs, ys = [], []
        for p0 in p_vals:
            cr = cool7.get((f"{cname}_{sec}", "quench", f"noisy_{p0:.2f}", 1000.0))
            if cr:
                xs.append(100.0 * p0)
                ys.append(100.0 * cr["pheal"])
        s_heal.append({"xs": xs, "ys": ys, "color": col, "dash": dsh, "label": lbl})

    s_hot = []
    for tag, proto, col, dsh, lbl in [
        ("rottor_L2r1_s2_Z_err", "anneal", "#059669", "",    "508-Qubit Torus: Gradual Cooling (100% clean)"),
        ("rottor_L2r1_s2_Z_err", "quench", "#10b981", "5,4", "508-Qubit Torus: Sudden Cold Quench"),
        ("sq_L2r1_s3_X_err",     "anneal", "#2563eb", "",    "1,942-Qubit 3-Step: Gradual Cooling (63% clean)"),
        ("sq_L2r1_s3_X_err",     "quench", "#dc2626", "4,3", "1,942-Qubit 3-Step: Sudden Quench (14%, trapped)"),
    ]:
        xs, ys = [], []
        for t_val in [100.0, 400.0, 1500.0]:
            cr = cool7.get((tag, proto, "hot", t_val))
            if cr:
                xs.append(t_val)
                ys.append(100.0 * cr["p0"])
        s_hot.append({"xs": xs, "ys": ys, "color": col, "dash": dsh, "label": lbl})

    render_double(
        "fig_v8_7_healing_cooling",
        dict(xlim=(0.5, 12.5), ylim=(-2.0, 108.0),
             title="(a) Self-Healing of Noisy Codewords (β=6.5)",
             xlabel="Initial Physical Error Rate (%)", ylabel="Zero-Decoder Exact Recovery (%)",
             xticks=[(1,"1%"), (2,"2%"), (4,"4%"), (6,"6%"), (8,"8%"), (10,"10%"), (12,"12%")],
             yticks=[(0,"0%"), (25,"25%"), (50,"50%"), (75,"75%"), (100,"100%")],
             series=s_heal, xlog=False, ylog=False, leg_pos="tr",
             hlines=[(25.0, "#6b7280", "2-Qubit Random Guess Floor (25%)", "bl")]),
        dict(xlim=(80.0, 1800.0), ylim=(-2.0, 142.0),
             title="(b) Cooling From a Hot Random State (50% Noise)",
             xlabel="Bath Cooling Time (Log Scale)", ylabel="Defect-Free Ground States (%)",
             xticks=[(100,"100"), (400,"400"), (1500,"1,500")],
             yticks=[(0,"0%"), (25,"25%"), (50,"50%"), (75,"75%"), (100,"100%")],
             series=s_hot, xlog=True, ylog=False, leg_pos="tl"),
    )


if __name__ == "__main__":
    main()
