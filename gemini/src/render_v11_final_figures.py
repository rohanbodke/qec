#!/usr/bin/env python3
"""
render_v11_final_figures.py
Renders the 6 final publication-grade figures (V11) incorporating quarter-million-qubit scaling
and live 6-level syndrome-only thermal RG defect-density flow:
  1. fig_v11_1_classes.png:    (a) Relative Lifetime Growth vs N up to 189,988 / 322,159 qubits (10,250x full quantum!)
                               (b) Additive vs Multiplicative Energy Barriers & Power-Law Fit (Delta_eff ~ N^alpha)
  2. fig_v11_2_geometry.png:   4-card geometric mechanism & exact cell limits (c_cycle = 2/5 < 1)
  3. fig_v11_3_validation.png: External simulator validation (3D Toric Code & 18 Random Layer Codes)
  4. fig_v11_4_scaling.png:    2x2 flagship multi-scale & RG figure:
                               (a) Multi-scale 2x2 lifetime crossings (bounded threshold beta_c ≈ 1.86)
                               (b) Live 6-level syndrome-only thermal RG flow on 322,159 qubits (s=5->1)
                               (c) 3-level full (X,Z) quantum ladder (19 -> 1,942 -> 189,988 qubits: 10,250x gain)
                               (d) Provably contractive 2x2 (47,387q: 71x) & Torus (50,296q: 7.4x) full cycles
  5. fig_v11_5_tradeoffs.png:  (a) Pure cubic-grid wire stretching (321x loss) & (b) 2->1 bridge pre-balancing (57x gain)
  6. fig_v11_6_cooling.png:    (a) Zero-decoder storage self-healing & (b) Hot-state gradual cooling vs quench
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


def power_law_fit(ns, ds):
    pts = [(math.log(n), math.log(d)) for n, d in zip(ns, ds) if n > 0 and d is not None and d > 0]
    mx = sum(x for x, _ in pts) / len(pts)
    my = sum(y for _, y in pts) / len(pts)
    den = sum((x - mx) ** 2 for x, _ in pts)
    alpha = sum((x - mx) * (y - my) for x, y in pts) / den
    c0 = math.exp(my - alpha * mx)
    ss_tot = sum((y - my) ** 2 for _, y in pts)
    ss_res = sum((y - (my + alpha * (x - mx))) ** 2 for x, y in pts)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    return c0, alpha, r2


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
               hlines=None, vlines=None, varrows=None, notes=None, y_off=58):
    elems.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="#fafafa" stroke="#374151" stroke-width="1.5"/>')
    elems.append(f'<text x="{x0+pw/2:.1f}" y="{y0-13:.1f}" text-anchor="middle" font-size="14.6" font-weight="700">{html.escape(title)}</text>')
    elems.append(f'<text x="{x0+pw/2:.1f}" y="{y0+ph+41:.1f}" text-anchor="middle" font-size="13.4" font-weight="600">{html.escape(xlabel)}</text>')
    elems.append(f'<text x="{x0-y_off:.1f}" y="{y0+ph/2:.1f}" text-anchor="middle" font-size="13.4" font-weight="600" '
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
        elems.append(f'<text x="{px:.1f}" y="{y0+ph+20}" text-anchor="middle" font-size="12.0">{html.escape(xlbl)}</text>')

    for yv, ylbl in yticks:
        py = ty(yv)
        elems.append(f'<line x1="{x0}" y1="{py:.1f}" x2="{x0+pw}" y2="{py:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        elems.append(f'<line x1="{x0-5}" y1="{py:.1f}" x2="{x0}" y2="{py:.1f}" stroke="#374151" stroke-width="1.2"/>')
        elems.append(f'<text x="{x0-8}" y="{py+4.5:.1f}" text-anchor="end" font-size="12.0">{html.escape(ylbl)}</text>')

    if hlines:
        for yv, col, lbl, halign in hlines:
            py = ty(yv)
            elems.append(f'<line x1="{x0}" y1="{py:.1f}" x2="{x0+pw}" y2="{py:.1f}" stroke="{col}" stroke-width="1.6" stroke-dasharray="6,4"/>')
            hx = x0 + 10 if "l" in halign else x0 + pw - 10
            anch = "start" if "l" in halign else "end"
            hy = py + 15.5 if "b" in halign else py - 6.0
            elems.append(f'<text x="{hx:.1f}" y="{hy:.1f}" text-anchor="{anch}" font-size="11.5" font-weight="600" fill="{col}">{html.escape(lbl)}</text>')

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
        sw = s.get("lw", 2.9)
        pstr = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        if len(pts) >= 2:
            elems.append(f'<polyline fill="none" stroke="{s["color"]}" stroke-width="{sw}"{d_attr} points="{pstr}"/>')
        if not s.get("no_dots"):
            for px, py in pts:
                elems.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.7" fill="{s["color"]}" stroke="#ffffff" stroke-width="1.2"/>')

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

    if notes:
        for xv, yv, txt, col, anch in notes:
            elems.append(f'<text x="{tx(xv):.1f}" y="{ty(yv):.1f}" text-anchor="{anch}" font-size="11.6" font-weight="700" fill="{col}">{html.escape(txt)}</text>')

    leg_series = [s for s in series if s.get("label")]
    row_h = 19.6
    lh = row_h * len(leg_series) + 10
    lw = min(pw - 18.0, max(195.0, max(len(s["label"]) for s in leg_series) * 6.55 + 46.0))
    lx = x0 + 10 if "l" in leg_pos else x0 + pw - lw - 10
    ly = y0 + 10 if "t" in leg_pos else y0 + ph - lh - 10
    elems.append(f'<rect x="{lx:.1f}" y="{ly:.1f}" width="{lw:.1f}" height="{lh:.1f}" rx="6" fill="#ffffff" fill-opacity="0.96" stroke="#d1d5db" stroke-width="1.1"/>')
    for i, s in enumerate(leg_series):
        iy = ly + 15.8 + i * row_h
        d_attr = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
        elems.append(f'<line x1="{lx+8:.1f}" y1="{iy-4.0:.1f}" x2="{lx+29:.1f}" y2="{iy-4.0:.1f}" stroke="{s["color"]}" stroke-width="2.7"{d_attr}/>')
        if not s.get("no_dots"):
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


def render_quad(nm, p1, p2, p3, p4):
    W, H = 1180, 886
    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<style>text { font-family: -apple-system, Helvetica, Arial, sans-serif; fill: #1f2937; }</style>',
    ]
    draw_panel(elems, 86,  42, 472, 344, **p1)
    draw_panel(elems, 676, 42, 472, 344, **p2)
    draw_panel(elems, 86,  476, 472, 344, **p3)
    draw_panel(elems, 676, 476, 472, 344, **p4)
    save_svg_png(nm, elems, W, H)


def render_clean_geometry():
    W, H = 1180, 356
    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<style>text { font-family: -apple-system, Helvetica, Arial, sans-serif; fill: #1f2937; }</style>',
    ]
    cards = [
        (16,  14, 275, 328, "(a) 3-Way Bridge Splits Errors"),
        (307, 14, 275, 328, "(b) 2×2 Four-Arm Cross (Works)"),
        (598, 14, 275, 328, "(c) 3-Arm T-Bridge (Leaks)"),
        (889, 14, 275, 328, "(d) Scale-8 Cubic Grid (Slow)"),
    ]
    for cx, cy, cw, ch, ctitle in cards:
        elems.append(f'<rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" rx="10" fill="#f9fafb" stroke="#cbd5e1" stroke-width="1.5"/>')
        elems.append(f'<text x="{cx+cw/2:.1f}" y="{cy+28}" text-anchor="middle" font-size="14.0" font-weight="700">{html.escape(ctitle)}</text>')

    elems.append('<polygon points="36,242 136,202 268,202 168,242" fill="#dbeafe" stroke="#2563eb" stroke-width="1.8"/>')
    elems.append('<polygon points="36,134 136,94 268,94 168,134" fill="#dbeafe" fill-opacity="0.85" stroke="#2563eb" stroke-width="1.8"/>')
    elems.append('<polygon points="102,222 202,182 202,114 102,154" fill="#fde68a" fill-opacity="0.92" stroke="#d97706" stroke-width="2.2"/>')
    elems.append('<line x1="102" y1="154" x2="202" y2="114" stroke="#b91c1c" stroke-width="3.6"/>')
    elems.append('<circle cx="88" cy="127" r="8" fill="#dc2626" stroke="#ffffff" stroke-width="1.5"/>')
    elems.append('<circle cx="195" cy="109" r="8" fill="#dc2626" stroke="#ffffff" stroke-width="1.5"/>')
    elems.append('<circle cx="152" cy="167" r="8" fill="#dc2626" stroke="#ffffff" stroke-width="1.5"/>')
    elems.append('<text x="153.5" y="292" text-anchor="middle" font-size="12.5" font-weight="600" fill="#1f2937">1 defect splits into 2 at seam</text>')
    elems.append('<text x="153.5" y="320" text-anchor="middle" font-size="12.7" font-weight="700" fill="#047857">Costs +energy to cross</text>')

    bx0, by0, bsz = 369, 90, 152
    elems.append(f'<rect x="{bx0}" y="{by0}" width="{bsz}" height="{bsz}" fill="#eff6ff" stroke="#1d4ed8" stroke-width="2.4"/>')
    elems.append(f'<line x1="{bx0+bsz/2}" y1="{by0}" x2="{bx0+bsz/2}" y2="{by0+bsz}" stroke="#2563eb" stroke-width="5.5"/>')
    elems.append(f'<line x1="{bx0}" y1="{by0+bsz/2}" x2="{bx0+bsz}" y2="{by0+bsz/2}" stroke="#2563eb" stroke-width="5.5"/>')
    for wx, wy in [(bx0+bsz/2, by0), (bx0+bsz/2, by0+bsz), (bx0, by0+bsz/2), (bx0+bsz, by0+bsz/2), (bx0+bsz/2, by0+bsz/2)]:
        elems.append(f'<circle cx="{wx}" cy="{wy}" r="6.5" fill="#1e40af" stroke="#ffffff" stroke-width="1.5"/>')
    elems.append('<text x="444.5" y="292" text-anchor="middle" font-size="12.5" font-weight="600" fill="#1f2937">Contracts syndromes by 2/5</text>')
    elems.append('<text x="444.5" y="320" text-anchor="middle" font-size="12.7" font-weight="700" fill="#047857">174×–220× smaller cell</text>')

    cx0, cy0, csz = 660, 90, 152
    elems.append(f'<rect x="{cx0}" y="{cy0}" width="{csz}" height="{csz}" fill="#fef2f2" stroke="#b91c1c" stroke-width="2.4"/>')
    elems.append(f'<line x1="{cx0}" y1="{cy0+csz/2}" x2="{cx0+csz}" y2="{cy0+csz/2}" stroke="#2563eb" stroke-width="5.5"/>')
    elems.append(f'<line x1="{cx0+csz/2}" y1="{cy0}" x2="{cx0+csz/2}" y2="{cy0+csz/2}" stroke="#2563eb" stroke-width="5.5"/>')
    elems.append(f'<line x1="{cx0+csz/2}" y1="{cy0+csz/2}" x2="{cx0+csz/2}" y2="{cy0+csz}" stroke="#9ca3af" stroke-width="2.5" stroke-dasharray="5,5"/>')
    elems.append(f'<circle cx="{cx0+csz/2}" cy="{cy0+csz/2+26}" r="18" fill="#fee2e2" stroke="#dc2626" stroke-width="2.6" stroke-dasharray="4,3"/>')
    elems.append(f'<text x="{cx0+csz/2}" y="{cy0+csz/2+30}" text-anchor="middle" font-size="11.5" font-weight="700" fill="#b91c1c">Hole</text>')
    elems.append('<text x="735.5" y="292" text-anchor="middle" font-size="12.5" font-weight="600" fill="#1f2937">Cut arm opens sheet gap</text>')
    elems.append('<text x="735.5" y="320" text-anchor="middle" font-size="12.7" font-weight="700" fill="#b91c1c">Spawns 1,004 bad qubits</text>')

    dx0, dy0 = 916, 112
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
    elems.append('<text x="1026.5" y="292" text-anchor="middle" font-size="12.5" font-weight="600" fill="#1f2937">7-qubit wire between junctions</text>')
    elems.append('<text x="1026.5" y="320" text-anchor="middle" font-size="12.7" font-weight="700" fill="#b91c1c">Cuts lifetime 321×</text>')

    save_svg_png("fig_v11_2_geometry", elems, W, H)


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
    for fn in ("cluster_v8_results.txt", "cluster_v9_results.txt", "cluster_v9b_results.txt", "cluster_v9d_results.txt"):
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
    render_clean_geometry()
    kmc4, kmc6, kmc7, ffs7, cool7, kmc9, rg9 = load_all()

    def v7_best(tag, jm, dec, b, min_reps=3):
        if (tag, jm, dec, b) in kmc7 and kmc7[(tag, jm, dec, b)][2] == 0:
            return kmc7[(tag, jm, dec, b)][1]
        if (tag, jm, dec, b) in ffs7 and ffs7[(tag, jm, dec, b)][2] >= min_reps:
            return ffs7[(tag, jm, dec, b)][1]
        return None

    # --- 1. `fig_v11_1_classes.png`: Quarter-Million-Qubit Scaling & Power-Law Barrier Fit ---
    def v4_norm_curve(cnames, b):
        pts = [(kmc4[cn][b][0], kmc4[cn][b][1]) for cn in cnames if b in kmc4.get(cn, {})]
        y0 = pts[0][1]
        return [x for x, _ in pts], [y / y0 for _, y in pts]

    hx, hy = v4_norm_curve([f"haah_cubic_L{L}_Xerr.code" for L in [3, 5, 7, 9, 11, 13]], 4.25)
    bx, by = v4_norm_curve([f"bb_gross_{l}x{m}_Xerr.code" for l, m in [(6,6),(12,6),(18,12),(24,18),(30,18),(36,18),(48,18),(60,18)]], 4.25)
    tx_n, ty_n = v4_norm_curve([f"toric2d_L{L}_Xerr.code" for L in [4, 6, 10, 16, 24, 34]], 4.50)

    # Check if 5 nontrivial scales (`s = 1..5` up to 322,159q) completed with `gain_uf` at beta=2.22 or 2.28:
    has_s5_gain = all((f"sq_rx_iso_s{s}_X_err", "gain_uf", 2.22) in kmc9 for s in [1, 2, 3, 4, 5])
    if has_s5_gain:
        b_ref = 2.28 if all((f"sq_rx_iso_s{s}_X_err", "gain_uf", 2.28) in kmc9 for s in [1, 2, 3, 4, 5]) else 2.22
        w_pts = [kmc9[(f"sq_rx_iso_s{s}_X_err", "gain_uf", b_ref)] for s in [1, 2, 3, 4, 5]]
        w_gain = w_pts[-1][1] / w_pts[0][1]
        w_lbl = f"3D Welded 2×2 (5 scales to 322k: {w_gain:,.0f}×)"
    else:
        w_pts = [kmc9[(f"sq_rx_iso_s{s}_X_err", "uf", 2.25)] for s in [1, 2, 3, 4]]
        w_gain = w_pts[-1][1] / w_pts[0][1]
        w_lbl = f"3D Welded 2×2 (4 scales to 27k: {w_gain:,.0f}×)"
    w_x = [p[0] for p in w_pts]
    w_y = [p[1] / w_pts[0][1] for p in w_pts]

    # 3-Level Full (X,Z) Quantum Ladder (`19 -> 1,942 -> 189,988` qubits at beta=6.50, pure `uf`, zero bails!)
    q3_pts = [kmc9[(f"sq_L2r1_s{s}_Z_err", "uf", 6.50)] for s in [1, 3, 5]]
    q3_x = [p[0] for p in q3_pts]
    q3_y = [p[1] / q3_pts[0][1] for p in q3_pts]  # 1x -> 479x -> 10,250x!

    s_norm = [
        {"xs": q3_x, "ys": q3_y, "color": "#ea580c", "dash": "7,3", "label": f"3D Welded Full (X,Z) (to 190k: {q3_y[-1]:,.0f}×)"},
        {"xs": w_x,  "ys": w_y,  "color": "#be123c", "dash": "",    "label": w_lbl},
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

    # Barrier curve for Uniform 2x2 (`s = 0..4` or `0..5`) + empirical power-law fit Delta_eff ~ N^alpha!
    w_bN, w_bD = [], []
    if has_s5_gain and all((f"sq_rx_iso_s{s}_X_err", "gain_uf", b) in kmc9 for s in range(6) for b in (2.08, 2.15, 2.22)):
        bs_fit = [b for b in (2.08, 2.15, 2.22, 2.28) if all((f"sq_rx_iso_s{s}_X_err", "gain_uf", b) in kmc9 for s in range(6))]
        for s in range(6):
            w_bN.append(kmc9[(f"sq_rx_iso_s{s}_X_err", "gain_uf", bs_fit[0])][0])
            w_bD.append(ols_slope(bs_fit, [kmc9[(f"sq_rx_iso_s{s}_X_err", "gain_uf", b)][1] for b in bs_fit]))
    else:
        bs_fit = [1.75, 2.00, 2.25]
        for s in range(5):
            w_bN.append(kmc9[(f"sq_rx_iso_s{s}_X_err", "uf", 1.75)][0])
            w_bD.append(ols_slope(bs_fit, [kmc9[(f"sq_rx_iso_s{s}_X_err", "uf", b)][1] for b in bs_fit]))

    c0_w, alpha_w, r2_w = power_law_fit(w_bN[1:], w_bD[1:])

    # 3-Level Full (X,Z) Quantum Barrier (`19 -> 1,942 -> 189,988` over matched cold window [6.12..6.50] or [6.25..6.50])
    q3_bwin = [b for b in (6.12, 6.18, 6.25, 6.35, 6.50) if all((f"sq_L2r1_s{s}_Z_err", "uf", b) in kmc9 for s in (1, 3, 5))]
    if len(q3_bwin) < 2:
        q3_bwin = [6.25, 6.50]
    q3_bN = [19, 1942, 189988]
    q3_bD = [
        ols_slope([b for b in (6.00, 6.25, 6.50) if (f"sq_L2r1_s{s}_Z_err", "uf", b) in kmc9],
                  [kmc9[(f"sq_L2r1_s{s}_Z_err", "uf", b)][1] for b in (6.00, 6.25, 6.50) if (f"sq_L2r1_s{s}_Z_err", "uf", b) in kmc9])
        if s < 5 else ols_slope(q3_bwin, [kmc9[("sq_L2r1_s5_Z_err", "uf", b)][1] for b in q3_bwin])
        for s in (1, 3, 5)
    ]
    _, alpha_q, r2_q = power_law_fit(q3_bN, q3_bD)
    print(f"Power-law fits: 2x2 X-welded alpha={alpha_w:.3f} (R^2={r2_w:.4f}); Full (X,Z) 3-level alpha={alpha_q:.3f} (R^2={r2_q:.4f}), slopes={[round(x,2) for x in q3_bD]}")

    s_bar = [
        {"xs": w_bN,     "ys": w_bD,     "color": "#be123c", "dash": "",    "label": f"3D Welded 2×2 (exponent α={alpha_w:.2f}, R²={r2_w:.2f})"},
        {"xs": q3_bN,    "ys": q3_bD,    "color": "#ea580c", "dash": "7,3", "label": f"3D Welded Full (X,Z) (α={alpha_q:.2f}, R²={r2_q:.2f})"},
        {"xs": c_haah_N, "ys": c_haah_D, "color": "#7c3aed", "dash": "",    "label": "Haah 3D Cubic (log N, bends down)"},
        {"xs": c_bb_N,   "ys": c_bb_D,   "color": "#2563eb", "dash": "5,4", "label": "Scaled 2D Bivariate Bicycle (flat ≈ 5.4)"},
        {"xs": c_tor_N,  "ys": c_tor_D,  "color": "#4b5563", "dash": "3,3", "label": "2D Toric Code (flat ≤ 2.3)"},
    ]

    top_n_lbl = "322k" if has_s5_gain else "190k"
    render_double(
        "fig_v11_1_classes",
        dict(xlim=(15.0, 420000.0), ylim=(0.02, 2.2e7),
             title=f"(a) Relative Lifetime Growth Up to {top_n_lbl} Qubits",
             xlabel="Physical Qubit Count N (Log Scale)", ylabel="Lifetime Relative to Smallest Code",
             xticks=[(20,"20"), (200,"200"), (2000,"2k"), (20000,"20k"), (200000,"200k")],
             yticks=[(0.05,"0.05×"), (1.0,"1×"), (100.0,"10²×"), (10000.0,"10⁴×")],
             series=s_norm, xlog=True, ylog=True, leg_pos="tl",
             hlines=[(1.0, "#6b7280", "1× = No Gain From Growing N", "br")]),
        dict(xlim=(1.6, 420000.0), ylim=(0.0, max(17.5, max(w_bD) * 1.32)),
             title="(b) Additive vs. Multiplicative Energy Barriers",
             xlabel="Physical Qubit Count N (Log Scale)", ylabel="Effective Energy Barrier Δeff",
             xticks=[(2,"2"), (30,"30"), (500,"500"), (10000,"10k"), (200000,"200k")],
             yticks=[(0,"0"), (4,"4"), (8,"8"), (12,"12"), (16,"16")] if max(w_bD) <= 15 else [(0,"0"), (5,"5"), (10,"10"), (15,"15"), (20,"20")],
             series=s_bar, xlog=True, ylog=False, leg_pos="tl"),
    )

    # --- 3. `fig_v11_3_validation.png`: External Simulator Validation ---
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
        "fig_v11_3_validation",
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

    # --- 4. `fig_v11_4_scaling.png`: 2x2 Flagship Scaling & Live 322k-Qubit Thermal RG Figure ---
    # Panel (a): Multi-scale Uniform 2x2 (`s=1..5` up to 322,159q if available, else `s=1..4` + s=5 points)
    ms_series = []
    if has_s5_gain:
        for s, col, dsh, nlbl in [
            (5, "#7f1d1d", "",    "Scale 5 (322,159q)"),
            (4, "#be123c", "",    "Scale 4 (27,107q)"),
            (3, "#ea580c", "6,3", "Scale 3 (2,311q)"),
            (2, "#2563eb", "5,4", "Scale 2 (203q)"),
            (1, "#6b7280", "3,3", "Scale 1 (19q)"),
        ]:
            bs_cand = [1.60, 1.75, 1.90, 2.00, 2.08, 2.15, 2.22, 2.28]
            xs = [b for b in bs_cand if (f"sq_rx_iso_s{s}_X_err", "gain_uf", b) in kmc9]
            ys = [kmc9[(f"sq_rx_iso_s{s}_X_err", "gain_uf", b)][1] for b in xs]
            ms_series.append({"xs": xs, "ys": ys, "color": col, "dash": dsh, "label": nlbl})
        b_arr = xs[0] if not ms_series else ms_series[0]["xs"][-1]
        y_lo_a = kmc9[("sq_rx_iso_s1_X_err", "gain_uf", b_arr)][1]
        y_hi_a = kmc9[("sq_rx_iso_s5_X_err", "gain_uf", b_arr)][1]
        p_a = dict(xlim=(1.55, 2.46), ylim=(2.5, max(9e4, y_hi_a * 3.0)),
                   title="(a) 5 Scales to 322k: Bounded Threshold",
                   xlabel="Inverse Bath Temperature β = 1 / T", ylabel="Mean Sector Lifetime",
                   xticks=[(1.6,"1.6"), (1.8,"1.8"), (2.0,"2.0"), (2.2,"2.2"), (2.4,"2.4")],
                   yticks=[(1e1,"10¹"), (1e2,"10²"), (1e3,"10³"), (1e4,"10⁴"), (1e5,"10⁵")],
                   series=ms_series, xlog=False, ylog=True, leg_pos="tl",
                   vlines=[(1.92, "#374151", "Threshold β ≈ 1.9", "rb")],
                   varrows=[(b_arr, y_lo_a, y_hi_a, f"{y_hi_a/y_lo_a:,.0f}×", "#7f1d1d", "r", math.sqrt(y_lo_a * y_hi_a))])
    else:
        for s, col, dsh, nlbl in [
            (4, "#be123c", "",    "Scale 4 (27,107q: slope=13.2)"),
            (3, "#ea580c", "",    "Scale 3 (2,311q: slope=8.8)"),
            (2, "#2563eb", "6,3", "Scale 2 (203q: slope=4.6)"),
            (1, "#6b7280", "3,3", "Scale 1 (19q: slope=2.0)"),
        ]:
            bs_s = [b for b in (1.25, 1.50, 1.75, 1.90, 2.00, 2.10, 2.15, 2.25) if (f"sq_rx_iso_s{s}_X_err", "uf", b) in kmc9]
            ms_series.append({"xs": bs_s, "ys": [kmc9[(f"sq_rx_iso_s{s}_X_err", "uf", b)][1] for b in bs_s], "color": col, "dash": dsh, "label": nlbl})
        p_a = dict(xlim=(1.20, 2.48), ylim=(2.5, 9e4),
                   title="(a) 4 Scales: Bounded Threshold",
                   xlabel="Inverse Bath Temperature β = 1 / T", ylabel="Mean Sector Lifetime",
                   xticks=[(1.25,"1.25"), (1.5,"1.5"), (1.75,"1.75"), (2.0,"2.0"), (2.25,"2.25")],
                   yticks=[(1e1,"10¹"), (1e2,"10²"), (1e3,"10³"), (1e4,"10⁴")],
                   series=ms_series, xlog=False, ylog=True, leg_pos="tl",
                   vlines=[(1.86, "#374151", "Threshold β ≈ 1.86", "rb")],
                   varrows=[(2.25, kmc9[("sq_rx_iso_s1_X_err", "uf", 2.25)][1], kmc9[("sq_rx_iso_s4_X_err", "uf", 2.25)][1], "396×", "#be123c", "r", 7.0e2)])

    # Panel (b): Live 6-Level Syndrome-Only Thermal RG Flow on 322,159 Qubits (`sq_rx_iso_s5`, zero decoder!)
    # x = 0..4 corresponds to s = 5 -> 4 -> 3 -> 2 -> 1
    rg_series = []
    for b, col, dsh, lbl in [
        (1.35, "#dc2626", "5,4", "Hot Bath (β=1.35: expands upward)"),
        (1.86, "#ea580c", "5,4", "Above Tc (β=1.86: expands upward)"),
        (2.00, "#374151", "",    "Critical Separatrix (β=2.00: flat)"),
        (2.15, "#2563eb", "",    "Below Tc (β=2.15: contracts → 0)"),
        (2.35, "#059669", "",    "Cold Bath (β=2.35: contracts → 0)"),
    ]:
        rhos = rg9[("sq_rx_iso_s5", b)]  # index = scale s
        xs, ys = [], []
        for step_idx, s in enumerate([5, 4, 3, 2, 1]):
            if rhos[s] > 0:
                xs.append(float(step_idx))
                ys.append(rhos[s])
        rg_series.append({"xs": xs, "ys": ys, "color": col, "dash": dsh, "label": lbl})

    p_b = dict(
        xlim=(-0.2, 4.25), ylim=(1.2e-4, 55.0),
        title="(b) Live 322k-Qubit Syndrome RG Flow",
        xlabel="Coarse-Graining Scale (Microscopic → Macroscopic)",
        ylabel="Coarse-Grained Defect Density ρs",
        xticks=[(0,"s=5 (322k)"), (1,"s=4 (27k)"), (2,"s=3 (2.3k)"), (3,"s=2 (203)"), (4,"s=1 (19)")],
        yticks=[(3e-4,"3×10⁻⁴"), (3e-3,"3×10⁻³"), (3e-2,"3×10⁻²"), (3e-1,"0.3")],
        series=rg_series, xlog=False, ylog=True, leg_pos="tl",
        notes=[(4.12, 3.0e-4, "→ 0 defects at s≤1", "#047857", "end")],
    )

    # Panel (c): 3-Level Full (X,Z) Quantum Ladder (`19 -> 1,942 -> 189,988` qubits, pure `uf`, zero bails!)
    q_s1_bs = [4.50, 5.00, 5.50, 5.75, 6.00, 6.25, 6.50]
    q_s3_bs = [4.50, 5.00, 5.50, 5.75, 6.00, 6.25, 6.50]
    q_s5_bs = [b for b in (6.12, 6.18, 6.25, 6.35, 6.50) if ("sq_L2r1_s5_Z_err", "uf", b) in kmc9]
    p_c = dict(
        xlim=(4.40, 6.82), ylim=(6.0, 4.0e6),
        title="(c) 3-Level Full Quantum Ladder (to 190k)",
        xlabel="Inverse Bath Temperature β = 1 / T", ylabel="Quantum Memory Lifetime",
        xticks=[(4.5,"4.5"), (5.0,"5.0"), (5.5,"5.5"), (6.0,"6.0"), (6.5,"6.5")],
        yticks=[(1e1,"10¹"), (1e2,"10²"), (1e3,"10³"), (1e4,"10⁴"), (1e5,"10⁵"), (1e6,"10⁶")],
        series=[
            {"xs": q_s5_bs, "ys": [kmc9[("sq_L2r1_s5_Z_err", "uf", b)][1] for b in q_s5_bs],
             "color": "#be123c", "dash": "",    "label": f"Step 5 (189,988q: slope={q3_bD[2]:.1f})"},
            {"xs": q_s3_bs, "ys": [kmc9[("sq_L2r1_s3_Z_err", "uf", b)][1] for b in q_s3_bs],
             "color": "#059669", "dash": "",    "label": f"Step 3 (1,942q: slope={q3_bD[1]:.1f})"},
            {"xs": q_s1_bs, "ys": [kmc9[("sq_L2r1_s1_Z_err", "uf", b)][1] for b in q_s1_bs],
             "color": "#6b7280", "dash": "4,3", "label": f"Step 1 (19q: slope={q3_bD[0]:.1f})"},
        ],
        xlog=False, ylog=True, leg_pos="tl",
        varrows=[(6.50, kmc9[("sq_L2r1_s1_Z_err", "uf", 6.50)][1], kmc9[("sq_L2r1_s5_Z_err", "uf", 6.50)][1], "10,250×", "#be123c", "r", 3.5e3)],
    )

    # Panel (d): Provably Contractive Uniform 2x2 (`323 -> 47,387`q: 71x) & Boundaryless Torus (`508 -> 50,296`q: 7.4x)
    iso_bs = [5.00, 5.50, 6.00, 6.25, 6.50, 6.65]
    iso_s2 = [kmc9[("sq_iso_L2r1_s2_X_err", "uf", b)][1] for b in iso_bs]
    iso_s4 = [kmc9[("sq_iso_L2r1_s4_X_err", "uf", b)][1] for b in iso_bs]
    rot_bs = [5.50, 6.00, 6.25, 6.50, 6.65]
    rot_s2 = [v7_best("rottor_L2r1_s2_X_err", "uniform", "uf", b) if b <= 6.50 else kmc9[("rottor_L2r1_s2_X_err", "uf", b)][1] for b in rot_bs]
    rot_s4 = [v7_best("rottor_L2r1_s4_X_err", "uniform", "uf", b) if b <= 6.50 else kmc9[("rottor_L2r1_s4_X_err", "uf", b)][1] for b in rot_bs]

    p_d = dict(
        xlim=(4.92, 6.86), ylim=(3.0e2, 4.5e7),
        title="(d) Uniform 2×2 & Torus Full Cycles",
        xlabel="Inverse Bath Temperature β = 1 / T", ylabel="Quantum Memory Lifetime",
        xticks=[(5.0,"5.0"), (5.5,"5.5"), (6.0,"6.0"), (6.5,"6.5")],
        yticks=[(1e3,"10³"), (1e4,"10⁴"), (1e5,"10⁵"), (1e6,"10⁶"), (1e7,"10⁷")],
        series=[
            {"xs": rot_bs, "ys": rot_s4, "color": "#be123c", "dash": "",    "label": "Torus Cycle 2 (50,296q: 7.4×)"},
            {"xs": rot_bs, "ys": rot_s2, "color": "#f97316", "dash": "5,4", "label": "Torus Cycle 1 (508q)"},
            {"xs": iso_bs, "ys": iso_s4, "color": "#059669", "dash": "",    "label": "Uniform 2×2 Cycle 2 (47,387q: 71×)"},
            {"xs": iso_bs, "ys": iso_s2, "color": "#2563eb", "dash": "5,4", "label": "Uniform 2×2 Cycle 1 (323q)"},
        ],
        xlog=False, ylog=True, leg_pos="tl",
        varrows=[(6.65, iso_s2[-1], iso_s4[-1], "71×", "#047857", "r", 1.8e5)],
    )

    render_quad("fig_v11_4_scaling", p_a, p_b, p_c, p_d)

    # --- 5. `fig_v11_5_tradeoffs.png`: Cubic-Grid Stretching & Sector Balance ---
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
        {"xs": bal_bs, "ys": [v7_best("rottor_L2r1_s2_X_err", "uniform", "uf", b) if b <= 6.50 else kmc9[("rottor_L2r1_s2_X_err", "uf", b)][1] for b in bal_bs],
         "color": "#dc2626", "dash": "",    "label": "508-Qubit Torus: X-Errors (bottleneck)"},
    ]
    render_double(
        "fig_v11_5_tradeoffs",
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

    # --- 6. `fig_v11_6_cooling.png`: Storage Self-Healing vs Hot-State Cooling ---
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
        "fig_v11_6_cooling",
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
