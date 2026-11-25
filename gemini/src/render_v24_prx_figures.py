#!/usr/bin/env python3
"""
render_v24_prx_figures.py
Generates clean, untitled PRX-style vector PDFs + PNGs for the two new sections:
  - `figures/fig_v24_5_rebalance.{pdf,png}`:
      (a) Ultra-compact 3-step `(X, Z, X)` mirror-image bottleneck test (putting 2 bridges in
          the bottleneck middle Z step `X2 R_{3Z} X2` at N=4,566 vs outer X step `R_{3X} Z2 X2` at N=4,556,
          plus `R_3 X_2` at N=10,580).
      (b) Complete 4-step Welding-Order Quartet (`XZXZ`, `ZXZX`, `ZXXZ`, `XZZX`) across
          beta in {5.5, 6.0, 6.5, 7.0} on planar (K=1) and periodic torus (K=2) seeds.
      (c) Expanded (N, t_0.80) resource frontier at beta = 6.5 combining 3-step, L=2 palindromic,
          and 3D-continuable coarse-L=3 palindromic hierarchies (`C3 C2^{ZX}`, `RZ3 C2^{ZX}`).
  - `figures/fig_v24_6_decoders.{pdf,png}`:
      (a) Unrelaxed i.i.d. single-shot bit-flip recovery P_rec(p) on `C2 C2` comparing `UF`,
          `Hier-UF`, `Cool-UF`, and `BP-OSD`.
      (b) Exact 1-step `C2` thermal calibration (beta = 7.0: `UF` and `BP-OSD` overlaid on the
          exact analytical 8,192-state maximum-likelihood curve P^opt(t)).
      (c) Microscopic failure-mode anatomy near t_0.80^UF across temperature and architecture
          (zero-syndrome / T=0-quenchable wrong-basin states vs excited states unrescued by both
          decoders vs states rescued by BP-OSD).
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "cm",
    "font.size": 9.0,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.2,
    "legend.fontsize": 6.8,
    "axes.linewidth": 0.85,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

C_TEAL  = "#0f766e"
C_AMBER = "#b45309"
C_BLUE  = "#1d4ed8"
C_GREEN = "#059669"
C_RED   = "#dc2626"
C_PURP  = "#6d28d9"
C_SLATE = "#475569"
C_DARK  = "#0f172a"


def style_ax(ax, panel_lbl, x_off=-0.20, y_off=1.03):
    ax.tick_params(direction="in", which="both", top=True, right=True, pad=3.2)
    ax.grid(True, which="major", color="#e2e8f0", lw=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.text(
        x_off, y_off, f"({panel_lbl})",
        transform=ax.transAxes, fontsize=10.6, fontweight="bold", va="bottom", ha="left", color=C_DARK,
    )


def plot_arm_point(ax, x, dstat, color, marker="o", ms=5.0, label=None, hollow=False, zorder=4):
    y = dstat["t_80"]
    lo, hi = dstat["t_80_ci95"]
    mfc = "white" if hollow else color
    if dstat["t_80_crossed"]:
        ylo = max(0.0, y - lo)
        yhi = max(0.0, hi - y)
        ax.errorbar(
            [x], [y], yerr=[[ylo], [yhi]],
            fmt=marker, color=color, mfc=mfc, mec=color, mew=1.2, ms=ms,
            lw=1.05, capsize=2.1, label=label, zorder=zorder,
        )
    else:
        ylo = max(0.0, y - lo)
        ax.errorbar(
            [x], [y], yerr=[[ylo], [0.0]],
            fmt=marker, color=color, mfc="white", mec=color, mew=1.3, ms=ms,
            lw=1.05, capsize=2.1, label=label, zorder=zorder,
        )
        ax.annotate(
            "", xy=(x, y * 1.85), xytext=(x, y),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.15, mutation_scale=7.8),
            zorder=zorder + 1,
        )


def save_fig(fig, stem):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf_p = FIG_DIR / f"{stem}.pdf"
    png_p = FIG_DIR / f"{stem}.png"
    fig.savefig(pdf_p, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(png_p, dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved {pdf_p}")


def render_fig5_rebalance(v19):
    arms = v19["arms"]
    fig = plt.figure(figsize=(7.15, 2.75))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.05, 1.03, 1.07], wspace=0.34)

    # Panel (a): 3-step (X, Z, X) mirror-image bottleneck test at beta=6.5 & 7.0
    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a, "a", x_off=-0.22)
    r2_items = [
        ("v19_sq_3s_Z2X2Z2",     "$Z_2 X_2 Z_2$"),
        ("v19_sq_3s_X2Z2X2",     "$X_2 Z_2 X_2$"),
        ("v19_sq_3s_X2Z3r1X2",   "$X_2 C_{3Z} X_2$"),
        ("v19_sq_3s_X3r2Z2X2",   "$R_{3X} Z_2 X_2$"),
        ("v19_sq_3s_X2Z3r2X2",   "$X_2 R_{3Z} X_2$"),
        ("v19_sq_3s_X3r2Z3r2X2", "$R_3 X_2$"),
    ]
    for i, (tag, _) in enumerate(r2_items):
        k65 = f"{tag}__b6.50"
        k70 = f"{tag}__b7.00"
        if k65 in arms:
            plot_arm_point(ax_a, i - 0.14, arms[k65]["decoders"]["uf"], C_TEAL, "o", 4.7, label=("$\\beta = 6.5$" if i == 0 else None))
        if k70 in arms:
            plot_arm_point(ax_a, i + 0.14, arms[k70]["decoders"]["uf"], C_BLUE, "s", 4.5, label=("$\\beta = 7.0$" if i == 0 else None))
    ax_a.set_yscale("log")
    ax_a.set_xlim(-0.55, len(r2_items) - 0.45)
    ax_a.set_xticks(range(len(r2_items)))
    ax_a.set_xticklabels([lbl for _, lbl in r2_items], fontsize=7.4, rotation=32, ha="right")
    ax_a.set_ylabel("Recovery time $t_{0.80}$")
    ax_a.set_ylim(1.8e3, 1.4e6)
    ax_a.legend(loc="upper left", frameon=True, framealpha=0.92, edgecolor="#cbd5e1", fontsize=6.8)

    # Panel (b): Complete 4-step L=2 Welding-Order Quartet across beta in {5.5, 6.0, 6.5, 7.0}
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b, "b", x_off=-0.22)
    q_specs = [
        ("v19_sq_C2_XZXZ",  [5.50, 6.00, 6.50, 7.00], C_SLATE, "o", "-",  "Planar $XZXZ$"),
        ("v19_sq_C2_ZXZX",  [5.50, 6.00, 6.50, 7.00], C_AMBER, "D", "--", "Planar $ZXZX$"),
        ("v19_sq_C2_ZXXZ",  [5.50, 6.00, 6.50, 7.00], C_BLUE,  "^", "-.", "Planar $ZXXZ$"),
        ("v19_sq_C2_XZZX",  [5.50, 6.00, 6.50, 7.00], C_GREEN, "s", "-",  "Planar $XZZX$"),
        ("v19_tor_C2_XZZX", [5.50, 6.00, 6.50],       C_PURP,  "s", ":",  "Torus $XZZX$"),
    ]
    for tag, b_list, col, mk, ls, lbl in q_specs:
        xs, ys = [], []
        for b in b_list:
            k = f"{tag}__b{b:.2f}"
            if k in arms:
                d = arms[k]["decoders"]["uf"]
                xs.append(b)
                ys.append(d["t_80"])
                plot_arm_point(ax_b, b, d, col, mk, 4.4, label=None)
        if xs:
            ax_b.plot(xs, ys, color=col, ls=ls, lw=1.35, zorder=3)
            ax_b.plot([], [], color=col, marker=mk, ls=ls, lw=1.35, ms=4.4, label=lbl)
    ax_b.set_yscale("log")
    ax_b.set_xlim(5.38, 7.12)
    ax_b.set_xticks([5.50, 6.00, 6.50, 7.00])
    ax_b.set_xlabel("Inverse temperature $\\beta$")
    ax_b.set_ylabel("Recovery time $t_{0.80}$")
    ax_b.set_ylim(4.0e2, 3.2e8)
    ax_b.legend(loc="lower right", frameon=True, framealpha=0.93, edgecolor="#cbd5e1", fontsize=6.0, borderpad=0.28, handlelength=1.55)

    # Panel (c): Expanded (N, t_0.80) Frontier at beta = 6.5
    ax_c = fig.add_subplot(gs[0, 2])
    style_ax(ax_c, "c", x_off=-0.22)
    frontier_pts = [
        ("v19_sq_3s_X2Z2X2__b6.50",     C_SLATE, "D", "$X_2 Z_2 X_2$",       (1.32, 0.80), "left"),
        ("v19_sq_3s_X2Z3r2X2__b6.50",   C_PURP,  "D", "$X_2 R_{3Z} X_2$",    (1.32, 0.70), "left"),
        ("v19_sq_3s_X3r2Z3r2X2__b6.50", C_PURP,  "D", "$R_3 X_2$",           (0.74, 1.55), "right"),
        ("v19_sq_C2_XZXZ__b6.50",       C_TEAL,  "o", "$C_2 C_2$",           (1.26, 0.72), "left"),
        ("v19_sq_C2_ZXZX__b6.50",       C_AMBER, "o", "$C_2^{ZX}C_2^{ZX}$",  (0.72, 1.05), "right"),
        ("v19_sq_C2_XZZX__b6.50",       C_GREEN, "s", "$C_2 C_2^{ZX}$",      (0.70, 1.08), "right"),
        ("v19_sq_C3_C2__b6.50",         C_AMBER, "^", "$C_3 C_2$",           (0.74, 0.62), "right"),
        ("v19_sq_RX3_C2__b6.50",        C_BLUE,  "^", "$R_{X3} C_2$",        (0.72, 2.35), "center"),
        ("v19_sq_C3_C2ZX__b6.50",       C_GREEN, "P", "$C_3 C_2^{ZX}$",      (0.66, 0.90), "right"),
        ("v19_sq_RZ3_C2ZX__b6.50",      C_BLUE,  "P", "$R_{Z3} C_2^{ZX}$",   (0.80, 2.15), "center"),
    ]
    rebal_front = []
    std_front = []
    for k, col, mk, txt, mult, ha_align in frontier_pts:
        if k not in arms:
            continue
        a = arms[k]
        d = a["decoders"]["uf"]
        plot_arm_point(ax_c, a["N"], d, col, mk, 4.9)
        ax_c.text(a["N"] * mult[0], d["t_80"] * mult[1], txt, fontsize=6.6, color=C_DARK, ha=ha_align, va="center")
        rebal_front.append((a["N"], d["t_80"]))
        if k in ("v19_sq_C2_XZXZ__b6.50", "v19_sq_C3_C2__b6.50", "v19_sq_RX3_C2__b6.50"):
            std_front.append((a["N"], d["t_80"]))

    rebal_front.sort(key=lambda t: t[0])
    px, py, best = [], [], -1.0
    for nx, ny in rebal_front:
        if ny > best:
            px.append(nx)
            py.append(ny)
            best = ny
    ax_c.plot(px, py, ls="--", lw=0.95, color=C_GREEN, alpha=0.7, zorder=2)

    std_front.sort(key=lambda t: t[0])
    spx, spy, sbest = [], [], -1.0
    for nx, ny in std_front:
        if ny > sbest:
            spx.append(nx)
            spy.append(ny)
            sbest = ny
    ax_c.plot(spx, spy, ls=":", lw=0.95, color=C_SLATE, alpha=0.75, zorder=2)

    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.set_xlim(1.0e3, 2.5e5)
    ax_c.set_ylim(3.5e3, 2.8e8)
    ax_c.set_xlabel("Physical qubits $N$")
    ax_c.set_ylabel("Whole-memory $t_{0.80}$")

    save_fig(fig, "fig_v24_5_rebalance")


def render_fig6_decoders(v19, loc_quot):
    arms = v19["arms"]
    iid = v19["iid_arms"]
    fig = plt.figure(figsize=(7.15, 2.75))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.02, 1.04, 1.06], wspace=0.34)

    # Panel (a): Unrelaxed i.i.d. bit-flip recovery on C2_C2 (UF vs Hier-UF vs Cool-UF vs BP-OSD)
    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a, "a", x_off=-0.22)
    if "v19_sq_C2_XZXZ" in iid:
        ia = iid["v19_sq_C2_XZXZ"]
        pg = np.array(ia["p_grid"]) * 100.0
        dec_styles = [
            ("uf",    C_SLATE, "-",  "o", "UF ($2.18\\%$)"),
            ("hier",  C_AMBER, "--", "s", "Hier-UF ($2.37\\%$)"),
            ("cool",  C_BLUE,  "-.", "^", "Cool-UF ($4.51\\%$)"),
            ("bposd", C_GREEN, "-",  "D", "BP-OSD ($5.67\\%$)"),
        ]
        for d, col, ls, mk, lbl in dec_styles:
            ax_a.plot(pg, ia["decoders"][d]["curve_Q"], color=col, ls=ls, marker=mk, ms=3.4, markevery=3, lw=1.45, label=lbl)
    ax_a.axhline(0.80, color="#94a3b8", lw=0.8, ls=":")
    ax_a.set_xlim(0.4, 8.2)
    ax_a.set_ylim(0.08, 1.04)
    ax_a.set_xlabel("Unrelaxed error rate $p$ (%)")
    ax_a.set_ylabel("Single-shot $P_{\\mathrm{rec}}(p)$")
    ax_a.legend(loc="lower left", frameon=True, framealpha=0.93, edgecolor="#cbd5e1", fontsize=6.1, borderpad=0.28)

    # Panel (b): Exact 1-step C2 MLD calibration (beta=7.0)
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b, "b", x_off=-0.22)
    ec = loc_quot["exact_cell_curves_b70"]
    ax_b.plot(ec["t_X"], ec["PX_C2_full"], color=C_DARK, lw=1.85, ls="-", label="Optimal $P_X^{\\mathrm{opt}}$")
    ax_b.plot(ec["t_Q"], ec["PQ_C2_full"], color=C_SLATE, lw=1.65, ls="-", label="Optimal $P_Q^{\\mathrm{opt}}$")
    if "v19_sq_C2_1step__b7.00" in arms:
        c1 = arms["v19_sq_C2_1step__b7.00"]
        tg = c1["t_grid"]
        ax_b.plot(tg, c1["decoders"]["uf"]["p_X"], color=C_AMBER, lw=1.45, ls="--", label="UF $p_X$")
        ax_b.plot(tg, c1["decoders"]["bposd"]["p_X"], color=C_GREEN, lw=1.5, ls=":", label="BP-OSD $p_X$")
        ax_b.plot(tg, c1["decoders"]["bposd"]["p_Q"], color=C_RED, lw=1.4, ls="--", label="UF / BP-OSD $P_{\\mathrm{rec}}$")
    ax_b.axhline(0.80, color="#94a3b8", lw=0.8, ls=":")
    ax_b.set_xscale("log")
    ax_b.set_xlim(8.0, 4.0e5)
    ax_b.set_ylim(0.45, 1.03)
    ax_b.set_xlabel("Time $t$ ($1$-step $C_2$, $\\beta = 7.0$)")
    ax_b.set_ylabel("Recovery probability")
    ax_b.legend(loc="lower left", frameon=True, framealpha=0.93, edgecolor="#cbd5e1", fontsize=6.0, borderpad=0.28)

    # Panel (c): Microscopic failure anatomy near t_0.80^UF across temperature & architecture
    ax_c = fig.add_subplot(gs[0, 2])
    style_ax(ax_c, "c", x_off=-0.22)
    anat_keys = [
        ("v19_sq_C2_XZZX__b5.50",     "$C_2 C_2^{ZX}$ ($5.5$)"),
        ("v19_sq_C2_XZZX__b6.00",     "$C_2 C_2^{ZX}$ ($6.0$)"),
        ("v19_sq_C2_XZZX__b6.50",     "$C_2 C_2^{ZX}$ ($6.5$)"),
        ("v19_sq_C2_XZZX__b7.00",     "$C_2 C_2^{ZX}$ ($7.0$)"),
        ("v19_tor_C2_XZXZ__b7.00",    "Tor. $C_2 C_2$ ($7.0$)"),
        ("v19_sq_3s_X2Z3r2X2__b7.00", "$X_2 R_{3Z} X_2$ ($7.0$)"),
    ]
    present = [(k, lbl) for k, lbl in anat_keys if k in arms]
    xs = np.arange(len(present))
    f_gq = np.array([arms[k]["failure_anatomy_near_t80"]["frac_settled_wrong_basin_GQ"] for k, _ in present])
    f_ex = np.array([arms[k]["failure_anatomy_near_t80"]["frac_excited_unrescued"] for k, _ in present])
    f_bp = np.array([arms[k]["failure_anatomy_near_t80"]["frac_rescued_by_bposd"] for k, _ in present])

    ax_c.bar(xs, f_gq, width=0.58, color=C_TEAL, label="Wrong basin ($s{=}0$/quench)")
    ax_c.bar(xs, f_ex, width=0.58, bottom=f_gq, color=C_AMBER, label="Excited (both fail)")
    ax_c.bar(xs, f_bp, width=0.58, bottom=f_gq + f_ex, color=C_GREEN, label="Rescued by BP-OSD")
    ax_c.set_xticks(xs)
    ax_c.set_xticklabels([lbl for _, lbl in present], fontsize=6.8, rotation=32, ha="right")
    ax_c.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_c.set_ylim(0.0, 1.24)
    ax_c.set_ylabel("Share of UF failures near $t_{0.80}$")
    ax_c.legend(loc="upper left", ncol=1, frameon=True, framealpha=0.93, edgecolor="#cbd5e1", fontsize=5.7, borderpad=0.24, labelspacing=0.22)

    save_fig(fig, "fig_v24_6_decoders")


def main():
    with open(ROOT / "parsed_v19_rebalance_and_decoders.json") as f:
        v19 = json.load(f)
    with open(ROOT / "local_quotient_and_pruning_data.json") as f:
        loc_quot = json.load(f)
    render_fig5_rebalance(v19)
    render_fig6_decoders(v19, loc_quot)


if __name__ == "__main__":
    main()
