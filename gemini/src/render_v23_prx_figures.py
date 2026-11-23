#!/usr/bin/env python3
"""
render_v23_prx_figures.py
Renders four clean Physical Review X figures for
"Selective Simplification of Finite Welded Quantum Memories":
  - Serif + Computer Modern math typography matching REVTeX 4.2
  - Full 4-sided spines with inward ticks; bold (a), (b), (c), (d) tags ONLY (never subplot titles)
  - Minimal in-plot annotations; clean natural wording without internal jargon
  - Whole-trajectory 95% bootstrap intervals throughout

Figures:
  1. `fig_v23_1_frontier`: Module sheet geometry (C2, C3, R3), finite-size resource frontiers
     t_0.80 vs N at beta = 6.5 and beta = 7.0 (planar & torus), and standalone vs fine-constituent reversal.
  2. `fig_v23_2_pruning`: Full vs three-arm bridge cutback diagram, exact 1-step CSS quotient recovery
     curves P_X^opt(t) & P_Q^opt(t) at beta = 7.0, and 2-step C2 pruning & protected-subsystem curves at beta = 6.5.
  3. `fig_v23_3_mechanism`: Single-cohort V17 recovery vs no-prior-failure decomposition on planar C3C2,
     resource-matched L=4 offset (1+3) vs centered (2+2) standalone & coarse-C2 composed comparison,
     and controlled fine-C3 flat-sheet zero-energy slowdown vs seam controls.
  4. `fig_v23_4_transfer`: Open 1x1 vs collared 3x3 local interface survival across fine modules,
     prospective held-out Decision 1 & Decision 2 outcomes vs frozen interface predictions & baselines at beta = 6.5,
     and cold beta = 7.0 coarse-bridge simplification test.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "cm",
    "font.size": 9.2,
    "axes.labelsize": 9.8,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.85,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

C_C2 = "#0f766e"    # deep teal
C_C3 = "#b45309"    # warm ochre
C_R3 = "#1d4ed8"    # royal blue
C_RX3 = "#059669"   # emerald
C_RZ3 = "#dc2626"   # crimson
C_SLATE = "#475569" # neutral slate
C_PURP = "#6d28d9"  # deep violet


def style_ax(ax, tag, x_off=-0.16):
    ax.tick_params(direction="in", which="both", top=True, right=True, pad=3.5)
    ax.grid(True, which="major", color="#e2e8f0", linewidth=0.5, alpha=0.55)
    ax.set_axisbelow(True)
    ax.text(
        x_off, 1.03, f"({tag})",
        transform=ax.transAxes,
        fontsize=10.6,
        fontweight="bold",
        va="bottom",
        ha="left",
        color="#0f172a",
    )


def save_fig(fig, stem):
    pdf_p = FIG_DIR / f"{stem}.pdf"
    png_p = FIG_DIR / f"{stem}.png"
    fig.savefig(pdf_p, bbox_inches="tight", pad_inches=0.07)
    fig.savefig(png_p, dpi=300, bbox_inches="tight", pad_inches=0.07)
    plt.close(fig)
    print(f"Saved {pdf_p}")


def plot_ci_point(ax, x, y, ci, col, marker="o", ms=5.4, hollow=False, label=None, zorder=4):
    ylo = max(y - ci[0], 0.0)
    yhi = max(ci[1] - y, 0.0)
    ax.errorbar(
        [x], [y], yerr=[[ylo], [yhi]],
        fmt=marker, color=col, ms=ms,
        mfc=("white" if hollow else col), mew=1.25,
        ecolor=col, elinewidth=1.05, capsize=2.2,
        label=label, zorder=zorder,
    )


def draw_module_diagram(ax):
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 3.5)
    ax.axis("off")
    style_ax(ax, "a", x_off=-0.02)

    configs = [
        (0.3, "$C_2$ ($2{\\times}2$, 1 bridge)", 2, [1], C_C2),
        (3.8, "$C_3$ ($3{\\times}3$, 1 bridge)", 3, [1], C_C3),
        (7.3, "$R_3$ ($3{\\times}3$, 2 bridges)", 3, [1, 2], C_R3),
    ]
    w = 2.65
    y0 = 0.58
    for x0, lbl, L, ribs, col in configs:
        if L == 3 and ribs == [1]:
            # Subtle shading on 2x2 flat patch
            rect = mpatches.Rectangle((x0 + w / 3, y0 + w / 3), 2 * w / 3, 2 * w / 3, facecolor="#fef3c7", edgecolor="none", zorder=1)
            ax.add_patch(rect)
        # Outer square
        outer = mpatches.Rectangle((x0, y0), w, w, facecolor="none", edgecolor="#1e293b", lw=1.25, zorder=3)
        ax.add_patch(outer)
        for k in range(1, L):
            pos = k * w / L
            is_rib = k in ribs
            lw = 2.1 if is_rib else 0.75
            ls = "-" if is_rib else "--"
            c_line = col if is_rib else "#94a3b8"
            z = 4 if is_rib else 2
            ax.plot([x0, x0 + w], [y0 + pos, y0 + pos], ls=ls, lw=lw, color=c_line, zorder=z)
            ax.plot([x0 + pos, x0 + pos], [y0, y0 + w], ls=ls, lw=lw, color=c_line, zorder=z)
        ax.text(x0 + w / 2, 0.14, lbl, ha="center", va="center", fontsize=8.4, color="#0f172a")


def render_fig1_frontier(suiteA):
    fig = plt.figure(figsize=(7.15, 5.45))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[0.82, 1.18], hspace=0.34, wspace=0.32)

    ax_a = fig.add_subplot(gs[0, :2])
    draw_module_diagram(ax_a)

    # Panel (b): Standalone vs Fine-constituent reversal (planar beta=6.5)
    ax_b = fig.add_subplot(gs[0, 2])
    style_ax(ax_b, "b", x_off=-0.24)
    rd = suiteA["readouts"]
    groups = [
        ("1-cycle", [("v15_sq_C2", C_C2), ("v15_sq_C3", C_C3), ("v15_sq_R3", C_R3)]),
        ("Coarse $C_2$", [("v15_sq_pkg_CC", C_C2), ("v16_sq_C2C3", C_C3), ("v15_sq_pkg_CR", C_R3)]),
        ("Coarse $C_3$", [("v16_sq_C3C2", C_C2), ("v15_sq_L3_CC", C_C3), ("v15_sq_L3_CR", C_R3)]),
        ("Coarse $R_3$", [("v15_sq_pkg_RC", C_C2), ("v15_sq_L3_RC", C_C3), ("v15_sq_L3_RR", C_R3)]),
    ]
    offs = [-0.20, 0.0, 0.20]
    lbls = ["$C_2$", "$C_3$", "$R_3$"]
    for gi, (_, items) in enumerate(groups):
        for j, (tag, col) in enumerate(items):
            r = rd[tag]["6.50"]
            plot_ci_point(
                ax_b, gi + offs[j], r["t_80"], r["t_80_ci95"], col,
                marker=("D" if gi == 0 else "o"), ms=4.8,
                label=(lbls[j] if gi == 1 else None),
            )
    ax_b.set_yscale("log")
    ax_b.set_xticks([0, 1, 2, 3])
    ax_b.set_xticklabels(["1-cyc.", "$C_2 B$", "$C_3 B$", "$R_3 B$"], fontsize=8.0)
    ax_b.set_ylabel("Recovery time $t_{0.80}$")
    ax_b.set_ylim(8e2, 1.4e6)
    ax_b.legend(loc="upper left", ncol=3, columnspacing=0.6, handletextpad=0.2, frameon=True, framealpha=0.92, edgecolor="#cbd5e1", fontsize=6.5)

    # Bottom row: Resource frontiers (Planar b=6.5, Planar b=7.0, Torus b=6.5)
    panels = [
        (gs[1, 0], "c", "planar", "6.50", [
            ("v15_sq_C2", C_C2, "D", "$C_2$", (1.35, 0.85)),
            ("v15_sq_C3", C_C3, "D", None, None),
            ("v15_sq_R3", C_R3, "D", "$R_3$", (1.35, 1.05)),
            ("v15_sq_pkg_CC", C_C2, "o", "$C_2 C_2$", (0.23, 1.05)),
            ("v16_sq_C2C3", C_C3, "o", None, None),
            ("v15_sq_pkg_CR", C_R3, "o", "$C_2 R_3$", (1.30, 0.95)),
            ("v16_sq_C3C2", C_C2, "o", "$C_3 C_2$", (0.24, 0.62)),
            ("v15_sq_L3_CC", C_C3, "o", None, None),
            ("v15_sq_L3_CR", C_R3, "o", None, None),
            ("v15_sq_pkg_RC", C_C2, "o", "$R_3 C_2$", (1.30, 1.05)),
            ("v15_sq_L3_RC", C_C3, "o", None, None),
            ("v15_sq_L3_RR", C_R3, "o", "$R_3 R_3$", (0.20, 0.72)),
        ], (1.5e2, 1.5e6), (1.0e3, 5.5e5), "Planar, $\\beta = 6.5$"),
        (gs[1, 1], "d", "planar", "7.00", [
            ("v15_sq_pkg_CC", C_C2, "o", "$C_2 C_2$", (0.32, 1.15)),
            ("v16_sq_C3C2", C_C2, "o", "$C_3 C_2$", (0.33, 0.62)),
            ("v15_sq_pkg_RC", C_C2, "o", "$R_3 C_2$", (1.22, 1.05)),
            ("v15_sq_pkg_CR", C_R3, "o", "$C_2 R_3$", (1.22, 0.92)),
            ("v15_sq_L3_CC", C_C3, "o", "$C_3 C_3$", (0.28, 1.25)),
            ("v15_sq_L3_RC", C_C3, "o", None, None),
        ], (9.0e3, 1.1e6), (6.0e3, 1.8e7), "Planar, $\\beta = 7.0$"),
        (gs[1, 2], "e", "torus", "6.50", [
            ("v15_tor_C2", C_C2, "D", "$C_2$", (1.35, 0.85)),
            ("v15_tor_C3", C_C3, "D", None, None),
            ("v15_tor_R3", C_R3, "D", "$R_3$", (1.35, 1.05)),
            ("v15_tor_pkg_CC", C_C2, "o", "$C_2 C_2$", (0.22, 1.08)),
            ("v16_tor_C2C3", C_C3, "o", None, None),
            ("v15_tor_pkg_CR", C_R3, "o", None, None),
            ("v16_tor_C3C2", C_C2, "o", "$C_3 C_2$", (0.22, 0.72)),
            ("v15_tor_L3_CC", C_C3, "o", None, None),
            ("v15_tor_L3_CR", C_R3, "o", None, None),
            ("v15_tor_pkg_RC", C_C2, "o", "$R_3 C_2$", (1.30, 1.05)),
            ("v15_tor_L3_RC", C_C3, "o", None, None),
            ("v15_tor_L3_RR", C_R3, "o", "$R_3 R_3$", (0.20, 0.68)),
        ], (1.5e2, 1.5e6), (1.0e3, 5.5e5), "Torus, $\\beta = 6.5$"),
    ]

    for spec, tag, _, bstr, items, xlim, ylim, corner_lbl in panels:
        ax = fig.add_subplot(spec)
        style_ax(ax, tag, x_off=-0.22)
        pts_front = []
        for ctag, col, mk, txt, mult in items:
            if bstr not in rd.get(ctag, {}):
                continue
            r = rd[ctag][bstr]
            n_val = r["N"]
            y_val = r["t_80"]
            pts_front.append((n_val, y_val))
            plot_ci_point(ax, n_val, y_val, r["t_80_ci95"], col, marker=mk, ms=4.9)
            if txt and mult:
                ax.text(n_val * mult[0], y_val * mult[1], txt, fontsize=7.5, color="#0f172a")
        # Pareto point-estimate guide
        pts_front.sort(key=lambda t: t[0])
        par_x, par_y = [], []
        best = -1.0
        for nx, ny in pts_front:
            if ny > best:
                par_x.append(nx)
                par_y.append(ny)
                best = ny
        ax.plot(par_x, par_y, ls="--", lw=0.95, color="#64748b", alpha=0.8, zorder=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Physical qubits $N$")
        if tag == "c":
            ax.set_ylabel("Recovery time $t_{0.80}$")
        ax.text(0.05, 0.92, corner_lbl, transform=ax.transAxes, fontsize=8.0, color="#334155", va="top")

    save_fig(fig, "fig_v23_1_frontier")


def render_fig2_pruning(loc_data, v18):
    fig = plt.figure(figsize=(7.15, 2.55))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[0.88, 1.06, 1.06], wspace=0.33)

    # Panel (a): Full cross vs three-arm bridge cutback schematic
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_xlim(0, 4.4)
    ax_a.set_ylim(0, 3.6)
    ax_a.axis("off")
    style_ax(ax_a, "a", x_off=-0.04)

    for idx, (x0, lbl, pruned) in enumerate([(0.22, "Full cross", False), (2.38, "Three-arm", True)]):
        w = 1.75
        y0 = 0.85
        ax_a.add_patch(mpatches.Rectangle((x0, y0), w, w, facecolor="none", edgecolor="#1e293b", lw=1.2))
        # Horizontal bridge (both arms)
        ax_a.plot([x0, x0 + w], [y0 + w / 2, y0 + w / 2], color=C_C2, lw=2.2)
        # Vertical bridge (top arm deleted if pruned)
        ax_a.plot([x0 + w / 2, x0 + w / 2], [y0, y0 + w / 2], color=C_C2, lw=2.2)
        if not pruned:
            ax_a.plot([x0 + w / 2, x0 + w / 2], [y0 + w / 2, y0 + w], color=C_C2, lw=2.2)
        else:
            ax_a.plot([x0 + w / 2, x0 + w / 2], [y0 + w / 2, y0 + w], color="#cbd5e1", lw=1.3, ls=":")
        ax_a.text(x0 + w / 2, 0.42, lbl, ha="center", va="center", fontsize=8.3, color="#0f172a")

    # Panel (b): Exact 1-step CSS quotient recovery curves at beta=7.0
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b, "b", x_off=-0.22)
    ec = loc_data["exact_cell_curves_b70"]
    ax_b.plot(ec["t_X"], ec["PX_C2_full"], color=C_C2, lw=1.65, ls="-", label="Full $X$ ($N{=}19$)")
    ax_b.plot(ec["t_X"], ec["PX_C2_3arm"], color=C_C3, lw=1.55, ls="--", label="3-arm $X$ ($N{=}18$)")
    ax_b.plot(ec["t_Q"], ec["PQ_C2_full"], color=C_SLATE, lw=1.65, ls="-", label="Full $X{/}Z$ quantum")
    ax_b.plot(ec["t_Q"], ec["PQ_C2_3arm"], color=C_RZ3, lw=1.35, ls=":", label="3-arm $X{/}Z$ quantum")
    ax_b.axhline(0.80, color="#94a3b8", lw=0.8, ls="--")
    ax_b.set_xscale("log")
    ax_b.set_xlim(5.0, 3.0e5)
    ax_b.set_ylim(0.45, 1.02)
    ax_b.set_xlabel("Time $t$ ($\\beta = 7.0$)")
    ax_b.set_ylabel(r"Optimal recovery $P^{\mathrm{opt}}(t)$")
    ax_b.legend(loc="lower left", frameon=True, framealpha=0.92, edgecolor="#cbd5e1", fontsize=6.8)

    # Panel (c): Two-step C2 module pruning & protected subsystem recovery at beta=6.5
    ax_c = fig.add_subplot(gs[0, 2])
    style_ax(ax_c, "c", x_off=-0.22)
    arms = v18["arms"]
    series = [
        ("v18_sq_C2_full__b6.50",       C_C2,  "-",  "Full ($254, K{=}1$)"),
        ("v18_sq_C2_pruneX__b6.50",     C_C3,  "--", "$X$-pruned ($248, K{=}1$)"),
        ("v18_sq_C2_pruneZ_sub__b6.50", C_R3,  "-.", r"$Z$-pruned ($228, K_{\mathrm{tot}}{=}9$)"),
        ("v18_sq_C2_pruneB_sub__b6.50", C_RZ3, ":",  r"Both-pruned ($222, K_{\mathrm{tot}}{=}9$)"),
    ]
    for key, col, ls, lbl in series:
        a = arms[key]
        ax_c.plot(a["t_grid"], a["P_id"], color=col, ls=ls, lw=1.6, label=lbl)
    ax_c.axhline(0.80, color="#94a3b8", lw=0.8, ls="--")
    ax_c.set_xscale("log")
    ax_c.set_xlim(10.0, 5.0e4)
    ax_c.set_ylim(-0.02, 1.03)
    ax_c.set_xlabel("Time $t$ ($\\beta = 6.5$)")
    ax_c.set_ylabel(r"Target recovery $P_{\mathrm{rec}}(t)$")
    ax_c.legend(loc="lower left", frameon=True, framealpha=0.92, edgecolor="#cbd5e1", fontsize=5.8, borderpad=0.30, handlelength=1.45)

    save_fig(fig, "fig_v23_2_pruning")


def render_fig3_mechanism(suiteA, v17_old, v18):
    fig = plt.figure(figsize=(7.15, 2.65))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.05, 1.02, 0.98], wspace=0.33)

    # Panel (a): Single-cohort V17 healing decomposition on planar C3C2 (beta=6.5 & 7.0)
    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a, "a", x_off=-0.22)
    sc65 = suiteA["readouts"]["v16_sq_C3C2"]["6.50"]["v17_single_cohort"]
    tg65 = suiteA["readouts"]["v16_sq_C3C2"]["6.50"]["t_grid"]
    sc70 = suiteA["readouts"]["v16_sq_C3C2"]["7.00"]["v17_single_cohort"]
    tg70 = suiteA["readouts"]["v16_sq_C3C2"]["7.00"]["t_grid"]

    ax_a.plot(tg65, sc65["P_id"], color=C_C2, lw=1.65, ls="-", label=r"$P_{\mathrm{rec}}$ ($\beta{=}6.5$)")
    ax_a.plot(tg65, sc65["P_nocross"], color=C_C2, lw=1.2, ls="--", label="No prior fail ($6.5$)")
    ax_a.fill_between(tg65, sc65["P_nocross"], sc65["P_id"], color=C_C2, alpha=0.14)

    ax_a.plot(tg70, sc70["P_id"], color=C_R3, lw=1.65, ls="-", label=r"$P_{\mathrm{rec}}$ ($\beta{=}7.0$)")
    ax_a.plot(tg70, sc70["P_nocross"], color=C_R3, lw=1.2, ls="--", label="No prior fail ($7.0$)")
    ax_a.fill_between(tg70, sc70["P_nocross"], sc70["P_id"], color=C_R3, alpha=0.12)

    ax_a.axhline(0.80, color="#94a3b8", lw=0.8, ls=":")
    ax_a.set_xscale("log")
    ax_a.set_xlim(2.5e3, 1.5e7)
    ax_a.set_ylim(-0.02, 1.04)
    ax_a.set_xlabel("Time $t$")
    ax_a.set_ylabel("Probability")
    ax_a.legend(loc="lower left", frameon=True, framealpha=0.92, edgecolor="#cbd5e1", fontsize=6.3)

    # Panel (b): Resource-matched L=4 placement (standalone N=2,926 & composed C2+L4 N=261,476)
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b, "b", x_off=-0.22)
    arms = v18["arms"]
    pairs = [
        (0, "v18_sq_L4_off__b6.50",    "v18_sq_L4_cen__b6.50"),
        (1, "v18_sq_C2_L4off__b6.50",  "v18_sq_C2_L4cen__b6.50"),
    ]
    for xi, k_off, k_cen in pairs:
        a_off, a_cen = arms[k_off], arms[k_cen]
        plot_ci_point(ax_b, xi - 0.22, a_off["t_80"], a_off["t_80_ci95"], C_C3, "o", 5.2, label=("Offset $t_{0.80}$" if xi == 0 else None))
        plot_ci_point(ax_b, xi - 0.07, a_cen["t_80"], a_cen["t_80_ci95"], C_C2, "o", 5.2, label=("Centered $t_{0.80}$" if xi == 0 else None))
        plot_ci_point(ax_b, xi + 0.08, a_off["t_80_nocross"], a_off["t_80_nocross_ci95"], C_C3, "s", 4.8, hollow=True, label=("Offset no-fail" if xi == 0 else None))
        plot_ci_point(ax_b, xi + 0.23, a_cen["t_80_nocross"], a_cen["t_80_nocross_ci95"], C_C2, "s", 4.8, hollow=True, label=("Centered no-fail" if xi == 0 else None))
    ax_b.set_yscale("log")
    ax_b.set_xticks([0, 1])
    ax_b.set_xticklabels(["1-cyc. $L{=}4$\n($2{,}926$)", "Coarse $C_2{+}L{=}4$\n($261{,}476$)"], fontsize=7.9)
    ax_b.set_ylabel("Threshold time ($\\beta = 6.5$)")
    ax_b.set_ylim(1.2e2, 1.5e4)
    ax_b.legend(loc="upper right", frameon=True, framealpha=0.92, edgecolor="#cbd5e1", fontsize=6.6)

    # Panel (c): Kinetic slowdown supporting check G_{C3}(mu) vs flat/seam t_0.80(mu)
    ax_c = fig.add_subplot(gs[0, 2])
    style_ax(ax_c, "c", x_off=-0.22)
    rows = v17_old["causal_GC3"]["planar"]["fineflat_vs_fineseam"]
    mus = [r["mu"] for r in rows]
    ax_c.plot(mus, [r["t80_C3C3_flat"] for r in rows], "-o", color=C_C3, ms=4.5, lw=1.5, label="$C_3 C_3$ flat")
    ax_c.plot(mus, [r["t80_R3C3_flat"] for r in rows], "-s", color=C_R3, ms=4.5, lw=1.5, label="$R_3 C_3$ flat")
    ax_c.plot(mus, [r["t80_C3C3_seam"] for r in rows], "--o", color=C_C3, mfc="white", ms=4.3, lw=1.15, label="$C_3 C_3$ seam ctrl.")
    ax_c.plot(mus, [r["t80_R3C3_seam"] for r in rows], "--s", color=C_R3, mfc="white", ms=4.3, lw=1.15, label="$R_3 C_3$ seam ctrl.")
    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.invert_xaxis()
    ax_c.set_xlabel("Rate factor $\\mu$")
    ax_c.set_ylabel("Recovery time $t_{0.80}$")
    ax_c.set_ylim(1.2e3, 3.8e4)
    ax_c.legend(loc="upper left", frameon=True, framealpha=0.92, edgecolor="#cbd5e1", fontsize=6.6)

    save_fig(fig, "fig_v23_3_mechanism")


def render_fig4_transfer(frozen, v18):
    fig = plt.figure(figsize=(7.15, 2.65))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.02, 1.06, 0.96], wspace=0.33)

    # Panel (a): Open 1x1 vs Collared 3x3 local interface survival across fine modules (beta=6.5)
    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a, "a", x_off=-0.22)
    mods = ["C2", "C3", "RX3", "RZ3", "R3"]
    xlabels = ["$C_2$", "$C_3$", "$R_{X3}$", "$R_{Z3}$", "$R_3$"]
    s1 = frozen["standalone_1x1_b65"]
    s3 = frozen["collared_3x3_b65"]
    for i, m in enumerate(mods):
        plot_ci_point(ax_a, i - 0.14, s1[m]["t_80"], s1[m]["t_80_ci95"], C_SLATE, "D", 4.6, hollow=True, label=("Open $1{\\times}1$" if i == 0 else None))
        plot_ci_point(ax_a, i + 0.14, s3[m]["t_80"], s3[m]["t_80_ci95"], C_RX3 if m == "RX3" else C_C2, "o", 5.0, label=("Collared $3{\\times}3$" if i == 0 else None))
    ax_a.set_yscale("log")
    ax_a.set_xticks(range(len(mods)))
    ax_a.set_xticklabels(xlabels)
    ax_a.set_ylabel("Local patch $t_{0.80}$ ($\\beta = 6.5$)")
    ax_a.set_ylim(1.2e3, 7.5e5)
    ax_a.legend(loc="upper left", frameon=True, framealpha=0.92, edgecolor="#cbd5e1", fontsize=6.8)

    # Panel (b): Prospective Held-Out Simplifications at beta=6.5 (Observed vs Frozen Ratio Predictions)
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b, "b", x_off=-0.22)
    arms = v18["arms"]
    pts_b65 = [
        ("v18_sq_C2pX_C2__b6.50", C_PURP, "s", "$C_{2,\\mathrm{pX}} C_2$", (1.18, 0.72)),
        ("v18_sq_C2_C2__b6.50",   C_C2,   "o", "$C_2 C_2$",               (1.22, 1.12)),
        ("v18_sq_C2_C3__b6.50",   C_C3,   "o", "$C_2 C_3$",               (0.42, 1.25)),
        ("v18_sq_C2_RX3__b6.50",  C_RX3,  "s", "$C_2 R_{X3}$",            (0.36, 1.12)),
        ("v18_sq_C2_RZ3__b6.50",  C_RZ3,  "s", "$C_2 R_{Z3}$",            (0.38, 0.92)),
        ("v18_sq_C2_R3__b6.50",   C_R3,   "o", "$C_2 R_3$",               (0.45, 0.68)),
    ]
    for k, col, mk, txt, mult in pts_b65:
        a = arms[k]
        plot_ci_point(ax_b, a["N"], a["t_80"], a["t_80_ci95"], col, mk, 5.2)
        ax_b.text(a["N"] * mult[0], a["t_80"] * mult[1], txt, fontsize=7.4, color="#0f172a")
    ax_b.axhline(20000.0, color=C_RX3, lw=0.8, ls=":", alpha=0.8)
    ax_b.axhline(120000.0, color=C_PURP, lw=0.8, ls=":", alpha=0.8)
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlim(1.5e4, 2.3e5)
    ax_b.set_ylim(1.1e3, 4.8e5)
    ax_b.set_xlabel("Physical qubits $N$ ($\\beta = 6.5$)")
    ax_b.set_ylabel("Whole-memory $t_{0.80}$")

    # Panel (c): Cold beta=7.0 coarse-bridge simplification test (C3C2, RX3C2, RZ3C2, R3C2)
    ax_c = fig.add_subplot(gs[0, 2])
    style_ax(ax_c, "c", x_off=-0.22)
    pts_b70 = [
        ("v18_sq_C3_C2__b7.00",  C_C3,  "o", "$C_3 C_2$"),
        ("v18_sq_RX3_C2__b7.00", C_RX3, "s", "$R_{X3} C_2$"),
        ("v18_sq_RZ3_C2__b7.00", C_RZ3, "s", "$R_{Z3} C_2$"),
        ("v18_sq_R3_C2__b7.00",  C_R3,  "o", "$R_3 C_2$"),
    ]
    for idx, (k, col, mk, lbl) in enumerate(pts_b70):
        a = arms[k]
        if a["t_80_crossed"]:
            plot_ci_point(ax_c, idx, a["t_80"], a["t_80_ci95"], col, mk, 5.3)
        else:
            # Explicit lower bound marker + upward arrow
            ylo = max(a["t_80"] - a["t_80_ci95"][0], 0.0)
            ax_c.errorbar([idx], [a["t_80"]], yerr=[[ylo], [0.0]], fmt=mk, color=col, mfc="white", mew=1.3, ms=5.3, capsize=2.2)
            ax_c.annotate("", xy=(idx, a["t_80"] * 1.95), xytext=(idx, a["t_80"]), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.3, mutation_scale=8.5))
    ax_c.set_yscale("log")
    ax_c.set_xticks([0, 1, 2, 3])
    ax_c.set_xticklabels(["$C_3 C_2$\n($107\\mathrm{k}$)", "$R_{X3} C_2$\n($116\\mathrm{k}$)", "$R_{Z3} C_2$\n($124\\mathrm{k}$)", "$R_3 C_2$\n($134\\mathrm{k}$)"], fontsize=7.5)
    ax_c.set_ylabel("Whole-memory $t_{0.80}$ ($\\beta = 7.0$)")
    ax_c.set_ylim(3.5e5, 6.5e7)

    save_fig(fig, "fig_v23_4_transfer")


def main():
    with open(ROOT / "reanalyzed_v16_v17_suiteA.json") as f:
        suiteA = json.load(f)
    with open(ROOT / "parsed_v17_data.json") as f:
        v17_old = json.load(f)
    with open(ROOT / "local_quotient_and_pruning_data.json") as f:
        loc_data = json.load(f)
    with open(ROOT / "frozen_interface_predictions.json") as f:
        frozen = json.load(f)
    with open(ROOT / "parsed_v18_rework_data.json") as f:
        v18 = json.load(f)

    render_fig1_frontier(suiteA)
    render_fig2_pruning(loc_data, v18)
    render_fig3_mechanism(suiteA, v17_old, v18)
    render_fig4_transfer(frozen, v18)


if __name__ == "__main__":
    main()
