#!/usr/bin/env python3
"""
render_v26_3d_local_figures.py
Generates all 7 clean, untitled PRX-style figures (`figures/fig_v26_1..7_*.{pdf,png}`) for the
strictly 3D-local manuscript (`L >= 3` at every step, `dim_eff < 3`, unit check coupling `J = 1`):
  - Fig. 1 (`fig_v26_1_frontier`): 3D Euclidean packing cutoff (`L = 2` dim > 3 forbidden vs `L >= 3`
    3D-local), constructive R^3 density scaling, and pure-L=3 `{C3, R3}^2` baseline matrix.
  - Fig. 2 (`fig_v26_2_pruning`): 3D-local `L = 3` bridge-arm cutback schematic, exact 1-step `C3` vs
    `R3` quotient curves, and 2-step `C3` (`N = 1,036 -> 963`) pruning & subsystem collapse.
  - Fig. 3 (`fig_v26_3_mechanism`): Pure-L=3 post-crossing return bands, standalone `L = 4` matched
    bridge placement (`N = 2,926`), and `C3 C3` / `R3 C3` flat-patch zero-energy rate scaling.
  - Fig. 4 (`fig_v26_4_transfer`): Open 1x1 vs collared 3x3 local interface probes on `{C3, RX3, RZ3, R3}`
    and pure-L=3 whole-memory selective bridge cuts (`C3 RX3`, `C3 RZ3`, `C3 R3`, `RX3 RX3`, `R3 R3`)
    at beta = 6.5 and beta = 7.0.
  - Fig. 5 (`fig_v26_5_welding_order`): Novel 3D-local sequences at pure `L = 3`: 3-step `(X, Z, X)`
    bottleneck reversal (`16.8k..22.2k` qubits), 4-step `R3` welding-order quartet (`XZXZ`, `ZXZX`,
    `ZXXZ`, `XZZX` on the identical 298,896-face 3D complex), and the 3D-local `(N, t_0.80)` frontier.
  - Fig. 6 (`fig_v26_6_decoders`): Paired 4-decoder single-shot i.i.d. thresholds on pure-L=3
    `C3 R3^ZX`, exact analytical MLD calibration, and wrong-basin failure anatomy across temperature.
  - Fig. 7 (`fig_v26_7_benchmarks`): Direct `J = 1` cross-family comparison across 2D Toric, 1-cycle
    3D Welded, pure-L=3 2-cycle `XZXZ` and `XZZX` welded hierarchies, and Z^3-local Haah's Cubic Code.
"""

import json
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
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
    "legend.fontsize": 6.2,
    "axes.linewidth": 0.85,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

C_SLATE = "#475569"
C_AMBER = "#b45309"
C_RED   = "#dc2626"
C_BLUE  = "#1d4ed8"
C_GREEN = "#059669"
C_TEAL  = "#0f766e"
C_PURP  = "#6d28d9"
C_DARK  = "#0f172a"


def style_ax(ax, lbl, x_off=-0.21, y_off=1.03):
    ax.tick_params(direction="in", which="both", top=True, right=True, pad=3.2)
    ax.grid(True, which="major", color="#e2e8f0", lw=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.text(
        x_off, y_off, f"({lbl})",
        transform=ax.transAxes, fontsize=10.6, fontweight="bold", va="bottom", ha="left", color=C_DARK,
    )


def save_fig(fig, stem):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf_p = FIG_DIR / f"{stem}.pdf"
    png_p = FIG_DIR / f"{stem}.png"
    fig.savefig(pdf_p, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(png_p, dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved {pdf_p}")


def plot_dstat(ax, x, dstat, color, marker="o", ms=4.6, hollow=False, label=None, zorder=4):
    y = dstat["t_80"]
    lo, hi = dstat["t_80_ci95"]
    mfc = "white" if hollow else color
    if dstat.get("t_80_crossed", True):
        ylo = max(0.0, y - lo)
        yhi = max(0.0, hi - y)
        ax.errorbar(
            [x], [y], yerr=[[ylo], [yhi]],
            fmt=marker, color=color, mfc=mfc, mec=color, mew=1.2, ms=ms,
            lw=1.05, capsize=2.0, label=label, zorder=zorder,
        )
    else:
        ylo = max(0.0, y - lo)
        ax.errorbar(
            [x], [y], yerr=[[ylo], [0.0]],
            fmt=marker, color=color, mfc="white", mec=color, mew=1.3, ms=ms,
            lw=1.05, capsize=2.0, label=label, zorder=zorder,
        )
        ax.annotate(
            "", xy=(x, y * 1.85), xytext=(x, y),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.15, mutation_scale=7.5),
            zorder=zorder + 1,
        )


def render_fig1(geom, suiteA, v21):
    fig = plt.figure(figsize=(7.15, 2.75))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.02, 1.03, 1.08], wspace=0.34)

    # Panel (a): 3D Euclidean packing dimension vs subdivision L
    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a, "a", x_off=-0.22)
    Ls = np.array([2, 3, 4, 5, 6, 8])
    # Exact spectral radius of thin-bridge step matrix [[2L^2, 2L], [2rL, r]]
    def step_dim(L, r):
        tr = 2 * L * L + r
        det = 2 * L * L * r - 4 * r * L * L
        lam = 0.5 * (tr + math.sqrt(tr * tr - 4 * det))
        return math.log(lam) / math.log(L)

    d_r1 = [step_dim(int(L), 1) for L in Ls]
    d_r2 = [step_dim(int(L), 2) for L in Ls if L >= 3]
    ax_a.axhspan(3.0, 3.8, color="#fee2e2", alpha=0.65, zorder=1)
    ax_a.axhline(3.0, color=C_RED, ls="--", lw=1.1, zorder=2)
    ax_a.text(5.2, 3.07, "3D limit ($d_{\\mathrm{pack}} = 3$)", fontsize=6.3, color=C_RED, ha="center")
    ax_a.plot(Ls, d_r1, color=C_TEAL, marker="o", ms=4.6, lw=1.4, label="1 bridge ($r{=}1$)", zorder=4)
    ax_a.plot(Ls[1:], d_r2, color=C_BLUE, marker="s", ms=4.5, lw=1.4, ls="--", label="2 bridges ($r{=}2$)", zorder=4)
    ax_a.plot([2], [d_r1[0]], marker="o", ms=6.2, mfc="none", mec=C_RED, mew=1.5, zorder=5)
    ax_a.text(2.22, d_r1[0] + 0.04, "$C_2$ ($3.30 > 3$)", fontsize=6.0, color=C_RED)
    ax_a.text(3.18, d_r2[0] + 0.05, f"$R_3$ (${d_r2[0]:.2f}$)", fontsize=6.0, color=C_BLUE)
    ax_a.text(2.00, 2.52, f"$C_3$ (${d_r1[1]:.2f}$)", fontsize=6.0, color=C_TEAL, ha="left")
    ax_a.set_xlim(1.65, 8.35)
    ax_a.set_ylim(2.15, 3.72)
    ax_a.set_xticks([2, 3, 4, 5, 6, 8])
    ax_a.set_xlabel("Subdivision factor $L$")
    ax_a.set_ylabel("Packing dimension $\\log_L \\rho$")
    ax_a.legend(loc="lower right", frameon=True, framealpha=0.93, edgecolor="#cbd5e1", fontsize=6.0)

    # Panel (b): Constructive R^3 box qubit density N / L_blk^3 across hierarchy steps s = 2, 3, 4
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b, "b", x_off=-0.22)
    emb = geom["embeddings_R3"]
    s_vals = [2, 3, 4]
    l2_dens = [
        emb["L=2 1-cyc C2"]["N"] / (emb["L=2 1-cyc C2"]["L_blk"] ** 3),
        emb["L=2 3-step X2_Z2_X2"]["N"] / (emb["L=2 3-step X2_Z2_X2"]["L_blk"] ** 3),
        emb["L=2 2-cyc C2_C2 (NON-3D)"]["N"] / (emb["L=2 2-cyc C2_C2 (NON-3D)"]["L_blk"] ** 3),
    ]
    l3_c_dens = [
        emb["L=3 1-cyc C3"]["N"] / (emb["L=3 1-cyc C3"]["L_blk"] ** 3),
        emb["L=3 3-step X3r1_Z3r2_X3r1"]["N"] / (emb["L=3 3-step X3r1_Z3r2_X3r1"]["L_blk"] ** 3),
        emb["L=3 2-cyc C3_R3^ZX"]["N"] / (emb["L=3 2-cyc C3_R3^ZX"]["L_blk"] ** 3),
    ]
    l3_r_dens = [
        emb["L=3 1-cyc R3"]["N"] / (emb["L=3 1-cyc R3"]["L_blk"] ** 3),
        emb["L=3 3-step R3_X3r2"]["N"] / (emb["L=3 3-step R3_X3r2"]["L_blk"] ** 3),
        emb["L=3 2-cyc R3_R3"]["N"] / (emb["L=3 2-cyc R3_R3"]["L_blk"] ** 3),
    ]
    ax_b.plot(s_vals, l2_dens, color=C_RED, marker="^", ms=4.8, lw=1.45, ls="-", label="$L{=}2$ (diverges)")
    ax_b.plot(s_vals, l3_r_dens, color=C_BLUE, marker="s", ms=4.6, lw=1.4, ls="--", label="$L{=}3$ ($R_3$ family)")
    ax_b.plot(s_vals, l3_c_dens, color=C_GREEN, marker="o", ms=4.6, lw=1.4, ls="-", label="$L{=}3$ ($C_3 R_3^{ZX}$)")
    ax_b.set_xticks([2, 3, 4])
    ax_b.set_xlim(1.8, 4.2)
    ax_b.set_ylim(0.4, 7.2)
    ax_b.set_xlabel("Welding steps $s$")
    ax_b.set_ylabel("3D box density $N / L_{\\mathrm{blk}}^3$")
    ax_b.legend(loc="upper left", frameon=True, framealpha=0.93, edgecolor="#cbd5e1", fontsize=5.8)

    # Panel (c): Pure-L=3 baseline family (1-cycle C3, R3 and 2-cycle {C3, R3}^2 in BDL XZXZ) at beta=6.5
    ax_c = fig.add_subplot(gs[0, 2])
    style_ax(ax_c, "c", x_off=-0.22)
    rdA = suiteA["readouts"]
    v21_arms = v21["arms"]
    items = [
        ("C_3",    v21_arms["v21_L3_C3_full__b6.50"]["decoders"]["uf"], C_AMBER, "D"),
        ("R_3",    v21_arms["v21_L3_R3__b6.50"]["decoders"]["uf"],      C_BLUE,  "D"),
        ("C_3 C_3",rdA["v15_sq_L3_CC"]["6.50"], C_AMBER, "o"),
        ("R_3 C_3",rdA["v15_sq_L3_RC"]["6.50"], C_AMBER, "s"),
        ("C_3 R_3",v21_arms["v21_4s_C3_R3__b6.50"]["decoders"]["uf"], C_BLUE, "o"),
        ("R_3 R_3",v21_arms["v21_4s_R3_R3__b6.50"]["decoders"]["uf"], C_BLUE, "s"),
    ]
    for i, (lbl, st, col, mk) in enumerate(items):
        d_use = {"t_80": st["t_80"], "t_80_ci95": st["t_80_ci95"], "t_80_crossed": True}
        plot_dstat(ax_c, i, d_use, col, mk, 4.9)
    ax_c.axvline(1.5, color="#cbd5e1", ls=":", lw=0.9)
    ax_c.set_yscale("log")
    ax_c.set_xticks(range(len(items)))
    ax_c.set_xticklabels([f"${x[0]}$" for x in items], fontsize=7.6, rotation=28, ha="right")
    ax_c.set_ylim(1.1e3, 1.2e5)
    ax_c.set_ylabel("Planar $t_{0.80}$ ($\\beta = 6.5$)")
    ax_c.text(0.5, 7.5e4, "1-cyc.", fontsize=6.5, color=C_SLATE, ha="center")
    ax_c.text(3.5, 7.5e4, "2-cyc. $XZXZ$ ($L{=}3$)", fontsize=6.5, color=C_SLATE, ha="center")

    save_fig(fig, "fig_v26_1_frontier")


def render_fig2(v21):
    arms = v21["arms"]
    fig = plt.figure(figsize=(7.15, 2.65))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[0.92, 1.06, 1.12], wspace=0.35)

    # Panel (a): 3x3 single-bridge cell C3 full vs 1-arm interior cutback schematic
    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a, "a", x_off=-0.16)
    ax_a.set_xlim(-0.3, 7.2)
    ax_a.set_ylim(-0.7, 3.6)
    ax_a.axis("off")

    for x0, title, drop_arm in [(0.0, "Full $C_3$ cross", False), (4.1, "1-arm cutback", True)]:
        rect = mpatches.Rectangle((x0, 0.2), 2.8, 2.8, fill=True, facecolor="#f8fafc", edgecolor=C_DARK, lw=1.1)
        ax_a.add_patch(rect)
        # 2x2 flat patch shading
        fp = mpatches.Rectangle((x0 + 0.933, 0.2 + 0.933), 1.866, 1.866, fill=True, facecolor="#fef3c7", edgecolor="none", alpha=0.8)
        ax_a.add_patch(fp)
        for k in (1, 2):
            pos = k * (2.8 / 3.0)
            ax_a.plot([x0 + pos, x0 + pos], [0.2, 3.0], color="#cbd5e1", lw=0.75, zorder=2)
            ax_a.plot([x0, x0 + 2.8], [0.2 + pos, 0.2 + pos], color="#cbd5e1", lw=0.75, zorder=2)
        rb = 2.8 / 3.0
        # Horizontal rib at y = 0.2 + rb
        ax_a.plot([x0, x0 + 2.8], [0.2 + rb, 0.2 + rb], color=C_TEAL, lw=2.3, zorder=4)
        # Vertical rib at x = x0 + rb
        if not drop_arm:
            ax_a.plot([x0 + rb, x0 + rb], [0.2, 3.0], color=C_TEAL, lw=2.3, zorder=4)
        else:
            ax_a.plot([x0 + rb, x0 + rb], [0.2 + rb, 3.0], color=C_TEAL, lw=2.3, zorder=4)
            ax_a.plot([x0 + rb, x0 + rb], [0.2, 0.2 + rb], color=C_RED, lw=1.6, ls=":", zorder=4)
        ax_a.text(x0 + 1.4, -0.30, title, fontsize=6.6, ha="center", color=C_DARK)

    # Panel (b): 2-step L=3 C3 pruning recovery curves P_rec(t) at beta = 6.5 (n = 512)
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b, "b", x_off=-0.22)
    p_curves = [
        ("v21_L3_C3_full__b6.50",      C_TEAL,  "-",  "Full $C_3$ ($1{,}036, K{=}1$)"),
        ("v21_L3_C3_pruneX__b6.50",    C_BLUE,  "--", "Prune $X$ ($1{,}028, K{=}1$)"),
        ("v21_L3_C3_pruneZ__b6.50",    C_AMBER, "-.", "Prune $Z$ ($971, K{=}12$)"),
        ("v21_L3_C3_pruneBoth__b6.50", C_RED,   ":",  "Prune both ($963, K{=}12$)"),
    ]
    for k, col, ls, lbl in p_curves:
        if k in arms:
            tg = arms[k]["t_grid"]
            pq = arms[k]["decoders"]["uf"]["p_Q"]
            ax_b.plot(tg, pq, color=col, ls=ls, lw=1.55, label=lbl)
    ax_b.axhline(0.80, color="#94a3b8", ls=":", lw=0.8)
    ax_b.set_xscale("log")
    ax_b.set_xlim(8.0, 6.0e4)
    ax_b.set_ylim(0.05, 1.04)
    ax_b.set_xlabel("Time $t$ ($\\beta = 6.5$)")
    ax_b.set_ylabel("Target recovery $P_{\\mathrm{rec}}(t)$")
    ax_b.legend(loc="lower left", frameon=True, framealpha=0.93, edgecolor="#cbd5e1", fontsize=5.6, borderpad=0.24)

    # Panel (c): Standalone 1-cycle L=3 module lifetimes (X,Z vs reversed Z,X orders) at beta = 6.5
    ax_c = fig.add_subplot(gs[0, 2])
    style_ax(ax_c, "c", x_off=-0.22)
    m_list = [
        ("v21_L3_C3_full__b6.50", "$C_3$",        C_SLATE, "o"),
        ("v21_L3_C3ZX__b6.50",    "$C_3^{ZX}$",   C_GREEN, "s"),
        ("v21_L3_RZ3__b6.50",     "$R_{Z3}$",     C_AMBER, "^"),
        ("v21_L3_RX3__b6.50",     "$R_{X3}$",     C_BLUE,  "^"),
        ("v21_L3_R3__b6.50",      "$R_3$",        C_TEAL,  "o"),
        ("v21_L3_RZ3ZX__b6.50",   "$R_{Z3}^{ZX}$",C_PURP,  "s"),
        ("v21_L3_R3ZX__b6.50",    "$R_3^{ZX}$",   C_GREEN, "D"),
    ]
    xs_lbl = []
    for idx, (k, lbl, col, mk) in enumerate(m_list):
        xs_lbl.append(lbl)
        if k in arms:
            plot_dstat(ax_c, idx, arms[k]["decoders"]["uf"], col, mk, 4.8)
    ax_c.set_yscale("log")
    ax_c.set_xticks(range(len(m_list)))
    ax_c.set_xticklabels(xs_lbl, fontsize=7.4, rotation=28, ha="right")
    ax_c.set_ylim(1.2e3, 4.0e5)
    ax_c.set_ylabel("1-cycle $t_{0.80}$ ($\\beta = 6.5$)")

    save_fig(fig, "fig_v26_2_pruning")


def render_fig3(v21, v18, v14_meta):
    arms21 = v21["arms"]
    fig = plt.figure(figsize=(7.15, 2.70))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.06, 0.98, 1.06], wspace=0.35)

    # Panel (a): Pure-L=3 post-crossing return bands on C3_RX3 at beta=6.5 and 7.0
    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a, "a", x_off=-0.22)
    for k, col, b_lbl in [
        ("v21_4s_C3_RX3__b6.50", C_TEAL, "$\\beta = 6.5$"),
        ("v21_4s_C3_RX3__b7.00", C_BLUE, "$\\beta = 7.0$"),
    ]:
        if k in arms21:
            tg = np.array(arms21[k]["t_grid"])
            pq = np.array(arms21[k]["decoders"]["uf"]["p_Q"])
            pnc = np.array(arms21[k]["decoders"]["uf"]["p_Q_nocross"])
            ax_a.plot(tg, pq, color=col, ls="-", lw=1.5, label=f"$P_{{\\mathrm{{rec}}}}$ ({b_lbl})")
            ax_a.plot(tg, pnc, color=col, ls="--", lw=1.25, label=f"No prior fail ({b_lbl})")
            ax_a.fill_between(tg, pnc, pq, color=col, alpha=0.14)
    ax_a.axhline(0.80, color="#94a3b8", ls=":", lw=0.8)
    ax_a.set_xscale("log")
    ax_a.set_xlim(5.0e2, 4.0e6)
    ax_a.set_ylim(0.05, 1.04)
    ax_a.set_xlabel("Time $t$ (planar $C_3 R_{X3}$)")
    ax_a.set_ylabel("Probability")
    ax_a.legend(loc="lower left", frameon=True, framealpha=0.93, edgecolor="#cbd5e1", fontsize=5.5, borderpad=0.22)

    # Panel (b): Standalone 3D-local L = 4 resource-matched bridge placement (offset 1+3 vs centered 2+2, N = 2,926)
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b, "b", x_off=-0.22)
    a_off = v18["arms"]["v18_sq_L4_off__b6.50"]
    a_cen = v18["arms"]["v18_sq_L4_cen__b6.50"]
    pts_b = [
        (0, a_off["t_80"], a_off["t_80_ci95"], C_AMBER, "o", False, "Offset $t_{0.80}$"),
        (1, a_cen["t_80"], a_cen["t_80_ci95"], C_TEAL,  "o", False, "Centered $t_{0.80}$"),
        (2, a_off["t_80_nocross"], a_off["t_80_nocross_ci95"], C_AMBER, "s", True, "Offset no-fail"),
        (3, a_cen["t_80_nocross"], a_cen["t_80_nocross_ci95"], C_TEAL,  "s", True, "Centered no-fail"),
    ]
    for x, y, ci, col, mk, hol, lbl in pts_b:
        plot_dstat(ax_b, x, {"t_80": y, "t_80_ci95": ci, "t_80_crossed": True}, col, mk, 5.2, hollow=hol, label=lbl)
    ax_b.set_yscale("log")
    ax_b.set_xlim(-0.6, 3.6)
    ax_b.set_xticks([0, 1, 2, 3])
    ax_b.set_xticklabels(["Off.\n$t_{0.80}$", "Cen.\n$t_{0.80}$", "Off.\nno-fail", "Cen.\nno-fail"], fontsize=7.2)
    ax_b.set_ylim(1.0e2, 6.0e3)
    ax_b.set_ylabel("Matched $L{=}4$ time ($\\beta = 6.5$)")

    # Panel (c): Pure-L=3 C3_C3 & R3_C3 flat-patch zero-energy rate intervention mu
    ax_c = fig.add_subplot(gs[0, 2])
    style_ax(ax_c, "c", x_off=-0.22)
    rows = v14_meta["causal_GC3"]["planar"]["fineflat_vs_fineseam"]
    mus = [r["mu"] for r in rows]
    ax_c.plot(mus, [r["t80_C3C3_flat"] for r in rows], "-o", color=C_AMBER, ms=4.5, lw=1.45, label="$C_3 C_3$ flat")
    ax_c.plot(mus, [r["t80_R3C3_flat"] for r in rows], "-s", color=C_BLUE,  ms=4.5, lw=1.45, label="$R_3 C_3$ flat")
    ax_c.plot(mus, [r["t80_C3C3_seam"] for r in rows], "--o", color=C_AMBER, mfc="white", ms=4.3, lw=1.15, label="$C_3 C_3$ seam ctrl.")
    ax_c.plot(mus, [r["t80_R3C3_seam"] for r in rows], "--s", color=C_BLUE,  mfc="white", ms=4.3, lw=1.15, label="$R_3 C_3$ seam ctrl.")
    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.invert_xaxis()
    ax_c.set_ylim(1.2e3, 3.8e4)
    ax_c.set_xlabel("Zero-energy rate factor $\\mu$")
    ax_c.set_ylabel("Planar $t_{0.80}$ ($\\beta = 6.5$)")
    ax_c.legend(loc="upper left", frameon=True, framealpha=0.93, edgecolor="#cbd5e1", fontsize=5.6, borderpad=0.22)

    save_fig(fig, "fig_v26_3_mechanism")


def render_fig4(frozen, geom, v21):
    arms = v21["arms"]
    fig = plt.figure(figsize=(7.15, 2.70))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[0.98, 1.12, 1.02], wspace=0.35)

    # Panel (a): Open 1x1 vs Collared 3x3 local interface probes on pure-L=3 modules {C3, RX3, RZ3, R3}
    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a, "a", x_off=-0.22)
    mods = [("C3", "$C_3$"), ("RX3", "$R_{X3}$"), ("RZ3", "$R_{Z3}$"), ("R3", "$R_3$")]
    for i, (m, lbl) in enumerate(mods):
        s1 = frozen["standalone_1x1_b65"][m]
        s3 = frozen["collared_3x3_b65"][m]
        plot_dstat(ax_a, i - 0.14, s1, C_SLATE, "D", 4.6, hollow=True, label=("Open $1{\\times}1$" if i == 0 else None))
        plot_dstat(ax_a, i + 0.14, s3, C_TEAL,  "o", 4.7, hollow=False, label=("Collared $3{\\times}3$" if i == 0 else None))
    ax_a.set_yscale("log")
    ax_a.set_xticks(range(len(mods)))
    ax_a.set_xticklabels([x[1] for x in mods], fontsize=8.2)
    ax_a.set_ylim(1.1e3, 2.2e5)
    ax_a.set_ylabel("Local patch $t_{0.80}$ ($\\beta = 6.5$)")
    ax_a.legend(loc="upper left", frameon=True, framealpha=0.93, edgecolor="#cbd5e1", fontsize=6.0)

    # Panel (b): Pure-L=3 whole-memory selective bridge cuts at beta = 6.5
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b, "b", x_off=-0.22)
    b_pts = [
        ("v21_4s_C3_RZ3__b6.50",  C_RED,   "v", "$C_3 R_{Z3}$",    (0.96, 0.95), "right"),
        ("v21_4s_C3_R3__b6.50",   C_BLUE,  "o", "$C_3 R_3$",       (1.00, 1.68), "center"),
        ("v21_4s_C3_RX3__b6.50",  C_GREEN, "s", "$C_3 R_{X3}$",    (1.00, 0.46), "center"),
        ("v21_4s_RX3_RX3__b6.50", C_TEAL,  "D", "$R_{X3} R_{X3}$", (1.04, 1.38), "left"),
        ("v21_4s_R3_R3__b6.50",   C_SLATE, "o", "$R_3 R_3$",       (0.97, 0.48), "center"),
    ]
    for k, col, mk, txt, mult, ha_al in b_pts:
        if k in arms:
            nv = arms[k]["N"]
            d = arms[k]["decoders"]["uf"]
            plot_dstat(ax_b, nv / 1e5, d, col, mk, 4.9)
            ax_b.text((nv / 1e5) * mult[0], d["t_80"] * mult[1], txt, fontsize=6.4, color=C_DARK, ha=ha_al, va="center")
    ax_b.set_yscale("log")
    ax_b.set_xlim(3.8, 7.6)
    ax_b.set_ylim(7.0e2, 2.5e5)
    ax_b.set_xlabel("Physical qubits $N$ ($\\times 10^5$, $\\beta = 6.5$)")
    ax_b.set_ylabel("Whole-memory $t_{0.80}$")

    # Panel (c): Cold beta = 7.0 pure-L=3 selective cuts
    ax_c = fig.add_subplot(gs[0, 2])
    style_ax(ax_c, "c", x_off=-0.22)
    c_items = [
        ("v21_4s_C3_R3__b7.00",   "$C_3 R_3$\n(591k)",      C_BLUE,  "o"),
        ("v21_4s_C3_RX3__b7.00",  "$C_3 R_{X3}$\n(431k)",   C_GREEN, "s"),
        ("v21_4s_RX3_RX3__b7.00", "$R_{X3} R_{X3}$\n(470k)",C_TEAL,  "D"),
        ("v21_4s_R3_R3__b7.00",   "$R_3 R_3$\n(709k)",      C_SLATE, "o"),
    ]
    for i, (k, lbl, col, mk) in enumerate(c_items):
        if k in arms:
            plot_dstat(ax_c, i, arms[k]["decoders"]["uf"], col, mk, 5.0)
    ax_c.set_yscale("log")
    ax_c.set_xlim(-0.5, 3.5)
    ax_c.set_xticks(range(len(c_items)))
    ax_c.set_xticklabels([x[1] for x in c_items], fontsize=6.9)
    ax_c.set_ylabel("Whole-memory $t_{0.80}$ ($\\beta = 7.0$)")

    save_fig(fig, "fig_v26_4_transfer")


def render_fig5(v21):
    arms = v21["arms"]
    fig = plt.figure(figsize=(7.15, 2.75))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.05, 1.03, 1.08], wspace=0.34)

    # Panel (a): Pure-L=3 three-step (Z, X, Z) & (X, Z, X) hierarchies (N = 16.8k..22.2k)
    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a, "a", x_off=-0.22)
    s3_items = [
        ("v21_3s_Z3r1_X3r1_Z3r1", "$Z_{3,1} X_{3,1} Z_{3,1}$"),
        ("v21_3s_X3r1_Z3r1_X3r1", "$X_{3,1} Z_{3,1} X_{3,1}$"),
        ("v21_3s_X3r2_Z3r1_X3r1", "$X_{3,2} Z_{3,1} X_{3,1}$"),
        ("v21_3s_X3r1_Z3r2_X3r1", "$X_{3,1} Z_{3,2} X_{3,1}$"),
        ("v21_3s_X3r2_Z3r2_X3r2", "$R_3 X_{3,2}$"),
    ]
    for i, (tag, _) in enumerate(s3_items):
        k65 = f"{tag}__b6.50"
        k70 = f"{tag}__b7.00"
        if k65 in arms:
            plot_dstat(ax_a, i - 0.14, arms[k65]["decoders"]["uf"], C_TEAL, "o", 4.6, label=("$\\beta = 6.5$" if i == 0 else None))
        if k70 in arms:
            plot_dstat(ax_a, i + 0.14, arms[k70]["decoders"]["uf"], C_BLUE, "s", 4.5, label=("$\\beta = 7.0$" if i == 0 else None))
    ax_a.set_yscale("log")
    ax_a.set_xlim(-0.55, len(s3_items) - 0.45)
    ax_a.set_xticks(range(len(s3_items)))
    ax_a.set_xticklabels([x[1] for x in s3_items], fontsize=6.9, rotation=30, ha="right")
    ax_a.set_ylabel("3-step $t_{0.80}$ ($L = 3$)")
    ax_a.legend(loc="upper left", frameon=True, framealpha=0.93, edgecolor="#cbd5e1", fontsize=6.2)

    # Panel (b): Pure-L=3 4-step Welding-Order Quartet (`R3` quartet across beta in {5.5, 6.0, 6.5, 7.0})
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b, "b", x_off=-0.22)
    q_specs = [
        ("v21_4s_R3_R3",       [5.50, 6.00, 6.50, 7.00], C_SLATE, "o", "-",  "Planar $XZXZ$"),
        ("v21_4s_R3ZX_R3ZX",   [5.50, 6.00, 6.50, 7.00], C_AMBER, "D", "--", "Planar $ZXZX$"),
        ("v21_4s_R3ZX_R3",     [5.50, 6.00, 6.50, 7.00], C_BLUE,  "^", "-.", "Planar $ZXXZ$"),
        ("v21_4s_R3_R3ZX",     [5.50, 6.00, 6.50, 7.00], C_GREEN, "s", "-",  "Planar $XZZX$"),
        ("v21_tor_4s_R3_R3ZX", [5.50, 6.00, 6.50],       C_PURP,  "s", ":",  "Torus $XZZX$"),
    ]
    for tag, bs, col, mk, ls, lbl in q_specs:
        xs, ys = [], []
        for b in bs:
            k = f"{tag}__b{b:.2f}"
            if k in arms:
                d = arms[k]["decoders"]["uf"]
                xs.append(b); ys.append(d["t_80"])
                plot_dstat(ax_b, b, d, col, mk, 4.4)
        if xs:
            ax_b.plot(xs, ys, color=col, ls=ls, lw=1.35, zorder=3)
            ax_b.plot([], [], color=col, marker=mk, ls=ls, lw=1.35, ms=4.4, label=lbl)
    ax_b.set_yscale("log")
    ax_b.set_xlim(5.38, 7.12)
    ax_b.set_xticks([5.50, 6.00, 6.50, 7.00])
    ax_b.set_xlabel("Inverse temperature $\\beta$")
    ax_b.set_ylabel("Recovery time $t_{0.80}$ ($L = 3$)")
    ax_b.legend(loc="lower right", frameon=True, framealpha=0.93, edgecolor="#cbd5e1", fontsize=5.7, borderpad=0.24)

    # Panel (c): 3D-Local (N, t_0.80) Resource Frontier at beta = 6.5 (pure L = 3 across 1, 3, and 4 steps)
    ax_c = fig.add_subplot(gs[0, 2])
    style_ax(ax_c, "c", x_off=-0.22)
    f_pts = [
        ("v21_L3_C3_full__b6.50",       C_SLATE, "D", "$C_3$",             (1.38, 0.85), "left"),
        ("v21_L3_R3__b6.50",            C_TEAL,  "D", "$R_3$",             (1.42, 0.72), "left"),
        ("v21_L3_R3ZX__b6.50",          C_GREEN, "D", "$R_3^{ZX}$",        (1.08, 1.95), "left"),
        ("v21_3s_X3r1_Z3r2_X3r1__b6.50",C_PURP,  "^", "$X_{3,1}Z_{3,2}X_{3,1}$", (0.95, 2.15), "center"),
        ("v21_4s_R3_R3__b6.50",         C_SLATE, "o", "$R_3 R_3$",         (0.88, 0.42), "center"),
        ("v21_4s_C3_RX3__b6.50",        C_BLUE,  "o", "$C_3 R_{X3}$",      (0.68, 0.90), "right"),
        ("v21_4s_C3_R3ZX__b6.50",       C_GREEN, "s", "$C_3 R_3^{ZX}$",    (0.68, 0.88), "right"),
        ("v21_4s_RZ3_R3ZX__b6.50",      C_BLUE,  "s", "$R_{Z3} R_3^{ZX}$", (0.78, 2.40), "center"),
        ("v21_4s_R3_R3ZX__b6.50",       C_GREEN, "P", "$R_3 R_3^{ZX}$",    (0.64, 0.55), "right"),
    ]
    for k, col, mk, txt, mult, ha_al in f_pts:
        if k in arms:
            nv = arms[k]["N"]
            d = arms[k]["decoders"]["uf"]
            plot_dstat(ax_c, nv, d, col, mk, 4.7)
            ax_c.text(nv * mult[0], d["t_80"] * mult[1], txt, fontsize=5.8, color=C_DARK, ha=ha_al, va="center")
    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.set_xlim(5.5e2, 1.5e6)
    ax_c.set_ylim(1.2e3, 4.5e7)
    ax_c.set_xlabel("Physical qubits $N$ ($L = 3$)")
    ax_c.set_ylabel("3D-local $t_{0.80}$ ($\\beta = 6.5$)")

    save_fig(fig, "fig_v26_5_welding_order")


def render_fig6(v21, v19, loc_quot):
    arms21 = v21["arms"]
    iid21 = v21["iid_arms"]
    arms19 = v19["arms"]
    fig = plt.figure(figsize=(7.15, 2.75))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.02, 1.04, 1.06], wspace=0.34)

    # Panel (a): Paired 4-decoder unrelaxed i.i.d. recovery on pure-L=3 C3_R3^ZX (422,296 qubits)
    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a, "a", x_off=-0.22)
    target_iid = "v21_4s_C3_R3ZX" if "v21_4s_C3_R3ZX" in iid21 else "v21_3s_X3r2_Z3r2_X3r2"
    if target_iid in iid21:
        ia = iid21[target_iid]
        pg = np.array(ia["p_grid"]) * 100.0
        for d, col, ls, mk, nm in [
            ("uf",    C_SLATE, "-",  "o", "UF"),
            ("hier",  C_AMBER, "--", "s", "Hier-UF"),
            ("cool",  C_BLUE,  "-.", "^", "Cool-UF"),
            ("bposd", C_GREEN, "-",  "D", "BP-OSD"),
        ]:
            p80 = ia["decoders"][d]["p_80"] * 100.0
            ax_a.plot(pg, ia["decoders"][d]["curve_Q"], color=col, ls=ls, marker=mk, ms=3.4, markevery=2, lw=1.45, label=f"{nm} (${p80:.2f}\\%$)")
    ax_a.axhline(0.80, color="#94a3b8", lw=0.8, ls=":")
    ax_a.set_xlim(0.4, 8.6)
    ax_a.set_ylim(0.05, 1.04)
    ax_a.set_xlabel("Unrelaxed error rate $p$ (%)")
    ax_a.set_ylabel("Single-shot $P_{\\mathrm{rec}}(p)$")
    ax_a.legend(loc="lower left", frameon=True, framealpha=0.93, edgecolor="#cbd5e1", fontsize=5.9, borderpad=0.25)

    # Panel (b): Exact 1-step MLD calibration (UF & BP-OSD vs exact analytical P^opt)
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b, "b", x_off=-0.22)
    ec = loc_quot["exact_cell_curves_b70"]
    ax_b.plot(ec["t_X"], ec["PX_C2_full"], color=C_DARK, lw=1.8, ls="-", label="Optimal $P_X^{\\mathrm{opt}}$")
    ax_b.plot(ec["t_Q"], ec["PQ_C2_full"], color=C_SLATE, lw=1.6, ls="-", label="Optimal $P_Q^{\\mathrm{opt}}$")
    if "v19_sq_C2_1step__b7.00" in arms19:
        c1 = arms19["v19_sq_C2_1step__b7.00"]
        tg = c1["t_grid"]
        ax_b.plot(tg, c1["decoders"]["uf"]["p_X"], color=C_AMBER, lw=1.4, ls="--", label="UF $p_X$")
        ax_b.plot(tg, c1["decoders"]["bposd"]["p_X"], color=C_GREEN, lw=1.45, ls=":", label="BP-OSD $p_X$")
        ax_b.plot(tg, c1["decoders"]["bposd"]["p_Q"], color=C_RED, lw=1.35, ls="--", label="UF / BP-OSD $P_{\\mathrm{rec}}$")
    ax_b.axhline(0.80, color="#94a3b8", lw=0.8, ls=":")
    ax_b.set_xscale("log")
    ax_b.set_xlim(8.0, 4.0e5)
    ax_b.set_ylim(0.45, 1.03)
    ax_b.set_xlabel("Time $t$ ($1$-step cell, $\\beta = 7.0$)")
    ax_b.set_ylabel("Recovery probability")
    ax_b.legend(loc="lower left", frameon=True, framealpha=0.93, edgecolor="#cbd5e1", fontsize=5.8, borderpad=0.24)

    # Panel (c): Microscopic failure anatomy near t_0.80^UF across pure-L=3 hierarchies
    ax_c = fig.add_subplot(gs[0, 2])
    style_ax(ax_c, "c", x_off=-0.22)
    anat_keys = [
        ("v21_4s_C3_R3ZX__b4.50",       "$C_3 R_3^{ZX}$ ($4.5$)"),
        ("v21_4s_C3_R3ZX__b5.50",       "$C_3 R_3^{ZX}$ ($5.5$)"),
        ("v21_4s_C3_R3ZX__b6.00",       "$C_3 R_3^{ZX}$ ($6.0$)"),
        ("v21_4s_C3_R3ZX__b6.50",       "$C_3 R_3^{ZX}$ ($6.5$)"),
        ("v21_3s_X3r1_Z3r2_X3r1__b7.00","$X_{3,1}Z_{3,2}X_{3,1}$ ($7.0$)"),
        ("v21_L3_R3ZX__b6.50",          "$R_3^{ZX}$ ($6.5$)"),
    ]
    present = [(k, lbl) for k, lbl in anat_keys if k in arms21]
    xs = np.arange(len(present))
    f_gq = np.array([arms21[k]["failure_anatomy_near_t80"]["frac_settled_wrong_basin_GQ"] for k, _ in present])
    f_ex = np.array([arms21[k]["failure_anatomy_near_t80"]["frac_excited_unrescued"] for k, _ in present])
    f_bp = np.array([arms21[k]["failure_anatomy_near_t80"]["frac_rescued_by_bposd"] for k, _ in present])

    ax_c.bar(xs, f_gq, width=0.58, color=C_TEAL, label="Wrong basin ($s{=}0$/quench)")
    ax_c.bar(xs, f_ex, width=0.58, bottom=f_gq, color=C_AMBER, label="Excited (both fail)")
    ax_c.bar(xs, f_bp, width=0.58, bottom=f_gq + f_ex, color=C_GREEN, label="Rescued by BP-OSD")
    ax_c.set_xticks(xs)
    ax_c.set_xticklabels([lbl for _, lbl in present], fontsize=6.5, rotation=32, ha="right")
    ax_c.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_c.set_ylim(0.0, 1.24)
    ax_c.set_ylabel("Share of UF failures near $t_{0.80}$")
    ax_c.legend(loc="upper left", frameon=True, framealpha=0.93, edgecolor="#cbd5e1", fontsize=5.6, borderpad=0.22)

    save_fig(fig, "fig_v26_6_decoders")


def render_fig7(v20, v21, v22):
    arms20 = v20["arms"]
    arms21 = v21["arms"]
    arms22 = v22["arms"]
    fig = plt.figure(figsize=(7.15, 2.85))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.04, 1.12, 1.00], wspace=0.34)

    def get_pt(src, k):
        m = arms20 if src == "v20" else (arms21 if src == "v21" else arms22)
        return m[k]["decoders"]["bposd"] if k in m else None

    # Panel (a): J = 1 temperature scaling across 2D & 3D-local families (explicit prior vs ours!)
    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a, "a", x_off=-0.22)
    curves_a = [
        (C_SLATE, "o", "-", "2D Toric (prior)", [
            (4.50, get_pt("v20", "v20_tor2d_L16__br4.50")),
            (5.00, get_pt("v20", "v20_tor2d_L16__br5.00")),
            (5.50, get_pt("v20", "v20_tor2d_L16__bq5.50")),
            (6.00, get_pt("v20", "v20_tor2d_L16__bq6.00")),
            (6.50, get_pt("v20", "v20_tor2d_L16__bq6.50")),
            (7.00, get_pt("v20", "v20_tor2d_L16__bq7.00")),
        ]),
        (C_AMBER, "D", "--", "Michnicki 1-cyc. (prior)", [
            (4.50, get_pt("v20", "v20_mich1c_L4__br4.50")),
            (5.00, get_pt("v20", "v20_mich1c_L4__br5.00")),
            (5.50, get_pt("v20", "v20_mich1c_L4__br5.50")),
            (6.00, get_pt("v20", "v20_mich1c_L4__br6.00")),
            (6.50, get_pt("v20", "v20_mich1c_L4__br6.50")),
            (7.00, get_pt("v20", "v20_mich1c_L4__br7.00")),
        ]),
        (C_BLUE, "^", "-.", "Ours: XZXZ cut ($L{=}3$)", [
            (4.50, get_pt("v21", "v21_4s_C3_RX3__b4.50")),
            (5.00, get_pt("v21", "v21_4s_C3_RX3__b5.00")),
            (5.50, get_pt("v21", "v21_4s_C3_RX3__b5.50")),
            (6.00, get_pt("v21", "v21_4s_C3_RX3__b6.00")),
            (6.50, get_pt("v21", "v21_4s_C3_RX3__b6.50")),
            (7.00, get_pt("v21", "v21_4s_C3_RX3__b7.00")),
        ]),
        (C_GREEN, "s", "-", "Ours: XZZX ($L{=}3$)", [
            (4.50, get_pt("v21", "v21_4s_C3_R3ZX__b4.50")),
            (5.00, get_pt("v21", "v21_4s_C3_R3ZX__b5.00")),
            (5.50, get_pt("v21", "v21_4s_C3_R3ZX__b5.50")),
            (6.00, get_pt("v21", "v21_4s_C3_R3ZX__b6.00")),
            (6.50, get_pt("v21", "v21_4s_C3_R3ZX__b6.50")),
            (7.00, get_pt("v21", "v21_4s_C3_R3ZX__b7.00")),
        ]),
        (C_PURP, "*", "-", "Ours: 3D Color ($L{=}3$)", [
            (4.00, get_pt("v22", "v22_cc2s_RX3__b4.00")),
            (4.50, get_pt("v22", "v22_cc2s_RX3__b4.50")),
            (5.00, get_pt("v22", "v22_cc2s_RX3__b5.00")),
            (5.50, get_pt("v22", "v22_cc2s_RX3__b5.50")),
            (6.00, get_pt("v22", "v22_cc2s_RX3__b6.00")),
            (6.50, get_pt("v22", "v22_cc2s_RX3__b6.50")),
        ]),
        (C_RED, "P", "-", "Haah Cubic (prior)", [
            (4.00, get_pt("v20", "v20_haah_L7__br4.00")),
            (4.25, get_pt("v20", "v20_haah_L7__br4.25")),
            (4.50, get_pt("v20", "v20_haah_L7__br4.50")),
            (5.00, get_pt("v20", "v20_haah_L7__br5.00")),
            (6.50, get_pt("v20", "v20_haah_L7__br6.50")),
        ]),
    ]
    for col, mk, ls, lbl, pts in curves_a:
        xs, ys = [], []
        ms = 5.6 if mk == "*" else 4.2
        for b, d in pts:
            if d is None:
                continue
            xs.append(b); ys.append(d["t_80"])
            plot_dstat(ax_a, b, d, col, mk, ms)
        if xs:
            ax_a.plot(xs, ys, color=col, ls=ls, lw=1.35, zorder=3)
            ax_a.plot([], [], color=col, marker=mk, ls=ls, lw=1.35, ms=ms, label=lbl)

    ax_a.set_yscale("log")
    ax_a.set_xlim(3.85, 7.18)
    ax_a.set_xticks([4.0, 5.0, 6.0, 7.0])
    ax_a.set_ylim(5.0, 6.5e9)
    ax_a.set_xlabel("Inverse temperature $\\beta$ ($J = 1$)")
    ax_a.set_ylabel("BP-OSD $t_{0.80}$")
    handles_a, labels_a = ax_a.get_legend_handles_labels()

    # Panel (b): t_0.80 vs N at J = 1, beta = 6.5 across strictly 2D & 3D-local families
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b, "b", x_off=-0.22)

    xs, ys = [], []
    for t in ["v20_tor2d_L8", "v20_tor2d_L16", "v20_tor2d_L32", "v20_tor2d_L64"]:
        k = f"{t}__bq6.50"
        if k in arms20:
            d = arms20[k]["decoders"]["bposd"]
            xs.append(arms20[k]["N"]); ys.append(d["t_80"])
            plot_dstat(ax_b, arms20[k]["N"], d, C_SLATE, "o", 4.2)
    ax_b.plot(xs, ys, color=C_SLATE, ls="-", lw=1.35, zorder=3)

    xs, ys = [], []
    for t in ["v20_mich1c_L2", "v20_mich1c_L3", "v20_mich1c_L4", "v20_mich1c_L6", "v20_mich1c_L8"]:
        k = f"{t}__br6.50"
        if k in arms20:
            d = arms20[k]["decoders"]["bposd"]
            xs.append(arms20[k]["N"]); ys.append(d["t_80"])
            plot_dstat(ax_b, arms20[k]["N"], d, C_AMBER, "D", 4.2)
    ax_b.plot(xs, ys, color=C_AMBER, ls="--", lw=1.35, zorder=3)

    # BDL original XZXZ baseline R_3 R_3 plotted hollow with explicit prior label
    if "v21_4s_R3_R3__b6.50" in arms21:
        nv = arms21["v21_4s_R3_R3__b6.50"]["N"]
        d = arms21["v21_4s_R3_R3__b6.50"]["decoders"]["bposd"]
        plot_dstat(ax_b, nv, d, C_BLUE, "^", 4.8, hollow=True)
        ax_b.text(nv * 0.90, d["t_80"] * 0.36, "BDL $R_3 R_3$ (prior)", fontsize=5.2, color=C_BLUE, ha="center", va="center")

    if "v21_4s_C3_RX3__b6.50" in arms21:
        nv = arms21["v21_4s_C3_RX3__b6.50"]["N"]
        d = arms21["v21_4s_C3_RX3__b6.50"]["decoders"]["bposd"]
        plot_dstat(ax_b, nv, d, C_BLUE, "^", 4.6)
        ax_b.text(nv * 0.58, d["t_80"] * 0.88, "$C_3 R_{X3}$", fontsize=5.4, color=C_DARK, ha="right", va="center")

    for k, txt, mult, ha_al in [
        ("v21_4s_C3_R3ZX__b6.50", "$C_3 R_3^{ZX}$", (0.68, 0.56), "center"),
        ("v21_4s_R3_R3ZX__b6.50", "$R_3 R_3^{ZX}$", (0.88, 2.45), "center"),
    ]:
        if k in arms21:
            nv = arms21[k]["N"]
            d = arms21[k]["decoders"]["bposd"]
            plot_dstat(ax_b, nv, d, C_GREEN, "s", 4.6)
            ax_b.text(nv * mult[0], d["t_80"] * mult[1], txt, fontsize=5.4, color=C_DARK, ha=ha_al, va="center")

    # Ours: strictly pure-L=3 3D-local welded color-code family overlay (dim_F < 3)
    for k, txt, mult, ha_al in [
        ("v22_cc2s_RX3__b6.50",          "Color $R_{X3}$", (1.35, 1.90), "left"),
        ("v22_cc2s_R3__b6.50",           "Color $R_3$",    (1.40, 0.55), "left"),
        ("v22_cc3s_X3r1Z3r2X3r1__b6.50", "Color 3s",       (1.36, 0.92), "left"),
    ]:
        if k in arms22:
            nv = arms22[k]["N"]
            d = arms22[k]["decoders"]["bposd"]
            plot_dstat(ax_b, nv, d, C_PURP, "*", 5.6)
            ax_b.text(nv * mult[0], d["t_80"] * mult[1], txt, fontsize=5.4, color=C_PURP, ha=ha_al, va="center")

    for k, txt, mult in [
        ("v20_haah_L7__br6.50",  "Haah $L{=}7$",  (1.38, 1.05)),
        ("v20_haah_L11__br6.50", "Haah $L{=}11$", (1.38, 0.88)),
    ]:
        if k in arms20:
            nv = arms20[k]["N"]
            d = arms20[k]["decoders"]["bposd"]
            plot_dstat(ax_b, nv, d, C_RED, "P", 4.8)
            ax_b.text(nv * mult[0], d["t_80"] * mult[1], txt, fontsize=5.5, color=C_RED, va="center")

    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlim(85.0, 2.0e6)
    ax_b.set_ylim(80.0, 6.5e9)
    ax_b.set_xlabel("Physical qubits $N$ ($\\beta = 6.5$)")
    ax_b.set_ylabel("BP-OSD $t_{0.80}$")
    ax_b.legend(
        handles_a, labels_a,
        loc="center left", bbox_to_anchor=(0.015, 0.42),
        frameon=True, framealpha=0.94, edgecolor="#cbd5e1",
        fontsize=4.75, borderpad=0.20, labelspacing=0.16, handlelength=1.28,
    )

    # Panel (c): Haah's 3D Cubic Code finite-size crossover vs lattice size L (J = 1)
    ax_c = fig.add_subplot(gs[0, 2])
    style_ax(ax_c, "c", x_off=-0.22)
    Ls_k2 = [5, 7, 9, 11, 13, 17]
    for mode_s, col, mk, ls, lbl in [
        ("bq6.50", C_AMBER, "o", "--", "$\\beta = 3.25$ ($K{=}2$)"),
        ("bq7.55", C_TEAL,  "s", "-.", "$\\beta = 3.78$ ($K{=}2$)"),
        ("br4.00", C_RED,   "^", "-",  "$\\beta = 4.00$ ($K{=}2$)"),
    ]:
        xs, ys = [], []
        for L in Ls_k2:
            k = f"v20_haah_L{L}__{mode_s}"
            if k in arms20:
                d = arms20[k]["decoders"]["bposd"]
                xs.append(L); ys.append(d["t_80"])
                plot_dstat(ax_c, L, d, col, mk, 4.3)
        if xs:
            ax_c.plot(xs, ys, color=col, ls=ls, lw=1.35, zorder=3)
            ax_c.plot([], [], color=col, marker=mk, ls=ls, lw=1.35, ms=4.3, label=lbl)

    for mode_s, col in [("bq6.50", C_AMBER), ("bq7.55", C_TEAL), ("br4.00", C_RED)]:
        k15 = f"v20_haah_L15__{mode_s}"
        if k15 in arms20:
            plot_dstat(ax_c, 15, arms20[k15]["decoders"]["bposd"], col, "v", 4.7, hollow=True)
    if "v20_haah_L15__bq6.50" in arms20:
        ax_c.text(14.9, arms20["v20_haah_L15__bq6.50"]["decoders"]["bposd"]["t_80"] * 0.34, "$L{=}15$\n($K{=}50$)", fontsize=5.5, color=C_DARK, ha="center")

    ax_c.set_yscale("log")
    ax_c.set_xlim(4.2, 18.0)
    ax_c.set_xticks([5, 7, 9, 11, 13, 15, 17])
    ax_c.set_ylim(2.2e2, 3.5e7)
    ax_c.set_xlabel("Cubic lattice size $L$")
    ax_c.set_ylabel("3D Haah $t_{0.80}$")
    ax_c.legend(loc="upper right", frameon=True, framealpha=0.93, edgecolor="#cbd5e1", fontsize=5.4, borderpad=0.22, labelspacing=0.18)

    save_fig(fig, "fig_v26_7_benchmarks")


def render_fig8(v20, v21, v22):
    arms20 = v20["arms"]
    arms21 = v21["arms"]
    arms22 = v22["arms"]
    iid22 = v22["iid_arms"]
    fig = plt.figure(figsize=(7.15, 2.75))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.08, 1.04, 1.00], wspace=0.35)

    # Panel (a): Unwelded 2D 4.8.8 color tori (L=3..12) vs 3D-Local Welded Color-Code hierarchies at beta = 6.5
    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a, "a", x_off=-0.22)

    xs, ys = [], []
    for L in (3, 6, 9, 12):
        k = f"v22_cc2d_L{L}__b6.50"
        if k in arms22:
            d = arms22[k]["decoders"]["bposd"]
            xs.append(arms22[k]["N"]); ys.append(d["t_80"])
            plot_dstat(ax_a, arms22[k]["N"], d, C_SLATE, "o", 4.3)
    if xs:
        ax_a.plot(xs, ys, color=C_SLATE, ls="-", lw=1.35, label="2D 4.8.8 ($L{=}3{\\to}12$)", zorder=3)

    for k, col, mk, hol, txt, mult, ha_al in [
        ("v22_cc1s_L3r2_sd__b6.50",      C_AMBER, "x", False, "1s SD",        (1.00, 1.95), "center"),
        ("v22_cc1s_L3r2__b6.50",         C_TEAL,  "D", False, "1s $R_3$",     (1.50, 1.15), "left"),
        ("v22_cc2s_mix32__b6.50",        C_GREEN, "D", True,  "$(3,2)$",      (1.00, 2.05), "center"),
        ("v22_cc2s_R3_sd__b6.50",        C_RED,   "v", False, "2s SD",        (1.00, 0.48), "center"),
        ("v22_cc2s_C3__b6.50",           C_PURP,  "^", False, "$C_3$",        (0.68, 1.25), "right"),
        ("v22_cc2s_RZ3__b6.50",          C_SLATE, "^", False, "$R_{Z3}$",     (1.42, 0.90), "left"),
        ("v22_cc2s_R3__b6.50",           C_PURP,  "P", False, "$R_3$",        (1.42, 0.72), "left"),
        ("v22_cc2s_RX3__b6.50",          C_PURP,  "*", False, "$R_{X3}$",     (0.65, 1.55), "right"),
        ("v22_cc3s_X3r1Z3r1X3r1__b6.50", C_BLUE,  "o", False, "3s $1{-}1{-}1$", (0.66, 0.78), "right"),
        ("v22_cc3s_X3r1Z3r2X3r1__b6.50", C_BLUE,  "s", False, "3s $1{-}2{-}1$", (0.66, 1.65), "right"),
        ("v22_cc3s_R3X3r2__b6.50",       C_BLUE,  "D", False, "3s $R_3X$",    (0.66, 0.56), "right"),
    ]:
        if k in arms22:
            nv = arms22[k]["N"]
            d = arms22[k]["decoders"]["bposd"]
            ms = 5.6 if mk == "*" else 4.5
            plot_dstat(ax_a, nv, d, col, mk, ms, hollow=hol)
            ax_a.text(nv * mult[0], d["t_80"] * mult[1], txt, fontsize=5.3, color=C_DARK, ha=ha_al, va="center")

    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlim(45.0, 2.6e5)
    ax_a.set_ylim(2.5e4, 2.0e8)
    ax_a.set_xlabel("Physical qubits $N$ ($\\beta = 6.5$)")
    ax_a.set_ylabel("BP-OSD $t_{0.80}$")
    ax_a.legend(loc="upper right", frameon=True, framealpha=0.94, edgecolor="#cbd5e1", fontsize=5.4, borderpad=0.22)

    # Panel (b): Temperature scaling of 3D Welded Color vs 3D Welded Toric (XZZX) vs 2D Color vs Haah
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b, "b", x_off=-0.22)
    b_curves = [
        (C_SLATE, "o", "-",  "2D Color ($L{=}12$)",     [(b, arms22.get(f"v22_cc2d_L12__b{b:.2f}")) for b in (4.0, 4.5, 5.0, 5.5, 6.0, 6.5)]),
        (C_GREEN, "s", "--", "Toric $XZZX$ ($422\\mathrm{k}$)", [(b, arms21.get(f"v21_4s_C3_R3ZX__b{b:.2f}")) for b in (4.5, 5.0, 5.5, 6.0, 6.5)]),
        (C_BLUE,  "^", "-.", "Color 3s ($82\\mathrm{k}$)", [(b, arms22.get(f"v22_cc3s_X3r1Z3r2X3r1__b{b:.2f}")) for b in (4.5, 5.0, 5.5, 6.0, 6.5)]),
        (C_PURP,  "*", "-",  "Color $R_{X3}$ ($4.1\\mathrm{k}$)", [(b, arms22.get(f"v22_cc2s_RX3__b{b:.2f}")) for b in (4.0, 4.5, 5.0, 5.5, 6.0, 6.5)]),
        (C_TEAL,  "D", "-",  "Color $(3,2)$ ($2.2\\mathrm{k}$)", [(b, arms22.get(f"v22_cc2s_mix32__b{b:.2f}")) for b in (4.0, 4.5, 5.0, 5.5, 6.0, 6.5)]),
        (C_RED,   "P", "-",  "3D Haah ($L{=}7$)",       [(b, arms20.get(f"v20_haah_L7__br{b:.2f}")) for b in (4.0, 4.5, 5.0, 6.5)]),
    ]
    for col, mk, ls, lbl, pts in b_curves:
        xs, ys = [], []
        ms = 5.5 if mk == "*" else 4.2
        for b, arm in pts:
            if arm is None:
                continue
            d = arm["decoders"]["bposd"]
            xs.append(b); ys.append(d["t_80"])
            plot_dstat(ax_b, b, d, col, mk, ms)
        if xs:
            ax_b.plot(xs, ys, color=col, ls=ls, lw=1.35, zorder=3)
            ax_b.plot([], [], color=col, marker=mk, ls=ls, lw=1.35, ms=ms, label=lbl)
    ax_b.set_yscale("log")
    ax_b.set_xlim(3.85, 6.68)
    ax_b.set_xticks([4.0, 4.5, 5.0, 5.5, 6.0, 6.5])
    ax_b.set_ylim(25.0, 5.0e9)
    ax_b.set_xlabel("Inverse temp. $\\beta$ ($J{=}1$)")
    ax_b.set_ylabel("BP-OSD $t_{0.80}$")
    ax_b.legend(loc="lower right", frameon=True, framealpha=0.94, edgecolor="#cbd5e1", fontsize=4.85, borderpad=0.20, labelspacing=0.16, handlelength=1.30)

    # Panel (c): Paired 4-decoder single-shot i.i.d. threshold on 3D-local welded color code `v22_cc2s_R3` (5,088 qubits)
    ax_c = fig.add_subplot(gs[0, 2])
    style_ax(ax_c, "c", x_off=-0.22)
    target_iid = "v22_cc2s_R3" if "v22_cc2s_R3" in iid22 else "v22_cc3s_X3r1Z3r2X3r1"
    if target_iid in iid22:
        ia = iid22[target_iid]
        pg = np.array(ia["p_grid"]) * 100.0
        for d, col, ls, mk, nm in [
            ("uf",    C_SLATE, "-",  "o", "UF"),
            ("hier",  C_AMBER, "--", "s", "Hier-UF"),
            ("cool",  C_BLUE,  "-.", "^", "Cool-UF"),
            ("bposd", C_PURP,  "-",  "D", "BP-OSD"),
        ]:
            p80 = ia["decoders"][d]["p_80"] * 100.0
            ax_c.plot(pg, ia["decoders"][d]["curve_Q"], color=col, ls=ls, marker=mk, ms=3.4, markevery=2, lw=1.45, label=f"{nm} (${p80:.2f}\\%$)")
    ax_c.axhline(0.80, color="#94a3b8", lw=0.8, ls=":")
    ax_c.set_xlim(0.4, 8.2)
    ax_c.set_ylim(0.0, 1.04)
    ax_c.set_xlabel("Bit/phase flip rate $p$ (%)")
    ax_c.set_ylabel("Single-shot $P_{\\mathrm{rec}}(p)$")
    ax_c.legend(loc="lower left", frameon=True, framealpha=0.94, edgecolor="#cbd5e1", fontsize=5.7, borderpad=0.24)

    save_fig(fig, "fig_v27_8_color_welded")


def render_fig9(v20, v21, v22, v23, ext_geom):
    arms20 = v20.get("arms", {})
    arms21 = v21.get("arms", {})
    arms22 = v22.get("arms", {})
    arms23 = v23.get("arms", {})
    if not arms23:
        return

    fig = plt.figure(figsize=(7.25, 2.85))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.05, 1.04, 1.08], wspace=0.42)

    # Panel (a): Regime 1 — Same-parity fixed-temperature scaling vs Haah's finite-L ceiling
    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a, "a", x_off=-0.22)

    # 1) Haah at beta=4.00 (turns downward across L=5..17)
    hx, hy = [], []
    for L in [5, 7, 9, 11, 13, 17]:
        k = f"v20_haah_L{L}__br4.00"
        if k in arms20:
            d = arms20[k]["decoders"]["bposd"]
            hx.append(L); hy.append(d["t_80"])
            plot_dstat(ax_a, L, d, C_RED, "^", 4.1)
    if hx:
        ax_a.plot(hx, hy, color=C_RED, ls="-", lw=1.3, label="Haah ($\\mathbb{T}^3$, $\\beta{=}4.0$)")
    if "v20_haah_L15__br4.00" in arms20:
        plot_dstat(ax_a, 15, arms20["v20_haah_L15__br4.00"]["decoders"]["bposd"], C_RED, "v", 4.3, hollow=True)

    # 2) Michnicki 1-cycle collapse (L=3..8, beta=6.5)
    mx, my = [], []
    for L in [2, 3, 4, 6, 8]:
        k = f"v20_mich1c_L{L}__br6.50"
        if k in arms20:
            d = arms20[k]["decoders"]["bposd"]
            mx.append(L); my.append(d["t_80"])
            plot_dstat(ax_a, L, d, C_AMBER, "D", 3.9)
    if mx:
        ax_a.plot(mx, my, color=C_AMBER, ls="--", lw=1.25, label="Michnicki 1-cyc. ($\\beta{=}6.5$)")

    # 3) Even-step Toric XZZX (s=2 -> s=4 at beta=6.5)
    tx, ty = [], []
    for Lval, k in [(3, "v21_L3_R3ZX__b6.50"), (81, "v21_4s_R3_R3ZX__b6.50")]:
        if k in arms21:
            d = arms21[k]["decoders"]["bposd"]
            tx.append(Lval); ty.append(d["t_80"])
            plot_dstat(ax_a, Lval, d, C_GREEN, "s", 4.5)
    if tx:
        ax_a.plot(tx, ty, color=C_GREEN, ls="-", lw=1.45, label="Toric $XZZX$ $s{=}2{\\to}4$ ($6.5$)")

    # 4) Odd-step Color (same parity s=1 -> s=3) at beta=6.5 and dispersed s=1 -> s=3 at beta=7.0
    for lbl, col, mk, ls, pts in [
        ("Color $1{\\to}3$ std ($6.5$)", C_PURP, "o", "-.", [(3, arms22.get("v22_cc1s_L3r2__b6.50")), (27, arms22.get("v22_cc3s_X3r1Z3r2X3r1__b6.50"))]),
        ("Color $1{\\to}3$ disp ($6.5$)", C_BLUE, "*", "-", [(3, arms23.get("v23_disp_1s_L3__b6.50")), (27, arms23.get("v23_disp_3s_L3__b6.50"))]),
        ("Color $1{\\to}3$ disp ($7.0$)", C_TEAL, "P", "-", [(3, arms23.get("v23_disp_1s_L3__b7.00")), (27, arms23.get("v23_disp_3s_L3__b7.00"))]),
    ]:
        px, py = [], []
        ms = 5.4 if mk == "*" else 4.3
        for Lval, arm in pts:
            if arm is None:
                continue
            d = arm["decoders"]["bposd"]
            px.append(Lval); py.append(d["t_80"])
            plot_dstat(ax_a, Lval, d, col, mk, ms)
        if px:
            ax_a.plot(px, py, color=col, ls=ls, lw=1.35, label=lbl)

    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlim(1.7, 110.0)
    ax_a.set_xticks([2, 3, 5, 9, 17, 27, 81])
    ax_a.set_xticklabels(["2", "3", "5", "9", "17", "27", "81"], fontsize=7.5)
    ax_a.set_ylim(1.2e2, 4.5e8)
    ax_a.set_xlabel("Box size $L_{\\mathrm{blk}}$ (fixed $\\beta$)")
    ax_a.set_ylabel("BP-OSD $t_{0.80}$")
    ax_a.legend(loc="upper right", frameon=True, framealpha=0.93, edgecolor="#cbd5e1", fontsize=4.6, borderpad=0.20, labelspacing=0.14)

    # Panel (b): Extension A1 — BDL T-junction dispersion vs + pillars across temperature
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b, "b", x_off=-0.22)
    b_series = [
        (C_SLATE, "o", "--", "2s $C_3$ ($+$ pillar, $3.8\\mathrm{k}$)",      [(b, arms22.get(f"v22_cc2s_C3__b{b:.2f}")) for b in (4.5, 5.0, 5.5, 6.0, 6.5)]),
        (C_TEAL,  "s", "-",  "2s Disp. ($T$ junc., $4.0\\mathrm{k}$)",      [(b, arms23.get(f"v23_disp_2s_L3__b{b:.2f}")) for b in (4.5, 5.0, 5.5, 6.0, 6.5, 7.0)]),
        (C_PURP,  "*", "-",  "2s $R_{X3}$ ($4.1\\mathrm{k}$)",               [(b, arms22.get(f"v22_cc2s_RX3__b{b:.2f}")) for b in (4.5, 5.0, 5.5, 6.0, 6.5)]),
        ("#db2777", "D", "-", "2s $R_X$ + Disp. ($4.3\\mathrm{k}$)",         [(b, arms23.get(f"v23_disp_2s_RX3__b{b:.2f}")) for b in (5.0, 5.5, 6.0, 6.5)]),
        (C_AMBER, "v", "--", "3s $1{-}1{-}1$ ($+$ pillar, $75\\mathrm{k}$)", [(b, arms22.get(f"v22_cc3s_X3r1Z3r1X3r1__b{b:.2f}")) for b in (4.5, 5.0, 5.5, 6.0, 6.5)]),
        (C_BLUE,  "^", "-.", "3s $1{-}2{-}1$ ($82\\mathrm{k}$)",             [(b, arms22.get(f"v22_cc3s_X3r1Z3r2X3r1__b{b:.2f}")) for b in (4.5, 5.0, 5.5, 6.0, 6.5)] + [(7.0, arms23.get("v23_cc3s_121_ref__b7.00"))]),
        (C_GREEN, "P", "-",  "3s Disp. ($T$ junc., $81\\mathrm{k}$)",      [(b, arms23.get(f"v23_disp_3s_L3__b{b:.2f}")) for b in (4.5, 5.0, 5.5, 6.0, 6.5, 7.0)]),
    ]
    for col, mk, ls, lbl, pts in b_series:
        xs, ys = [], []
        ms = 5.3 if mk == "*" else 4.1
        for b, arm in pts:
            if arm is None:
                continue
            d = arm["decoders"]["bposd"]
            xs.append(b); ys.append(d["t_80"])
            plot_dstat(ax_b, b, d, col, mk, ms)
        if xs:
            ax_b.plot(xs, ys, color=col, ls=ls, lw=1.35, zorder=3)
            ax_b.plot([], [], color=col, marker=mk, ls=ls, lw=1.35, ms=ms, label=lbl)
    ax_b.set_yscale("log")
    ax_b.set_xlim(4.35, 7.18)
    ax_b.set_xticks([4.5, 5.0, 5.5, 6.0, 6.5, 7.0])
    ax_b.set_ylim(3.0e2, 2.8e8)
    ax_b.set_xlabel("Inverse temp. $\\beta$ ($J = 1$)")
    ax_b.set_ylabel("BP-OSD $t_{0.80}$")
    ax_b.legend(loc="upper left", frameon=True, framealpha=0.93, edgecolor="#cbd5e1", fontsize=4.65, borderpad=0.20, labelspacing=0.14)

    # Panel (c): Regime 3 — Open-boundary planar 4.8.8 (`K=2`) + inset K(L) vs Haah
    ax_c = fig.add_subplot(gs[0, 2])
    style_ax(ax_c, "c", x_off=-0.22)
    px, py = [], []
    for t in ["v23_plan_cc2d_L6", "v23_plan_cc2d_L12"]:
        k = f"{t}__b6.50"
        if k in arms23:
            d = arms23[k]["decoders"]["bposd"]
            px.append(arms23[k]["N"]); py.append(d["t_80"])
            plot_dstat(ax_c, arms23[k]["N"], d, C_SLATE, "o", 4.4)
    if px:
        ax_c.plot(px, py, color=C_SLATE, ls="-", lw=1.35)
        ax_c.text(560.0, 3.6e3, "Open 2D\n($L{=}6,12$)", fontsize=5.2, color=C_SLATE, ha="center", va="top")

    for k, col, mk, txt, mult, ha_al in [
        ("v23_plan_cc1s_L3r2__b6.50",  C_AMBER, "D", "1s",        (1.00, 2.35), "center"),
        ("v23_plan_disp_2s_L3__b6.50", C_TEAL,  "s", "Disp. 2s",  (1.00, 0.34), "center"),
        ("v23_plan_cc2s_RX3__b6.50",   C_PURP,  "*", "$R_{X3}$",  (1.00, 2.35), "center"),
        ("v23_plan_cc2s_R3__b6.50",    C_BLUE,  "^", "$R_3$",     (1.30, 0.85), "left"),
    ]:
        if k in arms23:
            nv = arms23[k]["N"]
            d = arms23[k]["decoders"]["bposd"]
            plot_dstat(ax_c, nv, d, col, mk, 5.8 if mk == "*" else 4.6)
            ax_c.text(nv * mult[0], d["t_80"] * mult[1], txt, fontsize=5.6, color=C_DARK, ha=ha_al, va="center")

    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.set_xlim(140.0, 1.35e4)
    ax_c.set_ylim(2.0e2, 4.5e7)
    ax_c.set_xlabel("Planar qubits $N$ ($K{=}2$)")
    ax_c.set_ylabel("BP-OSD $t_{0.80}$ ($\\beta = 6.5$)")

    # Inset: Constant K (planar welded) vs oscillating K(L) (periodic Haah)
    if "haah_K_vs_L" in ext_geom:
        axins = ax_c.inset_axes([0.16, 0.53, 0.44, 0.41])
        hkl = sorted(ext_geom["haah_K_vs_L"].values(), key=lambda x: x["L"])
        ls_h = [x["L"] for x in hkl]
        ks_h = [x["K"] for x in hkl]
        axins.plot(ls_h, ks_h, color=C_RED, marker=".", ms=3.5, lw=1.0, label="Haah ($\\mathbb{T}^3$)")
        axins.axhline(2, color=C_PURP, ls="--", lw=1.1, label="Color ($K{=}2$)")
        axins.axhline(1, color=C_GREEN, ls=":", lw=1.1, label="Surf. ($K{=}1$)")
        axins.set_yscale("log")
        axins.set_xlim(2.5, 17.5)
        axins.set_ylim(0.7, 130.0)
        axins.set_xticks([3, 8, 15, 17])
        axins.tick_params(labelsize=5.4, pad=1.0)
        axins.set_xlabel("$L$", fontsize=5.8, labelpad=0.8)
        axins.set_ylabel("$K$", fontsize=5.8, labelpad=0.8)
        axins.legend(loc="upper left", frameon=False, fontsize=4.4, handlelength=1.0, labelspacing=0.1)

    save_fig(fig, "fig_v28_9_regimes_and_dispersion")


def main():
    with open(ROOT / "verified_3d_local_L3_geometry.json") as f:
        geom = json.load(f)
    with open(ROOT / "reanalyzed_v16_v17_suiteA.json") as f:
        suiteA = json.load(f)
    with open(ROOT / "parsed_v18_rework_data.json") as f:
        v18 = json.load(f)
    with open(ROOT / "parsed_v17_data.json") as f:
        v14 = json.load(f)
    with open(ROOT / "frozen_interface_predictions.json") as f:
        frozen = json.load(f)
    with open(ROOT / "parsed_v19_rebalance_and_decoders.json") as f:
        v19 = json.load(f)
    with open(ROOT / "local_quotient_and_pruning_data.json") as f:
        loc_quot = json.load(f)
    with open(ROOT / "parsed_v20_benchmarks.json") as f:
        v20 = json.load(f)
    with open(ROOT / "parsed_v21_3d_local.json") as f:
        v21 = json.load(f)
    v22_p = ROOT / "parsed_v22_color_welded.json"
    v22 = json.load(open(v22_p)) if v22_p.exists() else {"arms": {}, "iid_arms": {}}
    v23_p = ROOT / "parsed_v23_extensionA.json"
    v23 = json.load(open(v23_p)) if v23_p.exists() else {"arms": {}, "iid_arms": {}}
    ext_p = ROOT / "verified_v23_extensionA.json"
    ext_geom = json.load(open(ext_p)) if ext_p.exists() else {}

    render_fig1(geom, suiteA, v21)
    render_fig2(v21)
    render_fig3(v21, v18, v14)
    render_fig4(frozen, geom, v21)
    render_fig5(v21)
    render_fig6(v21, v19, loc_quot)
    render_fig7(v20, v21, v22)
    render_fig8(v20, v21, v22)
    render_fig9(v20, v21, v22, v23, ext_geom)


if __name__ == "__main__":
    main()
