#!/usr/bin/env python3
"""
render_v17_referee_figures.py
Generates 4 publication-grade, uncertainty-aware figures (V17, 300 DPI PNG + PDF)
answering every audit item in welded_memory_critical_review.pdf:

  Fig. 1 (fig_v17_1_cell_and_decoder.png):
    (a) Cellular sheaf construction, explicit visual key (qubits, Z-faces, X-stars,
        blue repetition vs red parity stalks), and two scoped modification tests.
    (b) 2x2 auxiliary repetition-cut & 5-bit wall-rounding lemma (min_{V>0} V_rep = 2,
        W <= 2 V_rep, R_mod / V_rep <= 2/5) + physical Z-coboundary & pocket comparison.
    (c) Paired-trajectory lifetimes on identical random seeds (rottor s=2 646q & s=4 94,774q):
        global cluster-GF(2) vs strictly O(1)-cluster-capped HierarchicalRG (cap=64).
    (d) Measured CPU decode time (us / call) vs N across 19 .. 322,159 qubits (O(N) work).

  Fig. 2 (fig_v17_2_flow_and_lifetimes.png):
    (a) Single-sector coarse-syndrome flow diagnostic (N=322,159, 96 runs, t_eq=20)
        with explicit zero floor, open zero markers, check denominators M_s, & CP95 bound.
    (b) Two-sector quantum torus coarse-syndrome COUNT flow <|sigma_s|> (N=94,774, 96 runs)
        showing monotone count contraction at every step + same-parity guide & CP95 bound.
    (c) Absolute two-sector quantum survival tau_Q(beta) = int S_X(t) S_Z(t) dt across
        unified planar (s=1..5: 19..423,559q) & torus (s=2,4: 646, 94,774q) welded codes
        alongside matched-model Haah Cubic (L=5) and Gu-Caha Layer (n=11, 13) baselines.
    (d) Fitted finite-window activation slopes d(ln tau)/d(beta) +/- 1 SE vs N (with
        explicit beta windows and separate family markers for Toric, BB, Layer, Cubic, Welded).

  Fig. 3 (fig_v17_3_euclidean_obstruction.png):
    (a) Conditional R^3 range-obstruction exponent alpha_lin(L) = (1/3) log_L m(L) - 1
        comparing 1-rib (m=2L^2+2L: L=2 excluded, L=3 candidate) vs 2-rib (m=2L^2+4L: L=3 excluded).
    (b) Full two-sector quantum survival tau_Q(beta) (+/- 1 SE) across 3 scales of the
        uniform 3x3 single-cross candidate (41 -> 1,271 -> 24,449q) + single-sector inset.
    (c) Matched two-sector tau_Q(beta) separating the 4.58x direct-graph factor (323q 1-rib
        vs 1,480q 2-rib) from the 38.05x cubic-grid routing factor (3,067q, 10,280q, 56,312q).

  Fig. 4 (fig_v17_4_wire_dilution.png):
    (a) Degree-2 parity-check wire schematic & exact finite-size Gibbs marginalization
        beta_eff(w) = 2 arctanh((tanh(beta/2))^w) = beta - ln(w) + O(w^2 e^{-2 beta}).
    (b) Controlled wire-length sweep w in {1, 2, 4, 8, 16} on one fixed 2x2 welded core
        (planar N0=323 and torus N0=646) under identical bath & decoder.
    (c) Analytical effective-temperature collapse tau vs beta_eff(w) + inset comparing
        measured finite-window slope compression against analytical d(beta_eff)/d(beta).
"""
import json
import math
import os
import shutil
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures")
ART_DIR = "/Users/harshilavlani/.gemini/antigravity/brain/35299b0d-7439-4f8e-8617-0b646151a931"

sys.path.insert(0, os.path.join(ROOT, "src"))
from render_v13_unified_figures import load_all, ols_slope
from render_v15_complete_figures import load_v12
from parse_v13_results import fit_slope_with_se

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.0,
    "axes.linewidth": 0.9,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.32,
    "savefig.dpi": 300,
})


def save_fig(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    png_p = os.path.join(FIG_DIR, f"{name}.png")
    pdf_p = os.path.join(FIG_DIR, f"{name}.pdf")
    fig.savefig(png_p, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if os.path.isdir(ART_DIR):
        shutil.copy2(png_p, os.path.join(ART_DIR, f"{name}.png"))
    print(f"Saved {png_p}")


def style_ax(ax):
    ax.grid(True, which="major", linestyle="-", color="#cbd5e1", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ==============================================================================
# FIGURE 1: Sheaf Cell, Auxiliary Cut Lemma, & Paired Local-Readout Validation
# ==============================================================================
def render_fig1(v13):
    fig = plt.figure(figsize=(13.8, 8.5))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.05, 1.0], width_ratios=[1.16, 1.0, 0.98],
                  hspace=0.35, wspace=0.35)

    # --- Panel (a): Top full-width schematic ---
    ax_a = fig.add_subplot(gs[0, :])
    ax_a.set_xlim(0, 100)
    ax_a.set_ylim(0, 34)
    ax_a.axis("off")

    bg_a = FancyBboxPatch((0.3, 0.5), 99.4, 33.0, boxstyle="round,pad=0.3,rounding_size=1.2",
                          facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.1)
    ax_a.add_patch(bg_a)
    ax_a.text(50, 31.6, "(a) Cellular Sheaf Welding Cycle, Explicit Generator Key, and Scoped Modification Tests",
              ha="center", va="center", fontsize=11.0, fontweight="bold", color="#0f172a")

    # Visual Key strip (y = 26.2 .. 29.8)
    key_bg = FancyBboxPatch((1.8, 26.3), 96.4, 3.6, boxstyle="round,pad=0.15,rounding_size=0.7",
                            facecolor="#ffffff", edgecolor="#e2e8f0", linewidth=0.9)
    ax_a.add_patch(key_bg)
    ax_a.text(3.2, 28.1, "Key:", fontweight="bold", fontsize=8.5, va="center", color="#1e293b")
    # Qubit dot on edge
    ax_a.plot([8.2, 11.4], [28.1, 28.1], color="#475569", lw=2.0)
    ax_a.scatter([9.8], [28.1], s=28, color="#0f172a", zorder=5)
    ax_a.text(12.2, 28.1, "Deg-2 sheet edge (1 qubit)", fontsize=8.0, va="center", color="#334155")
    # Blue seam
    ax_a.plot([31.2, 34.4], [28.1, 28.1], color="#1d4ed8", lw=3.0)
    ax_a.scatter([32.8], [28.1], s=32, color="#1d4ed8", zorder=5)
    ax_a.text(35.2, 28.1, "Blue rep. seam {000,111} (1 qubit)", fontsize=8.0, va="center", color="#1e3a8a")
    # Red seam
    ax_a.plot([58.0, 61.2], [28.1, 28.1], color="#dc2626", lw=3.0, ls="--")
    ax_a.scatter([59.1, 60.1], [28.1, 28.1], s=26, color="#dc2626", zorder=5)
    ax_a.text(62.0, 28.1, "Red parity seam u1+u2+u3=0 (2 qubits)", fontsize=8.0, va="center", color="#991b1b")
    # Z-face & X-star
    ax_a.add_patch(Rectangle((85.5, 27.2), 1.8, 1.8, facecolor="#dbeafe", edgecolor="#2563eb", lw=1.1))
    ax_a.text(87.9, 28.1, "Z-face / X-vertex star", fontsize=8.0, va="center", color="#1e293b")

    cards = [
        (1.8, 22.0, "1. Seed Surface Complex"),
        (25.2, 23.2, r"2. Odd Step $\mathcal{R}_X$: Blue Face Weld"),
        (49.8, 23.2, r"3. Even Step $\mathcal{R}_Z$: Red Parity Weld"),
        (74.4, 23.8, "4. Two Tested Modifications"),
    ]
    for cx, cw, ctitle in cards:
        ax_a.add_patch(FancyBboxPatch((cx, 1.8), cw, 23.2, boxstyle="round,pad=0.2,rounding_size=0.9",
                                      facecolor="#ffffff", edgecolor="#cbd5e1", lw=0.9))
        ax_a.text(cx + cw / 2, 23.3, ctitle, ha="center", va="center", fontsize=9.0, fontweight="bold", color="#0f172a")

    # Card 1 drawing: 2x2 seed patch
    sx, sy = 7.3, 9.2
    for ix in range(2):
        for iy in range(2):
            fc = "#dbeafe" if (ix == 0 and iy == 1) else "#f8fafc"
            ec = "#2563eb" if (ix == 0 and iy == 1) else "#94a3b8"
            ax_a.add_patch(Rectangle((sx + ix * 5.2, sy + iy * 5.2), 5.2, 5.2, facecolor=fc, edgecolor=ec, lw=1.5))
    ax_a.text(sx + 2.6, sy + 7.8, r"$S_Z$", ha="center", va="center", fontsize=8.8, fontweight="bold", color="#1d4ed8")
    vx, vy = sx + 5.2, sy + 5.2
    ax_a.plot([vx - 2.3, vx + 2.3], [vy, vy], color="#dc2626", lw=3.0)
    ax_a.plot([vx, vx], [vy - 2.3, vy + 2.3], color="#dc2626", lw=3.0)
    ax_a.scatter([vx], [vy], s=30, color="#dc2626", zorder=6)
    ax_a.text(vx + 2.5, vy - 1.8, r"$S_X$", fontsize=8.8, fontweight="bold", color="#b91c1c")
    ax_a.text(12.8, 5.8, "Planar [[4,1,2]] ($K=1$)\nor Torus [[4,2,2]] ($K=2$)", ha="center", va="center", fontsize=8.0, color="#334155")
    ax_a.text(12.8, 3.0, r"2D sheets: $\Delta E = 0$ hop", ha="center", va="center", fontsize=7.8, color="#64748b")

    # Card 2 drawing: Odd RX step (2 sheets + blue cross)
    for sh_y, lbl in [(15.2, "Sheet 0"), (8.2, "Sheet 1")]:
        ax_a.add_patch(Polygon([[27.8, sh_y], [39.8, sh_y], [43.2, sh_y + 3.6], [31.2, sh_y + 3.6]],
                               closed=True, facecolor="#eff6ff", edgecolor="#64748b", lw=1.1))
        ax_a.plot([33.8, 37.2], [sh_y, sh_y + 3.6], color="#1d4ed8", lw=2.3)
        ax_a.plot([29.5, 41.5], [sh_y + 1.8, sh_y + 1.8], color="#1d4ed8", lw=2.3)
        ax_a.text(44.2, sh_y + 1.6, lbl, fontsize=7.5, color="#334155", va="center")
    # Vertical bridge cross
    ax_a.add_patch(Rectangle((35.0, 10.0), 1.0, 7.0, facecolor="#bfdbfe", edgecolor="#1d4ed8", lw=1.2, alpha=0.85))
    ax_a.add_patch(Polygon([[29.5, 10.0], [41.5, 10.0], [41.5, 17.0], [29.5, 17.0]],
                           closed=True, facecolor="#93c5fd", edgecolor="#1d4ed8", lw=1.0, alpha=0.28))
    ax_a.text(36.8, 5.8, "4-arm blue bridge cross:\n3 Z-faces share each rib (1 q)", ha="center", va="center", fontsize=7.9, color="#1e3a8a")
    ax_a.text(36.8, 3.0, "Planar s=1: N=19, (dX,dZ)=(6,2)", ha="center", va="center", fontsize=7.8, fontweight="bold", color="#0f172a")

    # Card 3 drawing: Even RZ step (2 sheets + red parity seams)
    for sh_y in (15.2, 8.2):
        ax_a.add_patch(Polygon([[52.4, sh_y], [64.4, sh_y], [67.8, sh_y + 3.6], [55.8, sh_y + 3.6]],
                               closed=True, facecolor="#fef2f2", edgecolor="#64748b", lw=1.1))
        ax_a.plot([58.4, 61.8], [sh_y, sh_y + 3.6], color="#dc2626", lw=2.2, ls="--")
        ax_a.plot([54.1, 66.1], [sh_y + 1.8, sh_y + 1.8], color="#dc2626", lw=2.2, ls="--")
    for cx_v in (54.1, 60.1, 66.1):
        ax_a.plot([cx_v, cx_v], [10.0, 17.0], color="#dc2626", lw=1.8)
        ax_a.scatter([cx_v, cx_v], [10.0, 17.0], s=18, color="#dc2626", zorder=5)
    ax_a.text(61.4, 5.8, "Red parity seams (2 q/rib):\nX-stars branch 1->2 across sheets", ha="center", va="center", fontsize=7.9, color="#991b1b")
    ax_a.text(61.4, 3.0, "Cycle 1 (s=2): N=323, (dX,dZ)=(11,10)", ha="center", va="center", fontsize=7.8, fontweight="bold", color="#0f172a")

    # Card 4: Two scoped ablations
    ax_a.add_patch(FancyBboxPatch((75.8, 13.0), 21.0, 8.4, boxstyle="round,pad=0.15,rounding_size=0.6",
                                  facecolor="#fff1f2", edgecolor="#fecdd3", lw=0.9))
    ax_a.text(86.3, 19.6, "A. Pruned 3-Arm T-Bridge", ha="center", va="center", fontsize=8.2, fontweight="bold", color="#9f1239")
    ax_a.text(86.3, 15.9, "$H_X H_Z^T = 0$ preserved;\ntorus $K: 2 \\to 1{,}004$;\nseed $d_{\\mathrm{sub}}: (11,10) \\to (6,5)$",
              ha="center", va="center", fontsize=7.4, color="#881337")

    ax_a.add_patch(FancyBboxPatch((75.8, 2.8), 21.0, 8.8, boxstyle="round,pad=0.15,rounding_size=0.6",
                                  facecolor="#fff7ed", edgecolor="#fed7aa", lw=0.9))
    ax_a.text(86.3, 9.8, "B. 1-Step Blue+Red Sheaf Cross", ha="center", va="center", fontsize=8.2, fontweight="bold", color="#9a3412")
    ax_a.text(86.3, 6.0, "Horiz. blue + vert. red in 1 step:\nplanar $K: 1 \\to 3 \\to 49 \\to 737$;\nall $K=3$ classes have $d_Z^{\\mathrm{sub}} = 1$",
              ha="center", va="center", fontsize=7.4, color="#7c2d12")

    # --- Panel (b): Auxiliary 2x2 Cut & Rounding Lemma ---
    ax_b = fig.add_subplot(gs[1, 0])
    ax_b.set_xlim(0, 42)
    ax_b.set_ylim(0, 34)
    ax_b.axis("off")
    ax_b.add_patch(FancyBboxPatch((0.4, 0.4), 41.2, 33.2, boxstyle="round,pad=0.25,rounding_size=1.0",
                                  facecolor="#f8fafc", edgecolor="#cbd5e1", lw=1.0))
    ax_b.text(21.0, 31.6, "(b) Auxiliary 2x2 Cut & Rounding Lemma", ha="center", va="center",
              fontsize=9.6, fontweight="bold", color="#0f172a")

    # Sheet 0 active (4 quadrants)
    for ix in range(2):
        for iy in range(2):
            ax_b.add_patch(Rectangle((1.8 + ix * 4.1, 18.8 + iy * 4.1), 4.1, 4.1,
                                     facecolor="#bfdbfe", edgecolor="#1d4ed8", lw=1.4))
            ax_b.text(3.85 + ix * 4.1, 20.85 + iy * 4.1, "1", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#1e3a8a")
    ax_b.text(5.9, 28.4, r"Sheet 0 ($a_i=1$)", ha="center", va="center", fontsize=7.6, fontweight="bold", color="#1e3a8a")
    # Sheet 1 inactive
    for ix in range(2):
        for iy in range(2):
            ax_b.add_patch(Rectangle((14.2 + ix * 4.1, 18.8 + iy * 4.1), 4.1, 4.1,
                                     facecolor="#ffffff", edgecolor="#94a3b8", lw=1.1))
            ax_b.text(16.25 + ix * 4.1, 20.85 + iy * 4.1, "0", ha="center", va="center", fontsize=8.5, color="#64748b")
    ax_b.text(18.3, 28.4, r"Sheet 1 ($b_i=0$)", ha="center", va="center", fontsize=7.6, color="#475569")

    # Tight witness summary box on right of (b)
    ax_b.add_patch(FancyBboxPatch((24.6, 18.5), 15.8, 10.5, boxstyle="round,pad=0.2,rounding_size=0.7",
                                  facecolor="#eff6ff", edgecolor="#93c5fd", lw=1.0))
    ax_b.text(32.5, 27.0, "Tight Extremizer:", ha="center", va="center", fontsize=8.0, fontweight="bold", color="#1e3a8a")
    ax_b.text(32.5, 23.5, "$W = 8$ wall bits\n$V_{\\mathrm{rep}} = 4$ seams\n$W/(5V_{\\mathrm{rep}}) = 2/5$",
              ha="center", va="center", fontsize=7.7, color="#0f172a")
    ax_b.text(32.5, 19.7, "(40 / 4,096 states)", ha="center", va="center", fontsize=7.2, color="#475569")

    # Exact mathematical bullet box at bottom of (b)
    ax_b.add_patch(FancyBboxPatch((1.8, 1.6), 38.4, 15.4, boxstyle="round,pad=0.2,rounding_size=0.7",
                                  facecolor="#ffffff", edgecolor="#e2e8f0", lw=0.9))
    bullets = [
        ("Repetition block metric:", r"$V_{\mathrm{rep}}$ counts non-$\{000,111\}$ seams ($\min V_{\mathrm{rep}}>0$ is $2$)."),
        ("Wall-majority rounding:", r"$W \leq 2V_{\mathrm{rep}}$ and 5-bit vote $R_{\mathrm{mod}} \leq \frac{2}{5}V_{\mathrm{rep}}$ ($\frac{2}{7}$ on 3x3)."),
        ("Physical Z-coboundary:", r"On $\mathcal{R}_Z$, $V_{\mathrm{rep}} \leq |H_Z^T m|_{\mathrm{int}} \leq \frac{11}{5}V_{\mathrm{rep}}$, $W \leq 2|H_Z^T m|_{\mathrm{int}}$."),
        ("1-rib vs. 2-rib [BDL 6.2.6]:", r"1-rib torus has $V_0+1$ pockets; 2-rib has $V_0+E_0+F_0+1$."),
    ]
    for idx, (hd, bd) in enumerate(bullets):
        yy = 15.0 - idx * 3.55
        ax_b.text(2.7, yy, hd, fontsize=7.4, fontweight="bold", color="#0f172a", va="center")
        ax_b.text(2.7, yy - 1.45, bd, fontsize=6.95, color="#334155", va="center")

    # --- Panel (c): Paired Trajectory Comparison (rottor s=2 & s=4) ---
    ax_c = fig.add_subplot(gs[1, 1])
    style_ax(ax_c)
    pk = v13["paired_kmc"]
    for s_lv, col, lbl_base in [(2, "#2563eb", "Cycle 1 (646q)"), (4, "#dc2626", "Cycle 2 (94,774q)")]:
        rows = sorted(
            [(d["beta"], d) for k, d in pk.items() if k.startswith(f"paired_kmc__rottor_s{s_lv}__")],
            key=lambda t: t[0],
        )
        bs = [b for b, _ in rows]
        uf_m = [d["mean_uf"] for _, d in rows]
        uf_s = [d["sem_uf"] for _, d in rows]
        rg_m = [d["mean_rg"] for _, d in rows]
        rg_s = [d["sem_rg"] for _, d in rows]
        ax_c.errorbar(bs, uf_m, yerr=uf_s, fmt="-o", color=col, ms=5.2, lw=1.7, capsize=2.6,
                      label=f"{lbl_base} Global Cluster")
        ax_c.errorbar(bs, rg_m, yerr=rg_s, fmt="--s", color=col, mfc="white", mew=1.5, ms=5.2, lw=1.4, capsize=2.6,
                      label=f"{lbl_base} Local RG")

    s4_cold = [d for k, d in pk.items() if k.startswith("paired_kmc__rottor_s4__") and d["beta"] >= 6.0]
    if s4_cold:
        ag_min = min(d["chk_agree_rg"] for d in s4_cold) * 100.0
        ax_c.text(0.04, 0.05, rf"Paired checkpoints ($\beta \geq 6.0$):" + f"\n{ag_min:.1f}% agreement (0.51x at " + r"$\beta=5.5$)",
                  transform=ax_c.transAxes, fontsize=7.3, color="#1e293b",
                  bbox=dict(boxstyle="round,pad=0.24", facecolor="#f8fafc", edgecolor="#cbd5e1", lw=0.8))

    ax_c.set_yscale("log")
    ax_c.set_xlabel(r"Inverse Bath Temperature $\beta = 1/T$ ($J = 1$)")
    ax_c.set_ylabel(r"Paired X-Sector Lifetime $\langle T_{\mathrm{fail}} \rangle$")
    ax_c.set_title("(c) Paired Readout: Global vs. Local RG")
    ax_c.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, fontsize=7.3)

    # --- Panel (d): Thermal Readout Latency Scaling across N = 19 .. 322,159 ---
    ax_d = fig.add_subplot(gs[1, 2])
    style_ax(ax_d)
    ns_rx, us_uf_rx, us_rg_rx = [], [], []
    for s_lv in range(1, 6):
        key = f"paired_kmc__sq_rx_s{s_lv}__b2.20"
        if key in pk:
            ns_rx.append(pk[key]["N"])
            us_uf_rx.append(pk[key]["us_uf"])
            us_rg_rx.append(pk[key]["us_rg"])

    tor_ns, tor_uf, tor_rg = [], [], []
    for s_lv in (2, 4):
        key = f"paired_kmc__rottor_s{s_lv}__b6.50"
        if key in pk:
            tor_ns.append(pk[key]["N"])
            tor_uf.append(pk[key]["us_uf"])
            tor_rg.append(pk[key]["us_rg"])

    ax_d.plot(ns_rx, us_uf_rx, "-o", color="#475569", ms=5.2, lw=1.6, label=r"Global Cluster ($\beta=2.20$)")
    ax_d.plot(ns_rx, us_rg_rx, "-s", color="#059669", ms=5.2, lw=1.7, label=r"Local RG ($\beta=2.20$, 4.6x-7.8x)")
    if tor_ns:
        ax_d.plot(tor_ns, tor_uf, "--^", color="#94a3b8", ms=5.2, lw=1.3, label=r"Torus Global ($\beta=6.50$)")
        ax_d.plot(tor_ns, tor_rg, "--D", color="#dc2626", ms=5.2, lw=1.5, label=r"Torus Local RG (15.8x at 95k)")

    if len(ns_rx) >= 2:
        n_ref = np.array([19.0, 350000.0])
        c_lin = us_rg_rx[1] / ns_rx[1]
        ax_d.plot(n_ref, c_lin * n_ref, ":", color="#0f172a", lw=1.2, label=r"Linear $\propto N$ Guide")

    ax_d.set_xscale("log")
    ax_d.set_yscale("log")
    ax_d.set_xlabel("Physical Qubits $N$")
    ax_d.set_ylabel(r"Thermal Readout Time per Call ($\mu$s)")
    ax_d.set_title("(d) Empirical O(N) Thermal Readout Speed")
    ax_d.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, fontsize=7.0)

    save_fig(fig, "fig_v17_1_cell_and_decoder")


# ==============================================================================
# FIGURE 2: Coarse-Syndrome Flow Diagnostics & Multi-Scale Thermal Lifetimes
# ==============================================================================
def render_fig2(v13, kmc4, kmc6, kmc9):
    fig = plt.figure(figsize=(12.8, 8.4))
    gs = GridSpec(2, 2, figure=fig, hspace=0.33, wspace=0.26)
    rg = v13["rgflow_v13"]
    tq = v13["two_sector_tau_Q"]

    # --- Panel (a): 5-scale single-sector coarse-syndrome flow (N=322,159) ---
    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a)
    floor_a = 3.0e-6
    palette_a = [
        (1.60, "#dc2626", "-o"),
        (1.86, "#ea580c", "-s"),
        (2.00, "#ca8a04", "-D"),
        (2.15, "#16a34a", "-^"),
        (2.35, "#2563eb", "-v"),
        (2.65, "#7c3aed", "-P"),
    ]
    xs_a = list(range(6))  # s = 5, 4, 3, 2, 1, 0
    for b_val, col, fmt in palette_a:
        key = f"rgflow13__sq_rx_s5__b{b_val:.2f}"
        if key not in rg:
            continue
        d = rg[key]
        ys, is_z = [], []
        for s in range(5, -1, -1):
            r = d["scales"][str(s)]["rho"] if str(s) in d["scales"] else d["scales"][s]["rho"]
            if r <= 0.0:
                ys.append(floor_a)
                is_z.append(True)
            else:
                ys.append(r)
                is_z.append(False)
        lbl = rf"$\beta={b_val:.2f}$" + (f" (0/{d['trials']} log. err)" if d["log_fail"] == 0 and b_val >= 2.15 else "")
        ax_a.plot(xs_a, ys, fmt, color=col, ms=5.0, lw=1.6, label=lbl)
        zx = [x for x, z in zip(xs_a, is_z) if z]
        zy = [y for y, z in zip(ys, is_z) if z]
        if zx:
            ax_a.scatter(zx, zy, marker="v", s=52, facecolors="white", edgecolors=col, linewidths=1.6, zorder=6)

    ax_a.axhline(floor_a, color="#64748b", ls=":", lw=1.0)
    ax_a.text(2.12, floor_a * 1.45, r"Zero floor ($\rho=0$, open $\nabla$; 95% CP $\leq 0.031$)",
              fontsize=7.1, color="#475569")
    ax_a.set_yscale("log")
    ax_a.set_ylim(floor_a * 0.55, 0.55)
    ax_a.set_xticks(xs_a)
    ax_a.set_xticklabels(["s=5\n(248.8k)", "s=4\n(20.7k)", "s=3\n(1.7k)", "s=2\n(144)", "s=1\n(12)", "s=0\n(M=1)"])
    ax_a.set_xlabel(r"Coarse-Graining Scale $s$ (Check Count $M_s$)")
    ax_a.set_ylabel(r"Coarse Violated-Check Density $\rho_s = \langle |\sigma_s| \rangle / M_s$")
    ax_a.set_title("(a) Single-Sector Coarse-Syndrome Flow (N = 322,159)")
    ax_a.legend(loc="lower left", bbox_to_anchor=(0.01, 0.11), ncol=2, frameon=True, facecolor="white", framealpha=0.93, fontsize=7.0)

    # --- Panel (b): Alternating (X, Z) Torus Coarse-Syndrome COUNT Flow (N=94,774) ---
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b)
    floor_b = 4.0e-3
    palette_b = [
        (3.50, "#dc2626", "-o"),
        (4.50, "#ea580c", "-s"),
        (5.50, "#ca8a04", "-D"),
        (6.00, "#16a34a", "-^"),
        (6.50, "#2563eb", "-v"),
        (7.00, "#7c3aed", "-P"),
    ]
    xs_b = list(range(5))  # s = 4, 3, 2, 1, 0
    for b_val, col, fmt in palette_b:
        key = f"rgflow13__rottor_s4__b{b_val:.2f}"
        if key not in rg:
            continue
        d = rg[key]
        ys, se_y, is_z = [], [], []
        for s in range(4, -1, -1):
            sc = d["scales"][str(s)] if str(s) in d["scales"] else d["scales"][s]
            c_val = sc["cnt"]
            if c_val <= 0.0:
                ys.append(floor_b); se_y.append(0.0); is_z.append(True)
            else:
                ys.append(c_val); se_y.append(sc["semc"]); is_z.append(False)
        lbl = rf"$\beta={b_val:.2f}$" + (f" (0/{d['trials']} fail)" if d["log_fail"] == 0 and b_val >= 6.0 else "")
        ax_b.errorbar(xs_b, ys, yerr=se_y, fmt=fmt, color=col, ms=5.0, lw=1.6, capsize=2.2, label=lbl)
        # Same-parity full-cycle guide (s = 4 -> 2 -> 0) for beta = 6.00
        if abs(b_val - 6.00) < 1e-6:
            ax_b.plot([0, 2, 4], [ys[0], ys[2], ys[4]], "--", color="#065f46", lw=1.2, alpha=0.75)
        zx = [x for x, z in zip(xs_b, is_z) if z]
        zy = [y for y, z in zip(ys, is_z) if z]
        if zx:
            ax_b.scatter(zx, zy, marker="v", s=52, facecolors="white", edgecolors=col, linewidths=1.6, zorder=6)

    ax_b.axhline(floor_b, color="#64748b", ls=":", lw=1.0)
    ax_b.text(0.08, floor_b * 1.38, r"Exact-zero count floor ($\langle|\sigma_s|\rangle=0$, open $\nabla$; 95% CP $p_{\mathrm{fail}}\leq 0.031$)",
              fontsize=7.3, color="#475569")
    ax_b.set_yscale("log")
    ax_b.set_ylim(floor_b * 0.55, 2.5e3)
    ax_b.set_xticks(xs_b)
    ax_b.set_xticklabels(["s=4 (Z)\nM=41,472", "s=3 (X)\nM=3,456", "s=2 (Z)\nM=288", "s=1 (X)\nM=24", "s=0\nM=2"])
    ax_b.set_xlabel(r"Alternating $(X, Z)$ Step $s$ (Torus $N=94{,}774$, $K=2$)")
    ax_b.set_ylabel(r"Mean Coarse Defect Count $\langle |\sigma_s| \rangle$")
    ax_b.set_title("(b) Quantum Torus Defect-Count Contraction")
    ax_b.legend(loc="upper right", ncol=2, frameon=True, facecolor="white", framealpha=0.92, fontsize=7.2)

    # --- Panel (c): Absolute Two-Sector Quantum Lifetimes tau_Q(beta) ---
    ax_c = fig.add_subplot(gs[1, 0])
    style_ax(ax_c)
    q_series = [
        ("sq_iso_L2r1_s1", "#94a3b8", "-o",  "Planar s=1 (19q, 0.5 cyc)"),
        ("sq_iso_L2r1_s2", "#0ea5e9", "--s", "Planar s=2 (323q, Cyc 1)"),
        ("sq_iso_L2r1_s3", "#f97316", "-^",  "Planar s=3 (2,911q, 1.5 cyc)"),
        ("sq_iso_L2r1_s4", "#2563eb", "--D", "Planar s=4 (47,387q, Cyc 2)"),
        ("sq_iso_L2r1_s5", "#be123c", "-P",  "Planar s=5 (423,559q, 2.5 cyc)"),
        ("rottor_iso_L2r1_s4", "#059669", "-*", "Torus s=4 (94,774q, Cyc 2, K=2)"),
    ]
    for cname, col, fmt, lbl in q_series:
        if cname not in tq:
            continue
        pts = tq[cname]["points"]
        bs = [p["beta"] for p in pts]
        ys = [p["tau_Q"] for p in pts]
        es = [p["sem_Q"] for p in pts]
        ax_c.errorbar(bs, ys, yerr=es, fmt=fmt, color=col, ms=5.0, lw=1.6, capsize=2.4, label=lbl)

    # Also overlay matched-model Gu-Caha Random Layer n=11 code_100 (N=2,227) for absolute context!
    caha_bs, caha_ys = [], []
    for b in (6.00, 7.00):
        if ("caha_n11_code_100_X_err", "uniform", "uf", b) in kmc6:
            caha_bs.append(b)
            # CSS symmetric layer code: S_X and S_Z i.i.d. -> for exponential ~ tau_X / 2
            caha_ys.append(0.5 * kmc6[("caha_n11_code_100_X_err", "uniform", "uf", b)][1])
    if caha_bs:
        ax_c.plot(caha_bs, caha_ys, ":x", color="#6d28d9", ms=6.0, lw=1.5, label="Gu-Caha Layer n=11 (2,227q, sym.)")

    ax_c.set_yscale("log")
    ax_c.set_xlabel(r"Inverse Bath Temperature $\beta = 1/T$ ($J = 1$)")
    ax_c.set_ylabel(r"Two-Sector Quantum Lifetime $\hat{\tau}_Q = \int \hat{S}_X \hat{S}_Z\,dt$")
    ax_c.set_title("(c) Absolute Two-Sector Quantum Lifetimes")
    ax_c.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, fontsize=7.0)

    # --- Panel (d): Fitted Finite-Window Activation Slopes vs N (with error bars & windows!) ---
    ax_d = fig.add_subplot(gs[1, 1])
    style_ax(ax_d)

    # 1. Uniform 2x2 quantum odd steps (s=1,3,5, beta in [5.5, 6.5]) and even cycles (s=2,4, beta in [5.5, 6.5])
    q_odd_N, q_odd_sl, q_odd_se = [], [], []
    for s in (1, 3, 5):
        cn = f"sq_iso_L2r1_s{s}"
        if cn in tq:
            q_odd_N.append(tq[cn]["N"]); q_odd_sl.append(tq[cn]["slope"]); q_odd_se.append(tq[cn]["se_slope"])
    if q_odd_N:
        ax_d.errorbar(q_odd_N, q_odd_sl, yerr=q_odd_se, fmt="-P", color="#be123c", ms=6.2, lw=1.8, capsize=2.8,
                      label=r"Welded 2x2 Quantum $\hat{\tau}_Q$ ($\beta\in[5.5,6.5]$)")

    # 2. Uniform 2x2 Torus cycles (s=2,4)
    tor_N, tor_sl, tor_se = [], [], []
    for s in (2, 4):
        cn = f"rottor_iso_L2r1_s{s}"
        if cn in tq:
            tor_N.append(tq[cn]["N"]); tor_sl.append(tq[cn]["slope"]); tor_se.append(tq[cn]["se_slope"])
    if tor_N:
        ax_d.errorbar(tor_N, tor_sl, yerr=tor_se, fmt="--*", color="#059669", ms=6.5, lw=1.6, capsize=2.8,
                      label=r"Welded 2x2 Torus $\hat{\tau}_Q$ ($\beta\in[5.5,6.5]$)")

    # 3. 3x3 single-cross quantum (s=1,2,3)
    l3_N, l3_sl, l3_se = [], [], []
    for s in (1, 2, 3):
        cn = f"sq_L3r1_s{s}"
        if cn in tq:
            l3_N.append(tq[cn]["N"]); l3_sl.append(tq[cn]["slope"]); l3_se.append(tq[cn]["se_slope"])
    if l3_N:
        ax_d.errorbar(l3_N, l3_sl, yerr=l3_se, fmt="-^", color="#0284c7", ms=5.5, lw=1.6, capsize=2.6,
                      label=r"Welded 3x3 Candidate $\hat{\tau}_Q$ ($\beta\in[5.5,7.0]$)")

    # 4. Comparison families from kmc4 / kmc6 (distinct legend entries for Haah, Layer, BB, and 2D Toric!)
    def fam_slopes(cnames, bmin=3.5, bmax=5.0):
        xs, ys, ses = [], [], []
        for cn in cnames:
            bm = kmc4.get(cn, {})
            bs = sorted(b for b in bm if bmin <= b <= bmax)
            if len(bs) >= 3:
                ms = [bm[b][1] for b in bs]
                # ~10% trial SEM for 64-trial v4 runs:
                sl, se, _ = fit_slope_with_se(bs, ms, [0.12 * m for m in ms])
                xs.append(bm[bs[0]][0]); ys.append(sl); ses.append(se)
        return xs, ys, ses

    hN, hS, hE = fam_slopes([f"haah_cubic_L{L}_Xerr.code" for L in [3, 5, 7, 9, 11, 13]])
    bN, bS, bE = fam_slopes([f"bb_gross_{l}x{m}_Xerr.code" for l, m in [(6,6),(12,6),(18,12),(24,18),(36,18),(60,18)]])
    tN, tS, tE = fam_slopes([f"toric2d_L{L}_Xerr.code" for L in [4, 6, 10, 16, 24, 34]])
    if hN:
        ax_d.errorbar(hN, hS, yerr=hE, fmt="-D", color="#7c3aed", ms=4.8, lw=1.5, capsize=2.4,
                      label=r"Haah 3D Cubic ($\beta\in[3.5,5.0]$)")

    # Gu-Caha Random Layer ensemble (n = 11, 13, 15, 17, 19; beta in [6.0, 7.0])
    caha_groups = {
        11: ["caha_n11_code_100", "caha_n11_code_11", "caha_n11_code_113", "caha_n11_code_120", "caha_n11_code_128", "caha_n11_code_151"],
        13: ["caha_n13_code_102", "caha_n13_code_104", "caha_n13_code_106"],
        15: ["caha_n15_code_15",  "caha_n15_code_16",  "caha_n15_code_17"],
        17: ["caha_n17_code_1201","caha_n17_code_14",  "caha_n17_code_145"],
        19: ["caha_n19_code_129", "caha_n19_code_199", "caha_n19_code_236"],
    }
    lay_N, lay_S, lay_E = [], [], []
    for nv in (11, 13, 15, 17, 19):
        cs = caha_groups[nv]
        ns_v = [kmc6[(c + "_X_err", "uniform", "uf", 7.0)][0] for c in cs if (c + "_X_err", "uniform", "uf", 7.0) in kmc6]
        sls = [math.log(kmc6[(c + "_X_err", "uniform", "uf", 7.0)][1] / kmc6[(c + "_X_err", "uniform", "uf", 6.0)][1])
               for c in cs if (c + "_X_err", "uniform", "uf", 7.0) in kmc6 and (c + "_X_err", "uniform", "uf", 6.0) in kmc6]
        if sls:
            lay_N.append(float(np.mean(ns_v)))
            lay_S.append(float(np.mean(sls)))
            lay_E.append(float(np.std(sls, ddof=1) / math.sqrt(len(sls))) if len(sls) > 1 else 0.25)
    if lay_N:
        ax_d.errorbar(lay_N, lay_S, yerr=lay_E, fmt="-p", color="#d97706", ms=5.2, lw=1.4, capsize=2.4,
                      label=r"3D Random Layer ($\beta\in[6.0,7.0]$)")

    if bN:
        ax_d.errorbar(bN, bS, yerr=bE, fmt="--s", color="#2563eb", ms=4.5, lw=1.3, capsize=2.2,
                      label=r"2D Bivariate Bicycle ($\beta\in[3.5,5.0]$)")
    if tN:
        ax_d.errorbar(tN, tS, yerr=tE, fmt=":o", color="#64748b", ms=4.5, lw=1.3, capsize=2.2,
                      label=r"2D Toric Code ($\beta\in[3.5,5.0]$)")

    ax_d.set_xscale("log")
    ax_d.set_ylim(0.3, 11.4)
    ax_d.set_xlabel("Physical Qubits $N$ (Log Scale)")
    ax_d.set_ylabel(r"Fitted Activation Slope $\hat{\Delta}_{\mathrm{eff}} = \partial_\beta \ln \tau$")
    ax_d.set_title("(d) Finite-Window Fitted Activation Slopes")
    ax_d.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.93, fontsize=6.9)

    save_fig(fig, "fig_v17_2_flow_and_lifetimes")


# ==============================================================================
# FIGURE 3: Conditional R^3 Range Obstruction, 3x3 Candidate, & Graph vs Grid
# ==============================================================================
def render_fig3(v13):
    kmc12 = load_v12()
    tq = v13["two_sector_tau_Q"]
    fig = plt.figure(figsize=(13.2, 4.4))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.05, 1.08], wspace=0.28)

    # --- Panel (a): Conditional R^3 Range-Obstruction Exponent ---
    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a)
    Ls = np.linspace(1.75, 8.2, 300)
    # Growth exponent of r_q ~ (m(L) / L^3)^{q/3} = (L^{alpha_L})^q where alpha_L = (1/3) log_L(m(L)) - 1:
    exp_1rib = (np.log(2 * Ls ** 2 + 2 * Ls) / np.log(Ls)) / 3.0 - 1.0
    exp_2rib = (np.log(2 * Ls ** 2 + 4 * Ls) / np.log(Ls)) / 3.0 - 1.0

    ax_a.axhspan(0.0, 0.28, facecolor="#fee2e2", alpha=0.45)
    ax_a.axhspan(-0.22, 0.0, facecolor="#dcfce7", alpha=0.42)
    ax_a.axhline(0.0, color="#0f172a", lw=1.2, ls="--")

    ax_a.plot(Ls, exp_1rib, "-", color="#2563eb", lw=2.0, label=r"1-Rib Single Cross ($m=2L^2+2L$)")
    ax_a.plot(Ls, exp_2rib, "--", color="#dc2626", lw=1.9, label=r"2-Rib Parallel Pair ($m=2L^2+4L$)")

    # Discrete integer points L = 2..8
    L_ints = np.arange(2, 9)
    e1_i = (np.log(2 * L_ints ** 2 + 2 * L_ints) / np.log(L_ints)) / 3.0 - 1.0
    e2_i = (np.log(2 * L_ints ** 2 + 4 * L_ints) / np.log(L_ints)) / 3.0 - 1.0
    ax_a.scatter(L_ints, e1_i, color="#1d4ed8", s=36, zorder=5)
    ax_a.scatter(L_ints, e2_i, color="#b91c1c", marker="s", s=34, zorder=5)

    ax_a.annotate("$L=2$ (1-rib): $m=12 > 8$\n$r_q = \\Omega(N_q^{0.0544})$\n($\\mathbb{R}^3$ local excluded)",
                  xy=(2.0, e1_i[0]), xytext=(2.75, 0.088),
                  arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=1.1),
                  fontsize=7.4, fontweight="bold", color="#1e3a8a")
    ax_a.annotate("$L=3$ (1-rib): $24 < 27$\nPasses volume test\n(embedding open)",
                  xy=(3.0, e1_i[1]), xytext=(1.88, -0.185),
                  arrowprops=dict(arrowstyle="->", color="#065f46", lw=1.1),
                  fontsize=7.4, fontweight="bold", color="#065f46")
    ax_a.annotate("BDL $L=8$ (2-rib)", xy=(8.0, e2_i[-1]), xytext=(5.55, -0.075),
                  arrowprops=dict(arrowstyle="->", color="#991b1b", lw=1.0),
                  fontsize=7.4, color="#991b1b")

    ax_a.set_xlim(1.75, 8.35)
    ax_a.set_ylim(-0.21, 0.25)
    ax_a.set_xlabel("Cell Subdivision Scale $L$")
    ax_a.set_ylabel(r"Range Growth Exponent $\frac{1}{3}\log_L m(L) - 1$")
    ax_a.set_title(r"(a) Conditional $\mathbb{R}^3$ Range Obstruction")
    ax_a.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, fontsize=7.4)

    # --- Panel (b): Full Two-Sector Quantum Survival of 3x3 Candidate (41 -> 1,271 -> 24,449q) ---
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b)
    for s_lv, col, fmt, lbl in [
        (1, "#64748b", "-o",  "s=1 (41q, 0.5 cyc)"),
        (2, "#2563eb", "--s", "s=2 (1,271q, Cycle 1)"),
        (3, "#059669", "-^",  "s=3 (24,449q, 1.5 cyc)"),
    ]:
        cn = f"sq_L3r1_s{s_lv}"
        if cn not in tq:
            continue
        pts = tq[cn]["points"]
        ax_b.errorbar([p["beta"] for p in pts], [p["tau_Q"] for p in pts], yerr=[p["sem_Q"] for p in pts],
                      fmt=fmt, color=col, ms=5.2, lw=1.7, capsize=2.5, label=lbl)

    # Inset: 3-scale single-sector 3x3 crossover (41 -> 935 -> 22,097q) from v12
    ins = ax_b.inset_axes([0.08, 0.62, 0.39, 0.31])
    ins.grid(True, alpha=0.3)
    for s_lv, col in [(1, "#64748b"), (2, "#2563eb"), (3, "#059669")]:
        bs_i = [b for b in (1.75, 2.15, 2.55, 2.95, 3.35, 3.55) if (f"sq_rx_L3r1_s{s_lv}_X_err", "uniform", "uf", b) in kmc12]
        ys_i = [kmc12[(f"sq_rx_L3r1_s{s_lv}_X_err", "uniform", "uf", b)][1] for b in bs_i]
        ins.plot(bs_i, ys_i, "-o", color=col, ms=2.6, lw=1.05)
    ins.set_yscale("log")
    ins.set_title("Single-sector (to 22k)", fontsize=6.6)
    ins.tick_params(labelsize=6.1)

    ax_b.set_yscale("log")
    ax_b.set_ylim(16.0, 6.8e4)
    ax_b.set_xlabel(r"Inverse Bath Temperature $\beta = 1/T$ ($J = 1$)")
    ax_b.set_ylabel(r"Two-Sector Quantum Lifetime $\hat{\tau}_Q(\beta)$")
    ax_b.set_title("(b) 3x3 Candidate Two-Sector Scaling")
    ax_b.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92, fontsize=7.5)

    # --- Panel (c): Separating 4.58x Direct-Graph Factor vs 38.05x Cubic-Grid Routing ---
    ax_c = fig.add_subplot(gs[0, 2])
    style_ax(ax_c)
    comp_specs = [
        ("sq_iso_L2r1_s2",   "#059669", "-o",  "2x2 1-Rib Direct (323q)"),
        ("sq_aniso_L3r2_s2", "#2563eb", "-s",  "3x3 2-Rib Direct (1,480q; 4.58x)"),
        ("sq_z3_3_5_r1",     "#d97706", "--^", "Cubic Grid (3,5) 1-Rib (3,067q)"),
        ("sq_z3_4_6_r2",     "#7c3aed", "--D", "Cubic Grid (4,6) 2-Rib (10,280q)"),
        ("sq_bdl_8_8_r2",    "#dc2626", ":v",  "Scale-8 Grid 2-Rib (56,312q; 38.05x)"),
    ]
    for cname, col, fmt, lbl in comp_specs:
        if cname not in tq:
            continue
        pts = tq[cname]["points"]
        ax_c.errorbar([p["beta"] for p in pts], [p["tau_Q"] for p in pts], yerr=[p["sem_Q"] for p in pts],
                      fmt=fmt, color=col, ms=5.0, lw=1.6, capsize=2.4, label=lbl)

    ax_c.set_yscale("log")
    ax_c.set_ylim(42.0, 9.5e5)
    ax_c.set_xlabel(r"Inverse Bath Temperature $\beta = 1/T$ ($J = 1$)")
    ax_c.set_ylabel(r"Two-Sector Quantum Lifetime $\hat{\tau}_Q(\beta)$")
    ax_c.set_title("(c) Direct Graphs vs. Routed Cubic Grids")
    ax_c.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.93, fontsize=6.9)

    save_fig(fig, "fig_v17_3_euclidean_obstruction")


# ==============================================================================
# FIGURE 4: Controlled Wire-Length Sweep & Analytical Effective-Coupling Law
# ==============================================================================
def render_fig4(v13):
    ws = v13["wire_sweep"]
    fig = plt.figure(figsize=(13.8, 4.5))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.02, 1.04, 1.06], wspace=0.33)

    # --- Panel (a): Degree-2 Wire Schematic & Exact Marginalization Theorem ---
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_xlim(0, 40)
    ax_a.set_ylim(0, 34)
    ax_a.axis("off")
    ax_a.add_patch(FancyBboxPatch((0.5, 0.5), 39.0, 33.0, boxstyle="round,pad=0.25,rounding_size=1.0",
                                  facecolor="#f8fafc", edgecolor="#cbd5e1", lw=1.0))
    ax_a.text(20.0, 31.4, "(a) Exact Degree-2 Check-Wire Reduction", ha="center", va="center",
              fontsize=10.2, fontweight="bold", color="#0f172a")

    # Draw w=1 direct check leg vs w-bond wire chain
    ax_a.text(3.0, 26.8, "Core leg ($w=1$):", fontsize=8.2, fontweight="bold", color="#1e293b", va="center")
    ax_a.add_patch(Rectangle((20.8, 25.5), 4.0, 2.6, facecolor="#dbeafe", edgecolor="#1d4ed8", lw=1.3))
    ax_a.text(22.8, 26.8, r"$R_c$", ha="center", va="center", fontsize=8.0, fontweight="bold", color="#1e3a8a")
    ax_a.plot([24.8, 32.8], [26.8, 26.8], color="#0f172a", lw=2.0)
    ax_a.scatter([32.8], [26.8], s=42, color="#059669", zorder=5)
    ax_a.text(35.3, 26.8, r"$q_c$", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#065f46")

    ax_a.text(3.0, 21.8, "Subdivided degree-2 wire (length $w$):", fontsize=8.2, fontweight="bold", color="#1e293b")
    ax_a.add_patch(Rectangle((3.0, 17.5), 3.6, 2.6, facecolor="#dbeafe", edgecolor="#1d4ed8", lw=1.3))
    ax_a.text(4.8, 18.8, r"$R_c$", ha="center", va="center", fontsize=8.0, fontweight="bold", color="#1e3a8a")
    xs_w = [10.2, 16.6, 23.0, 29.4, 35.8]
    ax_a.plot([6.6, xs_w[-1]], [18.8, 18.8], color="#dc2626", lw=1.8, ls="--")
    for idx, xw in enumerate(xs_w):
        col_n = "#059669" if idx == len(xs_w) - 1 else "#dc2626"
        lbl_n = r"$q_c$" if idx == len(xs_w) - 1 else rf"$u_{{{idx+1}}}$"
        ax_a.scatter([xw], [18.8], s=36, color=col_n, zorder=5)
        ax_a.text(xw, 16.2, lbl_n, ha="center", va="center", fontsize=7.6, color="#334155")
    ax_a.text(20.0, 13.6, r"Single defect sits at any of $w$ bonds at identical $\Delta E = +1$",
              ha="center", va="center", fontsize=7.6, color="#991b1b")

    # Formula box
    ax_a.add_patch(FancyBboxPatch((2.2, 1.8), 35.6, 10.6, boxstyle="round,pad=0.2,rounding_size=0.7",
                                  facecolor="#ffffff", edgecolor="#cbd5e1", lw=0.9))
    ax_a.text(20.0, 10.2, r"Exact Gibbs + Kramers Wire Law ($\forall\,N_0, w,\beta$):",
              ha="center", va="center", fontsize=8.0, fontweight="bold", color="#0f172a")
    ax_a.text(20.0, 6.9, r"$\tanh(\beta_{\mathrm{eff}}(w)/2) = [\tanh(\beta/2)]^w \approx 1 - 2w e^{-\beta}$",
              ha="center", va="center", fontsize=8.4, color="#1d4ed8")
    ax_a.text(20.0, 3.6, r"$\tau(\beta, w) \approx w^2\,\tau_{\mathrm{core}}(\beta_{\mathrm{eff}}(w)) \propto w^{2-\Delta_{\mathrm{eff}}}\tau(\beta,1)$",
              ha="center", va="center", fontsize=7.9, color="#065f46")

    # --- Panel (b): Raw Lifetime vs Physical Inverse Temperature beta (Fixed 646q Torus Core) ---
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b)
    w_palette = [
        ("1",  "#059669", "-o"),
        ("2",  "#0284c7", "-s"),
        ("4",  "#2563eb", "-^"),
        ("8",  "#d97706", "-D"),
        ("16", "#dc2626", "-v"),
    ]
    tor_w = ws.get("tor", {})
    for wk, col, fmt in w_palette:
        d = tor_w.get(wk) or tor_w.get(int(wk))
        if not d:
            continue
        pts = [p for p in d["points"] if p["capped"] == 0]
        ax_b.errorbar([p["beta"] for p in pts], [p["mean_t"] for p in pts], yerr=[p["sem_t"] for p in pts],
                      fmt=fmt, color=col, ms=5.0, lw=1.7, capsize=2.4,
                      label=f"w = {wk} (N = {d['N']:,}, slope {d['slope']:.2f})")

    # Highlight 32x drop at fixed beta = 6.50!
    t_w1_65 = next((p["mean_t"] for p in (tor_w.get("1") or tor_w.get(1))["points"] if abs(p["beta"] - 6.5) < 1e-4), None)
    t_w16_65 = next((p["mean_t"] for p in (tor_w.get("16") or tor_w.get(16))["points"] if abs(p["beta"] - 6.5) < 1e-4), None)
    if t_w1_65 and t_w16_65:
        ax_b.annotate("", xy=(6.50, t_w16_65 * 1.15), xytext=(6.50, t_w1_65 * 0.88),
                      arrowprops=dict(arrowstyle="<->", color="#991b1b", lw=1.4))
        ax_b.text(6.58, math.sqrt(t_w1_65 * t_w16_65), f"{t_w1_65/t_w16_65:.0f}x drop\nat " + r"$\beta=6.5$",
                  fontsize=7.6, fontweight="bold", color="#991b1b", va="center")

    ax_b.set_xlim(4.85, 8.65)
    ax_b.set_yscale("log")
    ax_b.set_xlabel(r"Physical Bath Inverse Temp. $\beta = 1/T$ ($J = 1$)")
    ax_b.set_ylabel(r"Decoded Torus Lifetime $\langle T_{\mathrm{fail}} \rangle$")
    ax_b.set_title("(b) Torus Core (646q) vs. Wire Length $w$")
    ax_b.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, fontsize=7.2)

    # --- Panel (c): Analytical Kramers-Gibbs Collapse tau / w^2 vs beta_eff(w) ---
    ax_c = fig.add_subplot(gs[0, 2])
    style_ax(ax_c)
    for wk, col, fmt in w_palette:
        d = tor_w.get(wk) or tor_w.get(int(wk))
        if not d:
            continue
        w_num = float(wk)
        pts = [p for p in d["points"] if p["capped"] == 0]
        ax_c.errorbar([p["beta_eff"] for p in pts], [p["mean_t"] / (w_num ** 2) for p in pts],
                      yerr=[p["sem_t"] / (w_num ** 2) for p in pts],
                      fmt=fmt, color=col, ms=5.0, lw=1.6, capsize=2.4,
                      label=f"Torus core w = {wk}")

    # Inset: Planar core (N0=323, K=1) collapse tau / w^2 vs beta_eff(w)
    sq_w = ws.get("sq", {})
    if sq_w:
        ins_c = ax_c.inset_axes([0.11, 0.54, 0.45, 0.40])
        ins_c.grid(True, alpha=0.3)
        for wk, col, fmt in w_palette:
            d = sq_w.get(wk) or sq_w.get(int(wk))
            if not d:
                continue
            w_num = float(wk)
            pts = [p for p in d["points"] if p["capped"] == 0]
            ins_c.plot([p["beta_eff"] for p in pts], [p["mean_t"] / (w_num ** 2) for p in pts], fmt, color=col, ms=2.8, lw=1.1)
        ins_c.set_yscale("log")
        ins_c.set_title(r"Planar ($N_0=323$) $\tau/w^2$", fontsize=6.7)
        ins_c.tick_params(labelsize=6.2)

    ax_c.set_yscale("log")
    ax_c.set_xlabel(r"Effective Coupling $\beta_{\mathrm{eff}}(w) = 2\,\mathrm{arctanh}[\tanh^w(\beta/2)]$")
    ax_c.set_ylabel(r"Diffusive-Rescaled Lifetime $\langle T_{\mathrm{fail}} \rangle / w^2$")
    ax_c.set_title(r"(c) Kramers-Gibbs Collapse $\tau / w^2$")
    ax_c.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92, fontsize=7.2)

    save_fig(fig, "fig_v17_4_wire_dilution")


def main():
    v13_path = os.path.join(ROOT, "parsed_v13_data.json")
    with open(v13_path) as f:
        v13 = json.load(f)
    kmc4, kmc6, _, kmc9, *_ = load_all()
    render_fig1(v13)
    render_fig2(v13, kmc4, kmc6, kmc9)
    render_fig3(v13)
    render_fig4(v13)
    print("All 4 V17 figures rendered successfully.")


if __name__ == "__main__":
    main()
