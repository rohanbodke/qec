#!/usr/bin/env python3
"""
render_v19_focused_figures.py
Renders 5 clear, direct, publication-grade figures (PDF + 300 DPI PNG) for the V19 manuscript
"Finite-Size Thermal Protection in Compact Recursively Welded Codes":

  Consistent Visual Grammar Across All Figures:
    - 2x2 Planar family (`sq_iso_L2r1`): Royal Blue (#1d4ed8 / #2563eb)
    - 2x2 Torus family (`rottor_iso_L2r1`): Emerald Green (#059669)
    - 3x3 Single-Cross family (`sq_L3r1`): Violet (#7c3aed)
    - Completed cycles (s = 0, 2, 4): SOLID filled markers + solid lines
    - Half-cycles (s = 1, 3, 5): OPEN white-filled markers + dashed lines

  Figures:
    Fig. 1 (`fig_v19_1_construction`): BDL Correspondence, Compact Cell, & Local Bounds
    Fig. 2 (`fig_v19_2_lifetimes`):    Complete-Cycle Gains, Crossover, & Sector Bottlenecks (LEAD)
    Fig. 3 (`fig_v19_3_intervention`): Scale-Selective Coupling Intervention & Coarse Excursions
    Fig. 4 (`fig_v19_4_readout`):      Coarse Residual vs. Random Control, Paired Readout, & Cadence
    Fig. 5 (`fig_v19_5_implementation`): 3D Volume-Growth Excess, 3x3 Sector Test, & CSS Wire Collapse
"""

import json
import math
import os
import shutil
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures")
ART_DIR = "/Users/harshilavlani/.gemini/antigravity/brain/35299b0d-7439-4f8e-8617-0b646151a931"

C_PLANAR = "#1d4ed8"
C_TORUS = "#059669"
C_L3 = "#7c3aed"
C_MACRO = "#dc2626"
C_SUB = "#2563eb"
C_ALLC2 = "#d97706"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10.0,
        "axes.titlesize": 11.0,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.2,
        "xtick.labelsize": 9.1,
        "ytick.labelsize": 9.1,
        "legend.fontsize": 8.4,
        "axes.linewidth": 1.0,
        "grid.linewidth": 0.55,
        "grid.alpha": 0.35,
        "savefig.dpi": 300,
    }
)


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
# FIGURE 1: BDL Correspondence, Compact Cell, & Local Bounds
# ==============================================================================
def render_fig1():
    fig = plt.figure(figsize=(13.8, 4.65))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.07, 1.03, 1.03], wspace=0.16)

    # --- Panel (a): BDL Sec. 6.2 Correspondence & Odd X-Weld ---
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_xlim(0, 42)
    ax_a.set_ylim(0, 36)
    ax_a.axis("off")
    ax_a.add_patch(
        FancyBboxPatch(
            (0.5, 0.5),
            41.0,
            35.0,
            boxstyle="round,pad=0.25,rounding_size=1.0",
            facecolor="#f8fafc",
            edgecolor="#cbd5e1",
            lw=1.0,
        )
    )
    ax_a.text(21.0, 33.4, "(a) BDL Map & Odd X-Weld", ha="center", va="center", fontsize=10.8, fontweight="bold", color="#0f172a")

    ax_a.add_patch(
        FancyBboxPatch(
            (1.6, 26.0),
            38.8,
            5.5,
            boxstyle="round,pad=0.15,rounding_size=0.6",
            facecolor="#ffffff",
            edgecolor="#cbd5e1",
            lw=0.9,
        )
    )
    ax_a.text(
        21.0,
        29.8,
        r"Retained (BDL 6.2): sheet doubling + alternating $X/Z$ bridges",
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color="#0f172a",
    )
    ax_a.text(
        21.0,
        27.4,
        r"Simplified: $L=2$ single-rib graph cell (vs. $\ell=8$ two-rib $\mathbb{Z}^3$ grid)",
        ha="center",
        va="center",
        fontsize=7.4,
        color="#475569",
    )

    for sh_y, lbl in [(17.3, "Sheet 0"), (8.4, "Sheet 1")]:
        pts = [[5.5, sh_y], [24.5, sh_y], [30.0, sh_y + 4.8], [11.0, sh_y + 4.8]]
        ax_a.add_patch(Polygon(pts, closed=True, facecolor="#eff6ff", edgecolor="#64748b", lw=1.15))
        ax_a.plot([15.0, 20.5], [sh_y, sh_y + 4.8], color="#1d4ed8", lw=2.4)
        ax_a.plot([8.25, 27.25], [sh_y + 2.4, sh_y + 2.4], color="#1d4ed8", lw=2.4)
        ax_a.text(35.2, sh_y + 2.4, lbl, ha="center", va="center", fontsize=8.3, color="#334155", fontweight="bold")

    ax_a.add_patch(
        Polygon([[8.25, 10.8], [27.25, 10.8], [27.25, 19.7], [8.25, 19.7]], closed=True, facecolor="#93c5fd", edgecolor="#1d4ed8", lw=1.0, alpha=0.28)
    )
    ax_a.add_patch(
        Polygon([[15.0, 8.4], [20.5, 13.2], [20.5, 22.1], [15.0, 17.3]], closed=True, facecolor="#60a5fa", edgecolor="#1d4ed8", lw=1.0, alpha=0.30)
    )
    ax_a.scatter([17.75], [19.7], s=52, color="#1d4ed8", zorder=6)
    ax_a.annotate(
        "1 rib qubit in\n3 Z-face checks",
        xy=(17.75, 19.7),
        xytext=(2.2, 22.6),
        arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=1.2),
        fontsize=7.8,
        fontweight="bold",
        color="#1e3a8a",
    )

    ax_a.scatter([3.4], [5.2], s=36, color="#1d4ed8")
    ax_a.text(5.0, 5.2, "Physical qubit", va="center", fontsize=7.6, color="#1e293b")
    ax_a.add_patch(Rectangle((16.8, 4.3), 2.3, 1.8, facecolor="#dbeafe", edgecolor="#1d4ed8", lw=1.0))
    ax_a.text(19.9, 5.2, r"Face $Z$-check ($X$ defect $1 \to 2$)", va="center", fontsize=7.6, color="#1e293b")
    ax_a.text(21.0, 2.2, r"Seeds: planar $[[2, 1, (2, 1)]]$ ($d=1$); torus $[[4, 2, 2]]$ ($d=2$)", ha="center", va="center", fontsize=7.5, color="#475569")

    # --- Panel (b): Even Z-Weld & Controlled Ablations ---
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_xlim(0, 40)
    ax_b.set_ylim(0, 36)
    ax_b.axis("off")
    ax_b.add_patch(
        FancyBboxPatch(
            (0.5, 0.5),
            39.0,
            35.0,
            boxstyle="round,pad=0.25,rounding_size=1.0",
            facecolor="#f8fafc",
            edgecolor="#cbd5e1",
            lw=1.0,
        )
    )
    ax_b.text(20.0, 33.4, "(b) Even Z-Weld & Ablations", ha="center", va="center", fontsize=10.8, fontweight="bold", color="#0f172a")

    for sh_y in (24.4, 16.4):
        pts = [[6.2, sh_y], [24.2, sh_y], [29.2, sh_y + 4.2], [11.2, sh_y + 4.2]]
        ax_b.add_patch(Polygon(pts, closed=True, facecolor="#fef2f2", edgecolor="#64748b", lw=1.1))
        ax_b.plot([15.2, 20.2], [sh_y, sh_y + 4.2], color="#dc2626", lw=2.1, ls="--")
        ax_b.plot([8.7, 26.7], [sh_y + 2.1, sh_y + 2.1], color="#dc2626", lw=2.1, ls="--")
    for cx in (8.7, 17.7, 26.7):
        ax_b.plot([cx, cx], [18.5, 26.5], color="#dc2626", lw=1.8)
        ax_b.scatter([cx, cx], [18.5, 26.5], s=24, color="#dc2626", zorder=5)
    ax_b.text(34.3, 22.5, "2 parity q.\nbranch\nX-checks", ha="center", va="center", fontsize=8.0, fontweight="bold", color="#991b1b")
    ax_b.text(
        20.0,
        13.6,
        r"Complete 2-step cycle protects both sectors:" + "\n" + r"planar logical weight $(2, 1) \to (11, 10)$",
        ha="center",
        va="center",
        fontsize=7.9,
        fontweight="bold",
        color="#0f172a",
    )

    ax_b.add_patch(FancyBboxPatch((1.8, 1.8), 17.2, 9.5, boxstyle="round,pad=0.18,rounding_size=0.7", facecolor="#fff1f2", edgecolor="#fecdd3", lw=1.0))
    ax_b.text(10.4, 9.3, "3-Arm T-Bridge", ha="center", va="center", fontsize=8.3, fontweight="bold", color="#9f1239")
    ax_b.text(10.4, 5.0, "Fixed seed & CSS rules;\n8 wt-1 Z logicals,\ndressed dist. (6, 5)", ha="center", va="center", fontsize=7.4, color="#881337")

    ax_b.add_patch(FancyBboxPatch((21.0, 1.8), 17.2, 9.5, boxstyle="round,pad=0.18,rounding_size=0.7", facecolor="#fff7ed", edgecolor="#fed7aa", lw=1.0))
    ax_b.text(29.6, 9.3, "1-Step X+Z Cross", ha="center", va="center", fontsize=8.3, fontweight="bold", color="#9a3412")
    ax_b.text(29.6, 5.0, "Fixed seed & CSS rules;\ncenter seam clash,\ndressed " + r"$d_Z = 1$", ha="center", va="center", fontsize=7.4, color="#7c2d12")

    # --- Panel (c): Wall-Minority & Separate Majority Quotient ---
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.set_xlim(0, 40)
    ax_c.set_ylim(0, 36)
    ax_c.axis("off")
    ax_c.add_patch(
        FancyBboxPatch(
            (0.5, 0.5),
            39.0,
            35.0,
            boxstyle="round,pad=0.25,rounding_size=1.0",
            facecolor="#f8fafc",
            edgecolor="#cbd5e1",
            lw=1.0,
        )
    )
    ax_c.text(20.0, 33.4, "(c) Auxiliary Wall & Majority Bounds", ha="center", va="center", fontsize=10.8, fontweight="bold", color="#0f172a")

    ax_c.text(20.0, 29.8, r"Auxiliary dual face labels $a_i, b_i, c_i \in \{0, 1\}$ on wall $i$:", ha="center", va="center", fontsize=7.9, fontweight="bold", color="#1e293b")

    boxes = [
        (3.0, "#bfdbfe", "#1d4ed8", "1"),
        (9.2, "#bfdbfe", "#1d4ed8", "1"),
        (17.2, "#dbeafe", "#1e40af", "0"),
        (25.2, "#ffffff", "#94a3b8", "0"),
        (31.4, "#ffffff", "#94a3b8", "0"),
    ]
    for bx, fc, ec, val in boxes:
        ax_c.add_patch(Rectangle((bx, 21.6), 5.4, 4.1, facecolor=fc, edgecolor=ec, lw=1.5))
        ax_c.text(bx + 2.7, 23.65, val, ha="center", va="center", fontsize=9.4, fontweight="bold", color="#0f172a")
    ax_c.text(8.8, 27.0, r"Sheet 0 ($a$)", ha="center", va="center", fontsize=7.7, fontweight="bold", color="#1e3a8a")
    ax_c.text(19.9, 27.0, r"Bridge ($c$)", ha="center", va="center", fontsize=7.7, fontweight="bold", color="#1e40af")
    ax_c.text(31.0, 27.0, r"Sheet 1 ($b$)", ha="center", va="center", fontsize=7.7, fontweight="bold", color="#475569")

    ax_c.plot([8.4, 15.8], [21.4, 18.6], color="#dc2626", lw=1.8)
    ax_c.plot([19.5, 15.8], [21.4, 18.6], color="#dc2626", lw=1.8)
    ax_c.text(11.0, 17.2, r"Rib $(a,c)$ violated", ha="center", va="center", fontsize=7.7, fontweight="bold", color="#b91c1c")

    ax_c.plot([20.3, 27.6], [21.4, 18.6], color="#16a34a", lw=1.6, ls="--")
    ax_c.plot([31.0, 27.6], [21.4, 18.6], color="#16a34a", lw=1.6, ls="--")
    ax_c.text(29.8, 17.2, r"Rib $(b,c)$ clean", ha="center", va="center", fontsize=7.7, fontweight="bold", color="#15803d")

    ax_c.add_patch(FancyBboxPatch((1.8, 1.8), 36.4, 13.4, boxstyle="round,pad=0.2,rounding_size=0.8", facecolor="#ffffff", edgecolor="#cbd5e1", lw=1.0))
    ax_c.text(
        20.0,
        12.7,
        r"1. Wall minority (Prop. 1):  $W_{\mathrm{wall}} \leq 2\,V$",
        ha="center",
        va="center",
        fontsize=8.6,
        fontweight="bold",
        color="#1e3a8a",
    )
    ax_c.text(
        20.0,
        9.9,
        r"(tight on Sheet-0 flip: $W_{\mathrm{wall}} = 8,\ V = 4$; 4 extremizers)",
        ha="center",
        va="center",
        fontsize=7.5,
        color="#475569",
    )
    ax_c.text(
        20.0,
        6.5,
        r"2. Wall majority quotient (Prop. 2):  $C \leq \frac{2}{5}\,V$",
        ha="center",
        va="center",
        fontsize=8.6,
        fontweight="bold",
        color="#065f46",
    )
    ax_c.text(
        20.0,
        3.5,
        r"mod 4-wall flip $y_f$; physical basis $V \leq V_{\mathrm{phys}} \leq \frac{11}{5}V$",
        ha="center",
        va="center",
        fontsize=7.4,
        color="#475569",
    )

    save_fig(fig, "fig_v19_1_construction")


# ==============================================================================
# FIGURE 2 (LEAD): Complete-Cycle Protection, Crossover, & Sector Bottlenecks
# ==============================================================================
def render_fig2(v14):
    tq = v14["tau_q"]
    gains = v14["cycle_gains"]

    fig = plt.figure(figsize=(13.8, 4.5))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.08, 1.0, 1.02], wspace=0.27)

    # --- Panel (a): tau_Q vs N at fixed temperatures ---
    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a)

    beta_styles = [
        ("5.50", "o", 0.55, r"$\beta = 5.5$"),
        ("6.00", "s", 0.78, r"$\beta = 6.0$"),
        ("6.50", "D", 1.00, r"$\beta = 6.5$"),
    ]
    for b_str, mk, alph, b_lbl in beta_styles:
        p_ns, p_qs, p_ses, p_cyc = [], [], [], []
        for s in (1, 2, 3, 4, 5):
            cn = f"sq_iso_L2r1_s{s}"
            if cn in tq and b_str in tq[cn]:
                r = tq[cn][b_str]
                p_ns.append(r["N"])
                p_qs.append(r["tau_Q"])
                p_ses.append(r["sem_Q"])
                p_cyc.append(s % 2 == 0)
        ax_a.plot(p_ns, p_qs, "--", color=C_PLANAR, alpha=0.45 * alph, lw=1.2)
        c_ns = [n for n, ic in zip(p_ns, p_cyc) if ic]
        c_qs = [q for q, ic in zip(p_qs, p_cyc) if ic]
        ax_a.plot(c_ns, c_qs, "-", color=C_PLANAR, alpha=alph, lw=2.1, label=f"Planar {b_lbl}")
        for n, q, se, ic in zip(p_ns, p_qs, p_ses, p_cyc):
            ax_a.errorbar(
                [n],
                [q],
                yerr=[se],
                fmt=mk,
                color=C_PLANAR,
                alpha=alph,
                ms=6.0,
                mfc=(C_PLANAR if ic else "white"),
                mew=1.6,
                capsize=2.2,
            )

        t_ns, t_qs, t_ses, t_cyc = [], [], [], []
        for s in (1, 2, 3, 4):
            cn = f"rottor_iso_L2r1_s{s}"
            if cn in tq and b_str in tq[cn]:
                r = tq[cn][b_str]
                t_ns.append(r["N"])
                t_qs.append(r["tau_Q"])
                t_ses.append(r["sem_Q"])
                t_cyc.append(s % 2 == 0)
        tc_ns = [n for n, ic in zip(t_ns, t_cyc) if ic]
        tc_qs = [q for q, ic in zip(t_qs, t_cyc) if ic]
        ax_a.plot(tc_ns, tc_qs, "-", color=C_TORUS, alpha=alph, lw=2.0, label=f"Torus {b_lbl}")
        for n, q, se, ic in zip(t_ns, t_qs, t_ses, t_cyc):
            if ic:
                ax_a.errorbar([n], [q], yerr=[se], fmt=mk, color=C_TORUS, alpha=alph, ms=6.0, mfc=C_TORUS, capsize=2.2)

    ax_a.text(
        0.04,
        0.05,
        "Filled: completed cycles (s=2, 4)\nOpen: planar half-cycles (s=1, 3, 5)",
        transform=ax_a.transAxes,
        fontsize=7.6,
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#cbd5e1", alpha=0.92),
    )
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_ylim(15.0, 3.2e6)
    ax_a.set_xlabel("Physical Qubits N")
    ax_a.set_ylabel(r"Joint Lifetime $\tau_Q$")
    ax_a.set_title(r"(a) $\tau_Q$ vs. Size Across Temperatures")
    ax_a.legend(loc="upper left", ncol=2, frameon=True, facecolor="white", framealpha=0.93, fontsize=7.4)

    # --- Panel (b): Complete-Cycle Gain G_1(beta) & Torus Crossover ---
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b)
    ax_b.axhline(1.0, color="#0f172a", ls="--", lw=1.3)
    ax_b.axhspan(0.1, 1.0, facecolor="#fee2e2", alpha=0.28)
    ax_b.text(5.78, 0.23, r"No net gain ($G < 1$)", fontsize=7.9, color="#991b1b", fontweight="bold")

    for key, col, fmt, mfc, lbl in [
        ("planar_C1_to_C2", C_PLANAR, "-D", C_PLANAR, r"Planar $C_1 \to C_2$ ($146.7\times$ q.)"),
        ("torus_C1_to_C2", C_TORUS, "-o", C_TORUS, r"Torus $C_1 \to C_2$ ($146.7\times$ q.)"),
        ("planar_C2_to_s5_half", C_PLANAR, "--s", "white", r"Planar $C_2 \to s_5$ half ($8.9\times$ q.)"),
    ]:
        gd = gains.get(key, {})
        bs = sorted(float(k) for k in gd.keys())
        gs_v = [gd[f"{b:.2f}"]["gain"] for b in bs]
        ses_v = [gd[f"{b:.2f}"]["se_gain"] for b in bs]
        ax_b.errorbar(bs, gs_v, yerr=ses_v, fmt=fmt, color=col, mfc=mfc, mew=1.7, ms=6.0, lw=2.0, capsize=2.6, label=lbl)

    ax_b.annotate(
        "Torus crossover\n" + r"($\beta \approx 6.1$)",
        xy=(6.12, 1.0),
        xytext=(5.35, 3.3),
        arrowprops=dict(arrowstyle="->", color=C_TORUS, lw=1.3),
        fontsize=8.1,
        fontweight="bold",
        color="#065f46",
    )
    ax_b.annotate(
        r"$55.9\times$ at $\beta=6.5$",
        xy=(6.50, 55.9),
        xytext=(5.32, 17.5),
        arrowprops=dict(arrowstyle="->", color=C_PLANAR, lw=1.2),
        fontsize=8.0,
        fontweight="bold",
        color=C_PLANAR,
    )

    ax_b.set_yscale("log")
    ax_b.set_xlim(4.9, 6.62)
    ax_b.set_ylim(0.14, 130.0)
    ax_b.set_xlabel(r"Inverse Temperature $\beta = 1/T$")
    ax_b.set_ylabel(r"Lifetime Gain Ratio $G(\beta)$")
    ax_b.set_title("(b) Complete-Cycle Gain & Crossover")
    ax_b.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.94, fontsize=7.5)

    # --- Panel (c): Sector-Resolved Sawtooth Across Steps (Planar beta=6.5) ---
    ax_c = fig.add_subplot(gs[0, 2])
    style_ax(ax_c)
    steps = [1, 2, 3, 4, 5]
    txs, tzs, tqs, sems_q = [], [], [], []
    for s in steps:
        r = tq[f"sq_iso_L2r1_s{s}"]["6.50"]
        txs.append(r["tau_X"])
        tzs.append(r["tau_Z"])
        tqs.append(r["tau_Q"])
        sems_q.append(r["sem_Q"])

    ax_c.plot(steps, txs, "--^", color="#0284c7", ms=5.8, lw=1.5, label=r"$X$-sector $\tau_X$ (lower bnd if >cap)")
    ax_c.plot(steps, tzs, "--v", color="#dc2626", ms=5.8, lw=1.5, label=r"$Z$-sector $\tau_Z$ (lower bnd if >cap)")
    ax_c.plot(steps, tqs, "-", color=C_PLANAR, lw=2.2, label=r"Joint $\tau_Q$")
    for s, q, se in zip(steps, tqs, sems_q):
        ic = (s % 2 == 0)
        ax_c.errorbar([s], [q], yerr=[se], fmt="o", color=C_PLANAR, ms=6.8, mfc=(C_PLANAR if ic else "white"), mew=1.8, capsize=2.6, zorder=5)

    ax_c.annotate(
        r"$s=5$ $Z$-limited:" + "\n" + r"$1.18 \pm 0.28\times$" + "\n(not resolved)",
        xy=(4.95, tqs[-1] * 0.88),
        xytext=(3.28, 3.8e4),
        arrowprops=dict(arrowstyle="->", color="#991b1b", lw=1.2),
        fontsize=7.8,
        fontweight="bold",
        color="#991b1b",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#fecaca", alpha=0.92),
    )

    ax_c.set_yscale("log")
    ax_c.set_xticks(steps)
    ax_c.set_xticklabels(["s=1\n(19q)", "C1 (s=2)\n(323q)", "s=3\n(2.9k)", "C2 (s=4)\n(47.4k)", "s=5\n(424k)"])
    ax_c.set_ylim(30.0, 5.0e8)
    ax_c.set_xlabel("Recursion Step (Planar, " + r"$\beta = 6.5$" + ")")
    ax_c.set_ylabel("Mean Lifetime")
    ax_c.set_title("(c) Alternating Sector Bottleneck")
    ax_c.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.93, fontsize=7.5)

    save_fig(fig, "fig_v19_2_lifetimes")


# ==============================================================================
# FIGURE 3: Scale-Selective Coupling Intervention H(lambda; U) & Coarse Excursions
# ==============================================================================
def render_fig3(v14):
    tq = v14["tau_q"]
    inter = v14["interventions"]
    exc = v14["excursions"]

    fig = plt.figure(figsize=(13.8, 4.5))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.05, 1.0, 1.02], wspace=0.28)

    # --- Panel (a): tau_Q(lambda) at beta = 6.5 on Planar & Torus Cycle 2 ---
    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a)
    lams = [0.2, 0.4, 0.6, 0.8, 1.0]

    for cname, fam_lbl, ls_fam, mk_m, mk_s in [
        ("sq_iso_L2r1_s4", "Planar C2", "-", "o", "s"),
        ("rottor_iso_L2r1_s4", "Torus C2", "--", "^", "D"),
    ]:
        u_q = tq[cname]["6.50"]["tau_Q"]
        u_se = tq[cname]["6.50"]["sem_Q"]
        for tag_u, col, mk, u_lbl in [
            ("macro_weld", C_MACRO, mk_m, "Coarse C1 welds"),
            ("sub_weld", C_SUB, mk_s, "Matched C2 welds"),
        ]:
            ys = [inter[cname][tag_u]["6.50"][f"{l:.1f}"]["tau_Q"] for l in lams[:-1]] + [u_q]
            ses = [inter[cname][tag_u]["6.50"][f"{l:.1f}"]["sem_Q"] for l in lams[:-1]] + [u_se]
            ax_a.errorbar(
                lams,
                ys,
                yerr=ses,
                fmt=f"{ls_fam}{mk}",
                color=col,
                ms=5.8,
                lw=1.9,
                capsize=2.4,
                label=f"{fam_lbl}: {u_lbl}",
            )

    ax_a.axhline(tq["rottor_iso_L2r1_s2"]["6.50"]["tau_Q"], color=C_TORUS, ls=":", lw=1.3)
    ax_a.text(0.19, tq["rottor_iso_L2r1_s2"]["6.50"]["tau_Q"] * 1.32, "Torus C1 (646q)", fontsize=7.7, color=C_TORUS, fontweight="bold")
    ax_a.axhline(tq["sq_iso_L2r1_s2"]["6.50"]["tau_Q"], color="#475569", ls=":", lw=1.3)
    ax_a.text(0.19, tq["sq_iso_L2r1_s2"]["6.50"]["tau_Q"] * 1.32, "Planar C1 (323q)", fontsize=7.7, color="#475569", fontweight="bold")

    ax_a.set_yscale("log")
    ax_a.set_xlim(0.15, 1.05)
    ax_a.set_ylim(1.2, 2.8e6)
    ax_a.set_xlabel(r"Selected-Scale Check Coupling $\lambda$")
    ax_a.set_ylabel(r"Joint Lifetime $\tau_Q(\lambda)$ at $\beta = 6.5$")
    ax_a.set_title("(a) Equal-Budget Scale Weakening")
    ax_a.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.94, fontsize=7.4)

    # --- Panel (b): Scale Sensitivity Ratio tau(U_sub) / tau(U_macro) Across beta & Sectors ---
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b)
    betas = [5.50, 6.00, 6.25, 6.50]
    ax_b.axhline(1.0, color="#64748b", ls="--", lw=1.1)

    for cname, col_q, col_z, lbl in [
        ("sq_iso_L2r1_s4", C_PLANAR, "#dc2626", "Planar C2"),
        ("rottor_iso_L2r1_s4", C_TORUS, "#d97706", "Torus C2"),
    ]:
        rq_06 = [
            inter[cname]["sub_weld"][f"{b:.2f}"]["0.6"]["tau_Q"] / inter[cname]["macro_weld"][f"{b:.2f}"]["0.6"]["tau_Q"]
            for b in betas
        ]
        rz_04 = [
            inter[cname]["sub_weld"][f"{b:.2f}"]["0.4"]["tau_Z"] / inter[cname]["macro_weld"][f"{b:.2f}"]["0.4"]["tau_Z"]
            for b in betas
        ]
        ax_b.plot(betas, rq_06, "-o", color=col_q, ms=6.0, lw=2.0, label=f"{lbl} " + r"$\tau_Q$ ($\lambda=0.6$)")
        ax_b.plot(betas, rz_04, "--s", color=col_z, ms=5.6, lw=1.7, label=f"{lbl} " + r"$\tau_Z$ ($\lambda=0.4$)")

    ax_b.annotate(
        r"$66\times$ $Z$-sector sensitivity" + "\nto coarse C1 seams",
        xy=(6.50, 66.4),
        xytext=(5.82, 16.5),
        arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.2),
        fontsize=7.9,
        fontweight="bold",
        color="#991b1b",
    )

    ax_b.set_yscale("log")
    ax_b.set_xlim(5.42, 6.58)
    ax_b.set_ylim(0.85, 130.0)
    ax_b.set_xlabel(r"Inverse Temperature $\beta = 1/T$")
    ax_b.set_ylabel(r"Ratio $\tau(U_{\mathrm{sub}}) \,/\, \tau(U_{\mathrm{macro}})$")
    ax_b.set_title("(b) Coarse vs. Sub-Cell Sensitivity")
    ax_b.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.94, fontsize=7.4)

    # --- Panel (c): Coarse-Scale Cycle-1 Excursions Before Failure ---
    ax_c = fig.add_subplot(gs[0, 2])
    style_ax(ax_c)
    for cname, jtag, col, fmt, lbl in [
        ("rottor_iso_L2r1_s4", "uniform", C_TORUS, "-o", r"Torus C2 (uniform, $\lambda=1$)"),
        ("sq_iso_L2r1_s4", "uniform", C_PLANAR, "-D", r"Planar C2 (uniform, $\lambda=1$)"),
        ("rottor_iso_L2r1_s4", "sub_weld_l06", C_SUB, "--s", r"Torus C2 ($U_{\mathrm{sub}}, \lambda=0.6$)"),
        ("rottor_iso_L2r1_s4", "macro_weld_l06", C_MACRO, ":^", r"Torus C2 ($U_{\mathrm{macro}}, \lambda=0.6$)"),
    ]:
        bs_e, ms_e, ses_e = [], [], []
        for b in betas:
            k = f"excur__{cname}_X_err__{jtag}__b{b:.2f}"
            if k in exc:
                bs_e.append(b)
                ms_e.append(exc[k]["mean_n_exc"])
                ses_e.append(exc[k]["sem_n_exc"])
        ax_c.errorbar(bs_e, ms_e, yerr=ses_e, fmt=fmt, color=col, ms=5.8, lw=1.9, capsize=2.5, label=lbl)

    ax_c.annotate(
        r"$9.5 \pm 0.5$ coarse" + "\nexcursions healed\nbefore failure",
        xy=(6.50, 9.50),
        xytext=(5.55, 6.2),
        arrowprops=dict(arrowstyle="->", color=C_TORUS, lw=1.2),
        fontsize=7.9,
        fontweight="bold",
        color="#065f46",
    )

    ax_c.set_xlim(5.42, 6.58)
    ax_c.set_ylim(0.8, 11.4)
    ax_c.set_xlabel(r"Inverse Temperature $\beta = 1/T$")
    ax_c.set_ylabel("Mean Cycle-1 Excursions per Run")
    ax_c.set_title("(c) Repelled Coarse Excursions")
    ax_c.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.94, fontsize=7.4)

    save_fig(fig, "fig_v19_3_intervention")


# ==============================================================================
# FIGURE 4: Coarse Residual vs. Random Control, Paired Readout, & Cadence Audit
# ==============================================================================
def render_fig4(v14):
    rg14 = v14["rgflow14"]
    pk = v14["paired_kmc"]
    cad = v14["paired_cadence"]

    fig = plt.figure(figsize=(13.8, 4.5))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.05, 1.0, 1.02], wspace=0.28)

    # --- Panel (a): Pre-Seed-Solve Coarse Residual vs Matched-Weight Random Control ---
    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a)
    floor_a = 4e-3
    xs_a = list(range(5))

    for b_val, col, mk, lbl in [
        (5.50, "#d97706", "s", r"$\beta = 5.5$"),
        (6.00, C_PLANAR, "^", r"$\beta = 6.0$"),
        (6.50, C_TORUS, "o", r"$\beta = 6.5$"),
    ]:
        k = f"rgflow14__rottor_s4__t80.0__b{b_val:.2f}"
        if k not in rg14:
            continue
        d = rg14[k]
        qwt = int(round(d["mean_qwt"]))
        ys_t, ses_t, ys_i, ses_i = [], [], [], []
        for s in range(4, -1, -1):
            sc = d["scales"][str(s)] if str(s) in d["scales"] else d["scales"][s]
            ys_t.append(max(floor_a, sc["cnt"]))
            ses_t.append(sc["semc"] if sc["cnt"] > 0 else 0.0)
            ys_i.append(max(floor_a, sc["icnt"]))
            ses_i.append(sc["isemc"] if sc["icnt"] > 0 else 0.0)
        ax_a.errorbar(xs_a, ys_t, yerr=ses_t, fmt=f"-{mk}", color=col, ms=5.6, lw=1.9, capsize=2.3, label=f"Thermal {lbl} ({qwt} flips)")
        ax_a.errorbar(xs_a, ys_i, yerr=ses_i, fmt=f"--{mk}", color=col, mfc="white", mew=1.5, ms=5.2, lw=1.3, alpha=0.8, capsize=2.0, label=f"Random {qwt}-flip ref.")

    ax_a.axhline(floor_a, color="#64748b", ls=":", lw=1.1)
    ax_a.text(0.08, floor_a * 1.45, "0 pre-solve seed defects (0/96 runs)", fontsize=7.5, color="#475569")
    ax_a.set_yscale("log")
    ax_a.set_ylim(floor_a * 0.55, 8.0e3)
    ax_a.set_xticks(xs_a)
    ax_a.set_xticklabels(["s=4\n(41.5k)", "s=3\n(3.5k)", "s=2\n(288)", "s=1\n(24)", "Seed\n(2)"])
    ax_a.set_xlabel(r"Scale $s$ (Pre-Solve Checks, $t_{\mathrm{burn}}=80$)")
    ax_a.set_ylabel("Mean Violated Checks")
    ax_a.set_title("(a) Thermal vs. Random Coarse Flow")
    ax_a.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.93, fontsize=7.0)

    # --- Panel (b): Paired Local RG / Global UF Ratio with 95% CIs ---
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b)
    ax_b.axhline(1.0, color="#0f172a", ls="--", lw=1.2)
    ax_b.axhspan(0.90, 1.10, facecolor="#dcfce7", alpha=0.38, label=r"$\pm 10\%$ band")

    for s_lv, col, fmt, lbl in [
        (2, C_PLANAR, "-s", "Torus C1 (646q)"),
        (4, C_TORUS, "-o", "Torus C2 (94,774q)"),
    ]:
        rows = sorted([(d["beta"], d) for k, d in pk.items() if k.startswith(f"paired_kmc__rottor_s{s_lv}__")], key=lambda t: t[0])
        bs, rats, err_lo, err_hi = [], [], [], []
        for b, d in rows:
            bs.append(b)
            pr = d.get("paired_mean_ratio_rg_over_uf") or (d.get("paired_mean_ratio_strict_over_uf") if s_lv == 2 else None)
            if pr and "mean_ratio" in pr:
                r = pr["mean_ratio"]
                lo, hi = pr["ci95_ratio"]
            else:
                r = d["mean_rg"] / d["mean_uf"]
                se = r * math.hypot(d["sem_rg"] / d["mean_rg"], d["sem_uf"] / d["mean_uf"])
                lo, hi = max(0.05, r - 1.96 * se), r + 1.96 * se
            rats.append(r)
            err_lo.append(max(0.0, r - lo))
            err_hi.append(max(0.0, hi - r))
        ax_b.errorbar(bs, rats, yerr=[err_lo, err_hi], fmt=fmt, color=col, ms=6.0, lw=1.9, capsize=2.8, label=lbl)

    ax_b.set_xlim(4.9, 6.75)
    ax_b.set_ylim(0.32, 1.32)
    ax_b.set_xlabel(r"Inverse Temperature $\beta = 1/T$")
    ax_b.set_ylabel(r"Paired Lifetime Ratio $\bar{\tau}_{\mathrm{RG}} / \bar{\tau}_{\mathrm{UF}}$")
    ax_b.set_title("(b) Local vs. Global Readout (95% CI)")
    ax_b.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.93, fontsize=8.0)

    # --- Panel (c): Readout Cadence Sensitivity at beta = 6.5 ---
    ax_c = fig.add_subplot(gs[0, 2])
    style_ax(ax_c)
    ax_c.axvline(0.20, color="#64748b", ls=":", lw=1.2)
    ax_c.text(0.215, 6.5e4, "Baseline\n" + r"$\eta = 0.20$", fontsize=7.8, color="#475569", fontweight="bold")

    for tid, col, fmt, lbl in [
        ("cadence__sq_iso_L2r1_s4_X_err__b6.50_eta02", C_PLANAR, "-D", "Planar C2 (47,387q)"),
        ("cadence__rottor_iso_L2r1_s4_X_err__b6.50_eta02", C_TORUS, "-o", "Torus C2 (94,774q)"),
        ("cadence__rottor_iso_L2r1_s2_X_err__b6.50", C_TORUS, "--s", "Torus C1 (646q)"),
        ("cadence__sq_iso_L2r1_s2_X_err__b6.50", C_PLANAR, "--^", "Planar C1 (323q)"),
    ]:
        if tid not in cad:
            continue
        d = cad[tid]
        xs_c = [d["min_eta"], 0.05, 0.20, 0.50]
        ys_c = [d["m0"], d["m05"], d["m20"], d["m50"]]
        ses_c = [d["s0"], d["s05"], d["s20"], d["s50"]]
        ax_c.errorbar(xs_c, ys_c, yerr=ses_c, fmt=fmt, color=col, ms=5.8, lw=1.8, capsize=2.5, label=lbl)

    ax_c.set_yscale("log")
    ax_c.set_xlim(-0.02, 0.53)
    ax_c.set_ylim(7.0e3, 2.5e6)
    ax_c.set_xlabel(r"Checkpoint Interval Fraction $\eta$ ($\beta = 6.5$)")
    ax_c.set_ylabel("Detected Failure Time")
    ax_c.set_title("(c) Readout Cadence Sensitivity")
    ax_c.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.93, fontsize=7.5)

    save_fig(fig, "fig_v19_4_readout")


# ==============================================================================
# FIGURE 5: 3D Volume-Growth Excess, 3x3 Sector Balance, & CSS Wire Collapse
# ==============================================================================
def render_fig5(v14, v13):
    tq = v14["tau_q"]
    ws = v13["wire_sweep"]
    wb = v14["wire_matched_beff"]

    fig = plt.figure(figsize=(13.8, 4.6))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.04, 1.06], wspace=0.28)

    # --- Panel (a): 3D Volume-Growth Excess ---
    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a)
    Ls = np.linspace(1.75, 8.2, 300)
    exp_1rib = (np.log(2 * Ls ** 2 + 2 * Ls) / np.log(Ls)) / 3.0 - 1.0
    exp_2rib = (np.log(2 * Ls ** 2 + 4 * Ls) / np.log(Ls)) / 3.0 - 1.0

    ax_a.axhspan(0.0, 0.28, facecolor="#fee2e2", alpha=0.42)
    ax_a.axhspan(-0.22, 0.0, facecolor="#f1f5f9", alpha=0.65)
    ax_a.axhline(0.0, color="#0f172a", lw=1.3, ls="--")

    ax_a.plot(Ls, exp_1rib, "-", color=C_PLANAR, lw=2.2, label=r"1-Rib Cross ($2L^2 + 2L$)")
    ax_a.plot(Ls, exp_2rib, "--", color=C_MACRO, lw=2.0, label=r"2-Rib Cell ($2L^2 + 4L$)")

    L_ints = np.arange(2, 9)
    e1_i = (np.log(2 * L_ints ** 2 + 2 * L_ints) / np.log(L_ints)) / 3.0 - 1.0
    e2_i = (np.log(2 * L_ints ** 2 + 4 * L_ints) / np.log(L_ints)) / 3.0 - 1.0
    ax_a.scatter(L_ints, e1_i, color=C_PLANAR, s=42, zorder=5)
    ax_a.scatter(L_ints, e2_i, color=C_MACRO, marker="s", s=38, zorder=5)

    ax_a.text(5.3, 0.145, r"Must stretch links in $\mathbb{R}^3$", fontsize=8.0, fontweight="bold", color="#991b1b", ha="center")
    ax_a.text(5.8, -0.042, "Not excluded by volume test", fontsize=7.7, color="#475569", ha="center")
    ax_a.annotate(
        r"$L=3$ single cross ($24 < 27$):" + "\nsmallest uniform cell passing\nthis volume-count test",
        xy=(3.0, e1_i[1]),
        xytext=(2.05, -0.205),
        arrowprops=dict(arrowstyle="->", color=C_L3, lw=1.2),
        fontsize=7.5,
        fontweight="bold",
        color="#4c1d95",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#ddd6fe", alpha=0.92),
    )

    ax_a.set_xlim(1.75, 8.35)
    ax_a.set_ylim(-0.225, 0.24)
    ax_a.set_xlabel("Subdivision Scale L")
    ax_a.set_ylabel(r"Volume-Growth Excess $\frac{1}{3}\log_L m(L) - 1$")
    ax_a.set_title("(a) Necessary 3D Volume Count")
    ax_a.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.93, fontsize=7.8)

    # --- Panel (b): 3x3 Single-Cross Candidate & Step-3 Sector Bottleneck ---
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b)
    for s_lv, fmt, mfc, lw_v, lbl in [
        (1, "--o", "white", 1.5, "s=1 half-cycle (41q)"),
        (2, "-s", C_L3, 2.2, "Cycle 1 (1,271q)"),
        (3, "--^", "white", 1.9, "s=3 half-cycle (24,449q, Z-lim.)"),
    ]:
        cn = f"sq_L3r1_s{s_lv}"
        bs = sorted(float(k) for k in tq[cn].keys() if k != "_fit")
        qs = [tq[cn][f"{b:.2f}"]["tau_Q"] for b in bs]
        ses = [tq[cn][f"{b:.2f}"]["sem_Q"] for b in bs]
        ax_b.errorbar(bs, qs, yerr=ses, fmt=fmt, color=C_L3, mfc=mfc, mew=1.7, ms=5.8, lw=lw_v, capsize=2.5, label=lbl)

    bs3 = sorted(float(k) for k in tq["sq_L3r1_s3"].keys() if k != "_fit")
    xs3 = [tq["sq_L3r1_s3"][f"{b:.2f}"]["tau_X"] for b in bs3]
    ax_b.plot(bs3, xs3, ":d", color="#0284c7", ms=5.0, lw=1.5, label=r"$s=3\ \tau_X$ (2 $X$-welds)")

    ax_b.annotate(
        r"$s=3$ crosses $C_1$" + "\n" + r"at $\beta = 7.5$ ($2.3\times$)",
        xy=(7.50, tq["sq_L3r1_s3"]["7.50"]["tau_Q"]),
        xytext=(5.72, 2.6e5),
        arrowprops=dict(arrowstyle="->", color=C_L3, lw=1.2),
        fontsize=7.9,
        fontweight="bold",
        color="#4c1d95",
    )

    ax_b.set_yscale("log")
    ax_b.set_xlim(4.35, 7.68)
    ax_b.set_ylim(15.0, 2.2e8)
    ax_b.set_xlabel(r"Inverse Temperature $\beta = 1/T$")
    ax_b.set_ylabel("Lifetime")
    ax_b.set_title(r"(b) $3\times 3$ Candidate & Sector Balance")
    ax_b.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.94, fontsize=7.4)

    # --- Panel (c): CSS Wire Master Curve + Residuals Strip ---
    gs_c = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 2], height_ratios=[2.5, 1.0], hspace=0.12)
    ax_c = fig.add_subplot(gs_c[0, 0])
    ax_r = fig.add_subplot(gs_c[1, 0], sharex=ax_c)
    style_ax(ax_c)
    style_ax(ax_r)

    ax_c.axvspan(5.00, 5.70, facecolor="#ecfdf5", alpha=0.65)
    ax_r.axvspan(5.00, 5.70, facecolor="#ecfdf5", alpha=0.65)
    ax_c.text(5.35, 2.2e1, "5-w overlap", fontsize=7.3, color="#065f46", ha="center", fontweight="bold")

    w_palette = [
        ("1", "#059669", "-o"),
        ("2", "#0284c7", "-s"),
        ("4", "#2563eb", "-^"),
        ("8", "#d97706", "-D"),
        ("16", "#dc2626", "-v"),
    ]
    tor_w = ws.get("tor", {})
    for wk, col, fmt in w_palette:
        d = tor_w.get(wk) or tor_w.get(int(wk))
        if not d:
            continue
        w_num = float(wk)
        pts = [p for p in d["points"] if p["capped"] == 0]
        ax_c.errorbar(
            [p["beta_eff"] for p in pts],
            [p["mean_t"] / (w_num ** 2) for p in pts],
            yerr=[p["sem_t"] / (w_num ** 2) for p in pts],
            fmt=fmt,
            color=col,
            ms=5.0,
            lw=1.6,
            capsize=2.0,
            label=f"w={wk}",
        )

    ax_r.axhline(0.0, color="#0f172a", ls="--", lw=1.0)
    for wk, col, fmt in w_palette[1:]:
        w_int = int(wk)
        bx, ry = [], []
        for b0_s in ("5.00", "5.50", "6.00"):
            if b0_s in wb.get("tor", {}):
                pts_m = wb["tor"][b0_s]["points"]
                p_w = next((p for p in pts_m if p["w"] == w_int), None)
                if p_w:
                    bx.append(float(b0_s) + 0.018 * math.log2(w_int))
                    ry.append(math.log10(p_w["norm_tau_over_w2_t1"]))
        ax_r.plot(bx, ry, fmt[-1], color=col, ms=4.8)

    ax_c.set_yscale("log")
    ax_c.set_ylabel(r"$\tau(\beta, w) / w^2$")
    ax_c.set_title("(c) Torus CSS Wire Master Curve")
    ax_c.legend(loc="upper left", ncol=2, frameon=True, facecolor="white", framealpha=0.93, fontsize=7.3)
    plt.setp(ax_c.get_xticklabels(), visible=False)

    ax_r.set_ylim(-0.22, 0.22)
    ax_r.set_ylabel(r"$\Delta\log_{10}$")
    ax_r.set_xlabel(r"Exact $\beta_{\mathrm{eff}}(w) = 2\,\mathrm{arctanh}[\tanh^w(\beta/2)]$")

    save_fig(fig, "fig_v19_5_implementation")


def main():
    with open(os.path.join(ROOT, "parsed_v14_data.json")) as f:
        v14 = json.load(f)
    with open(os.path.join(ROOT, "parsed_v13_data.json")) as f:
        v13 = json.load(f)
    render_fig1()
    render_fig2(v14)
    render_fig3(v14)
    render_fig4(v14)
    render_fig5(v14, v13)
    print("All 5 V19 figures rendered successfully.")


if __name__ == "__main__":
    main()
