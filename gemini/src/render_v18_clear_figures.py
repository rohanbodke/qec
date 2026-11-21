#!/usr/bin/env python3
"""
render_v18_clear_figures.py
Renders 5 spacious, clean, high-readability publication figures (V18, PDF + 300 DPI PNG)
with concise titles, large typography, intuitive visual schematics, and zero cramped insets:

  Fig. 1 (fig_v18_1_welding):
    (a) Flat Sheet vs. Odd X-Weld
    (b) Even Z-Weld & Ablations
    (c) Wall-by-Wall 2/5 Contraction

  Fig. 2 (fig_v18_2_readout):
    (a) Local Cell Readout
    (b) Defect Contraction Across Scales
    (c) Local vs. Global Readout

  Fig. 3 (fig_v18_3_lifetimes):
    (a) Quantum Lifetime Across Scales
    (b) Effective Barrier vs. Qubit Count

  Fig. 4 (fig_v18_4_packing):
    (a) 3D Volume Packing Bound
    (b) 3x3 Candidate Quantum Lifetime

  Fig. 5 (fig_v18_5_wires):
    (a) Direct Graph vs. 3D Grid Wire
    (b) Lifetime vs. Wire Length
    (c) Collapse Under Effective Bath
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
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures")
ART_DIR = "/Users/harshilavlani/.gemini/antigravity/brain/35299b0d-7439-4f8e-8617-0b646151a931"

sys.path.insert(0, os.path.join(ROOT, "src"))
from render_v13_unified_figures import load_all
from parse_v13_results import fit_slope_with_se

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10.2,
    "axes.titlesize": 11.4,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.4,
    "xtick.labelsize": 9.3,
    "ytick.labelsize": 9.3,
    "legend.fontsize": 8.8,
    "axes.linewidth": 1.0,
    "grid.linewidth": 0.55,
    "grid.alpha": 0.35,
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
# FIGURE 1: Two-Step Welding Geometry & Wall-by-Wall 2/5 Contraction
# ==============================================================================
def render_fig1():
    fig = plt.figure(figsize=(13.4, 4.5))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.02, 1.04, 1.04], wspace=0.18)

    # --- Panel (a): Odd X-Weld (Blue 4-Arm Bridge Cross) ---
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_xlim(0, 40)
    ax_a.set_ylim(0, 34)
    ax_a.axis("off")
    ax_a.add_patch(FancyBboxPatch((0.5, 0.5), 39.0, 33.0, boxstyle="round,pad=0.25,rounding_size=1.1",
                                  facecolor="#f8fafc", edgecolor="#cbd5e1", lw=1.0))
    ax_a.text(20.0, 31.3, "(a) Odd Step: Face-Bridge X-Weld", ha="center", va="center",
              fontsize=11.0, fontweight="bold", color="#0f172a")

    # Draw Sheet 0 & Sheet 1 in clean isometric projection
    for sh_y, lbl in [(20.5, "Sheet 0"), (9.5, "Sheet 1")]:
        pts = [[5.5, sh_y], [25.5, sh_y], [31.5, sh_y + 5.6], [11.5, sh_y + 5.6]]
        ax_a.add_patch(Polygon(pts, closed=True, facecolor="#eff6ff", edgecolor="#64748b", lw=1.2))
        # 2x2 grid lines (blue cross in middle)
        ax_a.plot([15.5, 21.5], [sh_y, sh_y + 5.6], color="#1d4ed8", lw=2.6)
        ax_a.plot([8.5, 28.5], [sh_y + 2.8, sh_y + 2.8], color="#1d4ed8", lw=2.6)
        ax_a.text(35.2, sh_y + 2.8, lbl, ha="center", va="center", fontsize=8.8, color="#334155", fontweight="bold")

    # Vertical bridge cross between sheets
    ax_a.add_patch(Polygon([[8.5, 12.3], [28.5, 12.3], [28.5, 23.3], [8.5, 23.3]],
                           closed=True, facecolor="#93c5fd", edgecolor="#1d4ed8", lw=1.1, alpha=0.30))
    ax_a.add_patch(Polygon([[15.5, 9.5], [21.5, 15.1], [21.5, 26.1], [15.5, 20.5]],
                           closed=True, facecolor="#60a5fa", edgecolor="#1d4ed8", lw=1.1, alpha=0.32))
    # Highlight 1 rib qubit shared by 3 Z-faces
    ax_a.scatter([18.5], [23.3], s=55, color="#1d4ed8", zorder=6)
    ax_a.annotate("3 Z-faces share\n1 rib qubit", xy=(18.5, 23.3), xytext=(3.0, 26.8),
                  arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=1.2),
                  fontsize=8.3, fontweight="bold", color="#1e3a8a")

    ax_a.text(20.0, 5.4, r"Crossing $X$ defect splits $1 \to 2$ ($+$energy)",
              ha="center", va="center", fontsize=8.7, fontweight="bold", color="#1e3a8a")
    ax_a.text(20.0, 2.5, r"Flat 2D sheet alone: defects hop at $\Delta E = 0$",
              ha="center", va="center", fontsize=8.2, color="#64748b")

    # --- Panel (b): Even Z-Weld & Two Modifications ---
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_xlim(0, 40)
    ax_b.set_ylim(0, 34)
    ax_b.axis("off")
    ax_b.add_patch(FancyBboxPatch((0.5, 0.5), 39.0, 33.0, boxstyle="round,pad=0.25,rounding_size=1.1",
                                  facecolor="#f8fafc", edgecolor="#cbd5e1", lw=1.0))
    ax_b.text(20.0, 31.3, "(b) Even Z-Weld & Simplifications", ha="center", va="center",
              fontsize=10.9, fontweight="bold", color="#0f172a")

    for sh_y in (23.0, 15.2):
        pts = [[6.5, sh_y], [24.5, sh_y], [29.5, sh_y + 4.2], [11.5, sh_y + 4.2]]
        ax_b.add_patch(Polygon(pts, closed=True, facecolor="#fef2f2", edgecolor="#64748b", lw=1.1))
        ax_b.plot([15.5, 20.5], [sh_y, sh_y + 4.2], color="#dc2626", lw=2.2, ls="--")
        ax_b.plot([9.0, 27.0], [sh_y + 2.1, sh_y + 2.1], color="#dc2626", lw=2.2, ls="--")
    for cx in (9.0, 18.0, 27.0):
        ax_b.plot([cx, cx], [17.3, 25.1], color="#dc2626", lw=1.8)
        ax_b.scatter([cx, cx], [17.3, 25.1], s=24, color="#dc2626", zorder=5)
    ax_b.text(34.3, 21.2, "2 rib q.\nbranch\nX-stars", ha="center", va="center",
              fontsize=8.2, fontweight="bold", color="#991b1b")
    ax_b.text(20.0, 12.6, r"Full 2-step cycle: distance $(2,1) \to (11,10)$",
              ha="center", va="center", fontsize=8.5, fontweight="bold", color="#0f172a")

    # Two clean visual simplification cards at bottom
    ax_b.add_patch(FancyBboxPatch((2.0, 1.8), 17.2, 8.8, boxstyle="round,pad=0.18,rounding_size=0.7",
                                  facecolor="#fff1f2", edgecolor="#fecdd3", lw=1.0))
    ax_b.text(10.6, 8.5, "Cut 1 Bridge Arm", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#9f1239")
    ax_b.text(10.6, 4.6, "Opens interior gap;\ndistance " + r"$\to (6, 5)$", ha="center", va="center", fontsize=8.0, color="#881337")

    ax_b.add_patch(FancyBboxPatch((20.8, 1.8), 17.2, 8.8, boxstyle="round,pad=0.18,rounding_size=0.7",
                                  facecolor="#fff7ed", edgecolor="#fed7aa", lw=1.0))
    ax_b.text(29.4, 8.5, "1-Step X+Z Cross", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#9a3412")
    ax_b.text(29.4, 4.6, "Seam corner clash;\ndistance " + r"$\to (6, 1)$", ha="center", va="center", fontsize=8.0, color="#7c2d12")

    # --- Panel (c): Wall-by-Wall 2/5 Contraction (Zero Casework!) ---
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.set_xlim(0, 40)
    ax_c.set_ylim(0, 34)
    ax_c.axis("off")
    ax_c.add_patch(FancyBboxPatch((0.5, 0.5), 39.0, 33.0, boxstyle="round,pad=0.25,rounding_size=1.1",
                                  facecolor="#f8fafc", edgecolor="#cbd5e1", lw=1.0))
    ax_c.text(20.0, 31.3, "(c) Five-Edge Wall Contraction", ha="center", va="center",
              fontsize=11.0, fontweight="bold", color="#0f172a")

    ax_c.text(20.0, 28.0, "One outer wall = 5 sub-edges, 2 interior ribs:",
              ha="center", va="center", fontsize=8.4, fontweight="bold", color="#1e293b")

    boxes = [
        (3.0,  "#bfdbfe", "#1d4ed8", "1"),
        (9.2,  "#bfdbfe", "#1d4ed8", "1"),
        (17.2, "#dbeafe", "#1e40af", "0"),
        (25.2, "#ffffff", "#94a3b8", "0"),
        (31.4, "#ffffff", "#94a3b8", "0"),
    ]
    for bx, fc, ec, val in boxes:
        ax_c.add_patch(Rectangle((bx, 19.8), 5.4, 4.3, facecolor=fc, edgecolor=ec, lw=1.5))
        ax_c.text(bx + 2.7, 21.95, val, ha="center", va="center", fontsize=9.6, fontweight="bold", color="#0f172a")
    ax_c.text(8.8, 25.4, "Sheet 0", ha="center", va="center", fontsize=7.8, fontweight="bold", color="#1e3a8a")
    ax_c.text(19.9, 25.4, "Bridge", ha="center", va="center", fontsize=7.8, fontweight="bold", color="#1e40af")
    ax_c.text(31.0, 25.4, "Sheet 1", ha="center", va="center", fontsize=7.8, fontweight="bold", color="#475569")

    ax_c.plot([8.4, 15.8], [19.6, 16.7], color="#dc2626", lw=1.8)
    ax_c.plot([19.5, 15.8], [19.6, 16.7], color="#dc2626", lw=1.8)
    ax_c.text(11.2, 15.3, "Rib 0 violated", ha="center", va="center", fontsize=8.0, fontweight="bold", color="#b91c1c")

    ax_c.plot([20.3, 27.6], [19.6, 16.7], color="#16a34a", lw=1.6, ls="--")
    ax_c.plot([31.0, 27.6], [19.6, 16.7], color="#16a34a", lw=1.6, ls="--")
    ax_c.text(29.8, 15.3, "Rib 1 clean", ha="center", va="center", fontsize=8.0, fontweight="bold", color="#15803d")

    ax_c.add_patch(FancyBboxPatch((2.2, 1.8), 35.6, 11.8, boxstyle="round,pad=0.2,rounding_size=0.8",
                                  facecolor="#ffffff", edgecolor="#cbd5e1", lw=1.0))
    ax_c.text(20.0, 11.2, "0 ribs violated " + r"$\to$" + " 0 minority wall bits\n1 rib violated " + r"$\to$" + " at most 2 minority bits",
              ha="center", va="center", fontsize=8.2, color="#334155")
    ax_c.text(20.0, 6.3, r"Sum over 4 disjoint walls:  $W / 5 \leq \frac{2}{5}\,V$",
              ha="center", va="center", fontsize=9.3, fontweight="bold", color="#1e3a8a")
    ax_c.text(20.0, 3.1, "(Sheet 0 flip is tight: " + r"$W = 8,\ V = 4 \to 2/5$" + ")",
              ha="center", va="center", fontsize=7.8, color="#64748b")

    save_fig(fig, "fig_v18_1_welding")


# ==============================================================================
# FIGURE 2: Local Readout & Coarse Defect Contraction
# ==============================================================================
def render_fig2(v13):
    fig = plt.figure(figsize=(13.6, 4.4))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[0.95, 1.02, 1.05], wspace=0.27)
    rg = v13["rgflow_v13"]
    pk = v13["paired_kmc"]

    # --- Panel (a): Visual Step-by-Step Diagram of Local Cell Readout ---
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_xlim(0, 38)
    ax_a.set_ylim(0, 34)
    ax_a.axis("off")
    ax_a.add_patch(FancyBboxPatch((0.5, 0.5), 37.0, 33.0, boxstyle="round,pad=0.25,rounding_size=1.1",
                                  facecolor="#f8fafc", edgecolor="#cbd5e1", lw=1.0))
    ax_a.text(19.0, 31.3, "(a) Hierarchical Readout Steps", ha="center", va="center",
              fontsize=11.0, fontweight="bold", color="#0f172a")

    steps = [
        (22.2, "#eff6ff", "#93c5fd", "1. Clear Cell Interiors",
         "Push interior rib syndromes\nonto each cell's 4 outer walls"),
        (12.2, "#ecfdf5", "#6ee7b7", "2. Five-Edge Wall Vote",
         "Majority vote on each 5-edge\nwall " + r"$\to$" + " coarse syndrome one scale up"),
        (2.2,  "#fffbeb", "#fcd34d", "3. Seed Solve & Seam Clean",
         "Solve tiny seed code, lift flips,\nclear residual seam chains"),
    ]
    for sy, fc, ec, stitle, sbody in steps:
        ax_a.add_patch(FancyBboxPatch((2.4, sy), 33.2, 7.2, boxstyle="round,pad=0.18,rounding_size=0.8",
                                      facecolor=fc, edgecolor=ec, lw=1.1))
        ax_a.text(19.0, sy + 5.1, stitle, ha="center", va="center", fontsize=9.1, fontweight="bold", color="#0f172a")
        ax_a.text(19.0, sy + 2.1, sbody, ha="center", va="center", fontsize=8.1, color="#334155")

    for ay in (22.0, 12.0):
        ax_a.annotate("", xy=(19.0, ay - 2.4), xytext=(19.0, ay - 0.1),
                      arrowprops=dict(arrowstyle="->", color="#475569", lw=1.6))

    # --- Panel (b): Defect Count Contraction Across Scales (N = 94,774 Torus) ---
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b)
    floor_b = 4.0e-3
    palette_b = [
        (3.50, "#dc2626", "-o", r"$\beta = 3.5$ (warm)"),
        (4.50, "#ea580c", "-s", r"$\beta = 4.5$"),
        (5.50, "#ca8a04", "-D", r"$\beta = 5.5$"),
        (6.00, "#16a34a", "-^", r"$\beta = 6.0$"),
        (6.50, "#2563eb", "-v", r"$\beta = 6.5$"),
        (7.00, "#7c3aed", "-P", r"$\beta = 7.0$ (cold)"),
    ]
    xs_b = list(range(5))
    for b_val, col, fmt, lbl in palette_b:
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
        ax_b.errorbar(xs_b, ys, yerr=se_y, fmt=fmt, color=col, ms=5.5, lw=1.8, capsize=2.4, label=lbl)
        zx = [x for x, z in zip(xs_b, is_z) if z]
        zy = [y for y, z in zip(ys, is_z) if z]
        if zx:
            ax_b.scatter(zx, zy, marker="v", s=58, facecolors="white", edgecolors=col, linewidths=1.7, zorder=6)

    ax_b.axhline(floor_b, color="#64748b", ls=":", lw=1.1)
    ax_b.text(0.12, floor_b * 1.45, "Zero defects (0 / 96 runs)", fontsize=8.2, color="#475569")
    ax_b.set_yscale("log")
    ax_b.set_ylim(floor_b * 0.55, 3.2e4)
    ax_b.set_xticks(xs_b)
    ax_b.set_xticklabels(["s=4\n(41.5k)", "s=3\n(3.5k)", "s=2\n(288)", "s=1\n(24)", "Seed\n(2)"])
    ax_b.set_xlabel("Scale s (Check Count)")
    ax_b.set_ylabel("Mean Violated Checks")
    ax_b.set_title("(b) Defect Contraction (94,774q Torus)")
    ax_b.legend(loc="upper right", ncol=2, frameon=True, facecolor="white", framealpha=0.93, fontsize=7.8)

    # --- Panel (c): Paired Local vs Global Readout + Clean Runtime Inset ---
    ax_c = fig.add_subplot(gs[0, 2])
    style_ax(ax_c)
    for s_lv, col, lbl_base in [(2, "#2563eb", "Cycle 1 (646q)"), (4, "#dc2626", "Cycle 2 (94,774q)")]:
        rows = sorted(
            [(d["beta"], d) for k, d in pk.items() if k.startswith(f"paired_kmc__rottor_s{s_lv}__")],
            key=lambda t: t[0],
        )
        bs = [b for b, _ in rows]
        ax_c.errorbar(bs, [d["mean_uf"] for _, d in rows], yerr=[d["sem_uf"] for _, d in rows],
                      fmt="-o", color=col, ms=5.6, lw=1.9, capsize=2.6, label=f"{lbl_base} Global")
        ax_c.errorbar(bs, [d["mean_rg"] for _, d in rows], yerr=[d["sem_rg"] for _, d in rows],
                      fmt="--s", color=col, mfc="white", mew=1.6, ms=5.6, lw=1.6, capsize=2.6, label=f"{lbl_base} Local RG")

    # Clean non-overlapping bottom-right inset: linear O(N) runtime up to 322k qubits
    ins = ax_c.inset_axes([0.57, 0.08, 0.39, 0.30])
    ins.grid(True, alpha=0.3)
    ns_rx, uf_rx, rg_rx = [], [], []
    for s_lv in range(1, 6):
        k0 = f"paired_kmc__sq_rx_s{s_lv}__b2.20"
        if k0 in pk:
            ns_rx.append(pk[k0]["N"])
            uf_rx.append(pk[k0]["us_uf"] / 1e3)
            rg_rx.append(pk[k0]["us_rg"] / 1e3)
    ins.plot(ns_rx, uf_rx, "-o", color="#475569", ms=3.5, lw=1.3, label="Global")
    ins.plot(ns_rx, rg_rx, "-s", color="#059669", ms=3.5, lw=1.4, label="Local")
    ins.set_xscale("log"); ins.set_yscale("log")
    ins.set_title("Time (ms) vs. N", fontsize=7.6, fontweight="bold")
    ins.tick_params(labelsize=7.0)

    ax_c.set_yscale("log")
    ax_c.set_ylim(90.0, 1.2e7)
    ax_c.set_xlabel(r"Inverse Temperature $\beta = 1/T$")
    ax_c.set_ylabel("Memory Lifetime")
    ax_c.set_title("(c) Local vs. Global Readout")
    ax_c.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.93, fontsize=8.0)

    save_fig(fig, "fig_v18_2_readout")


# ==============================================================================
# FIGURE 3: Two-Sector Quantum Lifetimes & Effective Energy Barriers
# ==============================================================================
def render_fig3(v13, kmc4, kmc6):
    tq = v13["two_sector_tau_Q"]
    fig = plt.figure(figsize=(12.8, 4.6))
    gs = GridSpec(1, 2, figure=fig, wspace=0.24)

    # --- Panel (a): Quantum Lifetime vs Inverse Temperature ---
    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a)
    q_series = [
        ("sq_iso_L2r1_s1", "#94a3b8", "-o",  "Planar s=1 (19q)"),
        ("sq_iso_L2r1_s2", "#0ea5e9", "--s", "Planar Cycle 1 (323q)"),
        ("sq_iso_L2r1_s3", "#f97316", "-^",  "Planar s=3 (2,911q)"),
        ("sq_iso_L2r1_s4", "#2563eb", "--D", "Planar Cycle 2 (47,387q)"),
        ("sq_iso_L2r1_s5", "#be123c", "-P",  "Planar s=5 (423,559q)"),
        ("rottor_iso_L2r1_s4", "#059669", "-*", "Torus Cycle 2 (94,774q)"),
    ]
    for cname, col, fmt, lbl in q_series:
        if cname not in tq:
            continue
        pts = tq[cname]["points"]
        ax_a.errorbar([p["beta"] for p in pts], [p["tau_Q"] for p in pts], yerr=[p["sem_Q"] for p in pts],
                      fmt=fmt, color=col, ms=5.8, lw=1.9, capsize=2.6, label=lbl)

    ax_a.annotate("17,250x gain\n" + r"(19q $\to$ 423,559q)", xy=(6.50, 1.25e6), xytext=(5.68, 5.5e5),
                  arrowprops=dict(arrowstyle="->", color="#be123c", lw=1.3),
                  fontsize=8.3, fontweight="bold", color="#9f1239")

    ax_a.set_yscale("log")
    ax_a.set_xlabel(r"Inverse Temperature $\beta = 1/T$")
    ax_a.set_ylabel(r"Quantum Lifetime $\tau_Q$")
    ax_a.set_title("(a) Quantum Lifetime Across Scales")
    ax_a.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.93, fontsize=8.3)

    # --- Panel (b): Effective Energy Barrier vs Qubit Count N ---
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b)

    def fam_slopes(cnames, bmin=3.5, bmax=5.0):
        xs, ys, ses = [], [], []
        for cn in cnames:
            bm = kmc4.get(cn, {})
            bs = sorted(b for b in bm if bmin <= b <= bmax)
            if len(bs) >= 3:
                ms = [bm[b][1] for b in bs]
                sl, se, _ = fit_slope_with_se(bs, ms, [0.12 * m for m in ms])
                xs.append(bm[bs[0]][0]); ys.append(sl); ses.append(se)
        return xs, ys, ses

    hN, hS, hE = fam_slopes([f"haah_cubic_L{L}_Xerr.code" for L in [3, 5, 7, 9, 11]])
    bN, bS, bE = fam_slopes([f"bb_gross_{l}x{m}_Xerr.code" for l, m in [(6,6),(12,6),(18,12),(24,18),(36,18),(60,18)]])
    tN, tS, tE = fam_slopes([f"toric2d_L{L}_Xerr.code" for L in [4, 6, 10, 16, 24, 34]])

    q_odd_N = [tq[f"sq_iso_L2r1_s{s}"]["N"] for s in (1, 3, 5)]
    q_odd_S = [tq[f"sq_iso_L2r1_s{s}"]["slope"] for s in (1, 3, 5)]
    q_odd_E = [tq[f"sq_iso_L2r1_s{s}"]["se_slope"] for s in (1, 3, 5)]
    ax_b.errorbar(q_odd_N, q_odd_S, yerr=q_odd_E, fmt="-P", color="#be123c", ms=6.5, lw=2.1, capsize=2.8,
                  label="2x2 Welded Planar")

    tor_N = [tq[f"rottor_iso_L2r1_s{s}"]["N"] for s in (2, 4)]
    tor_S = [tq[f"rottor_iso_L2r1_s{s}"]["slope"] for s in (2, 4)]
    tor_E = [tq[f"rottor_iso_L2r1_s{s}"]["se_slope"] for s in (2, 4)]
    ax_b.errorbar(tor_N, tor_S, yerr=tor_E, fmt="--*", color="#059669", ms=7.0, lw=1.9, capsize=2.8,
                  label="2x2 Welded Torus")

    l3_N = [tq[f"sq_L3r1_s{s}"]["N"] for s in (1, 2, 3)]
    l3_S = [tq[f"sq_L3r1_s{s}"]["slope"] for s in (1, 2, 3)]
    l3_E = [tq[f"sq_L3r1_s{s}"]["se_slope"] for s in (1, 2, 3)]
    ax_b.errorbar(l3_N, l3_S, yerr=l3_E, fmt="-^", color="#0284c7", ms=5.8, lw=1.8, capsize=2.6,
                  label="3x3 Welded Candidate")

    if hN:
        ax_b.errorbar(hN, hS, yerr=hE, fmt="-D", color="#7c3aed", ms=5.2, lw=1.6, capsize=2.4,
                      label="3D Haah Cubic")

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
        ax_b.errorbar(lay_N, lay_S, yerr=lay_E, fmt="-p", color="#d97706", ms=5.5, lw=1.6, capsize=2.4,
                      label="3D Random Layer")
    if bN:
        ax_b.errorbar(bN, bS, yerr=bE, fmt="--s", color="#2563eb", ms=4.8, lw=1.4, capsize=2.2,
                      label="2D Bivariate Bicycle")
    if tN:
        ax_b.errorbar(tN, tS, yerr=tE, fmt=":o", color="#64748b", ms=4.8, lw=1.4, capsize=2.2,
                      label="2D Toric Code")

    ax_b.set_xscale("log")
    ax_b.set_ylim(0.3, 12.3)
    ax_b.set_xlabel("Physical Qubits N")
    ax_b.set_ylabel(r"Effective Barrier Slope $\Delta_{\mathrm{eff}}$")
    ax_b.set_title("(b) Effective Barrier vs. System Size")
    ax_b.legend(loc="upper left", ncol=2, frameon=True, facecolor="white", framealpha=0.93, fontsize=7.9)

    save_fig(fig, "fig_v18_3_lifetimes")


# ==============================================================================
# FIGURE 4: 3D Space Packing Bound & 3x3 Candidate Scaling
# ==============================================================================
def render_fig4(v13):
    tq = v13["two_sector_tau_Q"]
    fig = plt.figure(figsize=(12.4, 4.4))
    gs = GridSpec(1, 2, figure=fig, wspace=0.25)

    # --- Panel (a): 3D Volume Packing Bound ---
    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a)
    Ls = np.linspace(1.75, 8.2, 300)
    exp_1rib = (np.log(2 * Ls ** 2 + 2 * Ls) / np.log(Ls)) / 3.0 - 1.0
    exp_2rib = (np.log(2 * Ls ** 2 + 4 * Ls) / np.log(Ls)) / 3.0 - 1.0

    ax_a.axhspan(0.0, 0.28, facecolor="#fee2e2", alpha=0.45)
    ax_a.axhspan(-0.22, 0.0, facecolor="#dcfce7", alpha=0.42)
    ax_a.axhline(0.0, color="#0f172a", lw=1.3, ls="--")

    ax_a.plot(Ls, exp_1rib, "-", color="#2563eb", lw=2.2, label=r"Single Cross ($2L^2 + 2L$ faces)")
    ax_a.plot(Ls, exp_2rib, "--", color="#dc2626", lw=2.0, label=r"Two-Rib Cell ($2L^2 + 4L$ faces)")

    L_ints = np.arange(2, 9)
    e1_i = (np.log(2 * L_ints ** 2 + 2 * L_ints) / np.log(L_ints)) / 3.0 - 1.0
    e2_i = (np.log(2 * L_ints ** 2 + 4 * L_ints) / np.log(L_ints)) / 3.0 - 1.0
    ax_a.scatter(L_ints, e1_i, color="#1d4ed8", s=44, zorder=5)
    ax_a.scatter(L_ints, e2_i, color="#b91c1c", marker="s", s=40, zorder=5)

    ax_a.annotate("2x2 Single Cross (12 > 8):\nmust stretch links in 3D",
                  xy=(2.0, e1_i[0]), xytext=(2.8, 0.105),
                  arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=1.2),
                  fontsize=8.5, fontweight="bold", color="#1e3a8a")
    ax_a.annotate("3x3 Single Cross (24 < 27):\nsmallest cell fitting " + r"$L^3$" + " volume",
                  xy=(3.0, e1_i[1]), xytext=(2.35, -0.185),
                  arrowprops=dict(arrowstyle="->", color="#065f46", lw=1.2),
                  fontsize=8.5, fontweight="bold", color="#065f46")

    ax_a.set_xlim(1.75, 8.35)
    ax_a.set_ylim(-0.21, 0.25)
    ax_a.set_xlabel("Subdivision Scale L")
    ax_a.set_ylabel("3D Link-Stretch Exponent")
    ax_a.set_title("(a) 3D Space Packing Bound")
    ax_a.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.93, fontsize=8.4)

    # --- Panel (b): Clean Uncluttered 3x3 Candidate Quantum Lifetimes ---
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b)
    for s_lv, col, fmt, lbl in [
        (1, "#64748b", "-o",  "Step 1 (41 qubits)"),
        (2, "#2563eb", "--s", "Cycle 1 (1,271 qubits)"),
        (3, "#059669", "-^",  "Step 3 (24,449 qubits)"),
    ]:
        cn = f"sq_L3r1_s{s_lv}"
        if cn not in tq:
            continue
        pts = tq[cn]["points"]
        ax_b.errorbar([p["beta"] for p in pts], [p["tau_Q"] for p in pts], yerr=[p["sem_Q"] for p in pts],
                      fmt=fmt, color=col, ms=6.0, lw=2.0, capsize=2.8, label=lbl)

    ax_b.annotate("109x gain at " + r"$\beta = 7.0$" + "\n(685x at " + r"$\beta = 7.5$" + ")",
                  xy=(7.0, 2.54e4), xytext=(5.15, 1.3e4),
                  arrowprops=dict(arrowstyle="->", color="#059669", lw=1.3),
                  fontsize=8.5, fontweight="bold", color="#065f46")

    ax_b.set_yscale("log")
    ax_b.set_ylim(16.0, 5.5e4)
    ax_b.set_xlabel(r"Inverse Temperature $\beta = 1/T$")
    ax_b.set_ylabel(r"Quantum Lifetime $\tau_Q$")
    ax_b.set_title("(b) 3x3 Candidate Quantum Lifetime")
    ax_b.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.93, fontsize=8.6)

    save_fig(fig, "fig_v18_4_packing")


# ==============================================================================
# FIGURE 5: Why 3D Cubic Grids Add Wires & Universal Effective-Bath Collapse
# ==============================================================================
def render_fig5(v13):
    ws = v13["wire_sweep"]
    fig = plt.figure(figsize=(13.6, 4.5))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.08, 1.0, 1.0], wspace=0.28)

    # --- Panel (a): Physical Explanation of Direct Graph vs Stretched 3D Grid Wire ---
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_xlim(0, 42)
    ax_a.set_ylim(0, 34)
    ax_a.axis("off")
    ax_a.add_patch(FancyBboxPatch((0.5, 0.5), 41.0, 33.0, boxstyle="round,pad=0.25,rounding_size=1.1",
                                  facecolor="#f8fafc", edgecolor="#cbd5e1", lw=1.0))
    ax_a.text(21.0, 31.3, "(a) Why 3D Grids Stretch Wires", ha="center", va="center",
              fontsize=11.0, fontweight="bold", color="#0f172a")

    # 1. Direct graph connection (w = 1)
    ax_a.add_patch(FancyBboxPatch((2.2, 22.0), 37.6, 6.8, boxstyle="round,pad=0.15,rounding_size=0.7",
                                  facecolor="#ecfdf5", edgecolor="#6ee7b7", lw=1.0))
    ax_a.text(3.6, 26.8, "Direct Graph (w = 1 link):", fontsize=8.6, fontweight="bold", color="#065f46", va="center")
    ax_a.add_patch(Rectangle((6.5, 22.9), 7.2, 2.5, facecolor="#dbeafe", edgecolor="#1d4ed8", lw=1.2))
    ax_a.text(10.1, 24.15, "Sheet", ha="center", va="center", fontsize=8.1, fontweight="bold", color="#1e3a8a")
    ax_a.plot([13.7, 25.5], [24.15, 24.15], color="#059669", lw=2.6)
    ax_a.scatter([19.6], [24.15], s=50, color="#059669", zorder=5)
    ax_a.add_patch(Rectangle((25.5, 22.9), 7.6, 2.5, facecolor="#dbeafe", edgecolor="#1d4ed8", lw=1.2))
    ax_a.text(29.3, 24.15, "Bridge", ha="center", va="center", fontsize=8.1, fontweight="bold", color="#1e3a8a")
    ax_a.text(36.6, 24.15, "1 spot", ha="center", va="center", fontsize=8.2, fontweight="bold", color="#065f46")

    # 2. Cubic grid wire (w links)
    ax_a.add_patch(FancyBboxPatch((2.2, 10.2), 37.6, 10.4, boxstyle="round,pad=0.15,rounding_size=0.7",
                                  facecolor="#fef2f2", edgecolor="#fecaca", lw=1.0))
    ax_a.text(3.6, 18.9, "Routed 3D Grid (chain of w wire links):", fontsize=8.5, fontweight="bold", color="#991b1b", va="center")
    ax_a.add_patch(Rectangle((3.6, 14.8), 6.2, 2.4, facecolor="#dbeafe", edgecolor="#1d4ed8", lw=1.2))
    ax_a.text(6.7, 16.0, "Sheet", ha="center", va="center", fontsize=7.9, fontweight="bold", color="#1e3a8a")
    ax_a.plot([9.8, 32.2], [16.0, 16.0], color="#dc2626", lw=2.2, ls="--")
    for wx in (13.6, 18.2, 22.8, 27.4):
        ax_a.scatter([wx], [16.0], s=36, color="#dc2626", zorder=5)
    ax_a.add_patch(Rectangle((32.2, 14.8), 6.5, 2.4, facecolor="#dbeafe", edgecolor="#1d4ed8", lw=1.2))
    ax_a.text(35.45, 16.0, "Bridge", ha="center", va="center", fontsize=7.9, fontweight="bold", color="#1e3a8a")
    ax_a.text(21.0, 12.1, r"Free hops ($\Delta E = 0$) across $w$ spots" + "\n" + r"multiply defect weight by $w$",
              ha="center", va="center", fontsize=7.8, fontweight="bold", color="#991b1b")

    # 3. Bottom intuitive takeaway box
    ax_a.add_patch(FancyBboxPatch((2.2, 1.4), 37.6, 7.8, boxstyle="round,pad=0.15,rounding_size=0.7",
                                  facecolor="#ffffff", edgecolor="#cbd5e1", lw=1.0))
    ax_a.text(21.0, 7.1, r"Acts like a warmer bath:  $\beta_{\mathrm{eff}}(w) \approx \beta - \ln w$",
              ha="center", va="center", fontsize=8.5, fontweight="bold", color="#1e3a8a")
    ax_a.text(21.0, 3.6, "Scale-8 cubic grid (56,312q) loses 440x\nlifetime vs. direct graph (1,480q)",
              ha="center", va="center", fontsize=7.8, color="#334155")

    # --- Panel (b): Lifetime vs Wire Length w on Fixed 646q Torus Core ---
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
                      fmt=fmt, color=col, ms=5.5, lw=1.9, capsize=2.5,
                      label=f"w = {wk} ({d['N']:,}q)")

    t_w1_65 = next((p["mean_t"] for p in (tor_w.get("1") or tor_w.get(1))["points"] if abs(p["beta"] - 6.5) < 1e-4), None)
    t_w16_65 = next((p["mean_t"] for p in (tor_w.get("16") or tor_w.get(16))["points"] if abs(p["beta"] - 6.5) < 1e-4), None)
    if t_w1_65 and t_w16_65:
        ax_b.annotate("", xy=(6.50, t_w16_65 * 1.15), xytext=(6.50, t_w1_65 * 0.88),
                      arrowprops=dict(arrowstyle="<->", color="#991b1b", lw=1.6))
        ax_b.text(6.62, math.sqrt(t_w1_65 * t_w16_65), "32x drop",
                  fontsize=8.3, fontweight="bold", color="#991b1b", va="center",
                  bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#fecaca", lw=0.8, alpha=0.92))

    ax_b.set_xlim(4.85, 8.65)
    ax_b.set_yscale("log")
    ax_b.set_xlabel(r"Bath Inverse Temperature $\beta = 1/T$")
    ax_b.set_ylabel("Torus Lifetime")
    ax_b.set_title("(b) Lifetime vs. Wire Length w")
    ax_b.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.93, fontsize=8.3)

    # --- Panel (c): Clean Master Collapse (No Crowded Inset!) ---
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
                      fmt=fmt, color=col, ms=5.6, lw=1.8, capsize=2.5,
                      label=f"w = {wk}")

    ax_c.text(0.06, 0.84, "All 5 wire lengths\ncollapse onto 1 curve",
              transform=ax_c.transAxes, fontsize=8.6, fontweight="bold", color="#065f46",
              bbox=dict(boxstyle="round,pad=0.28", facecolor="#ecfdf5", edgecolor="#6ee7b7", lw=0.9))

    ax_c.set_yscale("log")
    ax_c.set_xlabel(r"Effective Bath $\beta_{\mathrm{eff}}(w) \approx \beta - \ln w$")
    ax_c.set_ylabel(r"Rescaled Lifetime $\tau / w^2$")
    ax_c.set_title("(c) Effective-Bath Collapse")
    ax_c.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.93, fontsize=8.4)

    save_fig(fig, "fig_v18_5_wires")


def main():
    v13_path = os.path.join(ROOT, "parsed_v13_data.json")
    with open(v13_path) as f:
        v13 = json.load(f)
    kmc4, kmc6, *_ = load_all()
    render_fig1()
    render_fig2(v13)
    render_fig3(v13, kmc4, kmc6)
    render_fig4(v13)
    render_fig5(v13)
    print("All 5 V18 figures rendered successfully.")


if __name__ == "__main__":
    main()
