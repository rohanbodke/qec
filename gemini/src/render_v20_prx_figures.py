#!/usr/bin/env python3
"""
render_v20_prx_figures.py
Renders clean APS / Physical Review X (PRX) figures for the Selective Multi-Scale
Hierarchy manuscript ("Selective Multi-Scale Protection in Recursively Welded Quantum Memories"):
  - Serif + Computer Modern math typography matching REVTeX 4.2
  - Full 4-sided spines with inward ticks (top & right enabled)
  - Bold (a), (b), (c), (d) panel labels ONLY; zero subplot titles
  - Clean minimalist canvas: zero cluttered callout boxes or arrows inside data plots

Figures:
  1. `fig_v20_1_modules_and_claim1`: Local modules (C vs R), the four two-cycle
     architectures (CC, RC, CR, RR), and matched-temperature single-cycle lifetimes (Claim 1).
  2. `fig_v20_2_architectures_claim2`: Two-cycle storage performance across CC, RC, CR, RR
     for same-subdivision L=3 (C3 vs R3) and unit-patch (C2 vs R3), planar/torus RC-to-CR
     ratios, and storage-vs-qubit resource comparison (Claim 2).
  3. `fig_v20_3_mechanism_claim3`: Single-cycle effective inverse-temperature map Phi_M(beta),
     zero-parameter out-of-sample composition predictions Phi_A(Phi_B(beta)), coarse-cluster
     post-escape survival amplification, and supporting fixed-code check weakening (Claim 3).
  4. `fig_v20_4_reliability_and_scope`: Paired readout-cadence verification across RC and CR,
     mixed-hierarchy 3D volume-growth trajectory, and check-wire routing scaling.
"""

import json
import math
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
    "font.size": 9.5,
    "axes.labelsize": 10.2,
    "xtick.labelsize": 8.8,
    "ytick.labelsize": 8.8,
    "legend.fontsize": 7.9,
    "axes.linewidth": 0.85,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Consistent architectural palette across all figures
C_CC = "#64748b"   # slate gray
C_RC = "#1d4ed8"   # royal blue
C_CR = "#c2410c"   # burnt orange / rust
C_RR = "#047857"   # deep emerald
C_C3 = "#7c3aed"   # violet
C_C2 = "#0284c7"   # sky blue
C_MACRO = "#b91c1c"


def style_prx_ax(ax, label_letter, x_off=-0.16):
    ax.tick_params(direction="in", which="both", top=True, right=True, pad=4)
    ax.grid(True, which="major", color="#e2e8f0", linewidth=0.5, alpha=0.55)
    ax.set_axisbelow(True)
    ax.text(
        x_off, 1.03, f"({label_letter})",
        transform=ax.transAxes,
        fontsize=10.8,
        fontweight="bold",
        va="bottom",
        ha="left",
        color="#0f172a",
    )


def save_fig(fig, stem):
    pdf_path = FIG_DIR / f"{stem}.pdf"
    png_path = FIG_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved {png_path}")


def get_curve(v15, tag, bmin=4.5, bmax=7.5):
    pts = [p for p in v15["codes"][tag]["points"] if bmin - 1e-6 <= p["beta"] <= bmax + 1e-6]
    pts.sort(key=lambda d: d["beta"])
    bs = [p["beta"] for p in pts]
    ys = [p["tau_Q"] for p in pts]
    ses = [p["se_Q"] for p in pts]
    return bs, ys, ses


# ==============================================================================
# FIGURE 1: Local Modules (C vs R), Four Hierarchies, & Single-Cycle Lifetimes
# ==============================================================================
def render_fig1(v15):
    fig = plt.figure(figsize=(13.5, 3.85))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.05, 0.95, 1.12], wspace=0.22)

    # --- (a) Side-by-side 3D isometric module comparison: C (1-rib) vs R (2-rib) ---
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_xlim(0, 26)
    ax_a.set_ylim(0, 20)
    ax_a.axis("off")
    ax_a.text(0.1, 18.8, "(a)", fontsize=10.8, fontweight="bold", color="#0f172a")

    def draw_cell_3d(x0, y0, n_sub, rib_pos, accent_col, fill_col, label_bot):
        def pr(x, y, z):
            return x0 + 7.6 * x + 3.0 * y, y0 + 2.6 * y + 8.6 * z

        # Bottom sheet
        s0 = [pr(0, 0, 0), pr(1, 0, 0), pr(1, 1, 0), pr(0, 1, 0)]
        ax_a.add_patch(mpatches.Polygon(s0, closed=True, facecolor="#f8fafc", edgecolor="#475569", lw=1.0, alpha=0.92))
        for k in range(1, n_sub):
            u = k / n_sub
            a1, a2 = pr(u, 0, 0), pr(u, 1, 0)
            b1, b2 = pr(0, u, 0), pr(1, u, 0)
            ax_a.plot([a1[0], a2[0]], [a1[1], a2[1]], color="#cbd5e1", lw=0.75)
            ax_a.plot([b1[0], b2[0]], [b1[1], b2[1]], color="#cbd5e1", lw=0.75)

        # Interior vertical rib bridges
        for r in rib_pos:
            u = r / n_sub
            bh = [pr(0, u, 0), pr(1, u, 0), pr(1, u, 1), pr(0, u, 1)]
            bv = [pr(u, 0, 0), pr(u, 1, 0), pr(u, 1, 1), pr(u, 0, 1)]
            ax_a.add_patch(mpatches.Polygon(bh, closed=True, facecolor=fill_col, edgecolor=accent_col, lw=0.95, alpha=0.45))
            ax_a.add_patch(mpatches.Polygon(bv, closed=True, facecolor=fill_col, edgecolor=accent_col, lw=0.95, alpha=0.50))

        # Top sheet
        s1 = [pr(0, 0, 1), pr(1, 0, 1), pr(1, 1, 1), pr(0, 1, 1)]
        ax_a.add_patch(mpatches.Polygon(s1, closed=True, facecolor="#f8fafc", edgecolor="#475569", lw=1.0, alpha=0.68))
        for k in range(1, n_sub):
            u = k / n_sub
            a1, a2 = pr(u, 0, 1), pr(u, 1, 1)
            b1, b2 = pr(0, u, 1), pr(1, u, 1)
            ax_a.plot([a1[0], a2[0]], [a1[1], a2[1]], color="#cbd5e1", lw=0.75)
            ax_a.plot([b1[0], b2[0]], [b1[1], b2[1]], color="#cbd5e1", lw=0.75)

        # Highlight weld seams on both sheets
        for zv in (0, 1):
            for r in rib_pos:
                u = r / n_sub
                p1, p2 = pr(0, u, zv), pr(1, u, zv)
                q1, q2 = pr(u, 0, zv), pr(u, 1, zv)
                ax_a.plot([p1[0], p2[0]], [p1[1], p2[1]], color=accent_col, lw=1.9)
                ax_a.plot([q1[0], q2[0]], [q1[1], q2[1]], color=accent_col, lw=1.9)

        ax_a.text(x0 + 5.3, y0 - 1.9, label_bot, ha="center", va="top", fontsize=9.2, color="#0f172a")

    draw_cell_3d(1.5, 4.2, 3, [1], C_RC, "#dbeafe", r"Compact $C$ ($1$ rib)")
    draw_cell_3d(14.3, 4.2, 3, [1, 2], C_RR, "#d1fae5", r"Two-rib $R$ ($2$ ribs)")
    ax_a.text(4.8, 16.0, "Sheet 1", fontsize=8.0, color="#475569")
    ax_a.text(1.5, 2.9, "Sheet 0", fontsize=8.0, color="#475569")

    # --- (b) 2x2 Architecture Matrix: CC, RC, CR, RR ---
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_xlim(0, 20)
    ax_b.set_ylim(0, 20)
    ax_b.axis("off")
    ax_b.text(0.1, 18.8, "(b)", fontsize=10.8, fontweight="bold", color="#0f172a")

    def draw_arch_tile(x0, y0, w, coarse_ribs, fine_ribs, border_col, title_txt):
        ax_b.add_patch(mpatches.Rectangle((x0, y0), w, w, facecolor="#f8fafc", edgecolor="#475569", lw=1.15))
        # Fine sub-grid inside each quadrant/patch:
        for gx in range(3):
            for gy in range(3):
                sx, sy = x0 + gx * (w / 3.0), y0 + gy * (w / 3.0)
                sw = w / 3.0
                ax_b.add_patch(mpatches.Rectangle((sx, sy), sw, sw, facecolor="none", edgecolor="#cbd5e1", lw=0.55))
                for fr in fine_ribs:
                    fu = fr / 3.0
                    fcol = C_RR if len(fine_ribs) == 2 else C_CC
                    ax_b.plot([sx + fu * sw, sx + fu * sw], [sy + 0.08 * sw, sy + 0.92 * sw], color=fcol, lw=0.95, alpha=0.85)
                    ax_b.plot([sx + 0.08 * sw, sx + 0.92 * sw], [sy + fu * sw, sy + fu * sw], color=fcol, lw=0.95, alpha=0.85)
        # Coarse Cycle-1 ribs across whole tile:
        ccol = C_RR if len(coarse_ribs) == 2 else C_RC
        if len(coarse_ribs) == 1 and len(fine_ribs) == 1:
            ccol = C_CC
        elif len(coarse_ribs) == 1 and len(fine_ribs) == 2:
            ccol = C_CR
        elif len(coarse_ribs) == 2 and len(fine_ribs) == 1:
            ccol = C_RC
        for cr in coarse_ribs:
            cu = cr / 3.0
            ax_b.plot([x0, x0 + w], [y0 + cu * w, y0 + cu * w], color=ccol, lw=2.3)
            ax_b.plot([x0 + cu * w, x0 + cu * w], [y0, y0 + w], color=ccol, lw=2.3)
        ax_b.add_patch(mpatches.Rectangle((x0, y0), w, w, facecolor="none", edgecolor=border_col, lw=1.5))
        ax_b.text(x0 + 0.5 * w, y0 - 1.05, title_txt, ha="center", va="top", fontsize=9.1, fontweight="bold", color=border_col)

    tw = 6.6
    draw_arch_tile(1.8, 10.7, tw, [1],    [1],    C_CC, r"$\mathrm{CC}$")
    draw_arch_tile(11.2, 10.7, tw, [1, 2], [1],    C_RC, r"$\mathrm{RC}$")
    draw_arch_tile(1.8, 1.8,  tw, [1],    [1, 2], C_CR, r"$\mathrm{CR}$")
    draw_arch_tile(11.2, 1.8,  tw, [1, 2], [1, 2], C_RR, r"$\mathrm{RR}$")

    # --- (c) Claim 1: Matched-temperature 1-cycle lifetimes ---
    ax_c = fig.add_subplot(gs[0, 2])
    style_prx_ax(ax_c, "c", x_off=-0.18)

    curves_1 = [
        ("v15_sq_R3",  "-D",  C_RR, False, r"Planar $R_3$ ($1{,}480$)"),
        ("v15_tor_R3", "--D", C_RR, True,  r"Torus $R_3$ ($1{,}480$)"),
        ("v15_sq_C3",  "-s",  C_C3, False, r"Planar $C_3$ ($1{,}036$)"),
        ("v15_tor_C3", "--s", C_C3, True,  r"Torus $C_3$ ($1{,}036$)"),
        ("v15_sq_C2",  "-o",  C_C2, False, r"Planar $C_2$ ($254$)"),
        ("v15_tor_C2", "--o", C_C2, True,  r"Torus $C_2$ ($254$)"),
    ]
    for tag, fmt, col, hollow, lbl in curves_1:
        if tag not in v15["codes"]:
            continue
        bs, ys, ses = get_curve(v15, tag, 4.5, 7.25)
        ax_c.errorbar(
            bs, ys, yerr=ses, fmt=fmt, color=col, ms=5.0,
            mfc=("white" if hollow else col), mew=1.3,
            lw=(1.75 if not hollow else 1.3), capsize=2.0, label=lbl,
        )

    ax_c.set_yscale("log")
    ax_c.set_xlim(4.40, 7.38)
    ax_c.set_xlabel(r"Inverse temperature $\beta$")
    ax_c.set_ylabel(r"Single-cycle lifetime $\tau_Q$")
    ax_c.legend(loc="upper left", frameon=False, handlelength=2.0, ncol=1, labelspacing=0.28)

    save_fig(fig, "fig_v20_1_modules_and_claim1")


# ==============================================================================
# FIGURE 2: Two-Cycle Storage Performance & Placement Reversal (Claim 2)
# ==============================================================================
def render_fig2(v15):
    fig = plt.figure(figsize=(12.6, 7.5))
    gs = GridSpec(2, 2, figure=fig, wspace=0.25, hspace=0.28)

    # --- (a) Same-subdivision L=3 rib placement: CC, RC, CR, RR ---
    ax_a = fig.add_subplot(gs[0, 0])
    style_prx_ax(ax_a, "a", x_off=-0.15)

    for tag, fmt, col, lbl in [
        ("v15_sq_L3_RR", "-D", C_RR, r"$\mathrm{RR}$ ($709{,}408$)"),
        ("v15_sq_L3_CR", "-^", C_CR, r"$\mathrm{CR}$ ($591{,}208$)"),
        ("v15_sq_L3_RC", "-o", C_RC, r"$\mathrm{RC}$ ($499{,}088$)"),
        ("v15_sq_L3_CC", "--s", C_CC, r"$\mathrm{CC}$ ($415{,}004$)"),
    ]:
        if tag not in v15["codes"]:
            continue
        bs, ys, ses = get_curve(v15, tag, 4.5, 7.0)
        ax_a.errorbar(bs, ys, yerr=ses, fmt=fmt, color=col, ms=5.3, lw=1.8, capsize=2.1, label=lbl)

    bs1, ys1, _ = get_curve(v15, "v15_sq_R3", 4.5, 7.0)
    ax_a.plot(bs1, ys1, ":", color="#0f172a", lw=1.5, label=r"1-cycle $R_3$ ($1{,}480$)")

    ax_a.set_yscale("log")
    ax_a.set_xlim(4.40, 7.12)
    ax_a.set_xlabel(r"Inverse temperature $\beta$")
    ax_a.set_ylabel(r"Joint lifetime $\tau_Q$ ($C_3 / R_3$)")
    ax_a.legend(loc="upper left", frameon=False, handlelength=2.1)

    # --- (b) Unit-patch module placement: C2 vs R3 ---
    ax_b = fig.add_subplot(gs[0, 1])
    style_prx_ax(ax_b, "b", x_off=-0.15)

    for tag, fmt, col, lbl in [
        ("v15_sq_L3_RR",  "-D", C_RR, r"$\mathrm{RR}$ ($709{,}408$)"),
        ("v15_sq_pkg_RC", "-o", C_RC, r"$\mathrm{RC}$ ($133{,}864$)"),
        ("v15_sq_pkg_CR", "-^", C_CR, r"$\mathrm{CR}$ ($136{,}536$)"),
        ("v15_sq_pkg_CC", "--s", C_CC, r"$\mathrm{CC}$ ($25{,}148$)"),
    ]:
        if tag not in v15["codes"]:
            continue
        bs, ys, ses = get_curve(v15, tag, 4.5, 7.0)
        ax_b.errorbar(bs, ys, yerr=ses, fmt=fmt, color=col, ms=5.3, lw=1.8, capsize=2.1, label=lbl)

    ax_b.plot(bs1, ys1, ":", color="#0f172a", lw=1.5, label=r"1-cycle $R_3$ ($1{,}480$)")

    ax_b.set_yscale("log")
    ax_b.set_xlim(4.40, 7.12)
    ax_b.set_xlabel(r"Inverse temperature $\beta$")
    ax_b.set_ylabel(r"Joint lifetime $\tau_Q$ ($C_2 / R_3$)")
    ax_b.legend(loc="upper left", frameon=False, handlelength=2.1)

    # --- (c) RC / CR placement ratio across Planar & Torus ---
    ax_c = fig.add_subplot(gs[1, 0])
    style_prx_ax(ax_c, "c", x_off=-0.15)
    ax_c.axhline(1.0, color="#334155", ls=":", lw=1.15)

    ratio_specs = [
        ("pkg_planar_RC_over_CR", "-o",  C_RC, False, r"Planar $C_2/R_3$"),
        ("pkg_torus_RC_over_CR",  "--o", C_RC, True,  r"Torus $C_2/R_3$"),
        ("L3_planar_RC_over_CR",  "-^",  C_CR, False, r"Planar $C_3/R_3$"),
        ("L3_torus_RC_over_CR",   "--^", C_CR, True,  r"Torus $C_3/R_3$"),
    ]
    for rkey, fmt, col, hollow, lbl in ratio_specs:
        rows = v15["comparisons"].get(rkey, [])
        if not rows:
            continue
        bs = [r["beta"] for r in rows]
        rs = [r["ratio"] for r in rows]
        lo = [r["ratio"] - r["ci95"][0] for r in rows]
        hi = [r["ci95"][1] - r["ratio"] for r in rows]
        ax_c.errorbar(
            bs, rs, yerr=[lo, hi], fmt=fmt, color=col, ms=5.2,
            mfc=("white" if hollow else col), mew=1.35,
            lw=(1.8 if not hollow else 1.35), capsize=2.1, label=lbl,
        )

    ax_c.set_yscale("log")
    ax_c.set_xlim(4.40, 7.12)
    ax_c.set_xlabel(r"Inverse temperature $\beta$")
    ax_c.set_ylabel(r"Placement ratio $\tau_Q(\mathrm{RC}) \,/\, \tau_Q(\mathrm{CR})$")
    ax_c.legend(loc="best", frameon=False, handlelength=2.1)

    # --- (d) Storage lifetime vs physical qubit budget N at cold beta ---
    ax_d = fig.add_subplot(gs[1, 1])
    style_prx_ax(ax_d, "d", x_off=-0.15)

    pts_budget = [
        ("v15_sq_R3",     "*",  "#0f172a", r"1-cycle $R_3$"),
        ("v15_sq_pkg_CC", "s",  C_CC,      r"$C_2/R_3$ $\mathrm{CC}$"),
        ("v15_sq_pkg_RC", "o",  C_RC,      r"$C_2/R_3$ $\mathrm{RC}$"),
        ("v15_sq_pkg_CR", "^",  C_CR,      r"$C_2/R_3$ $\mathrm{CR}$"),
        ("v15_sq_L3_CC",  "s",  C_C3,      r"$C_3/R_3$ $\mathrm{CC}$"),
        ("v15_sq_L3_RC",  "o",  "#6366f1", r"$C_3/R_3$ $\mathrm{RC}$"),
        ("v15_sq_L3_CR",  "^",  "#ea580c", r"$C_3/R_3$ $\mathrm{CR}$"),
        ("v15_sq_L3_RR",  "D",  C_RR,      r"$\mathrm{RR}$"),
    ]
    for b_target, hollow, alpha_v in [(7.0, False, 1.0), (6.5, True, 0.72)]:
        for tag, mk, col, lbl in pts_budget:
            if tag not in v15["codes"]:
                continue
            pmap = {round(p["beta"], 2): p for p in v15["codes"][tag]["points"]}
            if b_target not in pmap:
                continue
            nq = v15["codes"][tag]["spec"]["N"]
            p = pmap[b_target]
            ax_d.errorbar(
                [nq], [p["tau_Q"]], yerr=[p["se_Q"]], fmt=mk, color=col,
                ms=(7.2 if mk == "*" else 6.0),
                mfc=("white" if hollow else col), mew=1.4,
                capsize=2.0, alpha=alpha_v,
                label=(lbl if not hollow else None),
            )

    ax_d.set_xscale("log")
    ax_d.set_yscale("log")
    ax_d.set_xlim(9e2, 1.2e6)
    ax_d.set_ylim(1.5e3, 2.5e7)
    ax_d.set_xlabel(r"Physical qubits $N$")
    ax_d.set_ylabel(r"Joint lifetime $\tau_Q$ ($\beta = 7.0$ filled, $6.5$ open)")
    ax_d.legend(loc="lower left", frameon=False, ncol=2, columnspacing=0.9, handletextpad=0.4, fontsize=7.6)

    save_fig(fig, "fig_v20_2_architectures_claim2")


# ==============================================================================
# FIGURE 3: Two-Scale Escape & Confinement Mechanism + 1-Cycle Prediction (Claim 3)
# ==============================================================================
def render_fig3(v15, v14):
    fig = plt.figure(figsize=(12.6, 7.5))
    gs = GridSpec(2, 2, figure=fig, wspace=0.25, hspace=0.28)

    span_specs = [
        ("v15_sq_pkg_RC", "-o",  C_RC, False, r"$C_2/R_3$ $\mathrm{RC}$"),
        ("v15_sq_pkg_CR", "-^",  C_CR, False, r"$C_2/R_3$ $\mathrm{CR}$"),
        ("v15_sq_L3_CR",  "--^", C_CR, True,  r"$C_3/R_3$ $\mathrm{CR}$"),
        ("v15_sq_L3_RC",  "--o", C_RC, True,  r"$C_3/R_3$ $\mathrm{RC}$"),
    ]

    # --- (a) Fine-to-coarse 10-cell excursion arrival time tau_10(beta) ---
    ax_a = fig.add_subplot(gs[0, 0])
    style_prx_ax(ax_a, "a", x_off=-0.15)

    for tag, fmt, col, hollow, lbl in span_specs:
        if tag not in v15["codes"]:
            continue
        pts = [p for p in v15["codes"][tag]["points"] if p["beta"] >= 5.0 and p.get("span_stats")]
        bs = [p["beta"] for p in pts]
        sp10 = [p["span_stats"]["mean_t_sp10"] for p in pts]
        ax_a.plot(
            bs, sp10, fmt, color=col, ms=5.3,
            mfc=("white" if hollow else col), mew=1.35,
            lw=(1.8 if not hollow else 1.4), label=lbl,
        )

    ax_a.set_yscale("log")
    ax_a.set_xlim(4.90, 7.12)
    ax_a.set_xlabel(r"Inverse temperature $\beta$")
    ax_a.set_ylabel(r"10-cell arrival time $\tau_{10}$")
    ax_a.legend(loc="upper left", frameon=False, handlelength=2.1)

    # --- (b) Survival ratio tau_X / tau_6 ---
    ax_b = fig.add_subplot(gs[0, 1])
    style_prx_ax(ax_b, "b", x_off=-0.15)

    for tag, fmt, col, hollow, lbl in span_specs:
        if tag not in v15["codes"]:
            continue
        pts = [p for p in v15["codes"][tag]["points"] if p["beta"] >= 5.0 and p.get("span_stats")]
        bs = [p["beta"] for p in pts]
        amps = [p["span_stats"]["survival_amplification_tau_over_sp6"] for p in pts]
        ax_b.plot(
            bs, amps, fmt, color=col, ms=5.3,
            mfc=("white" if hollow else col), mew=1.35,
            lw=(1.8 if not hollow else 1.4), label=lbl,
        )

    ax_b.set_yscale("log")
    ax_b.set_xlim(4.90, 7.12)
    ax_b.set_xlabel(r"Inverse temperature $\beta$")
    ax_b.set_ylabel(r"Survival ratio $\tau_X \,/\, \tau_6$")
    ax_b.legend(loc="upper left", frameon=False, handlelength=2.1)

    # --- (c) Two-cycle slope splitting vs single-cycle module slope difference ---
    ax_c = fig.add_subplot(gs[1, 0])
    style_prx_ax(ax_c, "c", x_off=-0.15)
    ax_c.axhline(0.0, color="#334155", ls=":", lw=1.15)

    sl = v15["slopes_cold"]
    fam_bars = [
        ("Planar\n" + r"$C_2/R_3$",     "v15_sq_pkg_RC",     "v15_sq_pkg_CR",     "v15_sq_R3",     "v15_sq_C2",     C_RC),
        ("Torus\n" + r"$C_2/R_3$",      "v15_tor_pkg_RC",    "v15_tor_pkg_CR",    "v15_tor_R3",    "v15_tor_C2",    C_RC),
        ("Rotated\n" + r"$C_2/R_3$",    "v15_rottor_pkg_RC", "v15_rottor_pkg_CR", "v15_rottor_R3", "v15_rottor_C2", C_RC),
        ("Isotropic\n" + r"$C_2/R_3$",  "v15_sq_iso_RC",     "v15_sq_iso_CR",     "v15_sq_R3_iso", "v15_sq_C2",     C_RC),
        ("Planar\n" + r"$C_3/R_3$",     "v15_sq_L3_RC",      "v15_sq_L3_CR",      "v15_sq_R3",     "v15_sq_C3",     C_CR),
        ("Torus\n" + r"$C_3/R_3$",      "v15_tor_L3_RC",     "v15_tor_L3_CR",     "v15_tor_R3",    "v15_tor_C3",    C_CR),
    ]
    xs_f, dy_2, se_2, dy_1 = [], [], [], []
    xlbls, cols_f = [], []
    for idx, (xl, rc_t, cr_t, r1_t, c1_t, col) in enumerate(fam_bars):
        if rc_t in sl and cr_t in sl and r1_t in sl and c1_t in sl:
            xs_f.append(idx)
            xlbls.append(xl)
            cols_f.append(col)
            dy_2.append(sl[rc_t]["slope"] - sl[cr_t]["slope"])
            se_2.append(math.hypot(sl[rc_t]["se_slope"], sl[cr_t]["se_slope"]))
            dy_1.append(sl[r1_t]["slope"] - sl[c1_t]["slope"])

    seen_c, seen_o = False, False
    for x, y2, s2, col in zip(xs_f, dy_2, se_2, cols_f):
        lbl_i = None
        if col == C_RC and not seen_c:
            lbl_i = r"2-cycle $\mathrm{RC} - \mathrm{CR}$ ($C_2/R_3$)"
            seen_c = True
        elif col == C_CR and not seen_o:
            lbl_i = r"2-cycle $\mathrm{RC} - \mathrm{CR}$ ($C_3/R_3$)"
            seen_o = True
        ax_c.errorbar(
            [x - 0.11], [y2], yerr=[s2], fmt=("o" if col == C_RC else "^"), color=col,
            ms=6.0, lw=1.8, capsize=2.6, label=lbl_i,
        )
    ax_c.plot(
        [x + 0.11 for x in xs_f], dy_1, "D", color="#0f172a", mfc="white", mew=1.5, ms=5.6,
        label=r"1-cycle $R - C$",
    )

    ax_c.set_xticks(xs_f)
    ax_c.set_xticklabels(xlbls, fontsize=7.9)
    ax_c.set_xlim(-0.55, len(xs_f) - 0.45)
    ax_c.set_ylim(-1.85, 3.35)
    ax_c.set_ylabel(r"Slope difference")
    ax_c.legend(loc="upper left", frameon=False, handlelength=1.5, fontsize=7.5, labelspacing=0.26)

    # --- (d) Supporting causal check: fixed-code coarse vs fine weld weakening ---
    ax_d = fig.add_subplot(gs[1, 1])
    style_prx_ax(ax_d, "d", x_off=-0.15)

    inter = v14["interventions"]
    tq14 = v14["tau_q"]
    lams = [0.2, 0.4, 0.6, 0.8, 1.0]
    for cname, fam_lbl, ls_fam, mk_m, mk_s in [
        ("sq_iso_L2r1_s4", "Planar", "-", "o", "s"),
        ("rottor_iso_L2r1_s4", "Torus", "--", "^", "D"),
    ]:
        u_q = tq14[cname]["6.50"]["tau_Q"]
        u_se = tq14[cname]["6.50"]["sem_Q"]
        for tag_u, col, mk, u_lbl in [
            ("macro_weld", C_MACRO, mk_m, "coarse"),
            ("sub_weld", C_RC, mk_s, "fine"),
        ]:
            ys = [inter[cname][tag_u]["6.50"][f"{l:.1f}"]["tau_Q"] for l in lams[:-1]] + [u_q]
            ses = [inter[cname][tag_u]["6.50"][f"{l:.1f}"]["sem_Q"] for l in lams[:-1]] + [u_se]
            ax_d.errorbar(
                lams, ys, yerr=ses, fmt=f"{ls_fam}{mk}", color=col,
                ms=5.0, lw=1.6, capsize=2.0, label=f"{fam_lbl} {u_lbl}",
            )

    ax_d.set_yscale("log")
    ax_d.set_xlim(0.15, 1.05)
    ax_d.set_xlabel(r"Check-energy factor $\lambda$ ($\beta = 6.5$)")
    ax_d.set_ylabel(r"Fixed-code lifetime $\tau_Q(\lambda)$")
    ax_d.legend(loc="lower right", frameon=False, handlelength=2.0)

    save_fig(fig, "fig_v20_3_mechanism_claim3")


# ==============================================================================
# FIGURE 4: Readout Cadence Verification & Mixed-Hierarchy Spatial Scope
# ==============================================================================
def render_fig4(v15, v13, certs):
    fig = plt.figure(figsize=(13.4, 3.85))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.0, 1.0], wspace=0.27)

    # --- (a) Paired checkpoint-cadence verification across RC and CR ---
    ax_a = fig.add_subplot(gs[0, 0])
    style_prx_ax(ax_a, "a", x_off=-0.16)

    cad = v15["paired_cadence"]
    for ctag, fmt, col, hollow, lbl in [
        ("v15_sq_pkg_RC", "-o",  C_RC, False, r"$C_2/R_3$ $\mathrm{RC}$"),
        ("v15_sq_pkg_CR", "-^",  C_CR, False, r"$C_2/R_3$ $\mathrm{CR}$"),
        ("v15_sq_L3_CR",  "--^", C_CR, True,  r"$C_3/R_3$ $\mathrm{CR}$"),
        ("v15_sq_L3_RC",  "--o", C_RC, True,  r"$C_3/R_3$ $\mathrm{RC}$"),
    ]:
        if ctag not in cad:
            continue
        b_use = 6.5 if 6.5 in cad[ctag] else (6.0 if 6.0 in cad[ctag] else None)
        # JSON keys may be strings if loaded from disk:
        if b_use is None:
            b_use = "6.5" if "6.5" in cad[ctag] else ("6.50" if "6.50" in cad[ctag] else next(iter(cad[ctag].keys()), None))
        if b_use is None:
            continue
        rec = cad[ctag][b_use]
        ax_a.errorbar(
            rec["etas"], rec["means"], yerr=rec["sems"], fmt=fmt, color=col,
            ms=5.3, mfc=("white" if hollow else col), mew=1.35,
            lw=(1.8 if not hollow else 1.4), capsize=2.1, label=lbl,
        )

    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_ylim(1.1e3, 4.8e6)
    ax_a.set_xlabel(r"Readout spacing $\eta$ ($\beta = 6.5$)")
    ax_a.set_ylabel(r"Paired mean $\tau_X(\eta)$")
    ax_a.legend(loc="upper left", frameon=False, ncol=2, columnspacing=0.9, handlelength=1.8, fontsize=7.5)

    # --- (b) Mixed-hierarchy 3D volume-growth trajectory F_q^(1/q) vs 27 ---
    ax_b = fig.add_subplot(gs[0, 1])
    style_prx_ax(ax_b, "b", x_off=-0.16)

    qs = [1, 2, 3, 4]
    ax_b.axhline(27.0, color="#0f172a", ls=":", lw=1.25, label=r"$L^3 = 27$ bound")

    mh = certs["selective_hierarchy_and_mixed_packing"]["mixed_hierarchy_L3_volume_growth"]
    f_rr_iso = [30 ** q for q in qs]
    f_cc_iso = [24 ** q for q in qs]
    f_rc_iso = mh["isotropic_RC_verified"]["face_trajectory"]
    f_cr_iso = mh["isotropic_CR_verified"]["face_trajectory"]
    f_rr_ani = mh["anisotropic_RR_verified"]["face_trajectory"]

    for traj, fmt, col, hollow, lbl in [
        (f_rr_iso, "-D",  C_MACRO, False, r"Uniform $R_3$ ($30$)"),
        (f_rc_iso, "-o",  C_RC,    False, r"Mixed $\mathrm{RC}$ ($26.83$)"),
        (f_cr_iso, "--^", C_CR,    True,  r"Mixed $\mathrm{CR}$ ($26.83$)"),
        (f_cc_iso, "-s",  C_CC,    False, r"Uniform $C_3$ ($24$)"),
        (f_rr_ani, ":D",  C_RR,    True,  r"Thin-bridge $\mathrm{RR}$"),
    ]:
        gm = [traj[q - 1] ** (1.0 / q) for q in qs]
        ax_b.plot(
            qs, gm, fmt, color=col, ms=5.3,
            mfc=("white" if hollow else col), mew=1.35, lw=1.7, label=lbl,
        )

    ax_b.set_xticks(qs)
    ax_b.set_ylim(18.0, 31.4)
    ax_b.set_xlabel(r"Recursion step $q$ ($L = 3$)")
    ax_b.set_ylabel(r"Geometric-mean face factor $F_q^{1/q}$")
    ax_b.legend(loc="lower left", frameon=False, ncol=2, columnspacing=0.8, handlelength=1.7, fontsize=7.1)

    # --- (c) Torus check-leg wire collapse ---
    ax_c = fig.add_subplot(gs[0, 2])
    style_prx_ax(ax_c, "c", x_off=-0.16)

    w_styles = [
        (1,  "o", C_RR,      r"$w = 1$"),
        (2,  "s", C_C2,      r"$w = 2$"),
        (4,  "^", C_RC,      r"$w = 4$"),
        (8,  "D", C_CR,      r"$w = 8$"),
        (16, "v", C_MACRO,   r"$w = 16$"),
    ]
    wire_src = v13["wire_sweep"]["tor"]
    for w, mk, col, lbl in w_styles:
        if str(w) not in wire_src:
            continue
        pts = sorted(wire_src[str(w)]["points"], key=lambda d: d["beta"])
        beffs = [d["beta_eff"] for d in pts]
        ys = [d["mean_t"] / (w * w) for d in pts]
        ses = [d["sem_t"] / (w * w) for d in pts]
        ax_c.errorbar(beffs, ys, yerr=ses, fmt=f"-{mk}", color=col, ms=4.9, lw=1.4, capsize=2.0, label=lbl)

    ax_c.set_yscale("log")
    ax_c.set_xlim(2.05, 6.7)
    ax_c.set_ylim(0.7, 4.5e5)
    ax_c.set_xlabel(r"Effective coupling $\beta_{\mathrm{eff}}(w)$")
    ax_c.set_ylabel(r"$\tau_X(\beta, w) \,/\, w^2$")
    ax_c.legend(loc="upper left", frameon=False, handlelength=2.0)

    save_fig(fig, "fig_v20_4_reliability_and_scope")


def main():
    v15 = json.loads((ROOT / "parsed_v15_data.json").read_text())
    v14 = json.loads((ROOT / "parsed_v14_data.json").read_text())
    v13 = json.loads((ROOT / "parsed_v13_data.json").read_text())
    certs = json.loads((ROOT / "certificates_v20.json").read_text())
    render_fig1(v15)
    render_fig2(v15)
    render_fig3(v15, v14)
    render_fig4(v15, v13, certs)
    print("All 4 V20 PRX-style figures rendered successfully.")


if __name__ == "__main__":
    main()
