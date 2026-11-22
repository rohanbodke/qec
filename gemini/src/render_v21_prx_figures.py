#!/usr/bin/env python3
"""
render_v21_prx_figures.py
Renders clean APS / Physical Review X (PRX) figures for
"The Thermal Value of Welding Across Scales":
  - Serif + Computer Modern math typography matching REVTeX 4.2
  - Full 4-sided spines with inward ticks (top & right enabled)
  - Bold (a), (b), (c), (d) panel labels ONLY; zero subplot titles
  - Explicit error bars / 95% CIs on all plotted statistics
  - Explicit upward arrows on any right-censored lower-bound point; zero ratios with censored denominators

Figures:
  1. `fig_v21_1_modules`: Sheet-region anatomy of all three modules (C2, C3, R3),
     two-cycle scale-placement architectures, and matched-temperature single-cycle lifetimes.
  2. `fig_v21_2_reversal`: Central scale-placement reversal ratio across beta,
     operational final-readout fidelity P_id(t) = p_X(t) p_Z(t) at beta = 6.5 (no intermediate stopping),
     and two-cycle lifetime curves across C3/R3, C2/R3, and C3/C2.
  3. `fig_v21_3_resources_growth`: Qubit-resource frontier (including all-compact CC and the
     withheld C3C2 comparator) under both operational t_0.80 and first-failure tau_Q, paired with
     the proven thin-bridge vs isotropic spectral volume-growth screen rho(T_blk) / L_blk^3.
  4. `fig_v21_4_mechanism_causal`: Detailed-balance-preserving zero-energy kinetic causal test
     (flat-region d_Z=2 qubits vs matched seam controls), decoder recovery-cluster arrival times,
     unbiased first-excursion return probabilities, and uncensored slope comparisons with two-axis CIs.
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
    "font.size": 9.4,
    "axes.labelsize": 10.1,
    "xtick.labelsize": 8.7,
    "ytick.labelsize": 8.7,
    "legend.fontsize": 7.7,
    "axes.linewidth": 0.85,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

C_CC = "#64748b"   # slate gray
C_RC = "#1d4ed8"   # royal blue
C_CR = "#c2410c"   # burnt orange / rust
C_RR = "#047857"   # deep emerald
C_C3 = "#7c3aed"   # violet
C_C2 = "#0284c7"   # sky blue
C_GOLD = "#b45309" # warm amber


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


def get_pts(data, tag, bmin=4.5, bmax=7.5):
    pts = [p for p in data["codes"][tag]["points"] if bmin - 1e-6 <= p["beta"] <= bmax + 1e-6]
    pts.sort(key=lambda d: d["beta"])
    return pts


def plot_lifetime_series(ax, data, tag, fmt, col, lbl, bmin=4.5, bmax=7.0, hollow=False, ms=5.1, lw=1.75):
    if tag not in data["codes"]:
        return
    pts = get_pts(data, tag, bmin, bmax)
    unc = [p for p in pts if not p["is_censored"]]
    cen = [p for p in pts if p["is_censored"]]

    if bs_all := [p["beta"] for p in pts]:
        ys_all = [p["tau_Q"] for p in pts]
        ls = "--" if "--" in fmt else ("-." if "-." in fmt else "-")
        ax.plot(bs_all, ys_all, ls=ls, color=col, lw=lw, alpha=0.9)

    mk = fmt.replace("--", "").replace("-.", "").replace("-", "")
    if unc:
        ax.errorbar(
            [p["beta"] for p in unc],
            [p["tau_Q"] for p in unc],
            yerr=[p["se_Q"] for p in unc],
            fmt=mk, color=col, ms=ms,
            mfc=("white" if hollow else col), mew=1.3,
            capsize=2.0, ls="none", label=lbl,
        )
    elif cen:
        ax.plot([], [], fmt, color=col, mfc=("white" if hollow else col), mew=1.3, ms=ms, lw=lw, label=lbl)

    for p in cen:
        y0 = p["tau_Q"]
        ax.plot([p["beta"]], [y0], marker=mk, color=col, mfc="white", mew=1.35, ms=ms)
        ax.annotate(
            "", xy=(p["beta"], y0 * 1.65), xytext=(p["beta"], y0),
            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2, mutation_scale=8),
        )


# ==============================================================================
# FIGURE 1: Local Regions in C2, C3, and R3, Two-Cycle Order, & 1-Cycle Lifetimes
# ==============================================================================
def render_fig1(data):
    fig = plt.figure(figsize=(13.6, 3.85))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.22, 0.88, 1.08], wspace=0.22)

    # --- (a) Explicit local sheet regions in C2 (2x2, 1 rib), C3 (3x3, 1 rib), R3 (3x3, 2 ribs) ---
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_xlim(-0.6, 32.6)
    ax_a.set_ylim(0.4, 15.2)
    ax_a.set_aspect("equal")
    ax_a.axis("off")
    ax_a.text(-0.4, 14.3, "(a)", fontsize=10.8, fontweight="bold", color="#0f172a")

    def draw_module_sheet(x0, y0, w, L, ribs, seam_col, title_top, sub_bot, shade_flat=False):
        ax_a.add_patch(mpatches.Rectangle((x0, y0), w, w, facecolor="#f8fafc", edgecolor="#334155", lw=1.2))
        if shade_flat:
            u1 = w / 3.0
            ax_a.add_patch(mpatches.Rectangle((x0, y0 + u1), u1, 2 * u1, facecolor="#fef3c7", edgecolor="none", alpha=0.85))
            ax_a.add_patch(mpatches.Rectangle((x0 + u1, y0), 2 * u1, u1, facecolor="#fef3c7", edgecolor="none", alpha=0.85))
            ax_a.add_patch(mpatches.Rectangle((x0 + u1, y0 + u1), 2 * u1, 2 * u1, facecolor="#fde68a", edgecolor="none", alpha=0.85))

        for k in range(1, L):
            uk = x0 + (k / L) * w
            vk = y0 + (k / L) * w
            is_r = k in ribs
            if not is_r:
                ax_a.plot([uk, uk], [y0, y0 + w], color="#d97706", ls="--", lw=1.15)
                ax_a.plot([x0, x0 + w], [vk, vk], color="#d97706", ls="--", lw=1.15)

        for r in ribs:
            ur = x0 + (r / L) * w
            vr = y0 + (r / L) * w
            ax_a.plot([ur, ur], [y0, y0 + w], color=seam_col, lw=2.4)
            ax_a.plot([x0, x0 + w], [vr, vr], color=seam_col, lw=2.4)

        ax_a.add_patch(mpatches.Rectangle((x0, y0), w, w, facecolor="none", edgecolor="#334155", lw=1.2))
        ax_a.text(x0 + 0.5 * w, y0 + w + 0.45, title_top, ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#0f172a")
        ax_a.text(x0 + 0.5 * w, y0 - 0.45, sub_bot, ha="center", va="top", fontsize=7.7, color="#334155")

    w_box = 7.6
    y_box = 3.2
    draw_module_sheet(0.2,  y_box, w_box, 2, [1],    C_C2, r"$C_2$ ($2{\times}2$, $1$ bridge)", "unit faces\n($82.7\\%$ seam)")
    draw_module_sheet(11.9, y_box, w_box, 3, [1],    C_C3, r"$C_3$ ($3{\times}3$, $1$ bridge)", "flat patches\n($57.5\\%$ seam)", shade_flat=True)
    draw_module_sheet(23.6, y_box, w_box, 3, [1, 2], C_RR, r"$R_3$ ($3{\times}3$, $2$ bridges)", "unit faces\n($93.0\\%$ seam)")

    u3 = w_box / 3.0
    ax_a.text(11.9 + 0.5 * u3, y_box + 0.5 * u3, r"$1{\times}1$", ha="center", va="center", fontsize=6.9, color="#475569")
    ax_a.text(11.9 + 0.5 * u3, y_box + 2.5 * u3, r"$1{\times}2$", ha="center", va="center", fontsize=6.9, color="#92400e")
    ax_a.text(11.9 + 2.5 * u3, y_box + 0.5 * u3, r"$2{\times}1$", ha="center", va="center", fontsize=6.9, color="#92400e")
    ax_a.text(11.9 + 2.5 * u3, y_box + 2.5 * u3, r"$2{\times}2$", ha="center", va="center", fontsize=7.4, fontweight="bold", color="#92400e")

    # --- (b) Coarse-to-fine two-cycle placement schematic ---
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_xlim(0, 20)
    ax_b.set_ylim(0, 20)
    ax_b.set_aspect("equal")
    ax_b.axis("off")
    ax_b.text(0.1, 18.8, "(b)", fontsize=10.8, fontweight="bold", color="#0f172a")

    def draw_arch_tile(x0, y0, w, Lc, coarse_ribs, Lf, fine_ribs, border_col, title_txt):
        ax_b.add_patch(mpatches.Rectangle((x0, y0), w, w, facecolor="#f8fafc", edgecolor="#475569", lw=1.1))
        for gx in range(Lc):
            for gy in range(Lc):
                sx, sy = x0 + gx * (w / Lc), y0 + gy * (w / Lc)
                sw = w / Lc
                if Lf == 3 and len(fine_ribs) == 1:
                    ax_b.add_patch(mpatches.Rectangle((sx + sw / 3.0, sy + sw / 3.0), 2 * sw / 3.0, 2 * sw / 3.0, facecolor="#fde68a", edgecolor="none", alpha=0.65))
                ax_b.add_patch(mpatches.Rectangle((sx, sy), sw, sw, facecolor="none", edgecolor="#cbd5e1", lw=0.5))
                fcol = C_RR if len(fine_ribs) == 2 else (C_C2 if Lf == 2 else C_C3)
                for fr in fine_ribs:
                    fu = fr / float(Lf)
                    ax_b.plot([sx + fu * sw, sx + fu * sw], [sy + 0.08 * sw, sy + 0.92 * sw], color=fcol, lw=0.9, alpha=0.85)
                    ax_b.plot([sx + 0.08 * sw, sx + 0.92 * sw], [sy + fu * sw, sy + fu * sw], color=fcol, lw=0.9, alpha=0.85)
        for cr in coarse_ribs:
            cu = cr / float(Lc)
            ax_b.plot([x0, x0 + w], [y0 + cu * w, y0 + cu * w], color=border_col, lw=2.2)
            ax_b.plot([x0 + cu * w, x0 + cu * w], [y0, y0 + w], color=border_col, lw=2.2)
        ax_b.add_patch(mpatches.Rectangle((x0, y0), w, w, facecolor="none", edgecolor=border_col, lw=1.45))
        ax_b.text(x0 + 0.5 * w, y0 - 0.95, title_txt, ha="center", va="top", fontsize=8.8, fontweight="bold", color=border_col)

    tw = 6.5
    draw_arch_tile(1.6,  10.8, tw, 3, [1, 2], 3, [1],    C_RC, r"$R_3 C_3$")
    draw_arch_tile(11.1, 10.8, tw, 3, [1],    3, [1, 2], C_CR, r"$C_3 R_3$")
    draw_arch_tile(1.6,  1.9,  tw, 3, [1, 2], 2, [1],    C_RC, r"$R_3 C_2$")
    draw_arch_tile(11.1, 1.9,  tw, 2, [1],    3, [1, 2], C_CR, r"$C_2 R_3$")

    # --- (c) Single-cycle module lifetimes ---
    ax_c = fig.add_subplot(gs[0, 2])
    style_prx_ax(ax_c, "c", x_off=-0.18)

    for tag, fmt, col, hollow, lbl in [
        ("v15_sq_R3",  "-D",  C_RR, False, r"Planar $R_3$ ($1{,}480$)"),
        ("v15_tor_R3", "--D", C_RR, True,  r"Torus $R_3$ ($1{,}480$)"),
        ("v15_sq_C2",  "-o",  C_C2, False, r"Planar $C_2$ ($254$)"),
        ("v15_tor_C2", "--o", C_C2, True,  r"Torus $C_2$ ($254$)"),
        ("v15_sq_C3",  "-s",  C_C3, False, r"Planar $C_3$ ($1{,}036$)"),
        ("v15_tor_C3", "--s", C_C3, True,  r"Torus $C_3$ ($1{,}036$)"),
    ]:
        plot_lifetime_series(ax_c, data, tag, fmt, col, lbl, 4.5, 7.25, hollow=hollow, ms=4.9, lw=(1.75 if not hollow else 1.3))

    ax_c.set_yscale("log")
    ax_c.set_xlim(4.40, 7.38)
    ax_c.set_xlabel(r"Inverse temperature $\beta$")
    ax_c.set_ylabel(r"Single-cycle failure time $\tau_Q$")
    ax_c.legend(loc="upper left", frameon=False, handlelength=2.0, labelspacing=0.25)

    save_fig(fig, "fig_v21_1_modules")


# ==============================================================================
# FIGURE 2: Central Scale-Placement Reversal & Operational Final Readout
# ==============================================================================
def render_fig2(data):
    fig = plt.figure(figsize=(12.8, 7.6))
    gs = GridSpec(2, 2, figure=fig, wspace=0.24, hspace=0.28)

    # --- (a) HERO PANEL: Placement ratio across C2/R3, C3/C2, and C3/R3 ---
    ax_a = fig.add_subplot(gs[0, 0])
    style_prx_ax(ax_a, "a", x_off=-0.15)
    ax_a.axhline(1.0, color="#334155", ls=":", lw=1.2)

    ratio_specs = [
        ("pkg_planar_RC_over_CR",   "-o",  C_RC, False, r"Planar $R_3 C_2 / C_2 R_3$"),
        ("pkg_torus_RC_over_CR",    "--o", C_RC, True,  r"Torus $R_3 C_2 / C_2 R_3$"),
        ("C3C2_planar_C3C2_over_C2C3", "-s", C_C3, False, r"Planar $C_3 C_2 / C_2 C_3$"),
        ("C3C2_torus_C3C2_over_C2C3",  "--s", C_C3, True, r"Torus $C_3 C_2 / C_2 C_3$"),
        ("L3_planar_RC_over_CR",    "-^",  C_CR, False, r"Planar $R_3 C_3 / C_3 R_3$"),
        ("L3_torus_RC_over_CR",     "--^", C_CR, True,  r"Torus $R_3 C_3 / C_3 R_3$"),
    ]
    for rkey, fmt, col, hollow, lbl in ratio_specs:
        rows = [r for r in data["comparisons"].get(rkey, []) if r["status"] != "omitted_denominator_censored"]
        if not rows:
            continue
        ls = "--" if "--" in fmt else "-"
        mk = fmt.replace("--", "").replace("-", "")
        unc = [r for r in rows if r["status"] == "uncensored"]
        cen = [r for r in rows if r["status"] == "lower_bound_numerator_censored"]
        if unc:
            ax_a.plot([r["beta"] for r in unc], [r["ratio"] for r in unc], ls=ls, color=col, lw=(1.8 if not hollow else 1.35))
            ax_a.errorbar(
                [r["beta"] for r in unc],
                [r["ratio"] for r in unc],
                yerr=[[r["ratio"] - r["ci95"][0] for r in unc], [r["ci95"][1] - r["ratio"] for r in unc]],
                fmt=mk, color=col, ms=5.1, mfc=("white" if hollow else col), mew=1.3, capsize=2.0, ls="none", label=lbl,
            )
        else:
            ax_a.plot([], [], fmt, color=col, mfc=("white" if hollow else col), mew=1.3, ms=5.1, label=lbl)
        for r in cen:
            # Plot conservative 95% lower confidence bound or empirical lower bound with upward arrow (no downward solid line!)
            ax_a.plot([r["beta"]], [r["ratio"]], marker=mk, color=col, mfc="white", mew=1.35, ms=5.1)
            ax_a.annotate(
                "", xy=(r["beta"], r["ratio"] * 2.1), xytext=(r["beta"], r["ratio"]),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.3, mutation_scale=8.5),
            )

    ax_a.set_yscale("log")
    ax_a.set_xlim(4.40, 7.14)
    ax_a.set_ylim(0.025, 290.0)
    ax_a.set_xlabel(r"Inverse temperature $\beta$")
    ax_a.set_ylabel("Placement ratio (coarse vs. fine)")
    ax_a.legend(loc="upper left", frameon=False, ncol=1, handlelength=2.0, labelspacing=0.22, fontsize=7.3)

    # --- (b) Operational Final-Readout Fidelity P_id(t) = p_X(t) p_Z(t) at beta = 6.5 ---
    ax_b = fig.add_subplot(gs[0, 1])
    style_prx_ax(ax_b, "b", x_off=-0.15)
    ax_b.axhline(0.80, color="#64748b", ls="--", lw=0.95)
    ax_b.axhline(0.25, color="#94a3b8", ls=":", lw=0.95)

    # Plot boundaryless Torus (K=2, floor = 0.25) so all curves span full [0.25, 1.0] dynamic range cleanly!
    frout_specs = [
        ("v15_tor_pkg_RC", "-",  C_RC, 2.0, r"Torus $R_3 C_2$ ($133{,}864$)"),
        ("v16_tor_C3C2",   "-",  C_C3, 1.85, r"Torus $C_3 C_2$ ($107{,}140$)"),
        ("v15_tor_R3",     ":",  "#0f172a", 1.75, r"1-cycle Torus $R_3$ ($1{,}480$)"),
        ("v15_tor_pkg_CR", "-",  C_CR, 1.9, r"Torus $C_2 R_3$ ($136{,}536$)"),
        ("v15_tor_L3_CR",  "--", C_RR, 1.65, r"Torus $C_3 R_3$ ($591{,}208$)"),
        ("v16_tor_C2C3",   "-.", C_GOLD, 1.6, r"Torus $C_2 C_3$ ($95{,}896$)"),
        ("v15_tor_L3_RC",  "--", "#475569", 1.55, r"Torus $R_3 C_3$ ($499{,}088$)"),
    ]
    for tag, ls, col, lw, lbl in frout_specs:
        rec = data["final_readout"].get(tag, {}).get("6.50")
        if not rec:
            continue
        tg = np.array(rec["t_grid"])
        pid = np.array(rec["P_id"])
        lo = np.array(rec["P_id_ci95_lo"])
        hi = np.array(rec["P_id_ci95_hi"])
        ax_b.plot(tg, pid, ls=ls, color=col, lw=lw, label=lbl)
        if ls == "-":
            ax_b.fill_between(tg, lo, hi, color=col, alpha=0.13, linewidth=0)

    ax_b.set_xscale("log")
    ax_b.set_xlim(1e2, 3.98e6)
    ax_b.set_ylim(0.14, 1.04)
    ax_b.set_xlabel(r"Storage time $t$")
    ax_b.set_ylabel(r"Logical recovery probability $P_{\mathrm{id}}(t)$")
    ax_b.legend(loc="upper right", frameon=False, handlelength=1.9, labelspacing=0.21, fontsize=7.1)

    # --- (c) Matched L=3 (C3 / R3) two-cycle lifetimes ---
    ax_c = fig.add_subplot(gs[1, 0])
    style_prx_ax(ax_c, "c", x_off=-0.15)

    for tag, fmt, col, lbl in [
        ("v15_sq_L3_RR", "-D", C_RR, r"$R_3 R_3$ ($709{,}408$)"),
        ("v15_sq_L3_CR", "-^", C_CR, r"$C_3 R_3$ ($591{,}208$)"),
        ("v15_sq_L3_CC", "--s", C_CC, r"$C_3 C_3$ ($415{,}004$)"),
        ("v15_sq_L3_RC", "-o", C_RC, r"$R_3 C_3$ ($499{,}088$)"),
    ]:
        plot_lifetime_series(ax_c, data, tag, fmt, col, lbl, 4.5, 7.0)

    pts_r3 = get_pts(data, "v15_sq_R3", 4.5, 7.0)
    ax_c.plot([p["beta"] for p in pts_r3], [p["tau_Q"] for p in pts_r3], ":", color="#0f172a", lw=1.55, label=r"1-cycle $R_3$ ($1{,}480$)")
    ax_c.set_yscale("log")
    ax_c.set_xlim(4.40, 7.12)
    ax_c.set_xlabel(r"Inverse temperature $\beta$")
    ax_c.set_ylabel(r"Mean failure time $\tau_Q$")
    ax_c.legend(loc="upper left", frameon=False, handlelength=2.1)

    # --- (d) Unit-patch C2/R3 & single-rib C3/C2 two-cycle lifetimes ---
    ax_d = fig.add_subplot(gs[1, 1])
    style_prx_ax(ax_d, "d", x_off=-0.15)

    for tag, fmt, col, hollow, lbl in [
        ("v15_sq_pkg_CC", "--s", C_CC, True,  r"$C_2 C_2$ ($25{,}148$)"),
        ("v15_sq_pkg_RC", "-o",  C_RC, False, r"$R_3 C_2$ ($133{,}864$)"),
        ("v16_sq_C3C2",   "-D",  C_C3, False, r"$C_3 C_2$ ($107{,}140$)"),
        ("v15_sq_pkg_CR", "-^",  C_CR, False, r"$C_2 R_3$ ($136{,}536$)"),
        ("v16_sq_C2C3",   "-.v", C_GOLD, False, r"$C_2 C_3$ ($95{,}896$)"),
    ]:
        plot_lifetime_series(ax_d, data, tag, fmt, col, lbl, 4.5, 7.0, hollow=hollow)

    ax_d.plot([p["beta"] for p in pts_r3], [p["tau_Q"] for p in pts_r3], ":", color="#0f172a", lw=1.55, label=r"1-cycle $R_3$ ($1{,}480$)")
    ax_d.set_yscale("log")
    ax_d.set_xlim(4.40, 7.12)
    ax_d.set_ylim(25.0, 4.5e7)
    ax_d.set_xlabel(r"Inverse temperature $\beta$")
    ax_d.set_ylabel(r"Mean failure time $\tau_Q$")
    ax_d.legend(loc="upper left", frameon=False, handlelength=2.0, labelspacing=0.24)

    save_fig(fig, "fig_v21_2_reversal")


# ==============================================================================
# FIGURE 3: Resource Frontier (Including CC & C3C2) and Spectral Volume Screen
# ==============================================================================
def render_fig3(data):
    fig = plt.figure(figsize=(12.4, 3.95))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.12, 1.0], wspace=0.25)

    # --- (a) Resource frontier at beta = 6.5 (tau_Q & operational t_0.80 vs N) ---
    ax_a = fig.add_subplot(gs[0, 0])
    style_prx_ax(ax_a, "a", x_off=-0.15)

    # Filled markers = pass thin-bridge 3D spectral volume screen; open square = fails screen (C2C2, 1-cycle C2)
    frontier_items = [
        ("v15_sq_C2",     "o", C_C2,   True,  r"1-cycle $C_2$"),
        ("v15_sq_C3",     "s", C_C3,   False, r"1-cycle $C_3$"),
        ("v15_sq_R3",     "D", C_RR,   False, r"1-cycle $R_3$"),
        ("v15_sq_pkg_CC", "s", C_CC,   True,  r"$C_2 C_2$"),
        ("v16_sq_C2C3",   "v", C_GOLD, False, r"$C_2 C_3$"),
        ("v16_sq_C3C2",   "P", C_C3,   False, r"$C_3 C_2$"),
        ("v15_sq_pkg_RC", "o", C_RC,   False, r"$R_3 C_2$"),
        ("v15_sq_pkg_CR", "^", C_CR,   False, r"$C_2 R_3$"),
        ("v15_sq_L3_CC",  "s", "#94a3b8", False, r"$C_3 C_3$"),
        ("v15_sq_L3_RC",  "o", "#60a5fa", False, r"$R_3 C_3$"),
        ("v15_sq_L3_CR",  "^", "#fb923c", False, r"$C_3 R_3$"),
        ("v15_sq_L3_RR",  "D", "#10b981", False, r"$R_3 R_3$"),
    ]
    for tag, mk, col, fails_screen, lbl in frontier_items:
        if tag not in data["codes"]:
            continue
        N = data["codes"][tag]["spec"]["N"]
        pts65 = [p for p in data["codes"][tag]["points"] if abs(p["beta"] - 6.50) < 1e-6]
        if not pts65:
            continue
        p = pts65[0]
        y = p["tau_Q"]
        lo, hi = y - p["ci95_Q"][0], p["ci95_Q"][1] - y
        ax_a.errorbar(
            [N], [y], yerr=[[max(0.0, lo)], [max(0.0, hi)]],
            fmt=mk, color=col, ms=6.6,
            mfc=("white" if fails_screen else col), mew=1.55,
            capsize=2.3, label=lbl,
        )

    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlim(170, 1.3e6)
    ax_a.set_ylim(3.2e2, 3.5e6)
    ax_a.set_xlabel(r"Physical qubits $N$")
    ax_a.set_ylabel(r"Mean failure time $\tau_Q$")
    ax_a.legend(loc="lower left", frameon=False, ncol=3, fontsize=7.2, handletextpad=0.35, columnspacing=0.8)

    # --- (b) Proven spectral volume-growth ratio rho(T_blk) / L_blk^3 ---
    ax_b = fig.add_subplot(gs[0, 1])
    style_prx_ax(ax_b, "b", x_off=-0.16)
    ax_b.axvline(1.0, color="#b91c1c", ls="--", lw=1.3)

    # Exact eigenvalues from Proposition 1:
    T_C2 = np.array([[8.0, 4.0], [4.0, 1.0]])
    T_C3 = np.array([[18.0, 6.0], [6.0, 1.0]])
    T_R3 = np.array([[18.0, 6.0], [12.0, 2.0]])
    def rho(M): return float(np.max(np.real(np.linalg.eigvals(M))))

    blocks = [
        (r"$C_2 C_2$", rho(np.linalg.matrix_power(T_C2, 4)) / (16.0 ** 3), (12.0 ** 4) / (16.0 ** 3)),
        (r"$R_3 C_2$ / $C_2 R_3$", rho(T_C2 @ T_C2 @ T_R3 @ T_R3) / (36.0 ** 3), (12.0 ** 2 * 30.0 ** 2) / (36.0 ** 3)),
        (r"$C_3 C_2$ / $C_2 C_3$", rho(T_C2 @ T_C2 @ T_C3 @ T_C3) / (36.0 ** 3), (12.0 ** 2 * 24.0 ** 2) / (36.0 ** 3)),
        (r"$R_3 R_3$", rho(np.linalg.matrix_power(T_R3, 4)) / (81.0 ** 3), (30.0 ** 4) / (81.0 ** 3)),
        (r"$R_3 C_3$ / $C_3 R_3$", rho(T_C3 @ T_C3 @ T_R3 @ T_R3) / (81.0 ** 3), (24.0 ** 2 * 30.0 ** 2) / (81.0 ** 3)),
        (r"$C_3 C_3$", rho(np.linalg.matrix_power(T_C3, 4)) / (81.0 ** 3), (24.0 ** 4) / (81.0 ** 3)),
    ]
    y_pos = np.arange(len(blocks))
    h = 0.34
    thin_vals = [b[1] for b in blocks]
    iso_vals = [b[2] for b in blocks]

    ax_b.barh(y_pos - h / 2, thin_vals, height=h, color=C_RC, alpha=0.88, label="Thin bridges")
    ax_b.barh(y_pos + h / 2, iso_vals,  height=h, color="#94a3b8", alpha=0.78, label="Isotropic bridges")

    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels([b[0] for b in blocks])
    ax_b.invert_yaxis()
    ax_b.set_xscale("log")
    ax_b.set_xlim(0.18, 8.5)
    ax_b.set_xlabel(r"Face-growth ratio per unit volume")
    ax_b.legend(loc="lower right", frameon=False, fontsize=7.7)

    save_fig(fig, "fig_v21_3_resources_growth")


# ==============================================================================
# FIGURE 4: Causal Zero-Energy Kinetic Test & Recovery-Cluster Diagnostics
# ==============================================================================
def render_fig4(data):
    fig = plt.figure(figsize=(12.8, 7.6))
    gs = GridSpec(2, 2, figure=fig, wspace=0.25, hspace=0.28)

    # --- (a) Detailed-balance-preserving zero-energy kinetic test (beta = 6.5) ---
    ax_a = fig.add_subplot(gs[0, 0])
    style_prx_ax(ax_a, "a", x_off=-0.15)
    ax_a.axvline(1.0, color="#64748b", ls=":", lw=1.05)

    kin_specs = [
        ("v15_sq_L3_RC", "flat",     "-o",  C_RC, False, r"$R_3 C_3$, flat-sheet qubits"),
        ("v15_sq_L3_RC", "seamctrl", "--o", C_RC, True,  r"$R_3 C_3$, seam control"),
        ("v15_sq_L3_CR", "flat",     "-^",  C_CR, False, r"$C_3 R_3$, flat-sheet qubits"),
        ("v15_sq_L3_CR", "seamctrl", "--^", C_CR, True,  r"$C_3 R_3$, seam control"),
    ]
    for tag, mkind, fmt, col, hollow, lbl in kin_specs:
        rows = sorted(data["kinetic_zero_e"].get(tag, {}).get(mkind, []), key=lambda d: d["mu"])
        if not rows:
            continue
        mus = [r["mu"] for r in rows]
        ys = [r["tau_X"] for r in rows]
        lo = [r["tau_X"] - r["ci95_X"][0] for r in rows]
        hi = [r["ci95_X"][1] - r["tau_X"] for r in rows]
        ax_a.errorbar(
            mus, ys, yerr=[lo, hi], fmt=fmt, color=col, ms=5.1,
            mfc=("white" if hollow else col), mew=1.3,
            lw=(1.8 if not hollow else 1.3), capsize=2.0, label=lbl,
        )

    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_ylim(1.8e3, 7.8e4)
    ax_a.set_xlabel(r"Zero-energy flip rate factor $\mu$")
    ax_a.set_ylabel(r"Mean failure time $\tau_X$")
    ax_a.legend(loc="center right", frameon=False, handlelength=2.0, fontsize=7.4)

    # --- (b) Decoder recovery-cluster arrival time A_10 +/- SE ---
    ax_b = fig.add_subplot(gs[0, 1])
    style_prx_ax(ax_b, "b", x_off=-0.15)

    for tag, fmt, col, lbl in [
        ("v15_sq_pkg_RC", "-o",  C_RC, r"$R_3 C_2$"),
        ("v15_sq_pkg_CR", "-^",  C_CR, r"$C_2 R_3$"),
        ("v16_sq_C3C2",   "-D",  C_C3, r"$C_3 C_2$"),
        ("v15_sq_L3_CR",  "--^", C_CR, r"$C_3 R_3$"),
        ("v16_sq_C2C3",   "-.v", C_GOLD, r"$C_2 C_3$"),
        ("v15_sq_L3_RC",  "--o", C_RC, r"$R_3 C_3$"),
    ]:
        pts = [p for p in get_pts(data, tag, 6.0, 7.0) if p.get("span_v16") and p["span_v16"]["mean_A10"] is not None]
        if not pts:
            continue
        bs = [p["beta"] for p in pts]
        ys = [p["span_v16"]["mean_A10"] for p in pts]
        ses = [p["span_v16"]["se_A10"] for p in pts]
        ax_b.errorbar(bs, ys, yerr=ses, fmt=fmt, color=col, ms=5.0, lw=1.65, capsize=2.0, label=lbl)

    ax_b.set_yscale("log")
    ax_b.set_xlim(5.92, 7.08)
    ax_b.set_xlabel(r"Inverse temperature $\beta$")
    ax_b.set_ylabel(r"Time to 10-cell cluster")
    ax_b.legend(loc="upper left", frameon=False, ncol=2, handlelength=2.0, fontsize=7.5)

    # --- (c) Conditional post-arrival residual survival E[T_X - A_6 | A_6 < T_X] ---
    ax_c = fig.add_subplot(gs[1, 0])
    style_prx_ax(ax_c, "c", x_off=-0.15)

    for tag, fmt, col, lbl in [
        ("v15_sq_pkg_RC", "-o",  C_RC, r"$R_3 C_2$"),
        ("v16_sq_C3C2",   "-D",  C_C3, r"$C_3 C_2$"),
        ("v15_sq_pkg_CR", "-^",  C_CR, r"$C_2 R_3$"),
        ("v15_sq_L3_CR",  "--^", C_CR, r"$C_3 R_3$"),
        ("v16_sq_C2C3",   "-.v", C_GOLD, r"$C_2 C_3$"),
        ("v15_sq_L3_RC",  "--o", C_RC, r"$R_3 C_3$"),
    ]:
        pts = [
            p for p in get_pts(data, tag, 6.0, 7.0)
            if (not p["is_censored"]) and p.get("span_v16") and p["span_v16"]["cond_residual_T_minus_A6"] is not None
        ]
        if not pts:
            continue
        bs = [p["beta"] for p in pts]
        ys = [p["span_v16"]["cond_residual_T_minus_A6"] for p in pts]
        ses = [p["span_v16"]["se_residual_6"] for p in pts]
        ax_c.errorbar(bs, ys, yerr=ses, fmt=fmt, color=col, ms=5.0, lw=1.65, capsize=2.0, label=lbl)

    ax_c.set_yscale("log")
    ax_c.set_xlim(5.92, 7.08)
    ax_c.set_xlabel(r"Inverse temperature $\beta$")
    ax_c.set_ylabel(r"Remaining time after 6-cell cluster")
    ax_c.legend(loc="upper left", frameon=False, ncol=2, handlelength=2.0, fontsize=7.5)

    # --- (d) Exploratory cold-slope comparison WITH 95% CIs on both axes ---
    ax_d = fig.add_subplot(gs[1, 1])
    style_prx_ax(ax_d, "d", x_off=-0.15)
    ax_d.axhline(0.0, color="#64748b", ls=":", lw=1.0)
    ax_d.plot([-0.2, 2.3], [-0.2, 2.3], "--", color="#94a3b8", lw=1.1, label="Equal slope difference")

    sc_items = [
        ("pkg_planar",  "o", C_RC,   r"Planar $C_2 / R_3$"),
        ("pkg_torus",   "s", C_RC,   r"Torus $C_2 / R_3$"),
        ("pkg_rottor",  "D", C_C2,   r"Rotated torus $C_2 / R_3$"),
        ("C3C2_planar", "P", C_C3,   r"Planar $C_2 / C_3$"),
        ("C3C2_torus",  "X", "#6d28d9", r"Torus $C_2 / C_3$"),
        ("L3_planar",   "^", C_CR,   r"Planar $C_3 / R_3$"),
        ("L3_torus",    "v", C_GOLD, r"Torus $C_3 / R_3$"),
    ]
    for skey, mk, col, lbl in sc_items:
        pair = data["slope_comparisons"].get(skey)
        if not pair or not pair["two_cycle"] or not pair["one_cycle"]:
            continue
        s1, s2 = pair["one_cycle"], pair["two_cycle"]
        ax_d.errorbar(
            [s1["diff"]], [s2["diff"]],
            xerr=[[s1["diff"] - s1["ci95"][0]], [s1["ci95"][1] - s1["diff"]]],
            yerr=[[s2["diff"] - s2["ci95"][0]], [s2["ci95"][1] - s2["diff"]]],
            fmt=mk, color=col, ms=6.2, capsize=2.2, label=lbl,
        )

    ax_d.set_ylim(-1.55, 2.55)
    ax_d.set_xlabel(r"Single-cycle slope difference")
    ax_d.set_ylabel(r"Two-cycle slope difference")
    ax_d.legend(loc="lower right", frameon=False, fontsize=7.2, labelspacing=0.22)

    save_fig(fig, "fig_v21_4_mechanism_causal")


def main():
    with open(ROOT / "parsed_v16_data.json") as f:
        data = json.load(f)
    render_fig1(data)
    render_fig2(data)
    render_fig3(data)
    render_fig4(data)


if __name__ == "__main__":
    main()
