#!/usr/bin/env python3
"""
render_v22_prx_figures.py
Renders four clean Physical Review X figures for
"The Thermal Value of Welding Across Scales":
  - Serif + Computer Modern math typography matching REVTeX 4.2
  - Full 4-sided spines with inward ticks; bold (a), (b), (c), (d) labels ONLY (no subplot titles)
  - Minimal in-plot text; crisp error bars / 95% CIs on all plotted statistics
  - Explicit upward arrows on right-censored lower-bound points (including planar R3C2 at beta = 7.0)

Figures:
  1. `fig_v22_1_modules`: Sheet partition of C2, C3, and R3, two-cycle hierarchy tiles
     (C3C2 vs C2C3 and R3C2 vs C2R3), and standalone single-cycle lifetimes.
  2. `fig_v22_2_matrix_reversal`: Full 3x3 coarse-by-fine operational matrix (t_0.80 at beta = 6.5),
     standalone vs fine-constituent reversal across all coarse contexts, swap placement ratios,
     and two-cycle lifetime curves.
  3. `fig_v22_3_resources_cold`: Operational storage vs physical qubits N at beta = 6.5,
     cold marginal coarse-bridge benefit G_B(beta) at beta = 6.5 vs beta = 7.0,
     passive recovery vs no-crossing survival decomposition, and periodic-continuation screen.
  4. `fig_v22_4_causal_kinetic`: Controlled fine-C3 zero-energy kinetic intervention t_0.80(mu)
     on R3C3 and C3C3 versus matched seam controls, and marginal coarse-bridge ratio G_{C3}(mu).
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "cm",
    "font.size": 9.3,
    "axes.labelsize": 10.0,
    "xtick.labelsize": 8.6,
    "ytick.labelsize": 8.6,
    "legend.fontsize": 7.6,
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


def get_pts(v16, tag, bmin=4.5, bmax=7.5):
    pts = [p for p in v16["codes"][tag]["points"] if bmin - 1e-6 <= p["beta"] <= bmax + 1e-6]
    pts.sort(key=lambda d: d["beta"])
    return pts


def plot_lifetime_series(ax, v16, tag, fmt, col, lbl, bmin=4.5, bmax=7.0, hollow=False, ms=5.1, lw=1.75):
    if tag not in v16["codes"]:
        return
    pts = get_pts(v16, tag, bmin, bmax)
    unc = [p for p in pts if not p["is_censored"]]
    cen = [p for p in pts if p["is_censored"]]

    ls = "--" if "--" in fmt else ("-." if "-." in fmt else "-")
    mk = fmt.replace("--", "").replace("-.", "").replace("-", "")

    # Connect only uncensored points with solid/dashed curve; connect last uncensored to censored with dotted guide:
    if unc:
        ax.plot([p["beta"] for p in unc], [p["tau_Q"] for p in unc], ls=ls, color=col, lw=lw, alpha=0.92)
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
        if unc and p["beta"] > unc[-1]["beta"]:
            ax.plot([unc[-1]["beta"], p["beta"]], [unc[-1]["tau_Q"], p["tau_Q"]], ls=":", color=col, lw=1.2, alpha=0.8)
        y0 = p["tau_Q"]
        ax.plot([p["beta"]], [y0], marker=mk, color=col, mfc="white", mew=1.4, ms=ms, zorder=5)
        ax.annotate(
            "", xy=(p["beta"], y0 * 2.8), xytext=(p["beta"], y0),
            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.35, mutation_scale=9),
            zorder=6,
        )


def fmt_short_time(val):
    if val >= 1e5:
        return f"{val / 1e3:.0f}k"
    if val >= 1e4:
        return f"{val / 1e3:.1f}k"
    return f"{val / 1e3:.1f}k"


# ==============================================================================
# FIGURE 1: Local Modules (C2, C3, R3), Two-Cycle Order, & 1-Cycle Lifetimes
# ==============================================================================
def render_fig1(v16):
    fig = plt.figure(figsize=(13.6, 3.85))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.22, 0.88, 1.08], wspace=0.22)

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
            if k not in ribs:
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
    draw_arch_tile(1.6,  10.8, tw, 3, [1],    2, [1],    C_C3, r"$C_3 C_2$")
    draw_arch_tile(11.1, 10.8, tw, 2, [1],    3, [1],    C_GOLD, r"$C_2 C_3$")
    draw_arch_tile(1.6,  1.9,  tw, 3, [1, 2], 2, [1],    C_RC, r"$R_3 C_2$")
    draw_arch_tile(11.1, 1.9,  tw, 2, [1],    3, [1, 2], C_CR, r"$C_2 R_3$")

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
        plot_lifetime_series(ax_c, v16, tag, fmt, col, lbl, 4.5, 7.25, hollow=hollow, ms=4.9, lw=(1.75 if not hollow else 1.3))

    ax_c.set_yscale("log")
    ax_c.set_xlim(4.40, 7.38)
    ax_c.set_xlabel(r"Inverse temperature $\beta$")
    ax_c.set_ylabel(r"Single-cycle failure time $\tau_Q$")
    ax_c.legend(loc="upper left", frameon=False, handlelength=2.0, labelspacing=0.25)

    save_fig(fig, "fig_v22_1_modules")


# ==============================================================================
# FIGURE 2: 3x3 Operational Matrix, Standalone Misranking, & Placement Reversal
# ==============================================================================
def render_fig2(v16, v17):
    fig = plt.figure(figsize=(12.8, 7.8))
    gs = GridSpec(2, 2, figure=fig, wspace=0.24, hspace=0.29)

    # --- (a) Full 3x3 Coarse-by-Fine Operational Matrix (Planar beta = 6.5, t_0.80) ---
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.tick_params(direction="out", top=False, right=False, length=3.5)
    ax_a.text(-0.15, 1.03, "(a)", transform=ax_a.transAxes, fontsize=10.8, fontweight="bold", va="bottom", ha="left", color="#0f172a")

    mods = ["C2", "C3", "R3"]
    mod_lbls = [r"$C_2$", r"$C_3$", r"$R_3$"]
    cells = v17["matrix_650"]["planar"]["cells"]
    mat = np.zeros((3, 3), dtype=np.float64)
    for i, A in enumerate(mods):
        for j, B in enumerate(mods):
            mat[i, j] = cells[f"{A}_{B}"]["t_80"]

    im = ax_a.imshow(mat, cmap="YlGnBu", norm=LogNorm(vmin=1.4e3, vmax=2.8e5), aspect="auto")
    for i, A in enumerate(mods):
        for j, B in enumerate(mods):
            rec = cells[f"{A}_{B}"]
            v = rec["t_80"]
            lo, hi = rec["t_80_ci95"]
            txt_col = "white" if v > 4.5e4 else "#0f172a"
            ax_a.text(j, i - 0.09, fmt_short_time(v), ha="center", va="center", fontsize=10.2, fontweight="bold", color=txt_col)
            ax_a.text(
                j, i + 0.21, f"[{fmt_short_time(lo)}, {fmt_short_time(hi)}]",
                ha="center", va="center", fontsize=7.2, color=txt_col, alpha=0.92,
            )

    ax_a.set_xticks([0, 1, 2])
    ax_a.set_xticklabels(mod_lbls, fontsize=9.8)
    ax_a.set_yticks([0, 1, 2])
    ax_a.set_yticklabels(mod_lbls, fontsize=9.8)
    ax_a.set_xlabel("Fine inner module")
    ax_a.set_ylabel("Coarse outer module")
    cbar = fig.colorbar(im, ax=ax_a, fraction=0.046, pad=0.035)
    cbar.set_label(r"Recovery time $t_{0.80}$ ($\beta = 6.5$)", fontsize=8.8)
    cbar.ax.tick_params(labelsize=8.0)

    # --- (b) Standalone Module Lifetime vs Fine-Constituent Performance (Planar beta = 6.5) ---
    ax_b = fig.add_subplot(gs[0, 1])
    style_prx_ax(ax_b, "b", x_off=-0.15)

    groups = [
        ("Standalone\n(1 cycle)", [("C2", "v15_sq_C2"), ("C3", "v15_sq_C3"), ("R3", "v15_sq_R3")]),
        (r"Coarse $C_2$", [("C2", "v15_sq_pkg_CC"), ("C3", "v16_sq_C2C3"), ("R3", "v15_sq_pkg_CR")]),
        (r"Coarse $C_3$", [("C2", "v16_sq_C3C2"), ("C3", "v15_sq_L3_CC"), ("R3", "v15_sq_L3_CR")]),
        (r"Coarse $R_3$", [("C2", "v15_sq_pkg_RC"), ("C3", "v15_sq_L3_RC"), ("R3", "v15_sq_L3_RR")]),
    ]
    f_styles = {
        "C2": ("o", C_C2, -0.20, r"Fine / standalone $C_2$"),
        "C3": ("s", C_C3,  0.00, r"Fine / standalone $C_3$"),
        "R3": ("D", C_RR, +0.20, r"Fine / standalone $R_3$"),
    }
    ax_b.axvline(0.5, color="#94a3b8", ls=":", lw=1.0)
    used_lbl = set()
    for gx, (_, items) in enumerate(groups):
        for mkey, tag in items:
            mk, col, dx, lbl = f_styles[mkey]
            rec = v17["std_readout"][tag]["6.50"]
            y = rec["t_80"]
            lo, hi = rec["t_80_ci95"]
            ax_b.errorbar(
                [gx + dx], [y], yerr=[[y - lo], [hi - y]],
                fmt=mk, color=col, ms=6.2, mew=1.25, capsize=2.8, lw=1.35,
                label=(lbl if mkey not in used_lbl else None),
            )
            used_lbl.add(mkey)

    ax_b.set_yscale("log")
    ax_b.set_xlim(-0.52, 3.52)
    ax_b.set_ylim(1.1e3, 4.2e5)
    ax_b.set_xticks([0, 1, 2, 3])
    ax_b.set_xticklabels([g[0] for g in groups], fontsize=8.5)
    ax_b.set_ylabel(r"Recovery time $t_{0.80}$ ($\beta = 6.5$)")
    ax_b.legend(loc="lower right", frameon=False, fontsize=7.6)

    # --- (c) Placement ratio across beta for all three module pairs ---
    ax_c = fig.add_subplot(gs[1, 0])
    style_prx_ax(ax_c, "c", x_off=-0.15)
    ax_c.axhline(1.0, color="#334155", ls=":", lw=1.2)

    ratio_specs = [
        ("C3C2_planar_C3C2_over_C2C3", "-s",  C_C3, False, r"Planar $C_3 C_2 / C_2 C_3$"),
        ("C3C2_torus_C3C2_over_C2C3",  "--s", C_C3, True,  r"Torus $C_3 C_2 / C_2 C_3$"),
        ("pkg_planar_RC_over_CR",      "-o",  C_RC, False, r"Planar $R_3 C_2 / C_2 R_3$"),
        ("pkg_torus_RC_over_CR",       "--o", C_RC, True,  r"Torus $R_3 C_2 / C_2 R_3$"),
        ("L3_planar_RC_over_CR",       "-^",  C_CR, False, r"Planar $R_3 C_3 / C_3 R_3$"),
        ("L3_torus_RC_over_CR",        "--^", C_CR, True,  r"Torus $R_3 C_3 / C_3 R_3$"),
    ]
    for rkey, fmt, col, hollow, lbl in ratio_specs:
        rows = [r for r in v16["comparisons"].get(rkey, []) if r["status"] != "omitted_denominator_censored"]
        if not rows:
            continue
        ls = "--" if "--" in fmt else "-"
        mk = fmt.replace("--", "").replace("-", "")
        unc = [r for r in rows if r["status"] == "uncensored"]
        cen = [r for r in rows if r["status"] == "lower_bound_numerator_censored"]
        if unc:
            ax_c.plot([r["beta"] for r in unc], [r["ratio"] for r in unc], ls=ls, color=col, lw=(1.8 if not hollow else 1.35))
            ax_c.errorbar(
                [r["beta"] for r in unc],
                [r["ratio"] for r in unc],
                yerr=[[r["ratio"] - r["ci95"][0] for r in unc], [r["ci95"][1] - r["ratio"] for r in unc]],
                fmt=mk, color=col, ms=5.0, mfc=("white" if hollow else col), mew=1.3, capsize=2.0, ls="none", label=lbl,
            )
        for r in cen:
            bx = r["beta"] + (0.035 if "planar" in rkey else 0.0)
            ax_c.plot([bx], [r["ratio"]], marker=mk, color=col, mfc="white", mew=1.4, ms=5.1, zorder=5)
            ax_c.annotate(
                "", xy=(bx, r["ratio"] * 2.5), xytext=(bx, r["ratio"]),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4, mutation_scale=9.0),
                zorder=6,
            )

    ax_c.set_yscale("log")
    ax_c.set_xlim(4.40, 7.14)
    ax_c.set_ylim(0.025, 290.0)
    ax_c.set_xlabel(r"Inverse temperature $\beta$")
    ax_c.set_ylabel("Placement ratio (coarse vs. fine)")
    ax_c.legend(loc="upper left", frameon=False, ncol=1, handlelength=2.0, labelspacing=0.22, fontsize=7.3)

    # --- (d) Two-cycle first-failure curves (with explicit lower-bound arrow on planar R3C2 at beta = 7.0) ---
    ax_d = fig.add_subplot(gs[1, 1])
    style_prx_ax(ax_d, "d", x_off=-0.15)

    for tag, fmt, col, hollow, lbl in [
        ("v15_sq_pkg_CC", "--s", C_CC, True,  r"$C_2 C_2$ ($25{,}148$)"),
        ("v15_sq_pkg_RC", "-o",  C_RC, False, r"$R_3 C_2$ ($133{,}864$)"),
        ("v16_sq_C3C2",   "-D",  C_C3, False, r"$C_3 C_2$ ($107{,}140$)"),
        ("v15_sq_pkg_CR", "-^",  C_CR, False, r"$C_2 R_3$ ($136{,}536$)"),
        ("v15_sq_L3_RR",  "--d", C_RR, False, r"$R_3 R_3$ ($709{,}408$)"),
        ("v16_sq_C2C3",   "-.v", C_GOLD, False, r"$C_2 C_3$ ($95{,}896$)"),
    ]:
        plot_lifetime_series(ax_d, v16, tag, fmt, col, lbl, 4.5, 7.0, hollow=hollow)

    pts_r3 = get_pts(v16, "v15_sq_R3", 4.5, 7.0)
    ax_d.plot([p["beta"] for p in pts_r3], [p["tau_Q"] for p in pts_r3], ":", color="#0f172a", lw=1.5, label=r"1-cycle $R_3$ ($1{,}480$)")
    ax_d.set_yscale("log")
    ax_d.set_xlim(4.40, 7.15)
    ax_d.set_ylim(20.0, 9.0e7)
    ax_d.set_xlabel(r"Inverse temperature $\beta$")
    ax_d.set_ylabel(r"Mean failure time $\tau_Q$")
    ax_d.legend(loc="upper left", frameon=False, handlelength=2.0, labelspacing=0.21, fontsize=7.2)

    save_fig(fig, "fig_v22_2_matrix_reversal")


# ==============================================================================
# FIGURE 3: Operational Resource Frontier, Cold Marginal Gain, Healing, & Growth
# ==============================================================================
def render_fig3(v16, v17):
    fig = plt.figure(figsize=(12.8, 7.8))
    gs = GridSpec(2, 2, figure=fig, wspace=0.25, hspace=0.29)

    # --- (a) Operational recovery time t_0.80 vs Physical Qubits N at beta = 6.5 (Planar) ---
    ax_a = fig.add_subplot(gs[0, 0])
    style_prx_ax(ax_a, "a", x_off=-0.15)

    # (tag, marker, color, open_marker, text_label, (dx_pts, dy_pts), ha)
    frontier_pts = [
        ("v15_sq_C2",     "o", C_C2,      True,  r"$C_2$",     (7,  2),   "left"),
        ("v15_sq_C3",     "s", C_C3,      False, r"$C_3$",     (7, -3),   "left"),
        ("v15_sq_R3",     "D", C_RR,      False, r"$R_3$",     (7,  2),   "left"),
        ("v15_sq_pkg_CC", "s", C_CC,      True,  r"$C_2 C_2$", (-7, 4),   "right"),
        ("v16_sq_C2C3",   "v", C_GOLD,    False, r"$C_2 C_3$", (-7, 2),   "right"),
        ("v16_sq_C3C2",   "D", C_C3,      False, r"$C_3 C_2$", (-7, -4),  "right"),
        ("v15_sq_pkg_RC", "o", C_RC,      False, r"$R_3 C_2$", (8,  4),   "left"),
        ("v15_sq_pkg_CR", "^", C_CR,      False, r"$C_2 R_3$", (7,  5),   "left"),
        ("v15_sq_L3_CC",  "s", "#475569", False, r"$C_3 C_3$", (-7, 4),   "right"),
        ("v15_sq_L3_RC",  "v", "#0369a1", False, r"$R_3 C_3$", (7, -2),   "left"),
        ("v15_sq_L3_CR",  "^", "#ea580c", False, r"$C_3 R_3$", (-7, -4),  "right"),
        ("v15_sq_L3_RR",  "D", "#065f46", False, r"$R_3 R_3$", (7, -5),   "left"),
    ]
    for tag, mk, col, open_mk, lbl, (dx, dy), ha_align in frontier_pts:
        N = v16["codes"][tag]["spec"]["N"]
        rec = v17["std_readout"][tag]["6.50"]
        y = rec["t_80"]
        lo, hi = rec["t_80_ci95"]
        ax_a.errorbar(
            [N], [y], yerr=[[y - lo], [hi - y]],
            fmt=mk, color=col, mfc=("white" if open_mk else col),
            mew=1.5, ms=6.2, capsize=2.5, lw=1.25,
        )
        ax_a.annotate(
            lbl, xy=(N, y), xytext=(dx, dy), textcoords="offset points",
            ha=ha_align, va="center", fontsize=8.0, color=col, fontweight="bold",
        )

    ax_a.plot([], [], "o", color="#334155", mfc="#334155", ms=5.5, label="Passes continuation screen")
    ax_a.plot([], [], "o", color="#334155", mfc="white", mew=1.4, ms=5.5, label="Exceeds continuation screen")
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlim(150.0, 1.85e6)
    ax_a.set_ylim(1.1e3, 4.8e5)
    ax_a.set_xlabel(r"Physical qubits $N$")
    ax_a.set_ylabel(r"Recovery time $t_{0.80}$ ($\beta = 6.5$)")
    ax_a.legend(loc="upper left", frameon=False, fontsize=7.4, handletextpad=0.4)

    # --- (b) Marginal Coarse-Bridge Benefit G_B(beta) = t_0.80(R3B) / t_0.80(C3B) at beta = 6.5 vs 7.0 ---
    ax_b = fig.add_subplot(gs[0, 1])
    style_prx_ax(ax_b, "b", x_off=-0.15)
    ax_b.axhline(1.0, color="#334155", ls=":", lw=1.2)

    cm = v17["cold_marginal"]
    gb_series = [
        (r"Planar, fine $C_2$ ($R_3 C_2 / C_3 C_2$)", C_RC,  "o", False, -0.05, [cm["planar"]["6.50"]["G_C2_t80"], cm["planar"]["7.00"]["G_C2_t80"]]),
        (r"Torus, fine $C_2$ ($R_3 C_2 / C_3 C_2$)",  C_RC,  "o", True,  +0.05, [cm["torus"]["6.50"]["G_C2_t80"],  cm["torus"]["7.00"]["G_C2_t80"]]),
        (r"Planar, fine $C_3$ ($R_3 C_3 / C_3 C_3$)", C_CR,  "s", False, -0.05, [cm["planar"]["6.50"]["G_C3_t80"], cm["planar"]["7.00"]["G_C3_t80"]]),
        (r"Torus, fine $C_3$ ($R_3 C_3 / C_3 C_3$)",  C_CR,  "s", True,  +0.05, [cm["torus"]["6.50"]["G_C3_t80"],  cm["torus"]["7.00"]["G_C3_t80"]]),
    ]
    xs_base = np.array([0.0, 1.0])
    for lbl, col, mk, hollow, dx, recs in gb_series:
        ys = [r["ratio"] for r in recs]
        lo = [r["ratio"] - r["ci95"][0] for r in recs]
        hi = [r["ci95"][1] - r["ratio"] for r in recs]
        ls = "--" if hollow else "-"
        ax_b.plot(xs_base + dx, ys, ls=ls, color=col, lw=1.55)
        ax_b.errorbar(
            xs_base + dx, ys, yerr=[lo, hi],
            fmt=mk, color=col, mfc=("white" if hollow else col),
            mew=1.45, ms=6.2, capsize=2.8, ls="none", label=lbl,
        )

    ax_b.set_yscale("log")
    ax_b.set_xlim(-0.35, 1.35)
    ax_b.set_ylim(0.28, 14.0)
    ax_b.set_xticks([0.0, 1.0])
    ax_b.set_xticklabels([r"$\beta = 6.5$", r"$\beta = 7.0$"], fontsize=9.5)
    ax_b.set_ylabel(r"Second coarse-bridge gain $G_B(\beta)$")
    ax_b.legend(loc="upper left", frameon=False, fontsize=7.3, labelspacing=0.25)

    # --- (c) Passive Recovery P_id(t) vs No-Crossing Survival P_nocross(t) on Planar R3C2 & C3C2 ---
    ax_c = fig.add_subplot(gs[1, 0])
    style_prx_ax(ax_c, "c", x_off=-0.15)
    ax_c.axhline(0.80, color="#64748b", ls="--", lw=0.9)
    ax_c.axhline(0.50, color="#94a3b8", ls=":", lw=0.9)

    for beta_s, tag, col, name_lbl in [
        ("6.50", "v16_sq_C3C2",   "#d97706", r"$C_3 C_2$ ($\beta=6.5$)"),
        ("6.50", "v15_sq_pkg_RC", "#0284c7", r"$R_3 C_2$ ($\beta=6.5$)"),
        ("7.00", "v16_sq_C3C2",   C_C3,      r"$C_3 C_2$ ($\beta=7.0$)"),
        ("7.00", "v15_sq_pkg_RC", "#1e3a8a", r"$R_3 C_2$ ($\beta=7.0$)"),
    ]:
        rec = v17["std_readout"][tag][beta_s]
        tg = np.array(rec["t_grid"])
        pid = np.array(rec["P_id"])
        pnc = np.array(rec["P_nocross"])
        lw_m = 1.9 if beta_s == "7.00" else 1.55
        ax_c.plot(tg, pid, ls="-", color=col, lw=lw_m, label=name_lbl)
        ax_c.plot(tg, pnc, ls="--", color=col, lw=1.15, alpha=0.78)

    ax_c.plot([], [], ls="-", color="#334155", lw=1.6, label="Final readout (with healing)")
    ax_c.plot([], [], ls="--", color="#334155", lw=1.15, label="No prior crossing")
    ax_c.set_xscale("log")
    ax_c.set_xlim(1e3, 1.58e7)
    ax_c.set_ylim(0.38, 1.03)
    ax_c.set_xlabel(r"Storage time $t$")
    ax_c.set_ylabel(r"Planar probability ($K = 1$)")
    ax_c.legend(loc="lower left", frameon=False, fontsize=6.9, labelspacing=0.20, handlelength=1.9)

    # --- (d) Periodic-Continuation Volume-Growth Screen rho(T_blk) / L_blk^3 ---
    ax_d = fig.add_subplot(gs[1, 1])
    style_prx_ax(ax_d, "d", x_off=-0.18)
    ax_d.axvline(1.0, color="#b91c1c", ls="--", lw=1.35, zorder=4)

    def pf(M):
        return float(np.max(np.real(np.linalg.eigvals(np.array(M, dtype=float)))))

    T_c2 = np.array([[8, 4], [4, 1]])
    T_c3 = np.array([[18, 6], [6, 1]])
    T_r3 = np.array([[18, 6], [12, 2]])

    families = [
        (r"$C_2 C_2$",          pf(np.linalg.matrix_power(T_c2, 4)) / (16.0 ** 3), (12.0 ** 4) / (16.0 ** 3)),
        (r"$R_3 C_2 / C_2 R_3$", pf(T_c2 @ T_c2 @ T_r3 @ T_r3) / (36.0 ** 3),       ((12.0 ** 2) * (30.0 ** 2)) / (36.0 ** 3)),
        (r"$C_3 C_2 / C_2 C_3$", pf(T_c2 @ T_c2 @ T_c3 @ T_c3) / (36.0 ** 3),       ((12.0 ** 2) * (24.0 ** 2)) / (36.0 ** 3)),
        (r"$R_3 R_3$",          pf(np.linalg.matrix_power(T_r3, 4)) / (81.0 ** 3), (30.0 ** 4) / (81.0 ** 3)),
        (r"$R_3 C_3 / C_3 R_3$", pf(T_c3 @ T_c3 @ T_r3 @ T_r3) / (81.0 ** 3),       ((24.0 ** 2) * (30.0 ** 2)) / (81.0 ** 3)),
        (r"$C_3 C_3$",          pf(np.linalg.matrix_power(T_c3, 4)) / (81.0 ** 3), (24.0 ** 4) / (81.0 ** 3)),
    ]
    y_pos = np.arange(len(families))[::-1]
    bh = 0.34
    ax_d.barh(y_pos + bh / 2, [f[1] for f in families], height=bh, color=C_RC, alpha=0.88, label="Thin bridges")
    ax_d.barh(y_pos - bh / 2, [f[2] for f in families], height=bh, color="#94a3b8", alpha=0.85, label="Isotropic bridges")

    ax_d.set_xscale("log")
    ax_d.set_xlim(0.22, 22.0)
    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels([f[0] for f in families], fontsize=8.4)
    ax_d.set_xlabel("Face-growth ratio per unit volume")
    ax_d.legend(loc="lower right", frameon=False, fontsize=7.6)

    save_fig(fig, "fig_v22_3_resources_cold")


# ==============================================================================
# FIGURE 4: Controlled Zero-Energy Kinetic Test & Marginal Coarse Benefit G_{C3}(mu)
# ==============================================================================
def render_fig4(v17):
    fig = plt.figure(figsize=(12.4, 3.95))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.05, 1.0], wspace=0.24)

    # --- (a) Operational recovery time t_0.80(mu) on Planar R3C3 and C3C3 (fineflat vs fineseam) ---
    ax_a = fig.add_subplot(gs[0, 0])
    style_prx_ax(ax_a, "a", x_off=-0.15)
    ax_a.axvline(1.0, color="#64748b", ls=":", lw=1.0)

    rows_p = sorted(v17["causal_GC3"]["planar"]["fineflat_vs_fineseam"], key=lambda r: r["mu"])
    mus = [r["mu"] for r in rows_p]

    curves_a = [
        ("t80_C3C3_flat", "t80_C3C3_flat_ci95", "-s",  C_C3, False, r"$C_3 C_3$, flat-sheet qubits"),
        ("t80_R3C3_flat", "t80_R3C3_flat_ci95", "-o",  C_RC, False, r"$R_3 C_3$, flat-sheet qubits"),
        ("t80_C3C3_seam", "t80_C3C3_seam_ci95", "--s", C_C3, True,  r"$C_3 C_3$, seam control"),
        ("t80_R3C3_seam", "t80_R3C3_seam_ci95", "--o", C_RC, True,  r"$R_3 C_3$, seam control"),
    ]
    for yk, cik, fmt, col, hollow, lbl in curves_a:
        ys = [r[yk] for r in rows_p]
        lo = [r[yk] - r[cik][0] for r in rows_p]
        hi = [r[cik][1] - r[yk] for r in rows_p]
        ls = "--" if "--" in fmt else "-"
        mk = fmt.replace("--", "").replace("-", "")
        ax_a.plot(mus, ys, ls=ls, color=col, lw=(1.85 if not hollow else 1.3))
        ax_a.errorbar(
            mus, ys, yerr=[lo, hi],
            fmt=mk, color=col, mfc=("white" if hollow else col),
            mew=1.35, ms=5.4, capsize=2.3, ls="none", label=lbl,
        )

    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlim(0.007, 1.45)
    ax_a.set_ylim(1.2e3, 3.6e4)
    ax_a.set_xlabel(r"Zero-energy flip rate factor $\mu$")
    ax_a.set_ylabel(r"Recovery time $t_{0.80}$ ($\beta = 6.5$)")
    ax_a.legend(loc="upper right", frameon=False, fontsize=7.4, labelspacing=0.24)

    # --- (b) Marginal Coarse-Bridge Benefit G_{C3}(mu) = t_0.80(R3C3; mu) / t_0.80(C3C3; mu) ---
    ax_b = fig.add_subplot(gs[0, 1])
    style_prx_ax(ax_b, "b", x_off=-0.15)
    ax_b.axhline(1.0, color="#b91c1c", ls="--", lw=1.3)
    ax_b.axvline(1.0, color="#64748b", ls=":", lw=1.0)

    rows_t = sorted(v17["causal_GC3"]["torus"]["fineflat_vs_fineseam"], key=lambda r: r["mu"])

    for rows, gk, fmt, col, hollow, xmult, lbl in [
        (rows_p, "G_C3_t80_flat", "-o",  C_RC,   False, 0.96, r"Planar $G_{C_3}(\mu)$, flat-sheet qubits"),
        (rows_p, "G_C3_t80_seam", "--o", C_RC,   True,  1.04, r"Planar $G_{C_3}(\mu)$, seam control"),
        (rows_t, "G_C3_t80_flat", "-D",  C_RR,   False, 0.96, r"Torus $G_{C_3}(\mu)$, flat-sheet qubits"),
        (rows_t, "G_C3_t80_seam", "--D", C_RR,   True,  1.04, r"Torus $G_{C_3}(\mu)$, seam control"),
    ]:
        xs = [r["mu"] * xmult for r in rows]
        ys = [r[gk]["ratio"] for r in rows]
        lo = [r[gk]["ratio"] - r[gk]["ci95"][0] for r in rows]
        hi = [r[gk]["ci95"][1] - r[gk]["ratio"] for r in rows]
        ls = "--" if "--" in fmt else "-"
        mk = fmt.replace("--", "").replace("-", "")
        ax_b.plot(xs, ys, ls=ls, color=col, lw=(1.8 if not hollow else 1.25))
        ax_b.errorbar(
            xs, ys, yerr=[lo, hi],
            fmt=mk, color=col, mfc=("white" if hollow else col),
            mew=1.35, ms=5.3, capsize=2.2, ls="none", label=lbl,
        )

    ax_b.set_xlim(0.007, 1.45)
    ax_b.set_xscale("log")
    ax_b.set_ylim(0.32, 1.52)
    ax_b.set_xlabel(r"Zero-energy flip rate factor $\mu$")
    ax_b.set_ylabel(r"Second coarse-bridge gain $G_{C_3}(\mu)$")
    ax_b.legend(loc="upper left", frameon=False, fontsize=7.3, labelspacing=0.23)

    save_fig(fig, "fig_v22_4_causal_kinetic")


def main():
    with open(ROOT / "parsed_v16_data.json") as f:
        v16 = json.load(f)
    with open(ROOT / "parsed_v17_data.json") as f:
        v17 = json.load(f)
    render_fig1(v16)
    render_fig2(v16, v17)
    render_fig3(v16, v17)
    render_fig4(v17)


if __name__ == "__main__":
    main()
