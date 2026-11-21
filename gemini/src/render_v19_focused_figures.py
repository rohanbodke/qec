#!/usr/bin/env python3
"""
Render clean APS / Physical Review X (PRX) style figures:
  - Serif + Computer Modern math typography matching REVTeX 4.2
  - Full 4-sided spines with inward ticks (top & right ticks enabled)
  - No subplot titles; clean bold (a), (b), (c) panel labels
  - Minimalist canvas: zero promotional callout arrows or text boxes inside plots
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
    "legend.fontsize": 8.0,
    "axes.linewidth": 0.85,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

C_PLANAR = "#1d4ed8"   # royal blue
C_TORUS  = "#047857"   # deep emerald
C_MACRO  = "#b91c1c"   # deep crimson
C_SUB    = "#1d4ed8"   # royal blue
C_3X3    = "#6d28d9"   # deep violet
C_SLATE  = "#475569"


def style_prx_ax(ax, label_letter):
    ax.tick_params(direction="in", which="both", top=True, right=True, pad=4)
    ax.grid(True, which="major", color="#e2e8f0", linewidth=0.5, alpha=0.55)
    ax.set_axisbelow(True)
    ax.text(
        -0.16, 1.03, f"({label_letter})",
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


# ==============================================================================
# FIGURE 1: Clean PRX Geometric Schematic (No Titles, No Callout Boxes)
# ==============================================================================
def render_fig1():
    fig = plt.figure(figsize=(13.2, 3.7))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.0, 1.08], wspace=0.12)

    def proj(x, y, z):
        return 3.6 + 9.2 * x + 3.8 * y, 3.2 + 3.2 * y + 12.2 * z

    def draw_sheet(ax, z_val, edge_col="#475569", face_col="#f8fafc", alpha=0.88):
        pts = [proj(0, 0, z_val), proj(1, 0, z_val), proj(1, 1, z_val), proj(0, 1, z_val)]
        ax.add_patch(mpatches.Polygon(pts, closed=True, facecolor=face_col, edgecolor=edge_col, lw=1.15, alpha=alpha))
        p1, p2 = proj(0.5, 0, z_val), proj(0.5, 1, z_val)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#cbd5e1", lw=0.85)
        q1, q2 = proj(0, 0.5, z_val), proj(1, 0.5, z_val)
        ax.plot([q1[0], q2[0]], [q1[1], q2[1]], color="#cbd5e1", lw=0.85)

    # --- (a) Odd X-weld geometry ---
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_xlim(0, 21)
    ax_a.set_ylim(0, 21)
    ax_a.axis("off")
    ax_a.text(0.4, 19.6, "(a)", fontsize=11.0, fontweight="bold", color="#0f172a")

    draw_sheet(ax_a, 0.06, face_col="#f1f5f9")
    b1 = [proj(0, 0.5, 0.06), proj(1, 0.5, 0.06), proj(1, 0.5, 0.90), proj(0, 0.5, 0.90)]
    b2 = [proj(0.5, 0, 0.06), proj(0.5, 1, 0.06), proj(0.5, 1, 0.90), proj(0.5, 0, 0.90)]
    ax_a.add_patch(mpatches.Polygon(b1, closed=True, facecolor="#dbeafe", edgecolor="#1d4ed8", lw=1.1, alpha=0.50))
    ax_a.add_patch(mpatches.Polygon(b2, closed=True, facecolor="#bfdbfe", edgecolor="#1d4ed8", lw=1.1, alpha=0.55))
    draw_sheet(ax_a, 0.90, face_col="#f8fafc", alpha=0.68)

    for z_v in (0.06, 0.90):
        p1, p2 = proj(0, 0.5, z_v), proj(1, 0.5, z_v)
        q1, q2 = proj(0.5, 0, z_v), proj(0.5, 1, z_v)
        ax_a.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#1d4ed8", lw=2.3)
        ax_a.plot([q1[0], q2[0]], [q1[1], q2[1]], color="#1d4ed8", lw=2.3)

    rx, ry = proj(0.24, 0.5, 0.90)
    ax_a.scatter([rx], [ry], s=58, color="#1d4ed8", zorder=6)
    ax_a.annotate(
        r"$q_e$ ($3$ $Z$-faces)",
        xy=(rx, ry),
        xytext=(2.2, 18.2),
        arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=1.05),
        fontsize=8.8,
        color="#1e3a8a",
    )
    ax_a.text(16.9, 15.6, "Sheet 0", fontsize=8.6, color="#1e293b")
    ax_a.text(16.9, 5.2, "Sheet 1", fontsize=8.6, color="#1e293b")
    ax_a.text(10.8, 9.6, "Bridge", fontsize=8.4, color="#1e40af")

    # --- (b) Even Z-weld geometry ---
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_xlim(0, 21)
    ax_b.set_ylim(0, 21)
    ax_b.axis("off")
    ax_b.text(0.4, 19.6, "(b)", fontsize=11.0, fontweight="bold", color="#0f172a")

    draw_sheet(ax_b, 0.06, face_col="#fef2f2", alpha=0.82)
    b1e = [proj(0, 0.5, 0.06), proj(1, 0.5, 0.06), proj(1, 0.5, 0.90), proj(0, 0.5, 0.90)]
    b2e = [proj(0.5, 0, 0.06), proj(0.5, 1, 0.06), proj(0.5, 1, 0.90), proj(0.5, 0, 0.90)]
    ax_b.add_patch(mpatches.Polygon(b1e, closed=True, facecolor="#fee2e2", edgecolor="#b91c1c", lw=1.05, alpha=0.42))
    ax_b.add_patch(mpatches.Polygon(b2e, closed=True, facecolor="#fecaca", edgecolor="#b91c1c", lw=1.05, alpha=0.46))
    draw_sheet(ax_b, 0.90, face_col="#fef2f2", alpha=0.66)

    for z_v in (0.06, 0.90):
        p1, p2 = proj(0, 0.5, z_v), proj(1, 0.5, z_v)
        q1, q2 = proj(0.5, 0, z_v), proj(0.5, 1, z_v)
        ax_b.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#b91c1c", lw=2.1, ls="--")
        ax_b.plot([q1[0], q2[0]], [q1[1], q2[1]], color="#b91c1c", lw=2.1, ls="--")

    for xv, yv in [(0.5, 0.5), (0.0, 0.5), (1.0, 0.5)]:
        lo, hi = proj(xv, yv, 0.06), proj(xv, yv, 0.90)
        ax_b.plot([lo[0], hi[0]], [lo[1], hi[1]], color="#b91c1c", lw=1.6)

    q_a, q_b = proj(0.20, 0.5, 0.90), proj(0.31, 0.5, 0.90)
    ax_b.scatter([q_a[0], q_b[0]], [q_a[1], q_b[1]], s=48, color="#b91c1c", zorder=6)
    ax_b.annotate(
        r"$q_{e,1}, q_{e,2}$ (parity pair)",
        xy=q_a,
        xytext=(2.0, 18.2),
        arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.05),
        fontsize=8.8,
        color="#7f1d1d",
    )
    ax_b.text(16.9, 15.6, "Sheet 0", fontsize=8.6, color="#1e293b")
    ax_b.text(16.9, 5.2, "Sheet 1", fontsize=8.6, color="#1e293b")

    # --- (c) Auxiliary 5-bit outer wall diagram ---
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.set_xlim(0, 24)
    ax_c.set_ylim(0, 21)
    ax_c.axis("off")
    ax_c.text(0.4, 19.6, "(c)", fontsize=11.0, fontweight="bold", color="#0f172a")

    xs = [3.2, 6.9, 12.0, 17.1, 20.8]
    lbls = [r"$a_i$", r"$a_{i+1}$", r"$c_i$", r"$b_i$", r"$b_{i+1}$"]
    vals = ["1", "1", "0", "0", "0"]
    cols = ["#dbeafe", "#dbeafe", "#eff6ff", "#f8fafc", "#f8fafc"]
    ecols = ["#1d4ed8", "#1d4ed8", "#2563eb", "#64748b", "#64748b"]

    for x0, lb, v, fc, ec in zip(xs, lbls, vals, cols, ecols):
        ax_c.add_patch(mpatches.FancyBboxPatch((x0 - 1.35, 10.6), 2.7, 3.4, boxstyle="round,pad=0.1", facecolor=fc, edgecolor=ec, lw=1.3))
        ax_c.text(x0, 12.3, v, ha="center", va="center", fontsize=10.2, fontweight="bold", color="#0f172a")
        ax_c.text(x0, 15.1, lb, ha="center", va="center", fontsize=9.8, color="#1e293b")

    ax_c.text(5.05, 17.3, "Sheet 0", ha="center", fontsize=8.6, color="#1e3a8a")
    ax_c.text(12.0, 17.3, "Bridge", ha="center", fontsize=8.6, color="#1d4ed8")
    ax_c.text(18.95, 17.3, "Sheet 1", ha="center", fontsize=8.6, color="#334155")

    ax_c.plot([5.05, 8.6, 11.7], [10.3, 6.2, 10.3], color="#b91c1c", lw=1.6)
    ax_c.text(7.7, 4.4, r"$\chi(a_i, a_{i+1}, c_i) = 1$", ha="center", fontsize=8.8, color="#7f1d1d")

    ax_c.plot([12.3, 15.4, 18.95], [10.3, 6.2, 10.3], color="#047857", lw=1.4, ls="--")
    ax_c.text(16.5, 4.4, r"$\chi(b_i, b_{i+1}, c_i) = 0$", ha="center", fontsize=8.8, color="#065f46")

    save_fig(fig, "fig_v19_1_construction")


# ==============================================================================
# FIGURE 2: Completed-Cycle Lifetimes, Gains, & Sector Alternation
# ==============================================================================
def render_fig2(v14):
    tq = v14["tau_q"]
    gains = v14["cycle_gains"]

    fig = plt.figure(figsize=(13.4, 3.9))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.0, 1.0], wspace=0.28)

    # --- (a) Joint lifetime vs inverse temperature ---
    ax_a = fig.add_subplot(gs[0, 0])
    style_prx_ax(ax_a, "a")

    curves = [
        ("sq_iso_L2r1_s2", "--o", C_PLANAR, r"Planar $C_1$ ($323$)", True),
        ("sq_iso_L2r1_s4", "-D",  C_PLANAR, r"Planar $C_2$ ($47{,}387$)", False),
        ("rottor_iso_L2r1_s2", "--s", C_TORUS, r"Torus $C_1$ ($646$)", True),
        ("rottor_iso_L2r1_s4", "-o",  C_TORUS, r"Torus $C_2$ ($94{,}774$)", False),
    ]
    for cname, fmt, col, lbl, hollow in curves:
        bs, ys, ses = [], [], []
        for b_str in sorted([k for k in tq[cname].keys() if not k.startswith("_")], key=float):
            b = float(b_str)
            if 5.0 <= b <= 6.5:
                bs.append(b)
                ys.append(tq[cname][b_str]["tau_Q"])
                ses.append(tq[cname][b_str]["sem_Q"])
        ax_a.errorbar(
            bs, ys, yerr=ses, fmt=fmt, color=col, ms=5.4,
            mfc=("white" if hollow else col), mew=1.4,
            lw=(1.4 if hollow else 1.9), capsize=2.2, label=lbl,
        )

    ax_a.set_yscale("log")
    ax_a.set_xlim(4.92, 6.58)
    ax_a.set_ylim(2.2e2, 2.5e6)
    ax_a.set_xlabel(r"Inverse temperature $\beta$")
    ax_a.set_ylabel(r"Joint lifetime $\tau_Q$")
    ax_a.legend(loc="upper left", frameon=False, handlelength=2.1)

    # --- (b) Cycle gain G(beta) ---
    ax_b = fig.add_subplot(gs[0, 1])
    style_prx_ax(ax_b, "b")
    ax_b.axhline(1.0, color="#334155", ls=":", lw=1.1)

    for key, col, mk, ls, lbl in [
        ("planar_C1_to_C2", C_PLANAR, "D", "-", r"Planar $C_2 / C_1$"),
        ("torus_C1_to_C2", C_TORUS, "o", "-", r"Torus $C_2 / C_1$"),
        ("planar_C2_to_s5_half", C_PLANAR, "s", "--", r"Planar $s_5 / C_2$ (half)"),
    ]:
        b_strs = sorted(gains[key].keys(), key=float)
        bs = [float(b) for b in b_strs]
        gs_y = [gains[key][b]["gain"] for b in b_strs]
        ses = [gains[key][b]["se_gain"] for b in b_strs]
        ax_b.errorbar(
            bs, gs_y, yerr=ses, fmt=f"{ls}{mk}", color=col, ms=5.4,
            mfc=("white" if ls == "--" else col), mew=1.4,
            lw=(1.9 if ls == "-" else 1.4), capsize=2.2, label=lbl,
        )

    ax_b.set_yscale("log")
    ax_b.set_xlim(4.92, 6.58)
    ax_b.set_ylim(0.14, 105.0)
    ax_b.set_xlabel(r"Inverse temperature $\beta$")
    ax_b.set_ylabel(r"Lifetime ratio $G(\beta)$")
    ax_b.legend(loc="upper left", frameon=False, handlelength=2.1)

    # --- (c) Sector alternation across planar steps ---
    ax_c = fig.add_subplot(gs[0, 2])
    style_prx_ax(ax_c, "c")
    steps = [1, 2, 3, 4, 5]
    x_pos = np.arange(len(steps))
    txs, tzs, tqs, q_ses = [], [], [], []
    for s in steps:
        r = tq[f"sq_iso_L2r1_s{s}"]["6.50"]
        txs.append(r["tau_X"])
        tzs.append(r["tau_Z"])
        tqs.append(r["tau_Q"])
        q_ses.append(r["sem_Q"])

    ax_c.plot(x_pos, txs, "--^", color="#0284c7", lw=1.35, ms=5.6, label=r"$\tau_X$")
    ax_c.plot(x_pos, tzs, "--v", color=C_MACRO, lw=1.35, ms=5.6, label=r"$\tau_Z$")
    ax_c.errorbar(x_pos, tqs, yerr=q_ses, fmt="-o", color=C_PLANAR, lw=2.0, ms=5.8, capsize=2.2, label=r"$\tau_Q$")
    for idx, s in enumerate(steps):
        if s % 2 == 1:
            ax_c.plot([idx], [tqs[idx]], "o", color=C_PLANAR, mfc="white", mew=1.5, ms=6.0, zorder=5)

    ax_c.set_yscale("log")
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels([r"$s{=}1$", r"$C_1$", r"$s{=}3$", r"$C_2$", r"$s{=}5$"])
    ax_c.set_ylim(40, 4.5e8)
    ax_c.set_xlabel(r"Planar recursion step ($\beta = 6.5$)")
    ax_c.set_ylabel(r"Mean lifetime")
    ax_c.legend(loc="lower right", frameon=False, handlelength=2.1)

    save_fig(fig, "fig_v19_2_lifetimes")


# ==============================================================================
# FIGURE 3: Multi-Scale Check Coupling Test
# ==============================================================================
def render_fig3(v14):
    inter = v14["interventions"]
    tq = v14["tau_q"]
    exc = v14["excursions"]

    fig = plt.figure(figsize=(13.4, 3.9))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.0, 1.0], wspace=0.28)

    # --- (a) Coupling sweep tau_Q(lambda) at beta=6.5 ---
    ax_a = fig.add_subplot(gs[0, 0])
    style_prx_ax(ax_a, "a")
    lams = [0.2, 0.4, 0.6, 0.8, 1.0]

    for cname, fam_lbl, ls_fam, mk_m, mk_s in [
        ("sq_iso_L2r1_s4", "Planar", "-", "o", "s"),
        ("rottor_iso_L2r1_s4", "Torus", "--", "^", "D"),
    ]:
        u_q = tq[cname]["6.50"]["tau_Q"]
        u_se = tq[cname]["6.50"]["sem_Q"]
        for tag_u, col, mk, u_lbl in [
            ("macro_weld", C_MACRO, mk_m, r"$U_{\mathrm{macro}}$"),
            ("sub_weld", C_SUB, mk_s, r"$U_{\mathrm{sub}}$"),
        ]:
            ys = [inter[cname][tag_u]["6.50"][f"{l:.1f}"]["tau_Q"] for l in lams[:-1]] + [u_q]
            ses = [inter[cname][tag_u]["6.50"][f"{l:.1f}"]["sem_Q"] for l in lams[:-1]] + [u_se]
            ax_a.errorbar(
                lams, ys, yerr=ses, fmt=f"{ls_fam}{mk}", color=col,
                ms=5.2, lw=1.65, capsize=2.2, label=f"{fam_lbl} {u_lbl}",
            )

    ax_a.axhline(tq["rottor_iso_L2r1_s2"]["6.50"]["tau_Q"], color=C_TORUS, ls=":", lw=1.15)
    ax_a.axhline(tq["sq_iso_L2r1_s2"]["6.50"]["tau_Q"], color=C_SLATE, ls=":", lw=1.15)

    ax_a.set_yscale("log")
    ax_a.set_xlim(0.15, 1.05)
    ax_a.set_ylim(1.2, 2.8e6)
    ax_a.set_xlabel(r"Coupling factor $\lambda$")
    ax_a.set_ylabel(r"Joint lifetime $\tau_Q(\lambda)$")
    ax_a.legend(loc="lower right", frameon=False, handlelength=2.1)

    # --- (b) Linear ratio tau_Q(U_sub) / tau_Q(U_macro) ---
    ax_b = fig.add_subplot(gs[0, 1])
    style_prx_ax(ax_b, "b")
    betas = [5.50, 6.00, 6.25, 6.50]
    ax_b.axhline(1.0, color="#334155", ls=":", lw=1.1)

    for cname, col_q, mk, lbl in [
        ("sq_iso_L2r1_s4", C_PLANAR, "D", r"Planar $C_2$"),
        ("rottor_iso_L2r1_s4", C_TORUS, "o", r"Torus $C_2$"),
    ]:
        rq_06 = [
            inter[cname]["sub_weld"][f"{b:.2f}"]["0.6"]["tau_Q"] / inter[cname]["macro_weld"][f"{b:.2f}"]["0.6"]["tau_Q"]
            for b in betas
        ]
        ax_b.plot(betas, rq_06, f"-{mk}", color=col_q, ms=5.6, lw=1.9, label=lbl)

    ax_b.set_xlim(5.42, 6.58)
    ax_b.set_ylim(0.8, 4.0)
    ax_b.set_xlabel(r"Inverse temperature $\beta$")
    ax_b.set_ylabel(r"$\tau_Q(U_{\mathrm{sub}}) \,/\, \tau_Q(U_{\mathrm{macro}})$")
    ax_b.legend(loc="upper left", frameon=False, handlelength=2.1)

    # --- (c) Coarse Cycle-1 excursions before failure ---
    ax_c = fig.add_subplot(gs[0, 2])
    style_prx_ax(ax_c, "c")
    for cname, jtag, col, fmt, lbl in [
        ("rottor_iso_L2r1_s4", "uniform", C_TORUS, "-o", r"Torus ($\lambda=1$)"),
        ("sq_iso_L2r1_s4", "uniform", C_PLANAR, "-D", r"Planar ($\lambda=1$)"),
        ("rottor_iso_L2r1_s4", "sub_weld_l06", C_SUB, "--s", r"Torus ($U_{\mathrm{sub}}, \lambda=0.6$)"),
        ("rottor_iso_L2r1_s4", "macro_weld_l06", C_MACRO, ":^", r"Torus ($U_{\mathrm{macro}}, \lambda=0.6$)"),
    ]:
        prefix = f"excur__{cname}_X_err__{jtag}__b"
        pts = sorted([v for k, v in exc.items() if k.startswith(prefix)], key=lambda d: d["beta"])
        bs = [d["beta"] for d in pts]
        ms = [d["mean_n_exc"] for d in pts]
        ses = [d["sem_n_exc"] for d in pts]
        ax_c.errorbar(bs, ms, yerr=ses, fmt=fmt, color=col, ms=5.2, lw=1.7, capsize=2.2, label=lbl)

    ax_c.set_xlim(5.42, 6.58)
    ax_c.set_ylim(0.8, 11.0)
    ax_c.set_xlabel(r"Inverse temperature $\beta$")
    ax_c.set_ylabel(r"Healed coarse excursions")
    ax_c.legend(loc="upper left", frameon=False, handlelength=2.1)

    save_fig(fig, "fig_v19_3_intervention")


# ==============================================================================
# FIGURE 4: Spacious 2-Panel Readout & Coarse Confinement Figure
# ==============================================================================
def render_fig4(v14):
    flow = v14["rgflow14"]
    pdec = v14["paired_kmc"]

    fig = plt.figure(figsize=(10.2, 3.85))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.02, 1.0], wspace=0.26)

    # --- (a) Metastable thermal vs matched-weight random coarse flow ---
    ax_a = fig.add_subplot(gs[0, 0])
    style_prx_ax(ax_a, "a")
    scales_x = np.arange(5)

    for b_str, col, mk in [
        ("5.50", "#b45309", "s"),
        ("6.50", C_TORUS, "o"),
    ]:
        rec = flow[f"rgflow14__rottor_s4__t80.0__b{b_str}"]
        sc_keys = ["4", "3", "2", "1", "0"]
        th_y = [max(rec["scales"][sk]["cnt"], 4e-3) for sk in sc_keys]
        th_se = [rec["scales"][sk]["semc"] for sk in sc_keys]
        rn_y = [max(rec["scales"][sk]["icnt"], 4e-3) for sk in sc_keys]
        ax_a.errorbar(
            scales_x, th_y, yerr=th_se, fmt=f"-{mk}", color=col, lw=1.9, ms=5.6, capsize=2.2,
            label=rf"Thermal ($\beta={float(b_str):.1f}$)",
        )
        ax_a.plot(
            scales_x, rn_y, f"--{mk}", color=col, mfc="white", mew=1.4, lw=1.4, ms=5.4,
            label=rf"Random ($\beta={float(b_str):.1f}$ wt.)",
        )

    ax_a.axhline(4e-3, color="#64748b", ls=":", lw=0.95)
    ax_a.set_yscale("log")
    ax_a.set_xticks(scales_x)
    ax_a.set_xticklabels([r"$s{=}4$", r"$s{=}3$", r"$s{=}2$", r"$s{=}1$", "Seed"])
    ax_a.set_ylim(2.5e-3, 6e3)
    ax_a.set_xlabel(r"Coarse-graining scale")
    ax_a.set_ylabel(r"Violated checks")
    ax_a.legend(loc="upper right", frameon=False, handlelength=2.1)

    # --- (b) Paired hierarchical vs global readout ---
    ax_b = fig.add_subplot(gs[0, 1])
    style_prx_ax(ax_b, "b")
    ax_b.axhline(1.0, color="#334155", ls=":", lw=1.1)

    for short_c, col, mk, lbl in [
        ("rottor_s2", C_SUB, "s", r"Torus $C_1$ ($646$)"),
        ("rottor_s4", C_TORUS, "o", r"Torus $C_2$ ($94{,}774$)"),
    ]:
        prefix = f"paired_kmc__{short_c}__b"
        pts = sorted([v for k, v in pdec.items() if k.startswith(prefix)], key=lambda d: d["beta"])
        bs, rs, lo, hi = [], [], [], []
        for d in pts:
            bs.append(d["beta"])
            pr = d.get("paired_mean_ratio_rg_over_uf", {})
            if (not pr or "mean_ratio" not in pr) and d["mean_rg"] == d["mean_strict"]:
                pr = d.get("paired_mean_ratio_strict_over_uf", {})
            if pr and "mean_ratio" in pr:
                r = pr["mean_ratio"]
                rs.append(r)
                lo.append(r - pr["ci95_ratio"][0])
                hi.append(pr["ci95_ratio"][1] - r)
            else:
                r = d["mean_rg"] / d["mean_uf"]
                se = r * math.hypot(d["sem_rg"] / d["mean_rg"], d["sem_uf"] / d["mean_uf"])
                rs.append(r)
                lo.append(1.96 * se)
                hi.append(1.96 * se)
        ax_b.errorbar(bs, rs, yerr=[lo, hi], fmt=f"-{mk}", color=col, ms=5.6, lw=1.9, capsize=2.5, label=lbl)

    ax_b.set_xlim(4.88, 6.75)
    ax_b.set_ylim(0.30, 1.34)
    ax_b.set_xlabel(r"Inverse temperature $\beta$")
    ax_b.set_ylabel(r"$\bar{\tau}_{\mathrm{RG}} \,/\, \bar{\tau}_{\mathrm{UF}}$")
    ax_b.legend(loc="lower right", frameon=False, handlelength=2.1)

    save_fig(fig, "fig_v19_4_readout")


# ==============================================================================
# FIGURE 5: 3D Packing, Planar 3x3 Lifetimes, & Check-Wire Collapse
# ==============================================================================
def render_fig5(v14, v13):
    tq = v14["tau_q"]

    fig = plt.figure(figsize=(13.4, 3.9))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.0, 1.0], wspace=0.28)

    # --- (a) 3D ball-packing exponent ---
    ax_a = fig.add_subplot(gs[0, 0])
    style_prx_ax(ax_a, "a")
    Ls = np.arange(2, 9)
    ls_smooth = np.linspace(1.85, 8.2, 200)
    exc_1rib = [math.log(2 * L * L + 2 * L) / (3 * math.log(L)) - 1.0 for L in ls_smooth]
    exc_2rib = [math.log(2 * L * L + 4 * L) / (3 * math.log(L)) - 1.0 for L in ls_smooth]

    ax_a.axhline(0.0, color="#334155", ls=":", lw=1.1)
    ax_a.plot(ls_smooth, exc_1rib, "-", color=C_PLANAR, lw=1.8, label=r"One-rib ($2L^2+2L$)")
    ax_a.plot(ls_smooth, exc_2rib, "--", color=C_MACRO, lw=1.7, label=r"Two-rib ($2L^2+4L$)")
    ax_a.scatter(Ls, [math.log(2 * L * L + 2 * L) / (3 * math.log(L)) - 1.0 for L in Ls], color=C_PLANAR, s=34, zorder=5)
    ax_a.scatter(Ls, [math.log(2 * L * L + 4 * L) / (3 * math.log(L)) - 1.0 for L in Ls], color=C_MACRO, marker="s", s=32, zorder=5)

    ax_a.set_xlim(1.75, 8.3)
    ax_a.set_ylim(-0.22, 0.23)
    ax_a.set_xticks(Ls)
    ax_a.set_xlabel(r"Subdivision factor $L$")
    ax_a.set_ylabel(r"$\frac{1}{3}\log_L m(L) - 1$")
    ax_a.legend(loc="upper right", frameon=False, handlelength=2.1)

    # --- (b) Planar 3x3 lifetimes ---
    ax_b = fig.add_subplot(gs[0, 1])
    style_prx_ax(ax_b, "b")
    for cname, fmt, col, hollow, lbl in [
        ("sq_L3r1_s1", "--o", C_3X3, True, r"$s = 1$"),
        ("sq_L3r1_s2", "-s",  C_3X3, False, r"$C_1$ ($s = 2$)"),
        ("sq_L3r1_s3", "--^", C_3X3, True, r"$s = 3$"),
    ]:
        bs = sorted([float(b) for b in tq[cname].keys() if not b.startswith("_")])
        ys = [tq[cname][f"{b:.2f}"]["tau_Q"] for b in bs]
        ses = [tq[cname][f"{b:.2f}"]["sem_Q"] for b in bs]
        ax_b.errorbar(
            bs, ys, yerr=ses, fmt=fmt, color=col, ms=5.4,
            mfc=("white" if hollow else col), mew=1.4,
            lw=(1.9 if not hollow else 1.4), capsize=2.2, label=lbl,
        )

    bs3 = sorted([float(b) for b in tq["sq_L3r1_s3"].keys() if not b.startswith("_")])
    tx3 = [tq["sq_L3r1_s3"][f"{b:.2f}"]["tau_X"] for b in bs3]
    ax_b.plot(bs3, tx3, ":d", color="#0284c7", lw=1.45, ms=5.2, label=r"$s{=}3$ $\tau_X$")

    ax_b.set_yscale("log")
    ax_b.set_xlim(4.35, 7.68)
    ax_b.set_ylim(12, 2.5e8)
    ax_b.set_xlabel(r"Inverse temperature $\beta$")
    ax_b.set_ylabel(r"Mean lifetime")
    ax_b.legend(loc="upper left", frameon=False, handlelength=2.1)

    # --- (c) Torus check-wire collapse ---
    ax_c = fig.add_subplot(gs[0, 2])
    style_prx_ax(ax_c, "c")

    w_styles = [
        (1,  "o", C_TORUS, r"$w = 1$"),
        (2,  "s", "#0284c7", r"$w = 2$"),
        (4,  "^", "#2563eb", r"$w = 4$"),
        (8,  "D", "#b45309", r"$w = 8$"),
        (16, "v", C_MACRO, r"$w = 16$"),
    ]
    wire_src = v13["wire_sweep"]["tor"]
    for w, mk, col, lbl in w_styles:
        if str(w) not in wire_src:
            continue
        pts = sorted(wire_src[str(w)]["points"], key=lambda d: d["beta"])
        beffs = [d["beta_eff"] for d in pts]
        ys = [d["mean_t"] / (w * w) for d in pts]
        ses = [d["sem_t"] / (w * w) for d in pts]
        ax_c.errorbar(beffs, ys, yerr=ses, fmt=f"-{mk}", color=col, ms=5.0, lw=1.4, capsize=2.0, label=lbl)

    ax_c.set_yscale("log")
    ax_c.set_xlim(2.05, 6.7)
    ax_c.set_ylim(0.7, 4.5e5)
    ax_c.set_xlabel(r"Effective coupling $\beta_{\mathrm{eff}}(w)$")
    ax_c.set_ylabel(r"$\tau_X(\beta, w) \,/\, w^2$")
    ax_c.legend(loc="upper left", frameon=False, handlelength=2.1)

    save_fig(fig, "fig_v19_5_implementation")


def main():
    v14 = json.loads((ROOT / "parsed_v14_data.json").read_text())
    v13 = json.loads((ROOT / "parsed_v13_data.json").read_text())
    render_fig1()
    render_fig2(v14)
    render_fig3(v14)
    render_fig4(v14)
    render_fig5(v14, v13)
    print("All 5 PRX-style figures rendered successfully.")


if __name__ == "__main__":
    main()
