#!/usr/bin/env python3
"""
Render clean, spacious, low-clutter figures for V19:
  Fig 1: Compact Welded Cell (3 clean geometric schematics, zero text boxes)
  Fig 2: Cycle Lifetimes, Gains, & Sector Alternation (Arrhenius crossing, G(beta), sawtooth)
  Fig 3: Multi-Scale Check Coupling Test (lambda sweep, linear sensitivity ratio, healed excursions)
  Fig 4: Coarse Confinement & Hierarchical Readout (spacious 2-panel figure)
  Fig 5: Spatial Packing, 3x3 Lifetimes, & Check-Wire Collapse (full-height wire collapse)
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
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "mathtext.fontset": "dejavusans",
    "font.size": 9.5,
    "axes.titlesize": 10.6,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.6,
    "xtick.labelsize": 8.6,
    "ytick.labelsize": 8.6,
    "legend.fontsize": 8.1,
    "axes.linewidth": 0.95,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

C_PLANAR = "#1d4ed8"   # crisp royal blue
C_TORUS  = "#059669"   # emerald green
C_MACRO  = "#dc2626"   # crimson red
C_SUB    = "#2563eb"   # cobalt blue
C_3X3    = "#7c3aed"   # violet
C_SLATE  = "#475569"
C_GRID   = "#e2e8f0"


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color=C_GRID, linewidth=0.65, alpha=0.8)
    ax.set_axisbelow(True)


def save_fig(fig, stem):
    pdf_path = FIG_DIR / f"{stem}.pdf"
    png_path = FIG_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved {png_path}")


# ==============================================================================
# FIGURE 1: Clean 3-Panel Schematic (Zero Prose Boxes!)
# ==============================================================================
def render_fig1():
    fig = plt.figure(figsize=(13.4, 4.1))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.0, 1.05], wspace=0.14)

    def proj(x, y, z):
        return 4.5 + 8.6 * x + 3.6 * y, 5.2 + 3.0 * y + 11.5 * z

    def draw_sheet(ax, z_val, edge_col="#64748b", face_col="#f8fafc", alpha=0.88):
        pts = [proj(0, 0, z_val), proj(1, 0, z_val), proj(1, 1, z_val), proj(0, 1, z_val)]
        ax.add_patch(mpatches.Polygon(pts, closed=True, facecolor=face_col, edgecolor=edge_col, lw=1.25, alpha=alpha))
        for t in [0.5]:
            p1, p2 = proj(t, 0, z_val), proj(t, 1, z_val)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#cbd5e1", lw=0.9)
            q1, q2 = proj(0, t, z_val), proj(1, t, z_val)
            ax.plot([q1[0], q2[0]], [q1[1], q2[1]], color="#cbd5e1", lw=0.9)

    # --- Panel (a): Odd X-Weld ---
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_xlim(0, 22)
    ax_a.set_ylim(0, 24)
    ax_a.axis("off")

    ax_a.text(11.0, 22.6, "(a) Odd X-Weld", ha="center", va="center", fontsize=11.0, fontweight="bold", color="#0f172a")

    draw_sheet(ax_a, 0.08, face_col="#f1f5f9")
    # 4-arm vertical blue bridge cross
    b1 = [proj(0, 0.5, 0.08), proj(1, 0.5, 0.08), proj(1, 0.5, 0.92), proj(0, 0.5, 0.92)]
    b2 = [proj(0.5, 0, 0.08), proj(0.5, 1, 0.08), proj(0.5, 1, 0.92), proj(0.5, 0, 0.92)]
    ax_a.add_patch(mpatches.Polygon(b1, closed=True, facecolor="#dbeafe", edgecolor="#2563eb", lw=1.2, alpha=0.52))
    ax_a.add_patch(mpatches.Polygon(b2, closed=True, facecolor="#bfdbfe", edgecolor="#1d4ed8", lw=1.2, alpha=0.58))
    draw_sheet(ax_a, 0.92, face_col="#f8fafc", alpha=0.68)

    for z_v in (0.08, 0.92):
        p1, p2 = proj(0, 0.5, z_v), proj(1, 0.5, z_v)
        q1, q2 = proj(0.5, 0, z_v), proj(0.5, 1, z_v)
        ax_a.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#1d4ed8", lw=2.5)
        ax_a.plot([q1[0], q2[0]], [q1[1], q2[1]], color="#1d4ed8", lw=2.5)

    rx, ry = proj(0.25, 0.5, 0.92)
    ax_a.scatter([rx], [ry], s=68, color="#1d4ed8", zorder=6)
    ax_a.annotate(
        "1 rib qubit in 3 Z-faces\n(X defect splits 1 → 2)",
        xy=(rx, ry),
        xytext=(1.0, 19.2),
        arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=1.25),
        fontsize=8.3,
        color="#1e3a8a",
        fontweight="bold",
    )
    ax_a.text(17.5, 16.5, "Sheet 0", fontsize=8.5, fontweight="bold", color="#334155")
    ax_a.text(17.5, 6.8, "Sheet 1", fontsize=8.5, fontweight="bold", color="#334155")
    ax_a.text(11.0, 1.8, r"$2 \times 2$ cell + 4-arm bridge cross", ha="center", fontsize=8.4, color="#475569")

    # --- Panel (b): Even Z-Weld ---
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_xlim(0, 22)
    ax_b.set_ylim(0, 24)
    ax_b.axis("off")

    ax_b.text(11.0, 22.6, "(b) Even Z-Weld", ha="center", va="center", fontsize=11.0, fontweight="bold", color="#0f172a")

    draw_sheet(ax_b, 0.08, face_col="#fff1f2", alpha=0.82)
    b1e = [proj(0, 0.5, 0.08), proj(1, 0.5, 0.08), proj(1, 0.5, 0.92), proj(0, 0.5, 0.92)]
    b2e = [proj(0.5, 0, 0.08), proj(0.5, 1, 0.08), proj(0.5, 1, 0.92), proj(0.5, 0, 0.92)]
    ax_b.add_patch(mpatches.Polygon(b1e, closed=True, facecolor="#fee2e2", edgecolor="#dc2626", lw=1.1, alpha=0.42))
    ax_b.add_patch(mpatches.Polygon(b2e, closed=True, facecolor="#fecaca", edgecolor="#b91c1c", lw=1.1, alpha=0.46))
    draw_sheet(ax_b, 0.92, face_col="#fff1f2", alpha=0.65)

    for z_v in (0.08, 0.92):
        p1, p2 = proj(0, 0.5, z_v), proj(1, 0.5, z_v)
        q1, q2 = proj(0.5, 0, z_v), proj(0.5, 1, z_v)
        ax_b.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#dc2626", lw=2.3, ls="--")
        ax_b.plot([q1[0], q2[0]], [q1[1], q2[1]], color="#dc2626", lw=2.3, ls="--")

    # Two red parity qubits highlighted along a rib
    q_a, q_b = proj(0.22, 0.5, 0.92), proj(0.32, 0.5, 0.92)
    ax_b.scatter([q_a[0], q_b[0]], [q_a[1], q_b[1]], s=54, color="#dc2626", zorder=6)
    ax_b.annotate(
        "2 parity qubits per rib\n(X-checks branch 3-way)",
        xy=q_a,
        xytext=(1.0, 19.2),
        arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.25),
        fontsize=8.3,
        color="#991b1b",
        fontweight="bold",
    )
    for xv, yv in [(0.5, 0.5), (0.0, 0.5), (1.0, 0.5)]:
        lo, hi = proj(xv, yv, 0.08), proj(xv, yv, 0.92)
        ax_b.plot([lo[0], hi[0]], [lo[1], hi[1]], color="#dc2626", lw=1.8)

    ax_b.text(11.0, 1.8, "Full 2-step cycle welds both sectors", ha="center", fontsize=8.4, color="#475569")

    # --- Panel (c): Single-Cell Wall Contraction ---
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.set_xlim(0, 24)
    ax_c.set_ylim(0, 24)
    ax_c.axis("off")

    ax_c.text(12.0, 22.6, "(c) Wall Contraction", ha="center", va="center", fontsize=11.0, fontweight="bold", color="#0f172a")
    ax_c.text(12.0, 19.3, "5-bit outer wall coupled to 2 interior ribs:", ha="center", fontsize=8.5, color="#334155")

    xs = [3.0, 6.6, 12.0, 17.4, 21.0]
    vals = ["1", "1", "0", "0", "0"]
    cols = ["#dbeafe", "#dbeafe", "#eff6ff", "#f8fafc", "#f8fafc"]
    ecols = ["#1d4ed8", "#1d4ed8", "#2563eb", "#94a3b8", "#94a3b8"]
    for x0, v, fc, ec in zip(xs, vals, cols, ecols):
        ax_c.add_patch(mpatches.FancyBboxPatch((x0 - 1.35, 13.0), 2.7, 3.2, boxstyle="round,pad=0.12", facecolor=fc, edgecolor=ec, lw=1.5))
        ax_c.text(x0, 14.6, v, ha="center", va="center", fontsize=10.5, fontweight="bold", color="#0f172a")

    ax_c.text(4.8, 17.1, "Sheet 0", ha="center", fontsize=8.2, fontweight="bold", color="#1e40af")
    ax_c.text(12.0, 17.1, "Bridge", ha="center", fontsize=8.2, fontweight="bold", color="#1d4ed8")
    ax_c.text(19.2, 17.1, "Sheet 1", ha="center", fontsize=8.2, fontweight="bold", color="#475569")

    # Braces / lines to the two ribs
    ax_c.plot([4.8, 8.8, 11.7], [12.7, 9.6, 12.7], color="#dc2626", lw=1.8)
    ax_c.text(7.6, 8.2, "Rib 1 violated", ha="center", fontsize=8.2, fontweight="bold", color="#b91c1c")

    ax_c.plot([12.3, 15.2, 19.2], [12.7, 9.6, 12.7], color="#059669", lw=1.6, ls="--")
    ax_c.text(16.6, 8.2, "Rib 2 clean", ha="center", fontsize=8.2, fontweight="bold", color="#047857")

    ax_c.text(12.0, 4.7, r"Wall minority:  $W_{\mathrm{wall}} \leq 2\,V$", ha="center", fontsize=9.2, fontweight="bold", color="#1e3a8a")
    ax_c.text(12.0, 2.0, r"Rounded majority:  $C \leq \frac{2}{5}\,V$  (mod 4-wall flip)", ha="center", fontsize=8.7, fontweight="bold", color="#065f46")

    save_fig(fig, "fig_v19_1_construction")


# ==============================================================================
# FIGURE 2: Completed-Cycle Lifetimes (Arrhenius), Gains, & Sector Alternation
# ==============================================================================
def render_fig2(v14):
    tq = v14["tau_q"]
    gains = v14["cycle_gains"]

    fig = plt.figure(figsize=(13.6, 4.25))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.02, 1.0, 1.02], wspace=0.27)

    # --- Panel (a): Completed-Cycle Arrhenius Crossing (tau_Q vs beta) ---
    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a)

    curves = [
        ("sq_iso_L2r1_s2", "--o", C_PLANAR, "Planar C1 (323q)", True),
        ("sq_iso_L2r1_s4", "-D",  C_PLANAR, "Planar C2 (47.4kq)", False),
        ("rottor_iso_L2r1_s2", "--s", C_TORUS, "Torus C1 (646q)", True),
        ("rottor_iso_L2r1_s4", "-o",  C_TORUS, "Torus C2 (94.8kq)", False),
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
            bs, ys, yerr=ses, fmt=fmt, color=col, ms=6.0,
            mfc=("white" if hollow else col), mew=1.6,
            lw=(1.6 if hollow else 2.2), capsize=2.4, label=lbl,
        )

    ax_a.set_yscale("log")
    ax_a.set_xlim(4.92, 6.58)
    ax_a.set_ylim(2.5e2, 2.5e6)
    ax_a.set_xlabel(r"Inverse Temperature $\beta = 1/T$")
    ax_a.set_ylabel(r"Joint Lifetime $\tau_Q$")
    ax_a.set_title("(a) Completed-Cycle Lifetimes")
    ax_a.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.94, fontsize=7.8)

    # --- Panel (b): Complete-Cycle Gain Ratio G(beta) ---
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b)
    ax_b.axhline(1.0, color="#0f172a", ls="--", lw=1.2, zorder=2)
    ax_b.axhspan(0.12, 1.0, color="#fee2e2", alpha=0.22, zorder=1)

    for key, col, mk, ls, lbl in [
        ("planar_C1_to_C2", C_PLANAR, "D", "-", "Planar C1 → C2"),
        ("torus_C1_to_C2", C_TORUS, "o", "-", "Torus C1 → C2"),
        ("planar_C2_to_s5_half", C_PLANAR, "s", "--", "Planar half-step"),
    ]:
        b_strs = sorted(gains[key].keys(), key=float)
        bs = [float(b) for b in b_strs]
        gs_y = [gains[key][b]["gain"] for b in b_strs]
        ses = [gains[key][b]["se_gain"] for b in b_strs]
        ax_b.errorbar(
            bs, gs_y, yerr=ses, fmt=f"{ls}{mk}", color=col, ms=6.0,
            mfc=("white" if ls == "--" else col), mew=1.6,
            lw=(2.1 if ls == "-" else 1.5), capsize=2.5, label=lbl,
        )

    ax_b.annotate(
        r"$55.9\times$",
        xy=(6.50, 55.9),
        xytext=(5.88, 26.0),
        arrowprops=dict(arrowstyle="->", color=C_PLANAR, lw=1.2),
        fontsize=8.6,
        fontweight="bold",
        color=C_PLANAR,
    )
    ax_b.annotate(
        r"Crossover $\beta \approx 6.1$",
        xy=(6.12, 1.02),
        xytext=(5.12, 3.3),
        arrowprops=dict(arrowstyle="->", color=C_TORUS, lw=1.2),
        fontsize=8.1,
        fontweight="bold",
        color="#065f46",
    )

    ax_b.set_yscale("log")
    ax_b.set_xlim(4.92, 6.58)
    ax_b.set_ylim(0.14, 105.0)
    ax_b.set_xlabel(r"Inverse Temperature $\beta = 1/T$")
    ax_b.set_ylabel(r"Lifetime Gain $G(\beta)$")
    ax_b.set_title("(b) Cycle Gain & Crossover")
    ax_b.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.94, fontsize=7.8)

    # --- Panel (c): Alternating Sector Bottleneck (Planar beta=6.5) ---
    ax_c = fig.add_subplot(gs[0, 2])
    style_ax(ax_c)
    steps = [1, 2, 3, 4, 5]
    x_pos = np.arange(len(steps))
    txs, tzs, tqs, q_ses = [], [], [], []
    for s in steps:
        r = tq[f"sq_iso_L2r1_s{s}"]["6.50"]
        txs.append(r["tau_X"])
        tzs.append(r["tau_Z"])
        tqs.append(r["tau_Q"])
        q_ses.append(r["sem_Q"])

    ax_c.plot(x_pos, txs, "--^", color="#0284c7", lw=1.5, ms=6.2, label=r"$\tau_X$ (lower bnd if capped)")
    ax_c.plot(x_pos, tzs, "--v", color="#dc2626", lw=1.5, ms=6.2, label=r"$\tau_Z$ (lower bnd if capped)")
    ax_c.errorbar(x_pos, tqs, yerr=q_ses, fmt="-o", color=C_PLANAR, lw=2.3, ms=6.6, capsize=2.5, label=r"Joint $\tau_Q$")
    for idx, s in enumerate(steps):
        if s % 2 == 1:
            ax_c.plot([idx], [tqs[idx]], "o", color=C_PLANAR, mfc="white", mew=1.8, ms=6.8, zorder=5)

    ax_c.annotate(
        r"Z-limited ($1.18\times$)",
        xy=(4.0, tqs[4]),
        xytext=(2.35, 7.5e4),
        arrowprops=dict(arrowstyle="->", color="#991b1b", lw=1.2),
        fontsize=8.1,
        fontweight="bold",
        color="#991b1b",
    )

    ax_c.set_yscale("log")
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(["s=1\n(19q)", "C1\n(323q)", "s=3\n(2.9k)", "C2\n(47.4k)", "s=5\n(424k)"])
    ax_c.set_ylim(40, 4.5e8)
    ax_c.set_xlabel(r"Planar Step ($\beta = 6.5$)")
    ax_c.set_ylabel("Mean Lifetime")
    ax_c.set_title("(c) Sector Alternation")
    ax_c.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.94, fontsize=7.6)

    save_fig(fig, "fig_v19_2_lifetimes")


# ==============================================================================
# FIGURE 3: Multi-Scale Check Coupling Test
# ==============================================================================
def render_fig3(v14):
    inter = v14["interventions"]
    tq = v14["tau_q"]
    exc = v14["excursions"]

    fig = plt.figure(figsize=(13.6, 4.25))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.04, 0.98, 1.02], wspace=0.27)

    # --- Panel (a): Coupling Sweep tau_Q(lambda) at beta=6.5 ---
    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a)
    lams = [0.2, 0.4, 0.6, 0.8, 1.0]

    for cname, fam_lbl, ls_fam, mk_m, mk_s in [
        ("sq_iso_L2r1_s4", "Planar", "-", "o", "s"),
        ("rottor_iso_L2r1_s4", "Torus", "--", "^", "D"),
    ]:
        u_q = tq[cname]["6.50"]["tau_Q"]
        u_se = tq[cname]["6.50"]["sem_Q"]
        for tag_u, col, mk, u_lbl in [
            ("macro_weld", C_MACRO, mk_m, "coarse C1"),
            ("sub_weld", C_SUB, mk_s, "fine C2"),
        ]:
            ys = [inter[cname][tag_u]["6.50"][f"{l:.1f}"]["tau_Q"] for l in lams[:-1]] + [u_q]
            ses = [inter[cname][tag_u]["6.50"][f"{l:.1f}"]["sem_Q"] for l in lams[:-1]] + [u_se]
            ax_a.errorbar(
                lams, ys, yerr=ses, fmt=f"{ls_fam}{mk}", color=col,
                ms=5.8, lw=1.9, capsize=2.4, label=f"{fam_lbl}: {u_lbl}",
            )

    ax_a.axhline(tq["rottor_iso_L2r1_s2"]["6.50"]["tau_Q"], color=C_TORUS, ls=":", lw=1.3)
    ax_a.text(0.18, tq["rottor_iso_L2r1_s2"]["6.50"]["tau_Q"] * 1.35, "Torus C1", fontsize=7.9, color=C_TORUS, fontweight="bold")
    ax_a.axhline(tq["sq_iso_L2r1_s2"]["6.50"]["tau_Q"], color=C_SLATE, ls=":", lw=1.3)
    ax_a.text(0.18, tq["sq_iso_L2r1_s2"]["6.50"]["tau_Q"] * 1.35, "Planar C1", fontsize=7.9, color=C_SLATE, fontweight="bold")

    ax_a.set_yscale("log")
    ax_a.set_xlim(0.15, 1.05)
    ax_a.set_ylim(1.2, 2.8e6)
    ax_a.set_xlabel(r"Selected-Check Coupling $\lambda$")
    ax_a.set_ylabel(r"Joint Lifetime $\tau_Q(\lambda)$ ($\beta = 6.5$)")
    ax_a.set_title("(a) Equal-Budget Check Weakening")
    ax_a.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.94, fontsize=7.7)

    # --- Panel (b): Clean Linear Ratio tau_Q(U_sub) / tau_Q(U_macro) ---
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b)
    betas = [5.50, 6.00, 6.25, 6.50]
    ax_b.axhline(1.0, color="#64748b", ls="--", lw=1.2, label="Equal effect (1.0)")

    for cname, col_q, mk, lbl in [
        ("sq_iso_L2r1_s4", C_PLANAR, "D", "Planar C2"),
        ("rottor_iso_L2r1_s4", C_TORUS, "o", "Torus C2"),
    ]:
        rq_06 = [
            inter[cname]["sub_weld"][f"{b:.2f}"]["0.6"]["tau_Q"] / inter[cname]["macro_weld"][f"{b:.2f}"]["0.6"]["tau_Q"]
            for b in betas
        ]
        ax_b.plot(betas, rq_06, f"-{mk}", color=col_q, ms=6.4, lw=2.2, label=lbl)

    ax_b.annotate(
        r"$3.6\times$ stronger" + "\ncoarse-weld effect",
        xy=(6.50, 3.58),
        xytext=(5.62, 3.15),
        arrowprops=dict(arrowstyle="->", color=C_PLANAR, lw=1.2),
        fontsize=8.1,
        fontweight="bold",
        color=C_PLANAR,
    )

    ax_b.set_xlim(5.42, 6.58)
    ax_b.set_ylim(0.75, 4.15)
    ax_b.set_xlabel(r"Inverse Temperature $\beta = 1/T$")
    ax_b.set_ylabel(r"Ratio $\tau_Q(U_{\mathrm{sub}}) \,/\, \tau_Q(U_{\mathrm{macro}})$ ($\lambda = 0.6$)")
    ax_b.set_title("(b) Coarse vs. Fine Sensitivity")
    ax_b.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.94, fontsize=7.8)

    # --- Panel (c): Healed Coarse Cycle-1 Excursions ---
    ax_c = fig.add_subplot(gs[0, 2])
    style_ax(ax_c)
    for cname, jtag, col, fmt, lbl in [
        ("rottor_iso_L2r1_s4", "uniform", C_TORUS, "-o", "Torus (uniform)"),
        ("sq_iso_L2r1_s4", "uniform", C_PLANAR, "-D", "Planar (uniform)"),
        ("rottor_iso_L2r1_s4", "sub_weld_l06", C_SUB, "--s", r"Torus (fine C2, $\lambda=0.6$)"),
        ("rottor_iso_L2r1_s4", "macro_weld_l06", C_MACRO, ":^", r"Torus (coarse C1, $\lambda=0.6$)"),
    ]:
        prefix = f"excur__{cname}_X_err__{jtag}__b"
        pts = sorted(
            [v for k, v in exc.items() if k.startswith(prefix)],
            key=lambda d: d["beta"],
        )
        bs = [d["beta"] for d in pts]
        ms = [d["mean_n_exc"] for d in pts]
        ses = [d["sem_n_exc"] for d in pts]
        ax_c.errorbar(bs, ms, yerr=ses, fmt=fmt, color=col, ms=5.8, lw=1.9, capsize=2.4, label=lbl)

    ax_c.annotate(
        r"$9.5 \pm 0.5$ healed",
        xy=(6.50, 9.50),
        xytext=(5.60, 7.6),
        arrowprops=dict(arrowstyle="->", color=C_TORUS, lw=1.2),
        fontsize=8.2,
        fontweight="bold",
        color="#065f46",
    )

    ax_c.set_xlim(5.42, 6.58)
    ax_c.set_ylim(1.0, 11.2)
    ax_c.set_xlabel(r"Inverse Temperature $\beta = 1/T$")
    ax_c.set_ylabel("Coarse Excursions Before Failure")
    ax_c.set_title("(c) Healed Coarse Excursions")
    ax_c.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.94, fontsize=7.6)

    save_fig(fig, "fig_v19_3_intervention")


# ==============================================================================
# FIGURE 4: Spacious 2-Panel Readout & Coarse Confinement Figure
# ==============================================================================
def render_fig4(v14):
    flow = v14["rgflow14"]
    pdec = v14["paired_kmc"]

    fig = plt.figure(figsize=(10.8, 4.1))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.05, 1.0], wspace=0.25)

    # --- Panel (a): Metastable Thermal vs. Matched-Weight Random Coarse Flow ---
    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a)
    scales_x = np.arange(5)

    for b_str, col, mk in [
        ("5.50", "#d97706", "s"),
        ("6.50", C_TORUS, "o"),
    ]:
        rec = flow[f"rgflow14__rottor_s4__t80.0__b{b_str}"]
        sc_keys = ["4", "3", "2", "1", "0"]
        th_y = [max(rec["scales"][sk]["cnt"], 4e-3) for sk in sc_keys]
        th_se = [rec["scales"][sk]["semc"] for sk in sc_keys]
        rn_y = [max(rec["scales"][sk]["icnt"], 4e-3) for sk in sc_keys]
        flips = int(round(rec["mean_qwt"]))
        ax_a.errorbar(
            scales_x, th_y, yerr=th_se, fmt=f"-{mk}", color=col, lw=2.2, ms=6.2, capsize=2.4,
            label=rf"Thermal $\beta={float(b_str):.1f}$ ({flips} flips)",
        )
        ax_a.plot(
            scales_x, rn_y, f"--{mk}", color=col, mfc="white", mew=1.6, lw=1.5, ms=5.8, alpha=0.88,
            label=f"Random {flips}-flip control",
        )

    ax_a.axhline(4e-3, color="#64748b", ls=":", lw=1.0)
    ax_a.text(2.05, 6.2e-3, "0/96 pre-solve seed defects", fontsize=8.0, color="#065f46", fontweight="bold")

    ax_a.set_yscale("log")
    ax_a.set_xticks(scales_x)
    ax_a.set_xticklabels(["s=4\n(41.5k)", "s=3\n(3.5k)", "s=2\n(288)", "s=1\n(24)", "Seed\n(2)"])
    ax_a.set_ylim(2.5e-3, 6e3)
    ax_a.set_xlabel(r"Coarse Scale $s$ ($t_{\mathrm{burn}} = 80$, Pre-Seed Solve)")
    ax_a.set_ylabel("Mean Violated Checks")
    ax_a.set_title("(a) Thermal vs. Random Coarse Flow")
    ax_a.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.94, fontsize=7.9)

    # --- Panel (b): Paired Hierarchical vs. Global Readout ---
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b)
    ax_b.axhline(1.0, color="#0f172a", ls="--", lw=1.2)
    ax_b.axhspan(0.90, 1.10, color="#dcfce7", alpha=0.38, label=r"$\pm 10\%$ band")

    for short_c, col, mk, lbl in [
        ("rottor_s2", C_SUB, "s", "Torus C1 (646q)"),
        ("rottor_s4", C_TORUS, "o", "Torus C2 (94.8kq)"),
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
        ax_b.errorbar(bs, rs, yerr=[lo, hi], fmt=f"-{mk}", color=col, ms=6.2, lw=2.1, capsize=2.8, label=lbl)

    ax_b.text(5.98, 0.72, r"$11\text{–}18\times$ faster" + "\n" + r"for $\beta \geq 6.0$", fontsize=8.3, color="#065f46", fontweight="bold")

    ax_b.set_xlim(4.88, 6.75)
    ax_b.set_ylim(0.32, 1.32)
    ax_b.set_xlabel(r"Inverse Temperature $\beta = 1/T$")
    ax_b.set_ylabel(r"Paired Mean Ratio $\bar{\tau}_{\mathrm{RG}} \,/\, \bar{\tau}_{\mathrm{UF}}$")
    ax_b.set_title("(b) Hierarchical vs. Global Readout")
    ax_b.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.94, fontsize=8.0)

    save_fig(fig, "fig_v19_4_readout")


# ==============================================================================
# FIGURE 5: 3D Packing, 3x3 Lifetimes, & Full-Height Check-Wire Collapse
# ==============================================================================
def render_fig5(v14, v13):
    tq = v14["tau_q"]

    fig = plt.figure(figsize=(13.6, 4.25))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[0.98, 1.04, 1.04], wspace=0.27)

    # --- Panel (a): 3D Ball-Packing Condition ---
    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a)
    Ls = np.arange(2, 9)
    ls_smooth = np.linspace(1.85, 8.2, 200)
    exc_1rib = [math.log(2 * L * L + 2 * L) / (3 * math.log(L)) - 1.0 for L in ls_smooth]
    exc_2rib = [math.log(2 * L * L + 4 * L) / (3 * math.log(L)) - 1.0 for L in ls_smooth]

    ax_a.axhline(0.0, color="#0f172a", ls="--", lw=1.2)
    ax_a.axhspan(0.0, 0.25, color="#fee2e2", alpha=0.22)
    ax_a.axhspan(-0.23, 0.0, color="#dcfce7", alpha=0.20)

    ax_a.plot(ls_smooth, exc_1rib, "-", color=C_PLANAR, lw=2.1, label=r"1-Rib Cross ($2L^2+2L$)")
    ax_a.plot(ls_smooth, exc_2rib, "--", color=C_MACRO, lw=1.9, label=r"2-Rib Cell ($2L^2+4L$)")
    ax_a.scatter(Ls, [math.log(2 * L * L + 2 * L) / (3 * math.log(L)) - 1.0 for L in Ls], color=C_PLANAR, s=42, zorder=5)
    ax_a.scatter(Ls, [math.log(2 * L * L + 4 * L) / (3 * math.log(L)) - 1.0 for L in Ls], color=C_MACRO, marker="s", s=38, zorder=5)

    e3 = math.log(24) / (3 * math.log(3)) - 1.0
    ax_a.annotate(
        r"$L = 3$ ($24 < 27$)",
        xy=(3.0, e3),
        xytext=(3.55, -0.18),
        arrowprops=dict(arrowstyle="->", color=C_3X3, lw=1.3),
        fontsize=8.3,
        fontweight="bold",
        color=C_3X3,
    )
    ax_a.text(5.3, 0.08, "Link stretching required", ha="center", fontsize=8.0, fontweight="bold", color="#991b1b")
    ax_a.text(5.5, -0.05, "Volume-compatible", ha="center", fontsize=8.0, color="#065f46", fontweight="bold")

    ax_a.set_xlim(1.75, 8.3)
    ax_a.set_ylim(-0.22, 0.23)
    ax_a.set_xticks(Ls)
    ax_a.set_xlabel("Subdivision Scale L")
    ax_a.set_ylabel(r"Volume Excess $\frac{1}{3}\log_L m(L) - 1$")
    ax_a.set_title("(a) 3D Ball-Packing Condition")
    ax_a.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.94, fontsize=7.6)

    # --- Panel (b): Planar 3x3 Lifetimes ---
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b)
    for cname, fmt, col, hollow, lbl in [
        ("sq_L3r1_s1", "--o", C_3X3, True, "s=1 half-cycle (41q)"),
        ("sq_L3r1_s2", "-s",  C_3X3, False, "Cycle 1 (1,271q)"),
        ("sq_L3r1_s3", "--^", C_3X3, True, "s=3 half-cycle (24.4kq)"),
    ]:
        bs = sorted([float(b) for b in tq[cname].keys() if not b.startswith("_")])
        ys = [tq[cname][f"{b:.2f}"]["tau_Q"] for b in bs]
        ses = [tq[cname][f"{b:.2f}"]["sem_Q"] for b in bs]
        ax_b.errorbar(
            bs, ys, yerr=ses, fmt=fmt, color=col, ms=5.8,
            mfc=("white" if hollow else col), mew=1.6,
            lw=(2.2 if not hollow else 1.6), capsize=2.3, label=lbl,
        )

    bs3 = sorted([float(b) for b in tq["sq_L3r1_s3"].keys() if not b.startswith("_")])
    tx3 = [tq["sq_L3r1_s3"][f"{b:.2f}"]["tau_X"] for b in bs3]
    ax_b.plot(bs3, tx3, ":d", color="#0284c7", lw=1.6, ms=5.5, label=r"s=3 $\tau_X$ (lower bnd)")

    ax_b.annotate(
        r"Crosses C1 ($2.3\times$)",
        xy=(7.50, tq["sq_L3r1_s3"]["7.50"]["tau_Q"]),
        xytext=(5.45, 3.8e5),
        arrowprops=dict(arrowstyle="->", color=C_3X3, lw=1.2),
        fontsize=8.1,
        fontweight="bold",
        color="#5b21b6",
    )

    ax_b.set_yscale("log")
    ax_b.set_xlim(4.35, 7.68)
    ax_b.set_ylim(12, 2.5e8)
    ax_b.set_xlabel(r"Inverse Temperature $\beta = 1/T$")
    ax_b.set_ylabel("Mean Lifetime")
    ax_b.set_title(r"(b) Planar $3 \times 3$ Lifetimes")
    ax_b.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.94, fontsize=7.6)

    # --- Panel (c): Full-Height Torus Check-Wire Collapse ---
    ax_c = fig.add_subplot(gs[0, 2])
    style_ax(ax_c)

    def beta_eff(beta, w):
        t = math.tanh(0.5 * beta) ** w
        return 2.0 * math.atanh(min(max(t, 1e-15), 1.0 - 1e-15))

    w_styles = [
        (1,  "o", C_TORUS, "w = 1 (core)"),
        (2,  "s", "#0284c7", "w = 2"),
        (4,  "^", "#2563eb", "w = 4"),
        (8,  "D", "#d97706", "w = 8"),
        (16, "v", C_MACRO, "w = 16"),
    ]
    wire_src = v13["wire_sweep"]["tor"]
    for w, mk, col, lbl in w_styles:
        if str(w) not in wire_src:
            continue
        pts = sorted(wire_src[str(w)]["points"], key=lambda d: d["beta"])
        beffs = [d["beta_eff"] for d in pts]
        ys = [d["mean_t"] / (w * w) for d in pts]
        ses = [d["sem_t"] / (w * w) for d in pts]
        ax_c.errorbar(beffs, ys, yerr=ses, fmt=f"-{mk}", color=col, ms=5.4, lw=1.5, capsize=2.0, label=lbl)

    ax_c.axvspan(5.0, 5.72, color="#dcfce7", alpha=0.35)
    ax_c.text(4.75, 4.5, r"Matched-$\beta_{\mathrm{eff}}$ fit:" + "\n" + r"$\gamma = 2.02 \pm 0.05$", fontsize=8.1, color="#065f46", fontweight="bold")

    ax_c.set_yscale("log")
    ax_c.set_xlim(2.05, 6.7)
    ax_c.set_ylim(0.7, 4.5e5)
    ax_c.set_xlabel(r"Effective Wire Coupling $\beta_{\mathrm{eff}}(w)$")
    ax_c.set_ylabel(r"Rescaled Lifetime $\tau_X(\beta, w) \,/\, w^2$")
    ax_c.set_title("(c) Check-Wire Collapse")
    ax_c.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.94, fontsize=7.6)

    save_fig(fig, "fig_v19_5_implementation")


def main():
    v14 = json.loads((ROOT / "parsed_v14_data.json").read_text())
    v13 = json.loads((ROOT / "parsed_v13_data.json").read_text())
    render_fig1()
    render_fig2(v14)
    render_fig3(v14)
    render_fig4(v14)
    render_fig5(v14, v13)
    print("All 5 clean V19 figures rendered successfully.")


if __name__ == "__main__":
    main()
