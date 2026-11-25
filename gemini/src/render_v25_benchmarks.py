#!/usr/bin/env python3
"""
render_v25_benchmarks.py
Generates clean, fair, untitled PRX-style Figure 7 (`figures/fig_v25_7_benchmarks.{pdf,png}`)
under standard unit stabilizer coupling (J = 1 everywhere, zero temperature rescaling!),
focused on 3D-compatible passive quantum memory families under BP-OSD readout:
  (a) Recovery time t_0.80 vs physical inverse temperature beta in [4.0, 7.0] (J = 1) across:
      - 2D Toric Code (L = 16, N = 512)
      - 1-cycle 3D Welded Surface Code (L = 4, N = 2,926)
      - 2-cycle 3D-packable Alternating Welded C3 C2 (N = 107,140, XZXZ)
      - 2-cycle 3D-packable Symmetric Welded C3 C2^ZX (N = 87,652, XZZX)
      - Haah's 3D Cubic Code (L = 7, N = 686, Z^3 nearest-neighbor)
  (b) Recovery time t_0.80 vs physical qubits N at fixed J = 1, beta = 6.5 across 2D and 3D
      code families (highlighting how 1-cycle 3D Welded collapses back to 2D Toric as flat sheets
      grow, how 3D-packable symmetric welded hierarchies jump to > 3.3 x 10^7, and where strictly
      Z^3-local Haah Cubic Code sits at finite size).
  (c) Haah's 3D Cubic Code finite-size scaling t_0.80 vs lattice size L in {5, 7, 9, 11, 13, 15, 17}
      under standard J = 1 (at beta = 3.50 and beta = 4.00), illustrating entropic turnover at
      fixed temperature and the L = 15 (K = 50) degeneracy dip.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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
C_DARK  = "#0f172a"


def style_ax(ax, lbl, x_off=-0.22, y_off=1.03):
    ax.tick_params(direction="in", which="both", top=True, right=True, pad=3.2)
    ax.grid(True, which="major", color="#e2e8f0", lw=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.text(
        x_off, y_off, f"({lbl})",
        transform=ax.transAxes, fontsize=10.6, fontweight="bold", va="bottom", ha="left", color=C_DARK,
    )


def plot_dstat(ax, x, dstat, color, marker="o", ms=4.6, hollow=False, label=None, zorder=4):
    y = dstat["t_80"]
    lo, hi = dstat["t_80_ci95"]
    mfc = "white" if hollow else color
    if dstat["t_80_crossed"]:
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
            "", xy=(x, y * 1.90), xytext=(x, y),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.15, mutation_scale=7.5),
            zorder=zorder + 1,
        )


def render_fig7_benchmarks(v20, v19, suiteA):
    arms = v20["arms"]
    v19_arms = v19["arms"]
    rdA = suiteA["readouts"]

    fig = plt.figure(figsize=(7.15, 2.80))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.04, 1.08, 1.00], wspace=0.34)

    # Panel (a): Standard J = 1 temperature scaling across 2D & 3D families
    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a, "a", x_off=-0.22)

    def get_j1_point(src, key, dec="bposd"):
        if src == "v20" and key in arms:
            return arms[key]["decoders"][dec]
        if src == "v19" and key in v19_arms:
            return v19_arms[key]["decoders"][dec]
        return None

    curves_a = [
        (C_SLATE, "o", "-", "2D Toric ($L{=}16$)", [
            (4.50, get_j1_point("v20", "v20_tor2d_L16__br4.50")),
            (5.00, get_j1_point("v20", "v20_tor2d_L16__br5.00")),
            (5.50, get_j1_point("v20", "v20_tor2d_L16__bq5.50")),
            (6.00, get_j1_point("v20", "v20_tor2d_L16__bq6.00")),
            (6.50, get_j1_point("v20", "v20_tor2d_L16__bq6.50")),
            (7.00, get_j1_point("v20", "v20_tor2d_L16__bq7.00")),
        ]),
        (C_AMBER, "D", "--", "1-cyc. Welded ($L{=}4$)", [
            (4.50, get_j1_point("v20", "v20_mich1c_L4__br4.50")),
            (5.00, get_j1_point("v20", "v20_mich1c_L4__br5.00")),
            (5.50, get_j1_point("v20", "v20_mich1c_L4__br5.50")),
            (6.00, get_j1_point("v20", "v20_mich1c_L4__br6.00")),
            (6.50, get_j1_point("v20", "v20_mich1c_L4__br6.50")),
            (7.00, get_j1_point("v20", "v20_mich1c_L4__br7.00")),
        ]),
        (C_BLUE, "^", "-.", "$C_3 C_2$ ($XZXZ$)", [
            (4.50, get_j1_point("v20", "v19_sq_C3_C2__br4.50")),
            (5.00, get_j1_point("v20", "v19_sq_C3_C2__br5.00")),
            (5.50, get_j1_point("v20", "v19_sq_C3_C2__br5.50")),
            (6.00, get_j1_point("v20", "v19_sq_C3_C2__br6.00")),
            (6.50, get_j1_point("v19", "v19_sq_C3_C2__b6.50")),
            (7.00, get_j1_point("v19", "v19_sq_C3_C2__b7.00")),
        ]),
        (C_GREEN, "s", "-", "$C_3 C_2^{ZX}$ ($XZZX$)", [
            (4.50, get_j1_point("v20", "v20_weld_4s_C3C2ZX__br4.50")),
            (5.00, get_j1_point("v20", "v20_weld_4s_C3C2ZX__br5.00")),
            (5.50, get_j1_point("v20", "v20_weld_4s_C3C2ZX__br5.50")),
            (6.00, get_j1_point("v19", "v19_sq_C3_C2ZX__b6.00")),
            (6.50, get_j1_point("v19", "v19_sq_C3_C2ZX__b6.50")),
            (7.00, get_j1_point("v19", "v19_sq_C3_C2ZX__b7.00")),
        ]),
        (C_RED, "P", "-", "3D Haah ($L{=}7$)", [
            (4.00, get_j1_point("v20", "v20_haah_L7__br4.00")),
            (4.25, get_j1_point("v20", "v20_haah_L7__br4.25")),
            (4.50, get_j1_point("v20", "v20_haah_L7__br4.50")),
            (5.00, get_j1_point("v20", "v20_haah_L7__br5.00")),
            (6.50, get_j1_point("v20", "v20_haah_L7__br6.50")),
        ]),
    ]
    for col, mk, ls, lbl, pts in curves_a:
        xs, ys = [], []
        for b, d in pts:
            if d is None:
                continue
            xs.append(b)
            ys.append(d["t_80"])
            plot_dstat(ax_a, b, d, col, mk, 4.3)
        if xs:
            ax_a.plot(xs, ys, color=col, ls=ls, lw=1.35, zorder=3)
            ax_a.plot([], [], color=col, marker=mk, ls=ls, lw=1.35, ms=4.3, label=lbl)

    ax_a.set_yscale("log")
    ax_a.set_xlim(3.85, 7.18)
    ax_a.set_xticks([4.0, 5.0, 6.0, 7.0])
    ax_a.set_ylim(5.0, 6.5e9)
    ax_a.set_xlabel("Inverse temperature $\\beta$ ($J = 1$)")
    ax_a.set_ylabel("BP-OSD $t_{0.80}$")
    handles_a, labels_a = ax_a.get_legend_handles_labels()

    # Panel (b): t_0.80 vs N at fixed standard J = 1, beta = 6.5 across 2D & 3D families
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b, "b", x_off=-0.22)

    # 1. 2D Toric (J = 1, beta = 6.5)
    t2_tags = ["v20_tor2d_L8", "v20_tor2d_L16", "v20_tor2d_L32", "v20_tor2d_L64"]
    xs, ys = [], []
    for t in t2_tags:
        k = f"{t}__bq6.50"  # d_tot = 4.0 -> exact J = 1, beta = 6.50!
        if k in arms:
            d = arms[k]["decoders"]["bposd"]
            xs.append(arms[k]["N"]); ys.append(d["t_80"])
            plot_dstat(ax_b, arms[k]["N"], d, C_SLATE, "o", 4.3)
    ax_b.plot(xs, ys, color=C_SLATE, ls="-", lw=1.35, zorder=3)

    # 2. 1-cycle 3D Welded (J = 1, beta = 6.5)
    m1_tags = ["v20_mich1c_L2", "v20_mich1c_L3", "v20_mich1c_L4", "v20_mich1c_L6", "v20_mich1c_L8"]
    xs, ys = [], []
    for t in m1_tags:
        k = f"{t}__br6.50"
        if k in arms:
            d = arms[k]["decoders"]["bposd"]
            xs.append(arms[k]["N"]); ys.append(d["t_80"])
            plot_dstat(ax_b, arms[k]["N"], d, C_AMBER, "D", 4.3)
    ax_b.plot(xs, ys, color=C_AMBER, ls="--", lw=1.35, zorder=3)

    # 3. 2-cycle 3D-packable Alternating Welded (XZXZ, rho < L^3)
    alt_pts = [
        (v19_arms["v19_sq_C3_C2__b6.50"]["N"],  v19_arms["v19_sq_C3_C2__b6.50"]["decoders"]["bposd"],  "$C_3 C_2$",    (0.72, 0.90), "right"),
        (v19_arms["v19_sq_RX3_C2__b6.50"]["N"], v19_arms["v19_sq_RX3_C2__b6.50"]["decoders"]["bposd"], "$R_{X3} C_2$", (1.42, 0.96), "left"),
        (rdA["v15_sq_L3_RR"]["6.50"]["N"], {
            "t_80": rdA["v15_sq_L3_RR"]["6.50"]["t_80"],
            "t_80_crossed": True,
            "t_80_ci95": rdA["v15_sq_L3_RR"]["6.50"]["t_80_ci95"],
        }, "$R_3 R_3$", (0.95, 0.44), "center"),
    ]
    for nv, d, txt, mult, ha_al in alt_pts:
        plot_dstat(ax_b, nv, d, C_BLUE, "^", 4.6)
        ax_b.text(nv * mult[0], d["t_80"] * mult[1], txt, fontsize=5.9, color=C_DARK, ha=ha_al, va="center")

    # 4. 2-cycle 3D-packable Symmetric Welded (XZZX, rho < 36^3)
    sym_pts = [
        (v19_arms["v19_sq_C3_C2ZX__b6.50"]["N"],  v19_arms["v19_sq_C3_C2ZX__b6.50"]["decoders"]["bposd"],  "$C_3 C_2^{ZX}$",    (0.70, 0.85), "right"),
        (v19_arms["v19_sq_RZ3_C2ZX__b6.50"]["N"], v19_arms["v19_sq_RZ3_C2ZX__b6.50"]["decoders"]["bposd"], "$R_{Z3} C_2^{ZX}$", (0.95, 2.55), "center"),
    ]
    for nv, d, txt, mult, ha_al in sym_pts:
        plot_dstat(ax_b, nv, d, C_GREEN, "s", 4.7)
        ax_b.text(nv * mult[0], d["t_80"] * mult[1], txt, fontsize=5.9, color=C_DARK, ha=ha_al, va="center")

    # 5. Strictly Z^3-local Haah 3D Cubic Code (L = 7, 11 at J = 1, beta = 6.5)
    for k, txt, mult in [
        ("v20_haah_L7__br6.50",  "Haah $L{=}7$",  (1.38, 0.92)),
        ("v20_haah_L11__br6.50", "Haah $L{=}11$", (1.38, 0.86)),
    ]:
        if k in arms:
            nv = arms[k]["N"]
            d = arms[k]["decoders"]["bposd"]
            plot_dstat(ax_b, nv, d, C_RED, "P", 4.9)
            ax_b.text(nv * mult[0], d["t_80"] * mult[1], txt, fontsize=5.8, color=C_RED, va="center")

    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlim(85.0, 1.5e6)
    ax_b.set_ylim(80.0, 6.5e9)
    ax_b.set_xlabel("Physical qubits $N$ ($\\beta = 6.5$)")
    ax_b.set_ylabel("BP-OSD $t_{0.80}$")

    # Shared clean legend across panels (a) & (b) placed in the unobstructed pocket of panel (b)
    shared_lbls = [
        "2D Toric ($\\mathbb{Z}^2$)",
        "1-cyc. 3D Welded",
        "2-cyc. $XZXZ$ (3D)",
        "2-cyc. $XZZX$ (3D)",
        "3D Haah ($\\mathbb{Z}^3$)",
    ]
    ax_b.legend(
        handles_a, shared_lbls,
        loc="center left", bbox_to_anchor=(0.02, 0.53),
        frameon=True, framealpha=0.93, edgecolor="#cbd5e1",
        fontsize=5.3, borderpad=0.24, labelspacing=0.20, handlelength=1.45,
    )

    # Panel (c): Haah's 3D Cubic Code finite-size crossover vs lattice size L (standard J = 1)
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
            if k in arms:
                d = arms[k]["decoders"]["bposd"]
                xs.append(L); ys.append(d["t_80"])
                plot_dstat(ax_c, L, d, col, mk, 4.3)
        if xs:
            ax_c.plot(xs, ys, color=col, ls=ls, lw=1.35, zorder=3)
            ax_c.plot([], [], color=col, marker=mk, ls=ls, lw=1.35, ms=4.3, label=lbl)

    # Open markers at L = 15 (K = 50 degeneracy jump)
    for mode_s, col in [("bq6.50", C_AMBER), ("bq7.55", C_TEAL), ("br4.00", C_RED)]:
        k15 = f"v20_haah_L15__{mode_s}"
        if k15 in arms:
            plot_dstat(ax_c, 15, arms[k15]["decoders"]["bposd"], col, "v", 4.7, hollow=True)
    if "v20_haah_L15__bq6.50" in arms:
        ax_c.text(14.9, arms["v20_haah_L15__bq6.50"]["decoders"]["bposd"]["t_80"] * 0.34, "$L{=}15$\n($K{=}50$)", fontsize=5.5, color=C_DARK, ha="center")

    ax_c.set_yscale("log")
    ax_c.set_xlim(4.2, 18.0)
    ax_c.set_xticks([5, 7, 9, 11, 13, 15, 17])
    ax_c.set_ylim(2.2e2, 3.5e7)
    ax_c.set_xlabel("Cubic lattice size $L$")
    ax_c.set_ylabel("3D Haah $t_{0.80}$")
    ax_c.legend(loc="upper right", frameon=True, framealpha=0.93, edgecolor="#cbd5e1", fontsize=5.4, borderpad=0.22, labelspacing=0.18)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf_p = FIG_DIR / "fig_v25_7_benchmarks.pdf"
    png_p = FIG_DIR / "fig_v25_7_benchmarks.png"
    fig.savefig(pdf_p, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(png_p, dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved {pdf_p}")


def main():
    with open(ROOT / "parsed_v20_benchmarks.json") as f:
        v20 = json.load(f)
    with open(ROOT / "parsed_v19_rebalance_and_decoders.json") as f:
        v19 = json.load(f)
    with open(ROOT / "reanalyzed_v16_v17_suiteA.json") as f:
        suiteA = json.load(f)
    render_fig7_benchmarks(v20, v19, suiteA)


if __name__ == "__main__":
    main()
