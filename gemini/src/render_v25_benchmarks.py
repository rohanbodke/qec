#!/usr/bin/env python3
"""
render_v25_benchmarks.py
Generates clean, untitled PRX-style Figure 7 (`figures/fig_v25_7_benchmarks.{pdf,png}`)
comparing finite-size operational storage across five code families under BP-OSD readout:
  (a) Recovery time t_0.80 vs per-qubit normalized inverse temperature beta_q = beta * (d_tot / 4)
      for 2D Toric (L=16), Single-Scale 3D Welded (L=4), Haah 3D Cubic (L=7), and Palindromic 3D
      Welded C2 C2^ZX (planar & torus).
  (b) System-size scaling t_0.80 vs physical qubits N at fixed per-qubit coupling budget beta_q = 6.50
      across all five families (showing 2D Toric decay, single-scale 3D Welded flat-sheet turnover,
      Haah 3D Cubic entropic turnover + L=15 K=50 dip, 4D Toric 4D growth, and 3D multiscale
      rebalanced welded growth from N = 254 to 87,652).
  (c) Direct comparison across the three physical energy-normalization conventions:
      (1) Equal mean Hamiltonian coupling per qubit (beta_q = 6.50),
      (2) Equal bulk single-qubit excitation gap (beta * DeltaE_bulk = 13.0),
      (3) Unnormalized raw per-check J = 1 (beta = 6.50).
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
    "legend.fontsize": 6.4,
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
C_PURP  = "#6d28d9"
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


def plot_dstat(ax, x, dstat, color, marker="o", ms=4.7, hollow=False, label=None, zorder=4):
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


def render_fig7_benchmarks(v20, v19):
    arms = v20["arms"]
    v19_arms = v19["arms"]

    fig = plt.figure(figsize=(7.15, 2.80))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.02, 1.08, 1.04], wspace=0.34)

    # Panel (a): t_0.80 vs per-qubit normalized inverse temperature beta_q across 2D & 3D local families
    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a, "a", x_off=-0.22)
    bqs = [5.50, 6.00, 6.50, 7.00, 7.55]
    a_series = [
        ("v20_tor2d_L16",       bqs, C_SLATE, "o", "-",  "2D Toric ($L{=}16$)"),
        ("v20_mich1c_L4",       bqs, C_AMBER, "D", "--", "1-cyc. Weld ($L{=}4$)"),
        ("v20_haah_L7",         bqs, C_RED,   "^", "-.", "3D Haah ($L{=}7$)"),
        ("v20_weld_4s_sqXZZX",  bqs, C_GREEN, "s", "-",  "Planar $C_2 C_2^{ZX}$"),
        ("v20_weld_4s_torXZZX", bqs, C_PURP,  "s", ":",  "Torus $C_2 C_2^{ZX}$"),
    ]
    for tag, blist, col, mk, ls, lbl in a_series:
        xs, ys = [], []
        for bq in blist:
            if tag == "v20_weld_4s_sqXZZX" and abs(bq - 7.55) < 0.01 and "v19_sq_C2_XZZX__b6.50" in v19_arms:
                d = v19_arms["v19_sq_C2_XZZX__b6.50"]["decoders"]["bposd"]
            else:
                k = f"{tag}__bq{bq:.2f}"
                if k not in arms:
                    continue
                d = arms[k]["decoders"]["bposd"]
            xs.append(bq)
            ys.append(d["t_80"])
            plot_dstat(ax_a, bq, d, col, mk, 4.3)
        if xs:
            ax_a.plot(xs, ys, color=col, ls=ls, lw=1.35, zorder=3)
            ax_a.plot([], [], color=col, marker=mk, ls=ls, lw=1.35, ms=4.3, label=lbl)
    ax_a.set_yscale("log")
    ax_a.set_xlim(5.35, 7.72)
    ax_a.set_xticks([5.5, 6.0, 6.5, 7.0, 7.5])
    ax_a.set_ylim(30.0, 9.0e8)
    ax_a.set_xlabel("Per-qubit inv. temp. $\\beta_{\\mathrm{q}}$")
    ax_a.set_ylabel("BP-OSD $t_{0.80}$")
    ax_a.legend(loc="upper left", frameon=True, framealpha=0.93, edgecolor="#cbd5e1", fontsize=5.5, borderpad=0.22, labelspacing=0.20, handlelength=1.40)

    # Panel (b): System-size scaling t_0.80 vs N at fixed per-qubit energy budget beta_q = 6.50
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b, "b", x_off=-0.22)
    size_families = [
        (["v20_tor2d_L8", "v20_tor2d_L16", "v20_tor2d_L32", "v20_tor2d_L64"],
         C_SLATE, "o", "-", "2D Toric"),
        (["v20_mich1c_L2", "v20_mich1c_L3", "v20_mich1c_L4", "v20_mich1c_L6", "v20_mich1c_L8"],
         C_AMBER, "D", "--", "1-cyc. Welded"),
        (["v20_haah_L5", "v20_haah_L7", "v20_haah_L9", "v20_haah_L11", "v20_haah_L13", "v20_haah_L17"],
         C_RED, "^", "-.", "3D Haah"),
        (["v20_tor4d_L3", "v20_tor4d_L4", "v20_tor4d_L5"],
         C_BLUE, "P", "--", "4D Toric"),
        (["v20_mich1c_L2", "v20_weld_3s_X2R3ZX2", "v20_weld_3s_R3X2", "v20_weld_4s_sqXZZX", "v20_weld_4s_C3C2ZX"],
         C_GREEN, "s", "-", "Rebal. Welded"),
    ]
    for tag_list, col, mk, ls, lbl in size_families:
        xs, ys = [], []
        for tag in tag_list:
            k = f"{tag}__bq6.50"
            if k in arms:
                d = arms[k]["decoders"]["bposd"]
                n_val = arms[k]["N"]
                xs.append(n_val)
                ys.append(d["t_80"])
                plot_dstat(ax_b, n_val, d, col, mk, 4.3)
        if xs:
            ax_b.plot(xs, ys, color=col, ls=ls, lw=1.35, zorder=3)
            ax_b.plot([], [], color=col, marker=mk, ls=ls, lw=1.35, ms=4.3, label=lbl)

    if "v20_haah_L15__bq6.50" in arms:
        a15 = arms["v20_haah_L15__bq6.50"]
        plot_dstat(ax_b, a15["N"], a15["decoders"]["bposd"], C_RED, "v", 4.7, hollow=True)
        ax_b.text(a15["N"] * 1.05, a15["decoders"]["bposd"]["t_80"] * 0.48, "$L{=}15$", fontsize=5.6, color=C_RED, ha="center")

    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlim(80.0, 1.8e5)
    ax_b.set_ylim(75.0, 2.5e8)
    ax_b.set_xlabel("Physical qubits $N$ ($\\beta_{\\mathrm{q}} = 6.5$)")
    ax_b.set_ylabel("BP-OSD $t_{0.80}$")
    ax_b.legend(loc="center right", bbox_to_anchor=(0.99, 0.66), frameon=True, framealpha=0.93, edgecolor="#cbd5e1", fontsize=5.5, borderpad=0.22, labelspacing=0.20, handlelength=1.35)

    # Panel (c): Normalization sensitivity across 4 representative 2D/3D codes
    ax_c = fig.add_subplot(gs[0, 2])
    style_ax(ax_c, "c", x_off=-0.22)
    norm_rows = [
        ("2D Toric ($L{=}32$)",
         arms.get("v20_tor2d_L32__bq6.50", {}).get("decoders", {}).get("bposd"),
         arms.get("v20_tor2d_L32__bq6.50", {}).get("decoders", {}).get("bposd"),
         arms.get("v20_tor2d_L32__bq6.50", {}).get("decoders", {}).get("bposd")),
        ("1-cyc. Weld ($L{=}4$)",
         arms.get("v20_mich1c_L4__bq6.50", {}).get("decoders", {}).get("bposd"),
         arms.get("v20_mich1c_L4__br6.50", {}).get("decoders", {}).get("bposd"),
         arms.get("v20_mich1c_L4__br6.50", {}).get("decoders", {}).get("bposd")),
        ("3D Haah ($L{=}7$)",
         arms.get("v20_haah_L7__bq6.50", {}).get("decoders", {}).get("bposd"),
         arms.get("v20_haah_L7__bq6.50", {}).get("decoders", {}).get("bposd"),
         arms.get("v20_haah_L7__br6.50", {}).get("decoders", {}).get("bposd")),
        ("Planar $C_2 C_2^{ZX}$",
         arms.get("v20_weld_4s_sqXZZX__bq6.50", {}).get("decoders", {}).get("bposd"),
         v19_arms.get("v19_sq_C2_XZZX__b6.50", {}).get("decoders", {}).get("bposd"),
         v19_arms.get("v19_sq_C2_XZZX__b6.50", {}).get("decoders", {}).get("bposd")),
    ]
    for xi, (lbl, d_bq, d_bg, d_br) in enumerate(norm_rows):
        if d_bq:
            plot_dstat(ax_c, xi - 0.21, d_bq, C_TEAL, "o", 4.6, label=("$\\beta_{\\mathrm{q}}{=}6.5$" if xi == 0 else None))
        if d_bg:
            plot_dstat(ax_c, xi,        d_bg, C_BLUE, "s", 4.5, label=("$\\beta\\Delta E_1{=}13$" if xi == 0 else None))
        if d_br:
            plot_dstat(ax_c, xi + 0.21, d_br, C_RED,  "^", 4.7, label=("Raw $J{=}1$" if xi == 0 else None))

    ax_c.set_yscale("log")
    ax_c.set_xlim(-0.55, len(norm_rows) - 0.45)
    ax_c.set_xticks(range(len(norm_rows)))
    ax_c.set_xticklabels([r[0] for r in norm_rows], fontsize=6.9, rotation=28, ha="right")
    ax_c.set_ylim(60.0, 9.0e10)
    ax_c.set_ylabel("BP-OSD $t_{0.80}$")
    ax_c.legend(loc="upper left", frameon=True, framealpha=0.93, edgecolor="#cbd5e1", fontsize=5.5, borderpad=0.22, labelspacing=0.20)

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
    render_fig7_benchmarks(v20, v19)


if __name__ == "__main__":
    main()
