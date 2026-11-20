#!/usr/bin/env python3
"""
Generate 4 publication-grade 300-DPI figures on SubMIT from our Slurm simulation results.
"""
import os
import math
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9.2,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 220,
    "axes.grid": True,
    "grid.alpha": 0.28,
})

WORK_DIR = "/work/submit/harshila/6s980_qmem"
FIG_DIR = os.path.join(WORK_DIR, "figs_out")
os.makedirs(FIG_DIR, exist_ok=True)


def parse_kv(text):
    d = {}
    for tok in text.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            d[k] = v
    return d


kmc_q = defaultdict(dict)
thresh = defaultdict(list)
cool = defaultdict(list)
hybrid = defaultdict(list)

with open(os.path.join(WORK_DIR, "cluster_results_all.txt")) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith(("ERR", "TIMEOUT")):
            continue
        parts = line.split("|", 3)
        if len(parts) < 4:
            continue
        suite, cname, _, payload = parts
        kv = parse_kv(payload)
        if suite == "KMC_Q" and payload.startswith("KMC"):
            kmc_q[cname][float(kv["beta"])] = (
                int(kv["N"]), float(kv["mean_t"]), float(kv["sem_t"]),
                int(kv["capped"]), int(kv["trials"])
            )
        elif suite == "THRESH" and payload.startswith("THRESHOLD"):
            thresh[(cname, kv["dec"])].append((float(kv["p"]), float(kv["P_L"]), float(kv["sem"])))
        elif suite == "COOL" and payload.startswith("COOL"):
            cool[(cname, float(kv["beta"]))].append((float(kv["t"]), float(kv["syn_density"])))
        elif suite == "HYBRID" and payload.startswith("HYBRID"):
            hybrid[cname].append((float(kv["tau_ec"]), float(kv["P_fail"])))

cross = defaultdict(dict)
for fn in sorted(os.listdir(os.path.join(WORK_DIR, "results_fup"))):
    with open(os.path.join(WORK_DIR, "results_fup", fn)) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("CROSS|"):
                continue
            parts = line.split("|")
            cname = parts[1]
            kv = parse_kv(parts[-1])
            cross[cname][float(kv["beta"])] = (
                int(kv["N"]), float(kv["mean_t"]), float(kv["sem_t"]),
                int(kv["capped"]), int(kv["trials"])
            )

# =============================================================================
# FIGURE 1: 3D Quantum Memory Showdown — Worst-Sector Quantum Lifetime min(t_X, t_Z)
#           & Sawtooth Alternating Dynamics across Recursion Steps!
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.8))

betas = [3.0, 4.0, 5.0, 6.0, 7.0]

def get_min_sector(prefix):
    xs, ys = [], []
    for b in betas:
        cx, cz = f"{prefix}_Xerr.code", f"{prefix}_Zerr.code"
        if b in kmc_q[cx] and b in kmc_q[cz]:
            xs.append(b)
            ys.append(min(kmc_q[cx][b][1], kmc_q[cz][b][1]))
        elif b in kmc_q[cx]:
            xs.append(b)
            ys.append(kmc_q[cx][b][1])
    return xs, ys

series_f1 = [
    ("bdl_tor2_hyb21_s2", "3D BDL Torus Hybrid 2->1 rib (N=3160, Δ=(5,5))", "#d62728", "o-", 2.5),
    ("bdl_tor3_L2r1_s2",  "3D BDL 3x3 Torus 1-rib (N=2907, Δ=(4,5))",       "#ff7f0e", "s-", 2.2),
    ("bdl_tor2_L2r1_s2",  "3D BDL 2x2 Torus 1-rib (N=1292, Δ=(3,5))",       "#2ca02c", "^-", 2.0),
    ("bdl_sq_L3r2_s2",    "3D BDL Planar 2-rib L=3 (N=2152, Δ=(3,4))",      "#1f77b4", "D-", 2.0),
    ("bdl_sq_L3r1_s2",    "3D BDL Planar 1-rib L=3 (N=1271, Δ=(2,3))",      "#9467bd", "v--", 1.8),
    ("layer_913",         "3D Layer Code [[9,1,3]] (N=1853, Δ=(2,2))",      "#8c564b", "P--", 1.7),
    ("toric3d_L7",        "3D Toric Code L=7 (N=1029, min(t_X,t_Z)=t_Z)",   "#7f7f7f", "x:", 1.7),
    ("toric2d_L30",       "2D Toric Code L=30 (N=1800, Δ=(2,2))",           "#17becf", "+:", 1.7),
]
for pref, lbl, col, sty, lw in series_f1:
    bx, ty = get_min_sector(pref)
    ax1.semilogy(bx, ty, sty, color=col, linewidth=lw, markersize=6.5, label=lbl)

ax1.set_xlabel(r"Inverse Bath Temperature $\beta = 1/T$")
ax1.set_ylabel(r"Quantum Memory Lifetime $\min(\langle t_{\mathrm{fail},X}\rangle, \langle t_{\mathrm{fail},Z}\rangle)$")
ax1.set_title("(a) True Two-Sector 3D Quantum Memory Lifetime")
ax1.legend(loc="upper left", framealpha=0.92, fontsize=8.0)

# Panel (b): Sawtooth Entropic Dip across steps s=1, 2, 3 on 1-rib `bdl_sq_L2r1`
steps_s = [1, 2, 3]
for b, col, mk in [(4.0, "#1f77b4", "o"), (5.0, "#ff7f0e", "s"), (6.0, "#d62728", "^")]:
    tx = [kmc_q[f"bdl_sq_L2r1_s{s}_Xerr.code"][b][1] for s in steps_s]
    tz = [kmc_q[f"bdl_sq_L2r1_s{s}_Zerr.code"][b][1] for s in steps_s]
    ax2.semilogy(steps_s, tx, f"{mk}-", color=col, linewidth=2.2, markersize=7, label=rf"$X$-sector ($\beta={b:.0f}$)")
    ax2.semilogy(steps_s, tz, f"{mk}--", color=col, linewidth=1.7, markersize=6, alpha=0.75, label=rf"$Z$-sector ($\beta={b:.0f}$)")

ax2.set_xticks([1, 2, 3])
ax2.set_xticklabels([r"$s=1$ ($\mathcal{R}_X$)" + "\n$N=19$", r"$s=2$ ($\mathcal{R}_Z\mathcal{R}_X$)" + "\n$N=323$", r"$s=3$ ($\mathcal{R}_X\mathcal{R}_Z\mathcal{R}_X$)" + "\n$N=2911$"])
ax2.set_ylabel(r"Mean Memory Lifetime $\langle t_{\mathrm{fail}}\rangle$")
ax2.set_title(r"(b) Sawtooth Entropic Dip & Recovery (1-Rib $L=2$)")
ax2.legend(loc="upper left", ncol=2, fontsize=8.0, framealpha=0.92)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig1_quantum_lifetime_and_sawtooth.png"), dpi=240)
plt.close()

# =============================================================================
# FIGURE 2: Finite-Temperature Phase Transition Crossing beta_c (1-Rib vs 2-Rib)
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.6), sharey=True)

for ax, pref, title_str, n_map in [
    (ax1, "cls_l3_r1", r"(a) 1-Rib Dual-Self-Correcting ($\ell=3$: 1 rung/edge)", {0: 4, 2: 196, 4: 9604}),
    (ax2, "cls_l4_r2", r"(b) 2-Rib Dual-Self-Correcting ($\ell=4$: 2 rungs/edge)", {0: 4, 2: 400}),
]:
    styles = {0: ("o--", "#7f7f7f", "Step 0"), 2: ("s-", "#1f77b4", "Step 2 (1 full cycle)"), 4: ("^-", "#d62728", "Step 4 (2 full cycles)")}
    for st in sorted(n_map):
        cname = f"{pref}_step{st}.code"
        pts = sorted(cross[cname].items())
        if not pts:
            continue
        bx = [p[0] for p in pts]
        ty = [p[1][1] for p in pts]
        ey = [p[1][2] for p in pts]
        caps = [p[1][3] for p in pts]
        trs = [p[1][4] for p in pts]
        sty, col, slbl = styles[st]
        ax.errorbar(bx, ty, yerr=ey, fmt=sty, color=col, linewidth=2.2, markersize=6.5, capsize=3,
                    label=f"{slbl} ($N={n_map[st]}$)")
        for x, y, c, tr in zip(bx, ty, caps, trs):
            if c >= tr // 2:
                ax.annotate("↑ capped", (x, y), textcoords="offset points", xytext=(-18, 8), fontsize=8, color=col, fontweight="bold")
    ax.axvline(3.32, color="black", linestyle=":", alpha=0.6, label=r"Critical $\beta_c \approx 3.3$")
    ax.set_yscale("log")
    ax.set_xlabel(r"Inverse Bath Temperature $\beta = 1/T$")
    ax.set_title(title_str)
    ax.legend(loc="upper left", framealpha=0.92)

ax1.set_ylabel(r"Mean Memory Lifetime $\langle t_{\mathrm{fail}}\rangle$")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig2_phase_transition_crossing.png"), dpi=240)
plt.close()

# =============================================================================
# FIGURE 3: Decoder Showdown — Threshold P_L(p) & Thermal KMC Across 4 Decoders
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.6))

dec_styles = {
    "bposd":      ("o-",  "#d62728", "BP+OSD (Soft Global)"),
    "uf":         ("s-",  "#1f77b4", "Cluster Union-Find (Local Gauss)"),
    "rg_strict":  ("^--", "#2ca02c", "Sheaf-RG (Strict Monotonic)"),
    "rg_relaxed": ("v:",  "#9467bd", "Sheaf-RG (Relaxed 2x Push)"),
}
for dec in ("bposd", "uf", "rg_strict", "rg_relaxed"):
    sty, col, lbl = dec_styles[dec]
    pts1 = sorted(thresh[("bdl_sq_L3r1_s2_Xerr.code", dec)])
    pts2 = sorted(thresh[("bdl_sq_L3r2_s2_Xerr.code", dec)])
    ax1.plot([x[0]*100 for x in pts1], [x[1] for x in pts1], sty, color=col, linewidth=2.1, markersize=6, label=f"{lbl} (1-rib)")
    ax1.plot([x[0]*100 for x in pts2], [x[1] for x in pts2], sty, color=col, linewidth=1.4, markersize=5, alpha=0.48)

ax1.set_xlabel("Physical Pauli-$X$ Error Rate $p$ (%)")
ax1.set_ylabel("Logical Error Probability $P_L(p)$")
ax1.set_title("(a) i.i.d. Decoding Curves (Solid: 1-Rib, Faint: 2-Rib)")
ax1.legend(loc="lower right", fontsize=8.3, framealpha=0.92)

# Panel (b): Passive Thermal Cooling from T=infinity (Glassy Metastable Plateau at Cold Quench!)
cool_series = [
    (("cls_l4_r2_step3.code", 4.0), "2-Rib Step 3: Moderate Quench (β=4.0)", "s--", "#1f77b4"),
    (("cls_l4_r2_step3.code", 6.0), "2-Rib Step 3: Cold Quench (β=6.0, Glassy Trap!)", "s-", "#d62728"),
    (("cls_l3_r1_step3.code", 4.0), "1-Rib Step 3: Moderate Quench (β=4.0)", "o--", "#2ca02c"),
    (("cls_l3_r1_step3.code", 6.0), "1-Rib Step 3: Cold Quench (β=6.0, 3.4x Lower Trap!)", "o-", "#ff7f0e"),
    (("toric3d_L7_Xerr.code", 6.0), "3D Toric L=7 Membrane: Cold Quench (β=6.0 -> 0)", "^:", "#7f7f7f"),
]
for key, lbl, sty, col in cool_series:
    pts = sorted(cool[key])
    ax2.loglog([x[0] for x in pts], [max(x[1], 5e-5) for x in pts], sty, color=col, linewidth=2.0, markersize=4.5, label=lbl)

ax2.set_xlabel("Passive Bath Evolution Time $t$")
ax2.set_ylabel(r"Residual Syndrome Density $\langle |\sigma(t)| / M \rangle$")
ax2.set_title(r"(b) Passive Initialization from $T=\infty$ (Glassy Pinning)")
ax2.legend(loc="upper right", fontsize=8.0, framealpha=0.92)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig3_decoders_and_glassy_cooling.png"), dpi=240)
plt.close()

# =============================================================================
# FIGURE 4: Hybrid Lazy Active-Passive QEC Duty Cycle Benchmark (beta=5.0)
# =============================================================================
fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.5))
hyb_series = [
    ("bdl_tor3_L2r1_s2_Zerr.code", "3D BDL 3x3 Torus 1-Rib (Z-sector, N=2907)", "o-", "#2ca02c"),
    ("bdl_tor3_L2r1_s2_Xerr.code", "3D BDL 3x3 Torus 1-Rib (X-sector, N=2907)", "s-", "#d62728"),
    ("bdl_sq_L3r2_s2_Xerr.code",   "3D BDL Planar 2-Rib L=3 (X-sector, N=2152)", "^-", "#1f77b4"),
    ("layer_913_Xerr.code",        "3D Layer Code [[9,1,3]] (N=1853)",           "D--", "#ff7f0e"),
    ("toric2d_L20_Xerr.code",      "2D Toric Code L=20 (N=800)",                 "x:", "#7f7f7f"),
]
for cname, lbl, sty, col in hyb_series:
    pts = sorted(hybrid[cname])
    ax.semilogx([x[0] for x in pts], [x[1] for x in pts], sty, color=col, linewidth=2.2, markersize=7, label=lbl)

ax.set_xlabel(r"Interval Between Active QEC Readout/Correction Cycles $\tau_{\mathrm{EC}}$")
ax.set_ylabel("Cumulative Logical Failure Prob. (20 Cycles)")
ax.set_title(r"Hybrid Lazy Active + Passive Thermal Memory ($\beta = 5.0$)")
ax.legend(loc="lower right", fontsize=8.8, framealpha=0.92)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig4_hybrid_lazy_qec.png"), dpi=240)
plt.close()

print("Successfully generated 4 figures in", FIG_DIR)
