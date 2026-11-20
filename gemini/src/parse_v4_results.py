#!/usr/bin/env python3
"""
Parse & statistically analyze the V4 First-Principles Four-Class Self-Correction Campaign:
  1. Verify timeouts/errors/capping across all V4 tasks.
  2. Tabulate exact code stats (N, K, M_meta, w_max, qdeg, d, Delta) for all 4 symmetry classes.
  3. Compute Fixed-Temperature Scaling <t_mem(N, beta)> +/- SEM vs. N across all 4 classes.
  4. Extract Empirical Effective Energy Barrier Delta_eff(N) = d(ln <t>)/d(beta) across sizes!
  5. Summarize Chesi-Loss / BDL Sec. 5 Equilibrium Gibbs Confinement (<|Q_max|/N>) & 1-Flip Instability!
"""

import math
import os
from collections import defaultdict

WS = "/Users/harshilavlani/Downloads/6.S980_gemini"
IN_PATH = os.path.join(WS, "cluster_v4_results.txt")
OUT_PATH = os.path.join(WS, "parsed_v4_summary.txt")


def parse_kv(text):
    d = {}
    for tok in text.split():
        if "<=" in tok:
            k, v = tok.split("<=", 1)
            d[k] = v
        elif "=" in tok:
            k, v = tok.split("=", 1)
            d[k] = v
    return d


def fit_arrhenius_slope(beta_to_rec, bmin=3.5, bmax=6.0):
    """Fits ln(mean_t) = a + Delta_eff * beta over uncapped points (c == 0) in [bmin, bmax]."""
    pts = [
        (b, math.log(r["mean_t"]))
        for b, r in sorted(beta_to_rec.items())
        if bmin <= b <= bmax and r["capped"] == 0 and r["mean_t"] > 0
    ]
    if len(pts) < 2:
        return None, 0
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    if sxx < 1e-12:
        return None, n
    return sxy / sxx, n


def main():
    errs = []
    stats = {}
    kmc = defaultdict(dict)
    gibbs = defaultdict(dict)

    with open(IN_PATH) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(("ERR|", "TIMEOUT|")):
                errs.append(line)
                continue
            parts = line.split("|", 3)
            if len(parts) < 4:
                continue
            suite, label, dt_s, body = parts
            kv = parse_kv(body)
            if body.startswith("STATS "):
                stats[label] = {
                    "N": int(kv["N"]),
                    "M": int(kv["M"]),
                    "M_meta": int(kv["M_meta"]),
                    "K": int(kv["K"]),
                    "max_cw": int(kv["max_cw"]),
                    "mean_cw": float(kv["mean_cw"]),
                    "qdeg": int(kv["qdeg"]),
                    "d": int(kv["d"]),
                    "Delta": int(kv["Delta"]),
                }
            elif body.startswith("KMC "):
                b = float(kv["beta"])
                kmc[label][b] = {
                    "N": int(kv["N"]),
                    "K": int(kv["K"]),
                    "trials": int(kv["trials"]),
                    "mean_t": float(kv["mean_t"]),
                    "sem_t": float(kv["sem_t"]),
                    "median_t": float(kv["median_t"]),
                    "mean_flips": float(kv["mean_flips"]),
                    "capped": int(kv["capped"]),
                }
            elif body.startswith("GIBBS "):
                b = float(kv["beta"])
                gibbs[label][b] = {
                    "N": int(kv["N"]),
                    "K": int(kv["K"]),
                    "t_phys": float(kv["t_phys"]),
                    "syn_dens": float(kv["syn_dens"]),
                    "max_cl_q": float(kv["max_cl_q"]),
                    "max_cl_frac": float(kv["max_cl_frac"]),
                    "p_fail": float(kv["p_fail"]),
                    "p_instab1": float(kv["p_instab1"]),
                }

    lines = []
    lines.append("=== 0. V4 ZERO-TRUST EXECUTION AUDIT ===")
    lines.append(f"Total timeouts/errors: {len(errs)}")
    for e in errs[:30]:
        lines.append(f"  {e}")
    tot_kmc = sum(len(v) for v in kmc.values())
    capped_pts = [(c, b, r["capped"], r["trials"]) for c, v in kmc.items() for b, r in v.items() if r["capped"] > 0]
    lines.append(f"Total KMC (code, beta) points: {tot_kmc} | Points with any capped trial: {len(capped_pts)}")
    for c, b, cp, tr in capped_pts[:25]:
        lines.append(f"  CAPPED: {c} beta={b}: {cp}/{tr}")

    lines.append("\n=== 1. V4 EXACT CODE PARAMETERS ACROSS ALL 4 SYMMETRY CLASSES ===")
    for cn in sorted(stats):
        s = stats[cn]
        lines.append(
            f"  {cn:36s} | N={s['N']:6d} M={s['M']:6d} M_meta={s['M_meta']:4d} K={s['K']:2d} "
            f"w_max={s['max_cw']:2d} w_avg={s['mean_cw']:4.2f} qdeg={s['qdeg']:2d} d<={s['d']:4d} Delta<={s['Delta']:3d}"
        )

    lines.append("\n=== 2. FIXED-TEMPERATURE SCALING & EFFECTIVE BARRIER Delta_eff(N) = d(ln <t>)/d(beta) ===")
    for cn in sorted(kmc):
        bmap = kmc[cn]
        N = next(iter(bmap.values()))["N"]
        K = next(iter(bmap.values()))["K"]
        slope, nfit = fit_arrhenius_slope(bmap, 3.5, 6.0)
        sl_str = f"{slope:5.2f}({nfit})" if slope is not None else "  N/A   "
        pts_str = " | ".join(
            f"b={b:g}:{r['mean_t']:.3e}(c={r['capped']})"
            for b, r in sorted(bmap.items())
        )
        lines.append(f"  {cn:36s} | N={N:6d} K={K:2d} | Deff={sl_str} | {pts_str}")

    lines.append("\n=== 3. CHESI-LOSS / BDL SEC. 5 EQUILIBRIUM GIBBS CONFINEMENT & 1-FLIP INSTABILITY ===")
    for cn in sorted(gibbs):
        for b, g in sorted(gibbs[cn].items()):
            lines.append(
                f"  {cn:36s} | b={b:4.2f} t={g['t_phys']:5.1f} N={g['N']:6d} | "
                f"syn={g['syn_dens']:.3e} |Q_max|={g['max_cl_q']:6.2f} frac={g['max_cl_frac']:.3e} "
                f"P_fail={g['p_fail']:.4f} P_instab1={g['p_instab1']:.3e}"
            )

    with open(OUT_PATH, "w") as out:
        out.write("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} lines to {OUT_PATH}")


if __name__ == "__main__":
    main()
