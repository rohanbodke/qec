#!/usr/bin/env python3
"""
Lightweight parser & statistical analyzer for the SubMIT Slurm cluster results
(`cluster_results_all.txt`). Performs linear/power-law/Arrhenius fits and prints
summary tables for all 5 experimental suites.
"""

import glob
import math
import os
import re
from collections import defaultdict


def parse_kv(text):
    d = {}
    for tok in text.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            d[k] = v
    return d


def analyze_results(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    stats_rows = {}
    kmc_q = defaultdict(list)       # (code, dec) -> [(beta, N, mean_t, sem_t, median_t, capped, trials)]
    kmc_cls = defaultdict(list)     # code -> [(beta, N, mean_t, sem_t, median_t, capped, trials)]
    kmc_dec = defaultdict(list)     # (code, beta) -> [(dec, mean_t, sem_t, median_t, capped)]
    thresh = defaultdict(list)      # (code, dec) -> [(p, P_L, sem)]
    cool = defaultdict(list)        # (code, beta) -> [(t, syn_density)]
    hybrid = defaultdict(list)      # code -> [(tau_ec, T_tot, P_fail)]

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("ERR") or line.startswith("TIMEOUT"):
                continue
            parts = line.split("|", 3)
            if len(parts) < 4:
                continue
            suite, cname, dt_str, payload = parts
            kv = parse_kv(payload)

            if suite == "STATS" and payload.startswith("STATS"):
                stats_rows[cname] = kv
            elif suite == "KMC_Q" and payload.startswith("KMC"):
                kmc_q[(cname, kv["dec"])].append((
                    float(kv["beta"]), int(kv["N"]), float(kv["mean_t"]),
                    float(kv["sem_t"]), float(kv["median_t"]), int(kv["capped"]), int(kv["trials"])
                ))
            elif suite == "KMC_CLS" and payload.startswith("KMC"):
                kmc_cls[cname].append((
                    float(kv["beta"]), int(kv["N"]), float(kv["mean_t"]),
                    float(kv["sem_t"]), float(kv["median_t"]), int(kv["capped"]), int(kv["trials"])
                ))
            elif suite == "KMC_DEC" and payload.startswith("KMC"):
                kmc_dec[(cname, float(kv["beta"]))].append((
                    kv["dec"], float(kv["mean_t"]), float(kv["sem_t"]),
                    float(kv["median_t"]), int(kv["capped"])
                ))
            elif suite == "THRESH" and payload.startswith("THRESHOLD"):
                thresh[(cname, kv["dec"])].append((
                    float(kv["p"]), float(kv["P_L"]), float(kv["sem"])
                ))
            elif suite == "COOL" and payload.startswith("COOL"):
                cool[(cname, float(kv["beta"]))].append((
                    float(kv["t"]), float(kv["syn_density"])
                ))
            elif suite == "HYBRID" and payload.startswith("HYBRID"):
                hybrid[cname].append((
                    float(kv["tau_ec"]), float(kv["T_tot"]), float(kv["P_fail"])
                ))

    print("=== 1. NEW CODE STATS ===")
    for cname in sorted(stats_rows):
        print(f"  {cname:32s}: {stats_rows[cname]}")

    print("\n=== 2. 3D QUANTUM MEMORY KMC (<t_fail> vs beta) ===")
    for (cname, dec) in sorted(kmc_q):
        rows = sorted(kmc_q[(cname, dec)])
        N = rows[0][1]
        s_items = []
        for beta, _, mt, st, med, cap, tr in rows:
            cap_str = f"[cap {cap}/{tr}]" if cap > 0 else ""
            s_items.append(f"b={beta:.0f}: {mt:.2e}{cap_str}")
        # Fit effective Arrhenius barrier slope d(ln mean_t)/d(beta) on uncapped points!
        uncapped = [(b, math.log(mt)) for (b, _, mt, _, _, cap, tr) in rows if cap < tr // 2 and mt > 0]
        slope_str = ""
        if len(uncapped) >= 2:
            bx = [u[0] for u in uncapped]
            ly = [u[1] for u in uncapped]
            mx = sum(bx) / len(bx)
            my = sum(ly) / len(ly)
            num = sum((x - mx) * (y - my) for x, y in zip(bx, ly))
            den = sum((x - mx) ** 2 for x in bx)
            if den > 0:
                slope_str = f" | eff_slope d(ln t)/dbeta = {num/den:.2f}"
        print(f"  {cname:30s} (N={N:5d}): " + " | ".join(s_items) + slope_str)

    print("\n=== 3. 5-LEVEL 2D CLASSICAL DUAL-SELF-CORRECTING SCALING ===")
    for cname in sorted(kmc_cls):
        rows = sorted(kmc_cls[cname])
        N = rows[0][1]
        s_items = []
        for beta, _, mt, st, med, cap, tr in rows:
            cap_str = f"[c{cap}]" if cap > 0 else ""
            s_items.append(f"b={beta:.1f}:{mt:.2e}{cap_str}")
        print(f"  {cname:24s} (N={N:5d}): " + " | ".join(s_items))

    print("\n=== 4. DECODER SHOWDOWN (KMC: rg_strict vs rg_relaxed vs uf vs bposd) ===")
    for (cname, beta) in sorted(kmc_dec):
        rows = sorted(kmc_dec[(cname, beta)])
        items = [f"{dec}={mt:.2e}(c{cap})" for dec, mt, st, med, cap in rows]
        print(f"  {cname:28s} b={beta:.1f}: " + " | ".join(items))

    print("\n=== 5. DECODER THRESHOLD CURVES P_L(p) ===")
    for (cname, dec) in sorted(thresh):
        rows = sorted(thresh[(cname, dec)])
        items = [f"{p:.2f}:{pl:.3f}" for p, pl, _ in rows]
        print(f"  {cname:28s} [{dec:10s}]: " + " ".join(items))

    print("\n=== 6. PASSIVE COOLING FROM T=INF (p_init=0.5) ===")
    for (cname, beta) in sorted(cool):
        rows = sorted(cool[(cname, beta)])
        # Pick representative times t = 0.25, 4, 64, 1024, 65536, 1048576, 67108864
        targets = [0.25, 4.0, 64.0, 1024.0, 65536.0, 1048576.0, 67108864.0]
        picked = []
        for tg in targets:
            best = min(rows, key=lambda r: abs(r[0] - tg))
            picked.append(f"t={best[0]:.0e}:{best[1]:.4f}")
        print(f"  {cname:28s} b={beta:.1f}: " + " | ".join(picked))

    print("\n=== 7. HYBRID LAZY ACTIVE QEC + THERMAL BATH (beta=5.0, 20 cycles) ===")
    for cname in sorted(hybrid):
        rows = sorted(hybrid[cname])
        items = [f"tau={tau:.0f}(T={tt:.0e}):P={pf:.3f}" for tau, tt, pf in rows]
        print(f"  {cname:28s}: " + " | ".join(items))


if __name__ == "__main__":
    import sys
    analyze_results(sys.argv[1] if len(sys.argv) > 1 else "/Users/harshilavlani/Downloads/6.S980_gemini/cluster_results_all.txt")
