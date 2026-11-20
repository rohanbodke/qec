#!/usr/bin/env python3
"""
Parse V5 Results Directly Resolving Weaknesses #2 and #3:
  - Point 2: Boundaryless 3D-Local Rotated Torus [[4,2,2]] across 2 Full Cycles (s=0..4, N=4..50,296)
             + Intermediate 3D-Local Cycle-1 Sizes + Extended Cold Betas on Planar 2-Cycle (up to beta=7.0).
  - Point 3: Explicit Collision-Free Z^3 Crystal Embeddings ((3,5), (4,6), and BDL's literal ell=8 N=56,312!)
             quantifying the exact Z^3 Geometric Embedding Tax!
"""

import math
import os
from collections import defaultdict

WS = "/Users/harshilavlani/Downloads/6.S980_gemini"
IN_PATH = os.path.join(WS, "cluster_v5_results.txt")
OUT_PATH = os.path.join(WS, "parsed_v5_summary.txt")


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


def fit_slope(bmap, bmin=3.5, bmax=7.0):
    pts = [
        (b, math.log(r["mean_t"]))
        for b, r in sorted(bmap.items())
        if bmin <= b <= bmax and r["capped"] == 0 and r["mean_t"] > 0
    ]
    if len(pts) < 2:
        return None, 0
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    return (sxy / sxx if sxx > 1e-12 else None), n


def main():
    errs = []
    stats = {}
    kmc = defaultdict(dict)

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
            suite, label, _, body = parts
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
                    "capped": int(kv["capped"]),
                }

    lines = ["=== 0. V5 EXECUTION AUDIT ==="]
    lines.append(f"Total timeouts/errors: {len(errs)}")
    for e in errs[:30]:
        lines.append(f"  {e}")

    z3_audit = os.path.join(WS, "z3_embedding_audit.txt")
    if os.path.exists(z3_audit):
        lines.append("")
        with open(z3_audit) as zf:
            lines.extend(l.rstrip() for l in zf)

    lines.append("\n=== 1. V5 EXACT CODE STATS (Up to N = 56,312 BDL ell=8!) ===")
    for cn in sorted(stats):
        s = stats[cn]
        lines.append(
            f"  {cn:34s} | N={s['N']:6d} M={s['M']:6d} M_meta={s['M_meta']:4d} K={s['K']:2d} "
            f"w_max={s['max_cw']:2d} w_avg={s['mean_cw']:4.2f} qdeg={s['qdeg']:2d} d<={s['d']:4d} Delta<={s['Delta']:3d}"
        )

    lines.append("\n=== 2. V5 KMC LIFETIMES & EFFECTIVE BARRIERS (2-Cycle Boundaryless T^2 & Z^3 Crystal Codes) ===")
    for cn in sorted(kmc):
        bmap = kmc[cn]
        N = next(iter(bmap.values()))["N"]
        K = next(iter(bmap.values()))["K"]
        sl, nf = fit_slope(bmap, 4.0, 7.0)
        sl_s = f"{sl:5.2f}({nf})" if sl is not None else "  N/A   "
        pts_s = " | ".join(f"b={b:g}:{r['mean_t']:.3e}(c={r['capped']})" for b, r in sorted(bmap.items()))
        lines.append(f"  {cn:34s} | N={N:6d} K={K:2d} | Deff={sl_s} | {pts_s}")

    with open(OUT_PATH, "w") as out:
        out.write("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} lines to {OUT_PATH}")


if __name__ == "__main__":
    main()
