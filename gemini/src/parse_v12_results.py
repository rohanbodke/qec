#!/usr/bin/env python3
"""
parse_v12_results.py
Parses `cluster_v12_results.txt` and prints clean, structured benchmark summaries for:
  1A. 3D-Euclidean-Compatible `3x3` Single-Cross (`dim = log_3(24) = 2.893 < 3`) multi-scale lifetimes
  1B. Collision-free Z^3 cubic grids (`3,067q`, `10,280q`, `56,312q`): Uniform vs Bounded Wire-Energy Hills
  2A. Sub-2,500-qubit sector-balanced welded codes (`586q`, `790q`, `1,172q`, `1,580q` -> `129,148q`)
  2B. High-rate `{4,6}` (K=6) and `{4,5}` (K=6, K=10) hyperbolic square-tiling welded memories
"""

from collections import defaultdict
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_PATH = os.path.join(ROOT, "cluster_v12_results.txt")


def fit_slope(pts):
    """Linear least-squares slope of ln(mean_t) vs beta on low-T points."""
    if len(pts) < 2:
        return float("nan")
    xs = [b for b, _ in pts]
    ys = [math.log(t) for _, t in pts]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den > 0 else float("nan")


def main():
    if not os.path.exists(RES_PATH):
        print(f"Missing {RES_PATH}")
        sys.exit(1)

    rows = {}
    timeouts = []
    errors = []
    capped_rows = []
    bails_rows = []

    pat = re.compile(
        r"KMC dec=(\S+) beta=([\d.]+) N=(\d+) K=(\d+) trials=(\d+) "
        r"mean_t=([\deE+.-]+) sem_t=([\deE+.-]+) median_t=([\deE+.-]+) "
        r"mean_flips=([\deE+.-]+) bails=(\d+) capped=(\d+)"
    )

    with open(RES_PATH) as f:
        for raw in f:
            line = raw.strip()
            if not line or "|" not in line:
                continue
            tid, rhs = [p.strip() for p in line.split("|", 1)]
            if rhs.startswith("TIMEOUT"):
                timeouts.append(tid)
                continue
            if rhs.startswith("ERROR"):
                errors.append((tid, rhs))
                continue
            m = pat.search(rhs)
            if not m:
                continue
            dec, b, nq, k, tr, mt, st, med, mfl, bl, cap = m.groups()
            rec = {
                "tid": tid,
                "dec": dec,
                "beta": float(b),
                "N": int(nq),
                "K": int(k),
                "trials": int(tr),
                "mean_t": float(mt),
                "sem_t": float(st),
                "median_t": float(med),
                "mean_flips": float(mfl),
                "bails": int(bl),
                "capped": int(cap),
            }
            rows[tid] = rec
            if rec["capped"] > 0:
                capped_rows.append(rec)
            if rec["bails"] > 0:
                bails_rows.append(rec)

    print(f"Loaded {len(rows)} completed KMC tasks | timeouts={len(timeouts)} | errors={len(errors)} | bails>0={len(bails_rows)} | capped>0={len(capped_rows)}")
    if timeouts:
        print("  TIMEOUTS:", timeouts)
    if errors:
        print("  ERRORS:", errors)
    if capped_rows:
        for r in capped_rows:
            print(f"  CAPPED: {r['tid']} capped={r['capped']}/{r['trials']} mean_t={r['mean_t']:.3e} mean_flips={r['mean_flips']:.1f}")

    # Group by (code_sec, jsched)
    grouped = defaultdict(list)
    for tid, r in rows.items():
        parts = tid.split("__")
        if len(parts) >= 5:
            _, code_sec, jsched, dec, _ = parts
            grouped[(code_sec, jsched, dec)].append(r)

    print("\n=== SUMMARY BY SERIES ===")
    for key in sorted(grouped.keys()):
        code_sec, jsched, dec = key
        arr = sorted(grouped[key], key=lambda x: x["beta"])
        nq, k = arr[0]["N"], arr[0]["K"]
        tail = [(x["beta"], x["mean_t"]) for x in arr[-3:] if x["capped"] == 0]
        sl = fit_slope(tail)
        pts_s = ", ".join(
            f"b={x['beta']:.2f}: {x['mean_t']:.3e}±{x['sem_t']:.2e}" + (f"(cap={x['capped']})" if x["capped"] else "")
            for x in arr
        )
        print(f"{code_sec:28s} | {jsched:9s} | N={nq:7d} K={k:2d} | slope={sl:5.2f} | {pts_s}")

    # Compute quantum min(X_err, Z_err) for two-sector families
    fams = defaultdict(lambda: defaultdict(dict))
    for (code_sec, jsched, dec), arr in grouped.items():
        if code_sec.endswith("_X_err"):
            base = code_sec[:-6]
            sec = "X"
        elif code_sec.endswith("_Z_err"):
            base = code_sec[:-6]
            sec = "Z"
        else:
            continue
        for x in arr:
            fams[(base, jsched)][x["beta"]][sec] = x

    print("\n=== FULL TWO-SECTOR QUANTUM LIFETIMES tau_Q = min(tau_X, tau_Z) ===")
    for (base, jsched) in sorted(fams.keys()):
        bmap = fams[(base, jsched)]
        both = []
        for b in sorted(bmap.keys()):
            if "X" in bmap[b] and "Z" in bmap[b]:
                rx, rz = bmap[b]["X"], bmap[b]["Z"]
                tq = min(rx["mean_t"], rz["mean_t"])
                lim = "X" if rx["mean_t"] <= rz["mean_t"] else "Z"
                both.append((b, tq, lim, rx["N"], rx["K"], rx["mean_t"], rz["mean_t"], rx["capped"] + rz["capped"]))
        if both:
            nq, k = both[0][3], both[0][4]
            sl = fit_slope([(b, tq) for b, tq, *_ in both[-3:]])
            s_str = ", ".join(f"b={b:.2f}: {tq:.3e}({lim}; X={tx:.2e},Z={tz:.2e})" for b, tq, lim, _, _, tx, tz, _ in both)
            print(f"{base:22s} | {jsched:9s} | N={nq:7d} K={k:2d} ({nq/k:6.1f}q/log) | slope={sl:5.2f} | {s_str}")


if __name__ == "__main__":
    main()
