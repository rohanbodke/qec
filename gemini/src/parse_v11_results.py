#!/usr/bin/env python3
"""
parse_v11_results.py
Parses `cluster_v11_results.txt` (alongside v8/v9/v10u) and prints concise quantitative summaries for:
  1. 100% Unified Uniform 2x2 3-Level Quantum Crossover (`sq_iso_L2r1` s=1,3,5: 19 -> 2,911 -> 423,559q)
  2. 5-Scale Uniform 2x2 Single-Sector Crossover (`sq_rx_iso` s=1..5: 19 -> 322,159q) + Local RG vs Global UF
  3. Live Alternating (X, Z) Quantum Syndrome Zoom-Out Staircase (`rottor_iso_L2r1_s4` 94,774q & `sq_iso_L2r1_s5` 423,559q)
  4. Multi-Scale Cooling-Time Scaling Law t_cool(N) across 4 scales on 3 families (`rottor_iso_L2r1`, `sq_iso_L2r1`, `sq_rx_iso`)
"""

import math
import os

WS = "/Users/harshilavlani/Downloads/6.S980_gemini"


def parse_kv(text):
    d = {}
    for tok in text.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            d[k] = v
    return d


def ols_slope(xs, ys):
    pts = [(x, math.log(y)) for x, y in zip(xs, ys) if y is not None and y > 0]
    if len(pts) < 2:
        return None
    mx = sum(x for x, _ in pts) / len(pts)
    my = sum(y for _, y in pts) / len(pts)
    den = sum((x - mx) ** 2 for x, _ in pts)
    return sum((x - mx) * (y - my) for x, y in pts) / den if den > 1e-12 else None


def load_v11():
    kmc, rg, cool = {}, {}, {}
    errs, timeouts, capped_runs = [], [], []
    for fn in (
        "cluster_v8_results.txt",
        "cluster_v9_results.txt",
        "cluster_v9b_results.txt",
        "cluster_v10u_results.txt",
        "cluster_v11_results.txt",
    ):
        fp = os.path.join(WS, fn)
        if not os.path.exists(fp):
            continue
        with open(fp) as f:
            for raw in f:
                ln = raw.strip()
                if not ln or "|" not in ln:
                    continue
                tid, payload = [x.strip() for x in ln.split("|", 1)]
                if payload.startswith("TIMEOUT"):
                    if fn == "cluster_v11_results.txt":
                        timeouts.append(tid)
                    continue
                if payload.startswith("ERROR"):
                    if fn == "cluster_v11_results.txt":
                        errs.append((tid, payload))
                    continue
                kv = parse_kv(payload)
                parts = tid.split("__")
                if parts[0] == "kmc":
                    _, tag, jm, dec, bs = parts
                    b = float(bs[1:])
                    cap = int(kv.get("capped", "0"))
                    bails = int(kv.get("bails", "0"))
                    if bails > 0:
                        continue
                    if cap == 0:
                        kmc[(tag, dec, b)] = (int(kv["N"]), float(kv["mean_t"]), float(kv.get("sem_t", "0")))
                    elif fn == "cluster_v11_results.txt":
                        capped_runs.append((tid, cap, float(kv["mean_t"])))
                elif parts[0] in ("rg", "rgflow"):
                    _, rg_nm, bs = parts
                    b = float(bs[1:])
                    S_max = int(kv["S_max"])
                    rhos = [float(kv[f"rho_{s}"]) for s in range(S_max + 1)]
                    rg[(rg_nm, b)] = rhos
                elif parts[0] == "cool":
                    _, tag, proto, init_m, ts = parts
                    tt = float(ts[1:])
                    cool[(tag, proto, init_m, tt)] = {
                        "N": int(kv["N"]),
                        "K": int(kv["K"]),
                        "syn": float(kv["syn_dens"]),
                        "mcl": float(kv["mean_max_cl"]),
                        "p0": float(kv["p_zero_syn"]),
                        "pheal": float(kv["p_exact_heal"]),
                        "psurv": float(kv["p_log_survive"]),
                    }
    return kmc, rg, cool, errs, timeouts, capped_runs


def interp_tcool(t_pts, y_pts, target, decreasing=True):
    """Interpolate log10(t) where y crosses `target`."""
    for i in range(len(t_pts) - 1):
        t0, t1 = t_pts[i], t_pts[i + 1]
        y0, y1 = y_pts[i], y_pts[i + 1]
        if decreasing:
            if y0 >= target >= y1 and y0 > y1:
                f = (y0 - target) / (y0 - y1)
                return 10.0 ** (math.log10(t0) + f * (math.log10(t1) - math.log10(t0)))
        else:
            if y0 <= target <= y1 and y1 > y0:
                f = (target - y0) / (y1 - y0)
                return 10.0 ** (math.log10(t0) + f * (math.log10(t1) - math.log10(t0)))
    if decreasing and y_pts and y_pts[0] < target:
        return t_pts[0]
    if not decreasing and y_pts and y_pts[0] > target:
        return t_pts[0]
    return None


def main():
    kmc, rg, cool, errs, timeouts, capped_runs = load_v11()
    print(f"V11 Status: errors={len(errs)}, timeouts={len(timeouts)}, capped={len(capped_runs)}")
    for t, e in errs[:10]:
        print("  ERR:", t, e)
    for t in timeouts[:10]:
        print("  TIMEOUT:", t)
    for t, c, mt in capped_runs[:10]:
        print(f"  CAPPED: {t} (capped={c}, mean_t={mt:.3e})")

    print("\n=== 1A. UNIFORM 2x2 3-LEVEL QUANTUM (`sq_iso_L2r1` s=1, 3, 5; Z_err) ===")
    for s in (1, 3, 5):
        pts = sorted((b, kmc[(f"sq_iso_L2r1_s{s}_Z_err", "uf", b)][1]) for (tg, dc, b) in kmc if tg == f"sq_iso_L2r1_s{s}_Z_err" and dc == "uf")
        sl = ols_slope([b for b, _ in pts if b >= 6.0], [y for b, y in pts if b >= 6.0])
        print(f"  s={s} (slope>=6.0: {sl}): " + ", ".join(f"b={b:.2f}:{y:.3e}" for b, y in pts))

    print("\n=== 1B. 5-SCALE SINGLE-SECTOR (`sq_rx_iso` s=1..5) UF vs RG_DEC ===")
    for dec in ("uf", "rg_dec"):
        print(f"  Decoder = {dec}:")
        for s in (1, 2, 3, 4, 5):
            pts = sorted((b, kmc[(f"sq_rx_iso_s{s}_X_err", dec, b)][1]) for (tg, dc, b) in kmc if tg == f"sq_rx_iso_s{s}_X_err" and dc == dec)
            print(f"    s={s}: " + ", ".join(f"{b:.2f}:{y:.2e}" for b, y in pts))

    print("\n=== 1B-2. TORUS QUANTUM (`rottor_iso_L2r1` s=2, 4) UF vs RG_DEC ===")
    for dec in ("uf", "rg_dec"):
        for s in (2, 4):
            pts = sorted((b, kmc[(f"rottor_iso_L2r1_s{s}_X_err", dec, b)][1]) for (tg, dc, b) in kmc if tg == f"rottor_iso_L2r1_s{s}_X_err" and dc == dec)
            print(f"  dec={dec} s={s}: " + ", ".join(f"{b:.2f}:{y:.2e}" for b, y in pts))

    print("\n=== 1C. ALTERNATING (X, Z) QUANTUM SYNDROME ZOOM-OUT (`RGV2`) ===")
    for nm in ("rottor_iso_L2r1_s4", "sq_iso_L2r1_s5"):
        bs = sorted(b for (rn, b) in rg if rn == nm)
        print(f"  {nm}:")
        for b in bs:
            rh = rg[(nm, b)]
            print(f"    b={b:.2f}: " + " -> ".join(f"s{s}={rh[s]:.3e}" for s in range(len(rh) - 1, -1, -1)))

    print("\n=== 2. MULTI-SCALE COOLING SCALING LAW t_cool(N) ===")
    for fam in ("rottor_iso_L2r1", "sq_iso_L2r1", "sq_rx_iso"):
        print(f"  Family = {fam}:")
        for proto in ("anneal", "quench"):
            print(f"    proto = {proto}:")
            ns, tc_syn, tc_p0 = [], [], []
            for s in (1, 2, 3, 4):
                tag = f"{fam}_s{s}_X_err"
                ts = sorted(tt for (tg, pr, im, tt) in cool if tg == tag and pr == proto and im == "hot")
                if not ts:
                    continue
                N = cool[(tag, proto, "hot", ts[0])]["N"]
                syns = [cool[(tag, proto, "hot", tt)]["syn"] for tt in ts]
                p0s = [cool[(tag, proto, "hot", tt)]["p0"] for tt in ts]
                t_s = interp_tcool(ts, syns, 0.002, decreasing=True)
                t_p = interp_tcool(ts, p0s, 0.50, decreasing=False)
                ns.append(N); tc_syn.append(t_s); tc_p0.append(t_p)
                print(f"      s={s} (N={N:6d}): t_0.2%={t_s} | t_p50={t_p} | syn={[round(x,5) for x in syns]} | p0={[round(x,2) for x in p0s]}")
            valid_s = [(math.log(n), math.log(tc)) for n, tc in zip(ns, tc_syn) if tc is not None]
            if len(valid_s) >= 2:
                gam = ols_slope([x for x, _ in valid_s], [math.exp(y) for _, y in valid_s])
                print(f"      => Power-law exponent gamma (syn <= 0.2%): t_cool ~ N^{gam:.3f}")


if __name__ == "__main__":
    main()
