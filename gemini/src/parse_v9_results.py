#!/usr/bin/env python3
"""
parse_v9_results.py
Parses `cluster_v8_results.txt` + `cluster_v9_results.txt` + `cluster_v9b_results.txt` into `parsed_v9_summary.txt`:
  1. 6-Level (s = 0..5, up to 322,159 qubits!) 3D 2x2 Welded Scaling & Power-Law Exponent Fit (Delta_eff ~ N^alpha).
  2. Live 6-Level Syndrome-Only Thermal RG Defect-Density Flow (`s = 5 -> 4 -> 3 -> 2 -> 1 -> 0`) on 322,159 & 288,344 qubits.
  3. 5-Step Full Two-Sector Quantum Welded Ladder (`s = 1 -> 3 -> 5`, up to 189,988 planar & 379,976 torus qubits).
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
        return float("nan")
    mx = sum(x for x, _ in pts) / len(pts)
    my = sum(y for _, y in pts) / len(pts)
    den = sum((x - mx) ** 2 for x, _ in pts)
    return sum((x - mx) * (y - my) for x, y in pts) / den if den > 1e-12 else float("nan")


def power_law_fit(ns, ds):
    pts = [(math.log(n), math.log(d)) for n, d in zip(ns, ds) if n > 0 and not math.isnan(d) and d > 0]
    if len(pts) < 2:
        return float("nan"), float("nan")
    mx = sum(x for x, _ in pts) / len(pts)
    my = sum(y for _, y in pts) / len(pts)
    den = sum((x - mx) ** 2 for x, _ in pts)
    alpha = sum((x - mx) * (y - my) for x, y in pts) / den
    ss_tot = sum((y - my) ** 2 for _, y in pts)
    ss_res = sum((y - (my + alpha * (x - mx))) ** 2 for x, y in pts)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    return alpha, r2


def main():
    kmc = {}  # (tag, dec, beta) -> (N, mean_t, sem_t, capped)
    rg = {}
    errs = []

    for fname in ("cluster_v8_results.txt", "cluster_v9_results.txt", "cluster_v9b_results.txt", "cluster_v9d_results.txt"):
        fpath = os.path.join(WS, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            for raw in f:
                ln = raw.strip()
                if not ln or "|" not in ln:
                    continue
                tid, payload = [x.strip() for x in ln.split("|", 1)]
                if payload.startswith(("TIMEOUT", "ERROR")) or "VALID_REPS=0" in payload:
                    if fname in ("cluster_v9_results.txt", "cluster_v9b_results.txt"):
                        errs.append(f"[{fname}] {tid}: {payload[:80]}")
                    continue
                kv = parse_kv(payload)
                parts = tid.split("__")
                if parts[0] == "kmc":
                    if int(kv.get("bails", "0")) > 0:
                        continue
                    _, tag, jm, dec, bs = parts
                    kmc[(tag, dec, float(bs[1:]))] = (
                        int(kv["N"]),
                        float(kv["mean_t"]),
                        float(kv["sem_t"]),
                        int(kv["capped"]),
                    )
                elif parts[0] == "rg":
                    _, rg_nm, bs = parts
                    S_max = int(kv["S_max"])
                    N_top = int(kv["N_top"])
                    rhos = [float(kv[f"rho_{s}"]) for s in range(S_max + 1)]
                    rg[(rg_nm, float(bs[1:]))] = (S_max, N_top, rhos)

    def uncapped(tag, dec, b):
        if (tag, dec, b) in kmc and kmc[(tag, dec, b)][3] == 0:
            return kmc[(tag, dec, b)]
        return None

    out = []
    out.append(f"=== V9/V9B QUARTER-MILLION-QUBIT & LIVE RG FLOW SUMMARY (KMC={len(kmc)}, RG={len(rg)}, TO/ERR={len(errs)}) ===")

    all_bs = [1.00, 1.25, 1.50, 1.60, 1.75, 1.90, 2.00, 2.08, 2.10, 2.15, 2.20, 2.22, 2.25, 2.28, 2.35, 2.40, 2.50, 2.75, 3.00]
    for dec in ("uf", "maj_uf", "gain_uf"):
        for fam, title, slope_win in [
            ("sq_rx_iso", f"[{dec}] Uniform 2x2 Planar 6-Level (s=0..5, up to 322,159 qubits)", [1.75, 1.90, 2.00, 2.08, 2.10, 2.15, 2.22, 2.25]),
            ("rottor_rx", f"[{dec}] Thin-Bridge 2x2 Torus 6-Level (s=0..5, up to 288,344 qubits)", [1.75, 1.90, 2.00, 2.08, 2.10, 2.15, 2.22, 2.25]),
            ("sq_rx",     f"[{dec}] Thin-Bridge 2x2 Planar 6-Level (s=0..5, up to 144,172 qubits)", [1.75, 1.90, 2.00, 2.08, 2.10, 2.15, 2.22, 2.25]),
        ]:
            out.append(f"\n=== 1. {title} ===")
            fit_N, fit_D = [], []
            for s in range(6):
                tag = f"{fam}_s{s}_X_err"
                pts, sx, sy = [], [], []
                N_val = None
                for b in all_bs:
                    r = uncapped(tag, dec, b)
                    if r:
                        N_val, mt, st, _ = r
                        pts.append(f"b={b:.2f}:{mt:.3e}")
                        if 2.05 <= b <= 2.25:
                            sx.append(b); sy.append(mt)
                sl = ols_slope(sx, sy)
                if N_val is not None:
                    fit_N.append(N_val); fit_D.append(sl)
                if pts:
                    out.append(f"  s={s} (N={N_val}, Deff[2.05..2.25]={sl:.2f}): " + " | ".join(pts))

            if len(fit_N) >= 4:
                alpha_all, r2_all = power_law_fit(fit_N[1:], fit_D[1:])
                out.append(f"  ==> Empirical Power-Law Fit (nontrivial scales): Delta_eff(N) ~ N^{alpha_all:.3f} (R^2 = {r2_all:.4f})")

    # 2. Live 6-Level Syndrome-Only Thermal RG Flow (`s = 5 -> 4 -> 3 -> 2 -> 1 -> 0`)
    for rg_nm, title in [
        ("sq_rx_iso_s5", "LIVE 6-LEVEL RG FLOW: Uniform 2x2 Planar (N = 322,159 qubits)"),
        ("rottor_rx_s5", "LIVE 6-LEVEL RG FLOW: Boundaryless Torus (N = 288,344 qubits)"),
    ]:
        out.append(f"\n=== 2. {title} ===")
        for b in [0.80, 1.10, 1.35, 1.60, 1.75, 1.86, 2.00, 2.15, 2.35, 2.65]:
            if (rg_nm, b) in rg:
                S_max, N_top, rhos = rg[(rg_nm, b)]
                chain = " -> ".join(f"s={s}:{rhos[s]:.4e}" for s in range(S_max, -1, -1))
                out.append(f"  beta={b:.2f}: {chain}")

    # 3. 5-Step Full Two-Sector Quantum Welded Ladder (`s = 1 -> 3 -> 5`, up to 189,988 & 379,976 qubits!)
    q_bs = [4.50, 4.75, 5.00, 5.25, 5.50, 5.75, 6.00, 6.12, 6.15, 6.18, 6.25, 6.35, 6.50]
    for dec in ("uf", "maj_uf", "gain_uf"):
        for fam, title in [
            ("sq_L2r1",     f"[{dec}] 5-STEP FULL QUANTUM LADDER: Planar (19 -> 1,942 -> 189,988 qubits)"),
            ("rottor_L2r1", f"[{dec}] 5-STEP FULL QUANTUM LADDER: Boundaryless Torus (38 -> 3,884 -> 379,976 qubits)"),
        ]:
            out.append(f"\n=== 3. {title} ===")
            for s in (1, 3, 5):
                for sec in ("Z_err", "X_err"):
                    tag = f"{fam}_s{s}_{sec}"
                    pts, sx, sy = [], [], []
                    N_val = None
                    for b in q_bs:
                        if (tag, dec, b) in kmc:
                            n, mt, st, cap = kmc[(tag, dec, b)]
                            N_val = n
                            cs = "" if cap == 0 else f"(cap={cap})"
                            pts.append(f"b={b:.2f}:{mt:.3e}{cs}")
                            if cap == 0 and 5.50 <= b <= 6.50:
                                sx.append(b); sy.append(mt)
                    sl = ols_slope(sx, sy)
                    if pts:
                        out.append(f"  s={s} {sec} (N={N_val}, Deff[5.5..6.5]={sl:.2f}): " + " | ".join(pts))

    txt = "\n".join(out) + "\n"
    with open(os.path.join(WS, "parsed_v9_summary.txt"), "w") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
