#!/usr/bin/env python3
"""
parse_v6_results.py
Parses `/Users/harshilavlani/Downloads/6.S980_gemini/cluster_v6_results.txt` into structured tables:
  1. External Ground-Truth Validation:
     - 3D Toric Code One-Sector Loop Memory (X_err exponential growth vs Z_err 1/N decay)
     - Gu-Caha Random CSS Layer Codes (exact comparison vs published PRL CSVs + n=11..19 scaling)
  2. Forward Flux Sampling (FFS) Validation (FFS vs KMC at overlapping beta) + Deep Low-T Crossings (beta up to 9.0)
  3. Hayden S8.3 Leaner Welded Gadget Percolation Threshold:
     - 4-rib (`L2r1`) vs 3-rib Checkerboard (`cb3rib`) vs 2-rib Percolating (`cb2rib`)
     - Up to s=5 (N=189,988!) and Proven 2-Bridge s=3 (N=22,160!)
  4. Z^3 Cubic-Grid Wire-Hill Rescue (<J> == 1.000 budget-neutral & bounded [1, 2.2]) up to BDL N=56,312
  5. Soft-Junction Passive Cooling vs Hard Quench & Simulated Annealing
"""

from collections import defaultdict
import math
import os
import re

ROOT = "/Users/harshilavlani/Downloads/6.S980_gemini"
IN_FILE = os.path.join(ROOT, "cluster_v6_results.txt")
OUT_FILE = os.path.join(ROOT, "parsed_v6_summary.txt")


def parse_kv(text):
    d = {}
    for tok in text.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            d[k] = v
    return d


def fit_slope(pts):
    if len(pts) < 2:
        return float("nan")
    xs = [p[0] for p in pts]
    ys = [math.log(max(p[1], 1e-30)) for p in pts]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den < 1e-12:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def best_est(kmc, ffs, tag, jm, dec, beta):
    """Prefer uncapped KMC at moderate beta; use FFS where KMC is absent or capped."""
    kr = kmc.get((tag, jm, dec, beta))
    fr = ffs.get((tag, jm, dec, beta))
    if kr and kr["capped"] == 0:
        return kr["N"], kr["mean_t"], "KMC"
    if fr:
        return fr["N"], fr["mean_t"], "FFS"
    if kr:
        return kr["N"], kr["mean_t"], f"KMC(c={kr['capped']})"
    return None, None, None


def main():
    if not os.path.exists(IN_FILE):
        print(f"Missing {IN_FILE}")
        return

    kmc, ffs, cool, errors = {}, {}, {}, []
    with open(IN_FILE) as f:
        for raw in f:
            ln = raw.strip()
            if not ln or "|" not in ln:
                continue
            tid, payload = [x.strip() for x in ln.split("|", 1)]
            if payload.startswith("TIMEOUT") or payload.startswith("ERROR") or "VALID_REPS=0" in payload:
                errors.append((tid, payload))
                continue
            kv = parse_kv(payload)
            parts = tid.split("__")
            mode = parts[0]
            if mode == "kmc":
                _, tag, j_mode, dec, b_str = parts
                beta = float(b_str[1:])
                kmc[(tag, j_mode, dec, beta)] = {
                    "N": int(kv["N"]),
                    "K": int(kv["K"]),
                    "mean_t": float(kv["mean_t"]),
                    "sem_t": float(kv["sem_t"]),
                    "median_t": float(kv["median_t"]),
                    "capped": int(kv["capped"]),
                    "trials": int(kv["trials"]),
                }
            elif mode == "ffs":
                _, tag, j_mode, dec, b_str = parts
                beta = float(b_str[1:])
                ffs[(tag, j_mode, dec, beta)] = {
                    "N": int(kv["N"]),
                    "K": int(kv["K"]),
                    "Phi0": float(kv["Phi0"]),
                    "Pfail": float(kv["Pfail"]),
                    "mean_t": float(kv["mean_t"]),
                    "sem_t": float(kv["sem_t"]),
                    "reps": int(kv["valid_reps"]),
                    "peak_E": float(kv["movie_peak_E"]),
                }
            elif mode == "cool":
                _, tag, proto, init_m, t_str = parts
                ttot = int(t_str[1:])
                cool[(tag, proto, init_m, ttot)] = {
                    "N": int(kv["N"]),
                    "syn": float(kv["syn_dens"]),
                    "mcl": float(kv["mean_max_cl"]),
                    "p0": float(kv["p_zero_syn"]),
                    "psurv": float(kv["p_log_survive"]),
                }

    out = []
    out.append(f"=== V6 AMBITIOUS CAMPAIGN SUMMARY (KMC={len(kmc)}, FFS={len(ffs)}, COOL={len(cool)}, ERR/TO={len(errors)}) ===\n")
    if errors:
        out.append("WARNINGS / TIMEOUTS:")
        for t, p in errors[:25]:
            out.append(f"  {t}: {p}")
        out.append("")

    # 1A. 3D Toric Code
    out.append("=== 1A. 3D TORIC CODE ONE-SECTOR BENCHMARK (Gu-Caha Appendix E.3) ===")
    for sec, dec, betas in [("X_err", "sweep_uf", [1.50, 1.75, 2.00, 2.25]), ("Z_err", "uf", [2.00, 3.00, 4.00])]:
        out.append(f"  Sector {sec} (dec={dec}):")
        for L in [3, 4, 5, 6, 7]:
            tag = f"toric3d_L{L}_{sec}"
            row_s, pts, N_val = [], [], None
            for b in betas:
                r = kmc.get((tag, "uniform", dec, b))
                if r:
                    N_val = r["N"]
                    row_s.append(f"b={b:.2f}: {r['mean_t']:.3e} (c={r['capped']})")
                    if r["capped"] == 0:
                        pts.append((b, r["mean_t"]))
            if row_s:
                sl = fit_slope(pts[-3:])
                out.append(f"    L={L} (N={N_val:4d}, Deff={sl:5.2f}): " + " | ".join(row_s))
    out.append("")

    # 1B. Gu-Caha Random Layer Codes
    out.append("=== 1B. GU-CAHA RANDOM LAYER CODES (PRL 136, 210601) ===")
    by_n = defaultdict(list)
    for (tag, jm, dec, b), r in sorted(kmc.items()):
        m = re.match(r"caha_n(\d+)_(.+)_X_err", tag)
        if m and jm == "uniform":
            by_n[(int(m.group(1)), b)].append((m.group(2), r["N"], r["mean_t"], r["sem_t"], r["capped"]))
    for n_in in [11, 13, 15, 17, 19]:
        parts_s, Ns = [], []
        for b in [6.00, 7.00, 8.00]:
            lst = by_n.get((n_in, b), [])
            if lst:
                Ns = [x[1] for x in lst]
                avg_t = sum(x[2] for x in lst) / len(lst)
                tot_c = sum(x[4] for x in lst)
                parts_s.append(f"b={b:.1f}(KMC): {avg_t:.3e}(c={tot_c})")
        for b in [8.00, 9.00]:
            flst = [fr["mean_t"] for (tg, jm, dc, bb), fr in ffs.items() if tg.startswith(f"caha_n{n_in}_") and bb == b]
            if flst:
                parts_s.append(f"b={b:.1f}(FFS): {sum(flst)/len(flst):.3e}")
        if parts_s:
            out.append(f"  n={n_in} (<N>={sum(Ns)//max(1,len(Ns)):5d}): " + " | ".join(parts_s))
    if (11, 6.00) in by_n:
        out.append("  Individual n=11 codes at beta=6.0:")
        for cname, Nv, mt, st, cp in by_n[(11, 6.00)]:
            out.append(f"    {cname:12s} (N={Nv}): {mt:.1f} +- {st:.1f} (c={cp})")
    out.append("")

    # 2A. KMC vs FFS Cross-Validation
    out.append("=== 2A. HEAD-TO-HEAD KMC vs FFS+COMMITTOR VALIDATION ===")
    for (tag, jm, dec, b), fr in sorted(ffs.items()):
        kr = kmc.get((tag, jm, dec, b))
        if kr and kr["capped"] == 0:
            ratio = fr["mean_t"] / kr["mean_t"]
            out.append(
                f"  {tag:28s} b={b:.2f} (N={fr['N']:6d}): KMC={kr['mean_t']:.3e}+-{kr['sem_t']:.2e} | "
                f"FFS={fr['mean_t']:.3e}+-{fr['sem_t']:.2e} | FFS/KMC={ratio:.2f}"
            )
    out.append("")

    # 2B. Haah Cubic Code Deep Low-T (KMC + FFS)
    out.append("=== 2B. HAAH 3D CUBIC CODE A (KMC + FFS UP TO BETA=8.0) ===")
    for L in [5, 7, 9, 11]:
        tag = f"cubic_L{L}_X_err"
        row_s, pts, Nv = [], [], None
        for b in [4.25, 5.00, 6.00, 7.00, 8.00]:
            n_q, val, src = best_est(kmc, ffs, tag, "uniform", "uf", b)
            if val is not None:
                Nv = n_q
                row_s.append(f"b={b:.2f}({src}): {val:.3e}")
                pts.append((b, val))
        if row_s:
            out.append(f"  L={L:2d} (N={Nv:4d}, Deff={fit_slope(pts[-3:]):5.2f}): " + " | ".join(row_s))
    out.append("")

    # 3. Welded Quantum Codes: Separate X & Z AND Worst-Case min(t_X, t_Z) Across Lean Gadgets & Scales!
    out.append("=== 3. WELDED QUANTUM HIERARCHIES (SEPARATE X/Z & min(t_X, t_Z), KMC + FFS) ===")
    families = [
        ("Boundaryless [[4,2,2]] Full 4-Rib", "rottor_L2r1", [2, 3, 4], [5.50, 6.00, 6.50, 7.00, 7.50, 8.00, 8.50]),
        ("Boundaryless [[4,2,2]] Lean 3-Rib (Checkerboard T)", "rottor_cb3rib", [2, 3, 4], [5.50, 6.00, 6.50, 7.00, 7.50, 8.00, 8.50]),
        ("Boundaryless [[4,2,2]] Percolating 2-Rib (Control)", "rottor_cb2rib", [2, 3, 4], [5.50, 6.00, 6.50, 7.00, 7.50]),
        ("Planar Full 4-Rib (Up to s=5, N=189,988!)", "sq_L2r1", [2, 3, 4, 5], [5.50, 6.00, 6.50, 7.00, 7.50, 8.00]),
        ("Planar Lean 3-Rib (Up to s=5!)", "sq_cb3rib", [2, 3, 4, 5], [5.50, 6.00, 6.50, 7.00, 7.50, 8.00]),
        ("Planar Percolating 2-Rib (Control)", "sq_cb2rib", [2, 3, 4], [5.50, 6.00, 6.50, 7.00]),
        ("Planar Proven 2-Bridge (L=3, r=2)", "sq_L3r2", [2, 3], [5.50, 6.00, 6.50, 7.00, 7.50, 8.00]),
    ]
    for title, pref, s_list, betas in families:
        out.append(f"  [{title}]")
        for s in s_list:
            tx_tag, tz_tag = f"{pref}_s{s}_X_err", f"{pref}_s{s}_Z_err"
            row_q, pts_q, Nv = [], [], None
            for b in betas:
                nx, vx, sx = best_est(kmc, ffs, tx_tag, "uniform", "uf", b)
                nz, vz, sz = best_est(kmc, ffs, tz_tag, "uniform", "uf", b)
                if vx is not None and vz is not None:
                    Nv = nx
                    vmin = min(vx, vz)
                    lim = "X" if vx <= vz else "Z"
                    row_q.append(f"b={b:.2f}: {vmin:.2e}[{lim},{sx}]")
                    pts_q.append((b, vmin))
            if row_q:
                deff = fit_slope(pts_q[-3:])
                out.append(f"    s={s} (N={Nv:6d}, Deff={deff:5.2f}): " + " | ".join(row_q))
    out.append("")

    # 4. Z^3 Cubic-Grid Wire-Hill & Pinning-Trap Rescue!
    out.append("=== 4. Z^3 CUBIC-GRID RESCUE (UNIFORM vs HILL vs PINNING-TRAP <J>==1.000 & BOUNDED [1, 2]) ===")
    for zname in ["sq_z3_3_5_r1", "sq_z3_4_6_r2", "tor2_z3_4_6_r2", "sq_bdl_8_8_r2"]:
        out.append(f"  [{zname}]")
        for jm in ["uniform", "hill_norm", "pin_norm", "toll_norm", "hill_bnd", "pin_bnd"]:
            row_z, pts_z, Nv = [], [], None
            for b in [5.50, 6.00, 6.50, 7.00]:
                krx = kmc.get((f"{zname}_X_err", jm, "uf", b))
                krz = kmc.get((f"{zname}_Z_err", jm, "uf", b))
                if krx and krz:
                    Nv = krx["N"]
                    vmin = min(krx["mean_t"], krz["mean_t"])
                    lim = "X" if krx["mean_t"] <= krz["mean_t"] else "Z"
                    cp = krx["capped"] + krz["capped"]
                    row_z.append(f"b={b:.2f}: {vmin:.2e}[{lim},c={cp}]")
                    if cp == 0:
                        pts_z.append((b, vmin))
            if row_z:
                out.append(f"    {jm:10s} (N={Nv:5d}, Deff={fit_slope(pts_z[-3:]):5.2f}): " + " | ".join(row_z))
    out.append("")

    # 5. Soft-Junction Cooling
    out.append("=== 5. PASSIVE COOLING & STATE PREPARATION (QUENCH vs ANNEAL vs SOFT-JUNCTION) ===")
    for cname in ["rottor_L2r1_s2", "rottor_L2r1_s3", "sq_L2r1_s3", "sq_L2r1_s4"]:
        for sec in ["X_err", "Z_err"]:
            tag = f"{cname}_{sec}"
            for init_m in ["hot", "noisy_0.12"]:
                for ttot in [50, 250, 1000]:
                    rows_c = []
                    Nv = None
                    for proto in ["quench", "anneal", "soft_junc"]:
                        cr = cool.get((tag, proto, init_m, ttot))
                        if cr:
                            Nv = cr["N"]
                            rows_c.append(f"{proto}: syn={cr['syn']:.4f}, p0={cr['p0']:.2f}, surv={cr['psurv']:.2f}")
                    if rows_c:
                        out.append(f"  {tag:22s} (N={Nv:5d}) {init_m:10s} t={ttot:4d} -> " + " || ".join(rows_c))

    text = "\n".join(out) + "\n"
    with open(OUT_FILE, "w") as f:
        f.write(text)
    print(text[:5000])


if __name__ == "__main__":
    main()
