#!/usr/bin/env python3
"""
parse_v8_results.py
Parses `cluster_v8_results.txt` + `exact_soundness_v8_report.txt` into `parsed_v8_summary.txt`:
  1. 5-Scale 3D Welded Family (`s = 0, 1, 2, 3, 4` up to N = 30,507):
     - Consecutive crossover inverse temperatures beta_cross(s -> s+1) for s = 0->1, 1->2, 2->3, 3->4!
     - Fixed-temperature lifetime growth across all 5 scales!
     - Effective energy barrier slopes Delta_eff(s) across all 5 scales!
  2. Provably-Contractive (`c_cycle = 2/5`) Uniform 2x2 Two-Cycle Quantum Crossover (`N = 323` vs `N = 47,387`).
  3. Extended Cold Boundaryless Torus Crossover (`N = 508` vs `N = 50,296` up to beta = 6.75 / 7.00).
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


def main():
    kmc, ffs = {}, {}
    errs = []
    with open(os.path.join(WS, "cluster_v8_results.txt")) as f:
        for raw in f:
            ln = raw.strip()
            if not ln or "|" not in ln:
                continue
            tid, payload = [x.strip() for x in ln.split("|", 1)]
            if payload.startswith(("TIMEOUT", "ERROR")) or "VALID_REPS=0" in payload:
                errs.append(f"{tid}: {payload}")
                continue
            kv = parse_kv(payload)
            parts = tid.split("__")
            if parts[0] == "kmc":
                _, tag, jm, dec, bs = parts
                kmc[(tag, float(bs[1:]))] = (
                    int(kv["N"]),
                    float(kv["mean_t"]),
                    float(kv["sem_t"]),
                    int(kv["capped"]),
                )
            elif parts[0] == "ffs":
                _, tag, jm, dec, bs = parts
                vr, nr = [int(x) for x in kv["valid_reps"].split("/")]
                ffs[(tag, float(bs[1:]))] = (
                    int(kv["N"]),
                    float(kv["mean_t"]),
                    float(kv["sem_t"]),
                    vr,
                    nr,
                )

    def best_val(tag, b, min_reps=4):
        if (tag, b) in kmc and kmc[(tag, b)][3] == 0:
            n, mt, st, _ = kmc[(tag, b)]
            return n, mt, st, "KMC"
        if (tag, b) in ffs and ffs[(tag, b)][3] >= min_reps:
            n, mt, st, vr, nr = ffs[(tag, b)]
            return n, mt, st, f"FFS({vr}/{nr})"
        return None

    out = []
    out.append(f"=== V8 RIGOROUS MULTI-SCALE SUMMARY (KMC={len(kmc)}, FFS={len(ffs)}, TO/ERR={len(errs)}) ===")
    if errs:
        out.append("WARNINGS / TIMEOUTS:")
        for e in errs:
            out.append("  " + e)

    # 1. 5-Scale Pure R_X^s Families
    for fam, title, b_list, slope_bs in [
        ("rottor_rx", "Torus 5-Scale 2x2 Cross (K=2)", [1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.40, 2.50, 2.75, 3.00], [1.75, 2.00, 2.25]),
        ("sq_rx",     "Planar Thin-Bridge 5-Scale 2x2 Cross (K=1)", [1.25, 1.50, 1.75, 2.00, 2.25, 2.40, 2.50, 2.65, 2.75, 3.00], [1.75, 2.00, 2.25]),
        ("sq_rx_iso", "Planar Uniform 5-Scale 2x2 Cross (K=1)", [1.25, 1.50, 1.75, 2.00, 2.25, 2.40, 2.50, 2.65, 2.75, 3.00], [1.75, 2.00, 2.25]),
    ]:
        out.append(f"\n=== 1. {title} ===")
        for s in range(5):
            tag = f"{fam}_s{s}_X_err"
            pts = []
            s_xs, s_ys = [], []
            N_val = None
            for b in b_list:
                r = best_val(tag, b)
                if r:
                    N_val, mt, st, src = r
                    pts.append(f"b={b:.2f}:{mt:.3e}[{src}]")
                    if b in slope_bs:
                        s_xs.append(b)
                        s_ys.append(mt)
            sl = ols_slope(s_xs, s_ys)
            out.append(f"  s={s} (N={N_val}, Deff{slope_bs}={sl:.2f}): " + " | ".join(pts))

        # Estimate consecutive crossing inverse temperatures beta_cross(s -> s+1):
        out.append("  Consecutive Scale Crossings:")
        for s in range(4):
            t0, t1 = f"{fam}_s{s}_X_err", f"{fam}_s{s+1}_X_err"
            common_bs = [b for b in b_list if best_val(t0, b) and best_val(t1, b)]
            cross_b = None
            for i in range(len(common_bs) - 1):
                bA, bB = common_bs[i], common_bs[i + 1]
                dA = math.log(best_val(t1, bA)[1]) - math.log(best_val(t0, bA)[1])
                dB = math.log(best_val(t1, bB)[1]) - math.log(best_val(t0, bB)[1])
                if dA <= 0 <= dB and (dB - dA) > 1e-9:
                    cross_b = bA + (0.0 - dA) / (dB - dA) * (bB - bA)
                    break
            if cross_b is not None:
                out.append(f"    s={s} -> s={s+1}: beta_cross ≈ {cross_b:.2f}")
            else:
                # Check if s+1 already above s at lowest common beta!
                if common_bs:
                    b0 = common_bs[0]
                    r0, r1 = best_val(t0, b0)[1], best_val(t1, b0)[1]
                    out.append(f"    s={s} -> s={s+1}: at b={b0:.2f}, ratio t_{s+1}/t_{s} = {r1/r0:.2f}x")

    # 2. Uniform 2x2 Two-Cycle Quantum Crossover (`sq_iso_L2r1_s2` vs `sq_iso_L2r1_s4`)
    out.append("\n=== 2. PROVABLY CONTRACTIVE UNIFORM 2x2 TWO-CYCLE CROSSOVER (N=323 vs N=47,387) ===")
    for s in (2, 4):
        for sec in ("X_err", "Z_err"):
            tag = f"sq_iso_L2r1_s{s}_{sec}"
            pts = []
            xs, ys = [], []
            N_val = None
            for b in [5.00, 5.50, 6.00, 6.25, 6.50, 6.65, 6.75, 7.00]:
                r = best_val(tag, b, min_reps=3)
                if r:
                    N_val, mt, st, src = r
                    pts.append(f"b={b:.2f}:{mt:.3e}[{src}]")
                    if 5.50 <= b <= 6.65:
                        xs.append(b); ys.append(mt)
            sl = ols_slope(xs, ys)
            out.append(f"  s={s} {sec} (N={N_val}, Deff[5.5..6.65]={sl:.2f}): " + " | ".join(pts))

    # 3. Extended Cold Boundaryless Torus (`rottor_L2r1_s2` vs `rottor_L2r1_s4`)
    out.append("\n=== 3. EXTENDED COLD TORUS CYCLE 1 vs CYCLE 2 (b = 6.65, 6.75, 7.00) ===")
    for s in (2, 4):
        tag = f"rottor_L2r1_s{s}_X_err"
        pts = []
        for b in [6.65, 6.75, 7.00]:
            r = best_val(tag, b, min_reps=3)
            if r:
                pts.append(f"b={b:.2f}:{r[1]:.3e}+-{r[2]:.1e}[{r[3]}]")
        out.append(f"  rottor s={s}: " + " | ".join(pts))

    summary_txt = "\n".join(out) + "\n"
    with open(os.path.join(WS, "parsed_v8_summary.txt"), "w") as f:
        f.write(summary_txt)
    print(summary_txt)


if __name__ == "__main__":
    main()
