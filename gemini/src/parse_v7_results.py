#!/usr/bin/env python3
"""
parse_v7_results.py
Zero-trust parser for `cluster_v7_results.txt`:
  - Computes matched-window OLS slopes `Delta_eff` on identical beta grids (no unfair 2-point splices!)
  - Reports exact `FFS / KMC` overlap ratios with combined z-scores
  - Reports apples-to-apples `[[4,2,2]]` `2 -> 1` (`N=1,172`) vs `1 -> 2` (`N=1,292`) vs `1 -> 1` (`N=508`) vs `2 -> 2` (`N=2,960`)
  - Reports verified commuting `cb3rib` / `cb2rib` `O(1/N)` collapse times (`t_next = 0.0`)
  - Reports unified single-engine `Z^3` ratios (`(3,3)` vs `(4,6)` vs `(8,8)`, `pin_norm`, `hill_norm`, `hill_bnd`)
  - Reports 3D quantum hot cooling (`quench`, `anneal`, `soft_seam`) AND sub-threshold autonomous codeword self-healing (`p0 = 0.01..0.12`)
"""

import math
import os

WS = "/Users/harshilavlani/Downloads/6.S980_gemini"
IN_PATH = os.path.join(WS, "cluster_v7_results.txt")
OUT_PATH = os.path.join(WS, "parsed_v7_summary.txt")


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
    num = sum((x - mx) * (y - my) for x, y in pts)
    den = sum((x - mx) ** 2 for x, _ in pts)
    return num / den if den > 1e-12 else float("nan")


def main():
    if not os.path.exists(IN_PATH):
        print(f"Missing {IN_PATH}")
        return

    kmc, ffs, cool, errs = {}, {}, {}, []
    with open(IN_PATH) as f:
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
            mode = parts[0]
            if mode == "kmc":
                _, tag, jm, dec, bs = parts
                kmc[(tag, jm, dec, float(bs[1:]))] = {
                    "N": int(kv["N"]),
                    "K": int(kv["K"]),
                    "trials": int(kv["trials"]),
                    "mean_t": float(kv["mean_t"]),
                    "sem_t": float(kv["sem_t"]),
                    "capped": int(kv["capped"]),
                }
            elif mode == "ffs":
                _, tag, jm, dec, bs = parts
                vr, nr = [int(x) for x in kv["valid_reps"].split("/")]
                ffs[(tag, jm, dec, float(bs[1:]))] = {
                    "N": int(kv["N"]),
                    "K": int(kv["K"]),
                    "mean_t": float(kv["mean_t"]),
                    "sem_t": float(kv["sem_t"]),
                    "valid_reps": vr,
                    "num_reps": nr,
                    "Pfail": float(kv["Pfail"]),
                    "Phi0": float(kv["Phi0"]),
                }
            elif mode == "cool":
                _, tag, proto, init_m, ts = parts
                cool[(tag, proto, init_m, float(ts[1:]))] = {
                    "N": int(kv["N"]),
                    "K": int(kv["K"]),
                    "syn": float(kv["syn_dens"]),
                    "p0": float(kv["p_zero_syn"]),
                    "pheal": float(kv["p_exact_heal"]),
                    "psurv": float(kv["p_log_survive"]),
                }

    out = []
    out.append(f"=== V7 ZERO-TRUST RESCUE SUMMARY (KMC={len(kmc)}, FFS={len(ffs)}, COOL={len(cool)}, ERR/TO={len(errs)}) ===")
    if errs:
        out.append("WARNINGS / TIMEOUTS:")
        for e in errs:
            out.append(f"  {e}")

    # 1. Head-to-head FFS vs KMC validation
    out.append("\n=== 1. EXACT GROUND-TO-GROUND FFS vs DIRECT KMC VALIDATION ===")
    for (tag, jm, dec, b), kr in sorted(kmc.items()):
        if kr["capped"] > 0:
            continue
        fr = ffs.get((tag, jm, dec, b))
        if fr:
            ratio = fr["mean_t"] / kr["mean_t"]
            comb_sem = math.hypot(kr["sem_t"], fr["sem_t"])
            z = abs(fr["mean_t"] - kr["mean_t"]) / comb_sem if comb_sem > 0 else 0.0
            out.append(
                f"  {tag:28s} b={b:.2f} (N={kr['N']:5d}): KMC={kr['mean_t']:.3e}+-{kr['sem_t']:.2e} | "
                f"FFS={fr['mean_t']:.3e}+-{fr['sem_t']:.2e} (reps={fr['valid_reps']}/{fr['num_reps']}) | "
                f"FFS/KMC={ratio:.2f} ({z:.2f} sigma)"
            )

    # 2. Cubic Code L=5, 7 KMC + Rescued FFS
    out.append("\n=== 2. HAAH 3D CUBIC CODE A (RESCUED FFS & KMC) ===")
    for L in [5, 7]:
        tag = f"cubic_L{L}_X_err"
        bs_all = [4.25, 4.50, 4.75, 5.00, 5.50, 6.00, 6.50, 7.00, 7.50, 8.00]
        pts_ffs = [(b, ffs[(tag, "uniform", "uf", b)]["mean_t"]) for b in bs_all if (tag, "uniform", "uf", b) in ffs]
        deff_ffs = ols_slope([x for x, _ in pts_ffs], [y for _, y in pts_ffs])
        segs = []
        for b in bs_all:
            kr = kmc.get((tag, "uniform", "uf", b))
            fr = ffs.get((tag, "uniform", "uf", b))
            if kr and kr["capped"] == 0:
                segs.append(f"b={b:.2f}(KMC):{kr['mean_t']:.3e}")
            if fr:
                segs.append(f"b={b:.2f}(FFS,{fr['valid_reps']}/{fr['num_reps']}):{fr['mean_t']:.3e}+-{fr['sem_t']:.1e}")
        out.append(f"  L={L} (Deff_FFS={deff_ffs:.2f}): " + " | ".join(segs))

    # 3. Matched-Window Two-Cycle Crossings & Sector Asymmetry
    out.append("\n=== 3. TWO-CYCLE CROSSINGS (MATCHED WINDOWS) & X/Z ASYMMETRY ===")
    def min_sector(pref_s, b):
        tx_k = kmc.get((f"{pref_s}_X_err", "uniform", "uf", b))
        tz_k = kmc.get((f"{pref_s}_Z_err", "uniform", "uf", b))
        tx_f = ffs.get((f"{pref_s}_X_err", "uniform", "uf", b))
        tz_f = ffs.get((f"{pref_s}_Z_err", "uniform", "uf", b))
        vx, sx = (tx_k["mean_t"], "KMC") if (tx_k and tx_k["capped"] == 0) else ((tx_f["mean_t"], f"FFS({tx_f['valid_reps']}/{tx_f['num_reps']})") if tx_f else (None, ""))
        vz, sz = (tz_k["mean_t"], "KMC") if (tz_k and tz_k["capped"] == 0) else ((tz_f["mean_t"], f"FFS({tz_f['valid_reps']}/{tz_f['num_reps']})") if tz_f else (None, ""))
        # At even steps (s=2, s=4) on unpruned codes, Z_err >> 50 * X_err, so X_err strictly determines min(t_X, t_Z)
        if vx is not None and vz is None and "cb" not in pref_s and (pref_s.endswith("_s2") or pref_s.endswith("_s4")):
            return vx, f"X,{sx}"
        if vx is not None and vz is not None:
            return (vx, f"X,{sx}") if vx <= vz else (vz, f"Z,{sz}")
        return None, ""

    for fam, s_list, m_win in [
        ("rottor_L2r1", [2, 4], [5.50, 6.00, 6.25, 6.50]),
        ("sq_L2r1",     [2, 4], [5.50, 6.00, 6.25, 6.50]),
    ]:
        out.append(f"  [{fam} — Matched slope window {m_win}]")
        for s in s_list:
            ps = f"{fam}_s{s}"
            m_ys = [min_sector(ps, b)[0] for b in m_win]
            deff_m = ols_slope(m_win, m_ys)
            segs = []
            for b in [5.00, 5.50, 6.00, 6.25, 6.50, 6.75, 7.00, 7.50, 8.00]:
                v, src = min_sector(ps, b)
                if v is not None:
                    segs.append(f"b={b:.2f}:{v:.3e}[{src}]")
            out.append(f"    s={s} (Matched Deff={deff_m:.2f}): " + " | ".join(segs))

    out.append("  [rottor_L2r1_s2 Separate X vs Z Sector Asymmetry]")
    for b in [5.50, 6.00, 6.50, 7.00, 7.50, 8.00]:
        kx, fx = kmc.get(("rottor_L2r1_s2_X_err", "uniform", "uf", b)), ffs.get(("rottor_L2r1_s2_X_err", "uniform", "uf", b))
        kz, fz = kmc.get(("rottor_L2r1_s2_Z_err", "uniform", "uf", b)), ffs.get(("rottor_L2r1_s2_Z_err", "uniform", "uf", b))
        vx, sx = (kx["mean_t"], f"KMC+-{kx['sem_t']:.1e}") if (kx and kx["capped"] == 0) else ((fx["mean_t"], f"FFS({fx['valid_reps']}/{fx['num_reps']})+-{fx['sem_t']:.1e}") if fx else (None, ""))
        vz, sz = (kz["mean_t"], f"KMC+-{kz['sem_t']:.1e}") if (kz and kz["capped"] == 0) else ((fz["mean_t"], f"FFS({fz['valid_reps']}/{fz['num_reps']})+-{fz['sem_t']:.1e}") if fz else (None, ""))
        if vx and vz:
            out.append(f"    b={b:.2f}: t_X={vx:.3e} ({sx}) | t_Z={vz:.3e} ({sz}) | Z/X = {vz/vx:,.1f}x")

    # 4. Apples-to-apples `[[4,2,2]]` Hybrid `2->1` vs `1->2` vs `1->1` vs `2->2`
    out.append("\n=== 4. APPLES-TO-APPLES [[4,2,2]] HYBRID PRE-COMPENSATION ===")
    for hnm in ["rottor_L2r1_s2", "rottor_hyb12_s2", "rottor_hyb21_s2", "rottor_L3r2_s2", "sq_L3r2_s2"]:
        m_bs = [5.00, 5.50, 6.00, 6.50]
        m_ys = [min_sector(hnm, b)[0] for b in m_bs]
        deff = ols_slope(m_bs, m_ys)
        segs = []
        N_val = 0
        for b in [5.00, 5.50, 6.00, 6.50, 7.00, 7.50, 8.00]:
            kr = kmc.get((f"{hnm}_X_err", "uniform", "uf", b))
            if kr:
                N_val = kr["N"]
            v, src = min_sector(hnm, b)
            if v is not None:
                segs.append(f"b={b:.2f}:{v:.3e}[{src}]")
        out.append(f"  {hnm:18s} (N={N_val:4d}, Deff[5..6.5]={deff:.2f}): " + " | ".join(segs))

    # 5. Verified Commuting Pruned Gadgets (`bad_comm == 0`, `t_next = 0.0`)
    out.append("\n=== 5. VERIFIED COMMUTING PRUNED GADGETS (H_X H_Z^T == 0, t_next = 0.0) ===")
    for pref in ["rottor_cb3rib", "rottor_cb2rib", "sq_cb3rib", "sq_cb2rib"]:
        for s in [2, 3, 4]:
            ps = f"{pref}_s{s}"
            kz = kmc.get((f"{ps}_Z_err", "uniform", "uf", 6.50))
            if not kz:
                continue
            bs = [5.50, 6.00, 6.50, 7.00, 7.50, 8.00]
            ys = [min_sector(ps, b)[0] for b in bs]
            deff = ols_slope(bs, ys)
            segs = [f"b={b:.1f}:{y:.3e}" for b, y in zip(bs, ys) if y is not None]
            out.append(f"  {ps:18s} (N={kz['N']:5d}, K={kz['K']:4d}, Deff={deff:+.2f}): " + " | ".join(segs))

    # 6. Unified Single-Engine Z^3 Wire Stretching & Kramers Limit
    out.append("\n=== 6. UNIFIED SINGLE-ENGINE Z^3 STRETCHING & KRAMERS LIMIT ===")
    for zc in ["sq_z3_3_5_r1", "sq_z3_4_6_r2", "sq_bdl_8_8_r2"]:
        out.append(f"  [{zc}]")
        for jm in ["uniform", "pin_norm", "hill_norm", "hill_bnd"]:
            bs = [5.50, 6.00, 6.50, 7.00]
            vals = []
            N_val = 0
            for b in bs:
                kx = kmc.get((f"{zc}_X_err", jm, "uf", b))
                kz = kmc.get((f"{zc}_Z_err", jm, "uf", b))
                if kx:
                    N_val = kx["N"]
                if kx and kx["capped"] == 0:
                    v = min(kx["mean_t"], kz["mean_t"]) if (kz and kz["capped"] == 0) else kx["mean_t"]
                    vals.append((b, v, kx["sem_t"]))
            deff = ols_slope([x for x, _, _ in vals], [y for _, y, _ in vals])
            sstr = " | ".join(f"b={b:.2f}:{v:.3e}+-{se:.1e}" for b, v, se in vals)
            out.append(f"    {jm:10s} (N={N_val:5d}, Deff={deff:.2f}): {sstr}")

    # 7. 3D Quantum Cooling & Sub-Threshold Autonomous Self-Healing
    out.append("\n=== 7A. 3D QUANTUM COOLING FROM HOT RANDOM STATES (p0 = 0.50) ===")
    for cname in ["sq_L2r1_s2", "rottor_L2r1_s2", "rottor_hyb21_s2", "sq_L2r1_s3", "rottor_L2r1_s3"]:
        for sec in ["X_err", "Z_err"]:
            tag = f"{cname}_{sec}"
            for ttot in [100.0, 400.0, 1500.0]:
                segs = []
                N_v, K_v = 0, 0
                for pr in ["quench", "anneal", "soft_seam"]:
                    cr = cool.get((tag, pr, "hot", ttot))
                    if cr:
                        N_v, K_v = cr["N"], cr["K"]
                        segs.append(f"{pr}: syn={cr['syn']:.4f}, p0={cr['p0']:.2f}")
                if segs:
                    out.append(f"  {tag:22s} (N={N_v:4d},K={K_v}) t={int(ttot):4d} -> " + " || ".join(segs))

    out.append("\n=== 7B. 3D QUANTUM AUTONOMOUS CODEWORD SELF-HEALING vs INITIAL NOISE p0 (quench b=6.5, t=1000) ===")
    for cname in ["sq_L2r1_s2", "rottor_L2r1_s2", "rottor_hyb21_s2", "sq_L2r1_s3", "rottor_L2r1_s3"]:
        for sec in ["X_err", "Z_err"]:
            tag = f"{cname}_{sec}"
            segs = []
            N_v, K_v = 0, 0
            for p0 in ["0.01", "0.02", "0.03", "0.04", "0.06", "0.08", "0.12"]:
                cr = cool.get((tag, "quench", f"noisy_{p0}", 1000.0))
                if cr:
                    N_v, K_v = cr["N"], cr["K"]
                    segs.append(f"p={p0}:heal={cr['pheal']:.2f}(surv={cr['psurv']:.2f},p0={cr['p0']:.2f})")
            if segs:
                out.append(f"  {tag:22s} (N={N_v:4d},K={K_v}): " + " | ".join(segs))

    txt = "\n".join(out) + "\n"
    with open(OUT_PATH, "w") as f:
        f.write(txt)
    print(f"Wrote {len(out)} lines to {OUT_PATH}")


if __name__ == "__main__":
    main()
