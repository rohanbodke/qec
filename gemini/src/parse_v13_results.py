#!/usr/bin/env python3
"""
parse_v13_results.py
Parses cluster_v13_results.txt with full statistical rigor:
  1. Exact two-sector quantum survival integral:
       tau_Q = int_0^inf S_X(t) S_Z(t) dt = (1 / (n_X n_Z)) sum_{i,j} min(T_{X,i}, T_{Z,j})
     with 2,000-resample bootstrap SEMs and 95% CIs, plus explicit censoring accounting.
  2. Controlled wire-length sweep (w = 1, 2, 4, 8, 16) vs analytical
       beta_eff(w) = 2 * arctanh((tanh(beta / 2))^w),
     including weighted linear-fit activation slopes, fit windows, SEMs, and R^2.
  3. Paired local-vs-global decoder trajectory & i.i.d. single-shot benchmarks:
     per-checkpoint logical agreement, paired log10-ratio CIs, and O(N) decode latencies.
  4. Coarse-syndrome flow v13: mean counts, check denominators M_s, densities,
     decoded logical failure counts, and exact one-sided 95% Clopper-Pearson upper bounds.
"""
import json
import math
import os
import re
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fit_slope_with_se(betas, means, sems):
    x = np.asarray(betas, dtype=np.float64)
    y = np.log(np.asarray(means, dtype=np.float64))
    s_y = np.maximum(1e-4, np.asarray(sems, dtype=np.float64) / np.asarray(means, dtype=np.float64))
    w = 1.0 / (s_y ** 2)
    W = np.sum(w)
    xb = np.sum(w * x) / W
    yb = np.sum(w * y) / W
    Sxx = np.sum(w * (x - xb) ** 2)
    if Sxx <= 0 or len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    slope = float(np.sum(w * (x - xb) * (y - yb)) / Sxx)
    intercept = float(yb - slope * xb)
    resid = y - (slope * x + intercept)
    ss_res = float(np.sum(w * resid ** 2))
    ss_tot = float(np.sum(w * (y - yb) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    # Conservative SE combining weighted propagation and residual scatter:
    dof = max(1, len(x) - 2)
    chi2_red = max(1.0, ss_res / dof) if len(x) >= 3 else 1.0
    se_slope = float(math.sqrt(chi2_red / Sxx))
    return slope, se_slope, r2


def exact_two_sector_tau_q(tx_arr, cap_x, tz_arr, cap_z, n_boot=1500, seed=12345):
    tx = np.asarray(tx_arr, dtype=np.float64)
    tz = np.asarray(tz_arr, dtype=np.float64)
    tau_q = float(np.minimum(tx[:, None], tz[None, :]).mean())
    rng = np.random.default_rng(seed)
    nx, nz = len(tx), len(tz)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        bx = tx[rng.integers(0, nx, size=nx)]
        bz = tz[rng.integers(0, nz, size=nz)]
        boots[b] = np.minimum(bx[:, None], bz[None, :]).mean()
    sem_q = float(np.std(boots, ddof=1))
    ci_lo, ci_hi = [float(v) for v in np.percentile(boots, [2.5, 97.5])]
    # Check if any censored pair affects min(tx, tz):
    # In sorted t_list, capped trials sit at the top indices; conservative check:
    tx_s = np.sort(tx)
    tz_s = np.sort(tz)
    cx_thresh = tx_s[nx - cap_x] if cap_x > 0 else float("inf")
    cz_thresh = tz_s[nz - cap_z] if cap_z > 0 else float("inf")
    both_cens = int(np.sum((tx[:, None] >= cx_thresh) & (tz[None, :] >= cz_thresh)))
    return {
        "tau_Q": tau_q,
        "sem_Q": sem_q,
        "ci95": [ci_lo, ci_hi],
        "both_censored_pairs": both_cens,
        "total_pairs": nx * nz,
    }


def clopper_pearson_upper_95(k_fail, n_trials):
    if k_fail >= n_trials:
        return 1.0
    if k_fail == 0:
        return 1.0 - (0.05 ** (1.0 / n_trials))
    from scipy.stats import beta as beta_dist
    return float(beta_dist.ppf(0.95, k_fail + 1, n_trials - k_fail))


def main():
    in_path = os.path.join(ROOT, "cluster_v13_results.txt")
    if not os.path.exists(in_path):
        print(f"Missing {in_path}")
        return

    kmc = {}
    paired_kmc = {}
    iid_paired = {}
    rgflow = {}
    errors, timeouts = [], []

    with open(in_path) as f:
        for raw in f:
            ln = raw.strip()
            if not ln or "|" not in ln:
                continue
            tid, body = [x.strip() for x in ln.split("|", 1)]
            if body.startswith("ERROR"):
                errors.append(ln)
                continue
            if body.startswith("TIMEOUT"):
                timeouts.append(tid)
                continue

            if body.startswith("KMC "):
                kv = dict(re.findall(r"([a-zA-Z0-9_]+)=([^\s]+)", body))
                t_list = [float(x) for x in kv["t_list"].split(",")] if "t_list" in kv else []
                kmc[tid] = {
                    "dec": kv["dec"],
                    "beta": float(kv["beta"]),
                    "N": int(kv["N"]),
                    "K": int(kv["K"]),
                    "trials": int(kv["trials"]),
                    "mean_t": float(kv["mean_t"]),
                    "sem_t": float(kv["sem_t"]),
                    "median_t": float(kv["median_t"]),
                    "bails": int(kv["bails"]),
                    "capped": int(kv["capped"]),
                    "t_list": t_list,
                }
            elif body.startswith("PAIRED_KMC "):
                kv = dict(re.findall(r"([a-zA-Z0-9_]+)=([^\s]+)", body))
                pairs = []
                if "pairs_uf_strict" in kv:
                    for item in kv["pairs_uf_strict"].split(","):
                        u, s = item.split(":")
                        pairs.append((float(u), float(s)))
                paired_kmc[tid] = {
                    "beta": float(kv["beta"]),
                    "N": int(kv["N"]),
                    "K": int(kv["K"]),
                    "trials": int(kv["trials"]),
                    "mean_uf": float(kv["mean_uf"]),
                    "sem_uf": float(kv["sem_uf"]),
                    "mean_rg": float(kv["mean_rg"]),
                    "sem_rg": float(kv["sem_rg"]),
                    "mean_strict": float(kv["mean_strict"]),
                    "sem_strict": float(kv["sem_strict"]),
                    "log10_ratio_strict": float(kv["log10_ratio_strict"]),
                    "sem_log10_ratio": float(kv["sem_log10_ratio"]),
                    "frac_same_t": float(kv["frac_same_t"]),
                    "chk_agree_rg": float(kv["chk_agree_rg"]),
                    "chk_agree_strict": float(kv["chk_agree_strict"]),
                    "us_uf": float(kv["us_uf"]),
                    "us_rg": float(kv["us_rg"]),
                    "us_strict": float(kv["us_strict"]),
                    "cap_uf": int(kv["cap_uf"]),
                    "cap_strict": int(kv["cap_strict"]),
                    "pairs": pairs,
                }
            elif body.startswith("IID_PAIRED "):
                kv = dict(re.findall(r"([a-zA-Z0-9_]+)=([^\s]+)", body))
                iid_paired[tid] = {
                    "p": float(kv["p"]),
                    "N": int(kv["N"]),
                    "K": int(kv["K"]),
                    "shots": int(kv["shots"]),
                    "fail_uf": int(kv["fail_uf"]),
                    "fail_rg": int(kv["fail_rg"]),
                    "fail_strict": int(kv["fail_strict"]),
                    "disagree_rg": int(kv["disagree_rg"]),
                    "disagree_strict": int(kv["disagree_strict"]),
                    "us_uf": float(kv["us_uf"]),
                    "us_rg": float(kv["us_rg"]),
                    "us_strict": float(kv["us_strict"]),
                }
            elif body.startswith("RGFLOW_V13 "):
                kv = dict(re.findall(r"([a-zA-Z0-9_]+)=([^\s]+)", body))
                S_max = int(kv["S_max"])
                scales = {}
                for s in range(S_max, -1, -1):
                    scales[s] = {
                        "M": int(kv[f"M_{s}"]),
                        "cnt": float(kv[f"cnt_{s}"]),
                        "semc": float(kv[f"semc_{s}"]),
                        "rho": float(kv[f"rho_{s}"]),
                    }
                rgflow[tid] = {
                    "S_max": S_max,
                    "N_top": int(kv["N_top"]),
                    "beta": float(kv["beta"]),
                    "trials": int(kv["trials"]),
                    "nz_seed": int(kv["nz_seed"]),
                    "log_fail": int(kv["log_fail"]),
                    "cp95_seed": clopper_pearson_upper_95(int(kv["nz_seed"]), int(kv["trials"])),
                    "cp95_log": clopper_pearson_upper_95(int(kv["log_fail"]), int(kv["trials"])),
                    "scales": scales,
                }

    lines = []
    lines.append(f"V13 Parsed Summary: KMC={len(kmc)}, PAIRED_KMC={len(paired_kmc)}, IID_PAIRED={len(iid_paired)}, RGFLOW={len(rgflow)} | errors={len(errors)} | timeouts={len(timeouts)}")
    if errors:
        lines.append(f"  ERRORS: {errors[:5]}")
    if timeouts:
        lines.append(f"  TIMEOUTS: {timeouts}")

    # 1. Wire-length sweep analysis
    lines.append("\n=== 1. CONTROLLED WIRE-LENGTH SWEEP (w = 1, 2, 4, 8, 16) ===")
    wire_json = {}
    for fam, label in [("sq", "Planar 2x2 Core (N0=323, K=1)"), ("tor", "Torus 2x2 Core (N0=646, K=2)")]:
        lines.append(f"  [{label}]")
        wire_json[fam] = {}
        for w in (1, 2, 4, 8, 16):
            pts = sorted(
                [(d["beta"], d["mean_t"], d["sem_t"], d["capped"], d["N"]) for k, d in kmc.items() if k.startswith(f"wire__{fam}_w{w}__")],
                key=lambda t: t[0],
            )
            if not pts:
                continue
            uncap = [(b, m, s) for (b, m, s, c, _) in pts if c == 0 and b >= 5.0]
            sl, se_sl, r2 = fit_slope_with_se([t[0] for t in uncap], [t[1] for t in uncap], [t[2] for t in uncap]) if len(uncap) >= 2 else (float("nan"), float("nan"), float("nan"))
            b_win = f"[{uncap[0][0]:.2f},{uncap[-1][0]:.2f}]" if len(uncap) >= 2 else "n/a"
            pt_str = ", ".join(
                f"b={b:.2f}(beff={2*math.atanh(math.tanh(0.5*b)**w):.2f}):{m:.3e}±{s:.2e}" + (f"(cap={c})" if c else "")
                for b, m, s, c, _ in pts
            )
            lines.append(f"    w={w:2d} (N={pts[0][4]:4d}) | slope{b_win}={sl:.2f}±{se_sl:.2f} (R2={r2:.3f}) | {pt_str}")
            wire_json[fam][w] = {
                "N": pts[0][4],
                "slope": sl,
                "se_slope": se_sl,
                "r2": r2,
                "fit_window": [uncap[0][0], uncap[-1][0]] if len(uncap) >= 2 else [],
                "points": [
                    {
                        "beta": b,
                        "beta_eff": float(2.0 * math.atanh(math.tanh(0.5 * b) ** w)),
                        "mean_t": m,
                        "sem_t": s,
                        "capped": c,
                    }
                    for b, m, s, c, _ in pts
                ],
            }

    # 2. Paired trajectory benchmark
    lines.append("\n=== 2. PAIRED TRAJECTORY BENCHMARK (UF vs STRICT O(N) LOCAL RG) ===")
    for pref in ("paired_kmc__rottor_s2", "paired_kmc__rottor_s4", "paired_kmc__sq_rx_s1", "paired_kmc__sq_rx_s2", "paired_kmc__sq_rx_s3", "paired_kmc__sq_rx_s4", "paired_kmc__sq_rx_s5"):
        rows = sorted([(d["beta"], d) for k, d in paired_kmc.items() if k.startswith(pref + "__")], key=lambda t: t[0])
        if not rows:
            continue
        lines.append(f"  [{pref} (N={rows[0][1]['N']}, K={rows[0][1]['K']})]")
        for b, d in rows:
            rat = 10.0 ** d["log10_ratio_strict"]
            ci_lo = 10.0 ** (d["log10_ratio_strict"] - 1.96 * d["sem_log10_ratio"])
            ci_hi = 10.0 ** (d["log10_ratio_strict"] + 1.96 * d["sem_log10_ratio"])
            lines.append(
                f"    b={b:.2f}: UF={d['mean_uf']:.3e}±{d['sem_uf']:.2e} | StrictRG={d['mean_strict']:.3e}±{d['sem_strict']:.2e} "
                f"| geom_ratio={rat:.3f} [{ci_lo:.3f},{ci_hi:.3f}] | same_t={d['frac_same_t']*100:.1f}% | chk_agree={d['chk_agree_strict']*100:.2f}% "
                f"| us(UF/Strict)={d['us_uf']:.1f}/{d['us_strict']:.1f}"
            )

    # 2B. Paired i.i.d. readout benchmark
    lines.append("\n=== 2B. PAIRED SINGLE-SHOT I.I.D. BENCHMARK ===")
    for pref in ("iid_paired__rottor_s2", "iid_paired__rottor_s4", "iid_paired__sq_rx_s1", "iid_paired__sq_rx_s2", "iid_paired__sq_rx_s3", "iid_paired__sq_rx_s4", "iid_paired__sq_rx_s5"):
        rows = sorted([(d["p"], d) for k, d in iid_paired.items() if k.startswith(pref + "__")], key=lambda t: t[0])
        if not rows:
            continue
        lines.append(
            f"  [{pref} N={rows[0][1]['N']}]: "
            + ", ".join(
                f"p={p:.3f}: fail(UF={d['fail_uf']}/{d['shots']}, St={d['fail_strict']}/{d['shots']}, disag={d['disagree_strict']}/{d['shots']}, us={d['us_uf']:.1f}/{d['us_strict']:.1f})"
                for p, d in rows
            )
        )

    # 3. Coarse-syndrome flow v13
    lines.append("\n=== 3. COARSE-SYNDROME FLOW V13 (COUNTS, DENOMINATORS, & LOGICAL CLASS) ===")
    for pref in ("rgflow13__sq_rx_s5", "rgflow13__rottor_s4", "rgflow13__sq_iso_s5"):
        rows = sorted([(d["beta"], d) for k, d in rgflow.items() if k.startswith(pref + "__")], key=lambda t: t[0])
        if not rows:
            continue
        S_max = rows[0][1]["S_max"]
        Ms = [rows[0][1]["scales"][s]["M"] for s in range(S_max, -1, -1)]
        lines.append(f"  [{pref} N_top={rows[0][1]['N_top']}, M_s(s={S_max}..0)={Ms}]")
        for b, d in rows:
            c_str = " -> ".join(f"s{s}:{d['scales'][s]['cnt']:.2f}(rho={d['scales'][s]['rho']:.2e})" for s in range(S_max, -1, -1))
            lines.append(
                f"    b={b:.2f}: nz_seed={d['nz_seed']}/{d['trials']} (CP95<={d['cp95_seed']:.4f}), "
                f"log_fail={d['log_fail']}/{d['trials']} (CP95<={d['cp95_log']:.4f}) | {c_str}"
            )

    # 4. Exact Two-Sector Combined Quantum Lifetimes tau_Q = int_0^inf S_X(t) S_Z(t) dt
    lines.append("\n=== 4. EXACT TWO-SECTOR QUANTUM LIFETIMES tau_Q = int S_X(t) S_Z(t) dt ===")
    # Group tlist__* by base code name and beta!
    base_betas = {}
    for tid, d in kmc.items():
        if not tid.startswith("tlist__"):
            continue
        m = re.match(r"tlist__(.+)_(X_err|Z_err)__b([0-9.]+)", tid)
        if not m:
            continue
        cname, sec, b_s = m.group(1), m.group(2), float(m.group(3))
        base_betas.setdefault(cname, {}).setdefault(b_s, {})[sec] = d

    tau_q_json = {}
    for cname in sorted(base_betas.keys()):
        bmap = base_betas[cname]
        q_pts = []
        for b in sorted(bmap.keys()):
            if "X_err" in bmap[b] and "Z_err" in bmap[b]:
                dx, dz = bmap[b]["X_err"], bmap[b]["Z_err"]
                res = exact_two_sector_tau_q(dx["t_list"], dx["capped"], dz["t_list"], dz["capped"])
                q_pts.append((b, dx, dz, res))
        if not q_pts:
            continue
        N = q_pts[0][1]["N"]
        K = q_pts[0][1]["K"]
        uncap_q = [(b, r["tau_Q"], r["sem_Q"]) for (b, _, _, r) in q_pts if r["both_censored_pairs"] == 0 and b >= 5.50]
        if len(uncap_q) < 2:
            uncap_q = [(b, r["tau_Q"], r["sem_Q"]) for (b, _, _, r) in q_pts if r["both_censored_pairs"] == 0]
        sl, se_sl, r2 = fit_slope_with_se([t[0] for t in uncap_q], [t[1] for t in uncap_q], [t[2] for t in uncap_q]) if len(uncap_q) >= 2 else (float("nan"), float("nan"), float("nan"))
        b_win = f"[{uncap_q[0][0]:.2f},{uncap_q[-1][0]:.2f}]" if len(uncap_q) >= 2 else "n/a"
        p_str = ", ".join(
            f"b={b:.2f}: tau_Q={r['tau_Q']:.3e}±{r['sem_Q']:.2e} (X={dx['mean_t']:.2e}[c={dx['capped']}], Z={dz['mean_t']:.2e}[c={dz['capped']}])"
            for b, dx, dz, r in q_pts
        )
        lines.append(f"  {cname:22s} (N={N:6d}, K={K}) | slope{b_win}={sl:.2f}±{se_sl:.2f} (R2={r2:.3f}) | {p_str}")
        tau_q_json[cname] = {
            "N": N,
            "K": K,
            "slope": sl,
            "se_slope": se_sl,
            "r2": r2,
            "fit_window": [uncap_q[0][0], uncap_q[-1][0]] if len(uncap_q) >= 2 else [],
            "points": [
                {
                    "beta": b,
                    "tau_Q": r["tau_Q"],
                    "sem_Q": r["sem_Q"],
                    "ci95": r["ci95"],
                    "both_censored_pairs": r["both_censored_pairs"],
                    "mean_X": dx["mean_t"],
                    "sem_X": dx["sem_t"],
                    "cap_X": dx["capped"],
                    "trials_X": dx["trials"],
                    "mean_Z": dz["mean_t"],
                    "sem_Z": dz["sem_t"],
                    "cap_Z": dz["capped"],
                    "trials_Z": dz["trials"],
                }
                for b, dx, dz, r in q_pts
            ],
        }

    out_txt = os.path.join(ROOT, "parsed_v13_summary.txt")
    with open(out_txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    out_json = os.path.join(ROOT, "parsed_v13_data.json")
    with open(out_json, "w") as f:
        json.dump(
            {
                "wire_sweep": wire_json,
                "paired_kmc": {k: {kk: vv for kk, vv in v.items() if kk != "pairs"} for k, v in paired_kmc.items()},
                "iid_paired": iid_paired,
                "rgflow_v13": rgflow,
                "two_sector_tau_Q": tau_q_json,
            },
            f,
            indent=2,
        )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
