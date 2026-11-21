#!/usr/bin/env python3
"""
parse_v14_results.py
Unified statistical analysis of cluster_v13_results.txt + cluster_v14_results.txt
for the V19 manuscript "Finite-Size Thermal Protection in Compact Recursively Welded Codes":

  1. Complete-cycle quantum lifetime gains G_1(beta) = tau_Q(C_2, beta) / tau_Q(C_1, beta)
     with trajectory-level bootstrap SEMs, 95% CIs, sector lifetimes (tau_X, tau_Z),
     and exact full-cohort censoring brackets [tau_Q_lo, tau_Q_hi].
  2. Selected-scale coupling intervention H(lambda; U) across `macro_weld`, `sub_weld`,
     and `all_c2_weld` for lambda in {0.2, 0.4, 0.6, 0.8, 1.0} and beta in {5.5, 6.0, 6.25, 6.5}
     in both X and Z sectors and joint tau_Q.
  3. Coarse-scale excursion & escape statistics (`excursion_kmc`): first Cycle-1 crossing
     fraction, mean Cycle-1 excursions E[N_exc], attempt rate, and escape probability.
  4. Paired readout cadence audit (`paired_cadence`): eta in {0.0/0.02, 0.05, 0.20, 0.50}
     with paired ratio 95% CIs and delayed-persistent vs missed-healed breakdown.
  5. Paired local-vs-global decoder ratios (`paired_rg_kmc`) with paired cohort-mean ratio CIs.
  6. Coarse-syndrome flow burn-in sweep & matched-weight i.i.d. reference (`rg_flow_v14`).
  7. Exact matched-beta_eff wire exponent fit & master-curve residuals on overlapping support.
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
    dof = max(1, len(x) - 2)
    chi2_red = max(1.0, ss_res / dof) if len(x) >= 3 else 1.0
    se_slope = float(math.sqrt(chi2_red / Sxx))
    return slope, se_slope, r2


def exact_two_sector_tau_q(tx_arr, cap_x, tz_arr, cap_z, n_boot=2000, seed=12345):
    tx = np.sort(np.asarray(tx_arr, dtype=np.float64))
    tz = np.sort(np.asarray(tz_arr, dtype=np.float64))
    nx, nz = len(tx), len(tz)
    tau_lo = float(np.minimum(tx[:, None], tz[None, :]).mean())

    # Exact full-cohort deterministic censoring upper bound:
    if cap_x == 0 and cap_z == 0:
        tau_hi = tau_lo
    elif cap_x > 0 and cap_z == 0:
        tx_hi = tx.copy()
        tx_hi[nx - cap_x :] = np.inf
        tau_hi = float(np.minimum(tx_hi[:, None], tz[None, :]).mean())
    elif cap_z > 0 and cap_x == 0:
        tz_hi = tz.copy()
        tz_hi[nz - cap_z :] = np.inf
        tau_hi = float(np.minimum(tx[:, None], tz_hi[None, :]).mean())
    else:
        tau_hi = float("inf")

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        bx = tx[rng.integers(0, nx, size=nx)]
        bz = tz[rng.integers(0, nz, size=nz)]
        boots[b] = np.minimum(bx[:, None], bz[None, :]).mean()
    sem_q = float(np.std(boots, ddof=1))
    ci_lo, ci_hi = [float(v) for v in np.percentile(boots, [2.5, 97.5])]
    rel_bracket_gap = ((tau_hi - tau_lo) / tau_lo) if math.isfinite(tau_hi) and tau_lo > 0 else float("inf")
    return {
        "tau_Q": tau_lo,
        "sem_Q": sem_q,
        "ci95": [ci_lo, ci_hi],
        "bracket": [tau_lo, tau_hi],
        "rel_bracket_gap": rel_bracket_gap,
        "n_X": nx,
        "cap_X": cap_x,
        "n_Z": nz,
        "cap_Z": cap_z,
        "_boots": boots,
    }


def clopper_pearson_upper_95(k_fail, n_trials):
    if k_fail >= n_trials:
        return 1.0
    if k_fail == 0:
        return 1.0 - (0.05 ** (1.0 / n_trials))
    from scipy.stats import beta as beta_dist

    return float(beta_dist.ppf(0.95, k_fail + 1, n_trials - k_fail))


def paired_ratio_bootstrap(pairs, n_boot=2000, seed=54321):
    """Given list of (u_i, v_i), computes cohort-mean ratio mean(v)/mean(u) and paired 95% CI."""
    arr = np.asarray(pairs, dtype=np.float64)
    u, v = arr[:, 0], arr[:, 1]
    n = len(arr)
    r_hat = float(v.mean() / u.mean())
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = v[idx].mean() / max(1e-30, u[idx].mean())
    return {
        "mean_ratio": r_hat,
        "se_ratio": float(np.std(boots, ddof=1)),
        "ci95_ratio": [float(x) for x in np.percentile(boots, [2.5, 97.5])],
    }


def load_results_files(paths):
    kmc = {}
    paired_kmc = {}
    paired_cadence = {}
    excursions = {}
    iid_paired = {}
    rgflow13 = {}
    rgflow14 = {}
    errors, timeouts = [], []

    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for raw in f:
                ln = raw.strip()
                if not ln or "|" not in ln:
                    continue
                tid, body = [x.strip() for x in ln.split("|", 1)]
                if body.startswith("ERROR"):
                    errors.append(f"{os.path.basename(path)}:{ln}")
                    continue
                if body.startswith("TIMEOUT"):
                    timeouts.append(f"{os.path.basename(path)}:{tid}")
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
                    p_boot = paired_ratio_bootstrap(pairs) if pairs else {}
                    pairs_rg = []
                    if "pairs_uf_rg" in kv:
                        for item in kv["pairs_uf_rg"].split(","):
                            u, r = item.split(":")
                            pairs_rg.append((float(u), float(r)))
                    p_boot_rg = paired_ratio_bootstrap(pairs_rg) if pairs_rg else {}
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
                        "paired_mean_ratio_strict_over_uf": p_boot,
                        "paired_mean_ratio_rg_over_uf": p_boot_rg,
                    }
                elif body.startswith("PAIRED_CADENCE "):
                    kv = dict(re.findall(r"([a-zA-Z0-9_]+)=([^\s]+)", body))
                    quads = []
                    for item in kv["quads"].split(","):
                        q0, q1, q2, q3 = [float(x) for x in item.split(":")]
                        quads.append((q0, q1, q2, q3))
                    caps = [int(x) for x in kv["caps"].split(",")]
                    r20_0 = paired_ratio_bootstrap([(q[0], q[2]) for q in quads])
                    r05_0 = paired_ratio_bootstrap([(q[0], q[1]) for q in quads])
                    r50_0 = paired_ratio_bootstrap([(q[0], q[3]) for q in quads])
                    paired_cadence[tid] = {
                        "beta": float(kv["beta"]),
                        "min_eta": float(kv["min_eta"]),
                        "N": int(kv["N"]),
                        "K": int(kv["K"]),
                        "trials": int(kv["trials"]),
                        "m0": float(kv["m0"]),
                        "s0": float(kv["s0"]),
                        "m05": float(kv["m05"]),
                        "s05": float(kv["s05"]),
                        "m20": float(kv["m20"]),
                        "s20": float(kv["s20"]),
                        "m50": float(kv["m50"]),
                        "s50": float(kv["s50"]),
                        "same_0_20": int(kv["same_0_20"]),
                        "delayed_persist": int(kv["delayed_persist"]),
                        "missed_heal": int(kv["missed_heal"]),
                        "caps": caps,
                        "ratio_20_over_min": r20_0,
                        "ratio_05_over_min": r05_0,
                        "ratio_50_over_min": r50_0,
                    }
                elif body.startswith("EXCURSION_KMC "):
                    kv = dict(re.findall(r"([a-zA-Z0-9_]+)=([^\s]+)", body))
                    triples = []
                    for item in kv["triples"].split(","):
                        tf, t1, ne = item.split(":")
                        triples.append((float(tf), float(t1), int(ne)))
                    mt = float(kv["mean_t"])
                    mexc = float(kv["mean_n_exc"])
                    excursions[tid] = {
                        "beta": float(kv["beta"]),
                        "N": int(kv["N"]),
                        "K": int(kv["K"]),
                        "trials": int(kv["trials"]),
                        "mean_t": mt,
                        "sem_t": float(kv["sem_t"]),
                        "mean_t_first_s2": float(kv["mean_t_first_s2"]),
                        "mean_frac_first_s2": float(kv["mean_frac_first_s2"]),
                        "mean_n_exc": mexc,
                        "sem_n_exc": float(kv["sem_n_exc"]),
                        "attempt_rate_s2": mexc / max(1e-30, mt),
                        "escape_prob_per_exc": 1.0 / max(1.0, mexc),
                        "frac_multi_exc": sum(1 for _, _, ne in triples if ne >= 2) / max(1, len(triples)),
                        "capped": int(kv["capped"]),
                    }
                elif body.startswith("IID_PAIRED "):
                    kv = dict(re.findall(r"([a-zA-Z0-9_]+)=([^\s]+)", body))
                    iid_paired[tid] = {
                        "p": float(kv["p"]),
                        "N": int(kv["N"]),
                        "K": int(kv["K"]),
                        "shots": int(kv["shots"]),
                        "fail_uf": int(kv["fail_uf"]),
                        "fail_strict": int(kv["fail_strict"]),
                        "disagree_strict": int(kv["disagree_strict"]),
                        "us_uf": float(kv["us_uf"]),
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
                    rgflow13[tid] = {
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
                elif body.startswith("RGFLOW_V14 "):
                    kv = dict(re.findall(r"([a-zA-Z0-9_]+)=([^\s]+)", body))
                    S_max = int(kv["S_max"])
                    scales = {}
                    for s in range(S_max, -1, -1):
                        scales[s] = {
                            "M": int(kv[f"M_{s}"]),
                            "cnt": float(kv[f"cnt_{s}"]),
                            "semc": float(kv[f"semc_{s}"]),
                            "icnt": float(kv[f"icnt_{s}"]),
                            "isemc": float(kv[f"isemc_{s}"]),
                        }
                    rgflow14[tid] = {
                        "S_max": S_max,
                        "N_top": int(kv["N_top"]),
                        "beta": float(kv["beta"]),
                        "t_eq": float(kv["t_eq"]),
                        "trials": int(kv["trials"]),
                        "mean_qwt": float(kv["mean_qwt"]),
                        "nz_seed": int(kv["nz_seed"]),
                        "log_fail": int(kv["log_fail"]),
                        "iid_nz_seed": int(kv["iid_nz_seed"]),
                        "iid_log_fail": int(kv["iid_log_fail"]),
                        "cp95_seed": clopper_pearson_upper_95(int(kv["nz_seed"]), int(kv["trials"])),
                        "scales": scales,
                    }
    return kmc, paired_kmc, paired_cadence, excursions, iid_paired, rgflow13, rgflow14, errors, timeouts


def main():
    paths = [
        os.path.join(ROOT, "cluster_v13_results.txt"),
        os.path.join(ROOT, "cluster_v14_results.txt"),
    ]
    kmc, paired_kmc, paired_cadence, excursions, iid_paired, rgflow13, rgflow14, errors, timeouts = load_results_files(paths)

    lines = []
    lines.append(
        f"V14 Unified Summary: KMC={len(kmc)}, PAIRED_KMC={len(paired_kmc)}, CADENCE={len(paired_cadence)}, "
        f"EXCUR={len(excursions)}, RGFLOW14={len(rgflow14)} | errors={len(errors)} | timeouts={len(timeouts)}"
    )
    if errors:
        lines.append(f"  ERRORS: {errors}")
    if timeouts:
        lines.append(f"  TIMEOUTS: {timeouts}")

    # 1. Exact Two-Sector Lifetimes (`tlist__*`) + Complete-Cycle Gains G_1(beta)
    base_betas = {}
    for tid, d in kmc.items():
        m = re.match(r"tlist__(.+)_(X_err|Z_err)__b([0-9.]+)", tid)
        if m:
            cname, sec, b_v = m.group(1), m.group(2), round(float(m.group(3)), 2)
            base_betas.setdefault(cname, {}).setdefault(b_v, {})[sec] = d

    tau_q_table = {}
    boot_store = {}
    lines.append("\n=== 1. TWO-SECTOR LIFETIMES, SECTORS, & EXACT CENSORING BRACKETS ===")
    for cname in sorted(base_betas.keys()):
        tau_q_table[cname] = {}
        boot_store[cname] = {}
        pts = []
        for b_v in sorted(base_betas[cname].keys()):
            pair = base_betas[cname][b_v]
            if "X_err" not in pair or "Z_err" not in pair:
                continue
            dx, dz = pair["X_err"], pair["Z_err"]
            res = exact_two_sector_tau_q(
                dx["t_list"],
                dx["capped"],
                dz["t_list"],
                dz["capped"],
                seed=1000 + int(b_v * 100),
            )
            boots = res.pop("_boots")
            boot_store[cname][b_v] = boots
            res["N"] = dx["N"]
            res["K"] = dx["K"]
            res["tau_X"] = dx["mean_t"]
            res["sem_X"] = dx["sem_t"]
            res["tau_Z"] = dz["mean_t"]
            res["sem_Z"] = dz["sem_t"]
            res["limiting_sector"] = "X" if dx["mean_t"] < dz["mean_t"] else "Z"
            tau_q_table[cname][f"{b_v:.2f}"] = res
            pts.append((b_v, res))

        fit_pts = [(b, r["tau_Q"], r["sem_Q"]) for b, r in pts if r["rel_bracket_gap"] < 0.05 and b >= 5.0]
        sl, se_sl, r2 = (
            fit_slope_with_se([t[0] for t in fit_pts], [t[1] for t in fit_pts], [t[2] for t in fit_pts])
            if len(fit_pts) >= 2
            else (float("nan"), float("nan"), float("nan"))
        )
        tau_q_table[cname]["_fit"] = {
            "slope": sl,
            "se_slope": se_sl,
            "r2": r2,
            "window": [fit_pts[0][0], fit_pts[-1][0]] if len(fit_pts) >= 2 else [],
        }
        lines.append(
            f"  [{cname} (N={pts[0][1]['N']}, K={pts[0][1]['K']})] slope={sl:.2f}±{se_sl:.2f} win={tau_q_table[cname]['_fit']['window']}"
        )
        for b_v, r in pts:
            brk = (
                f" [brk +{r['rel_bracket_gap']*100:.2f}%]"
                if (r["cap_X"] > 0 or r["cap_Z"] > 0) and math.isfinite(r["rel_bracket_gap"])
                else (" [BOTH_CENS]" if not math.isfinite(r["rel_bracket_gap"]) else "")
            )
            lines.append(
                f"    b={b_v:.2f}: tau_Q={r['tau_Q']:.3e}±{r['sem_Q']:.2e}{brk} "
                f"| tau_X={r['tau_X']:.3e}[{r['cap_X']}/{r['n_X']}] "
                f"| tau_Z={r['tau_Z']:.3e}[{r['cap_Z']}/{r['n_Z']}] (lim={r['limiting_sector']})"
            )

    # Complete-cycle gain table G_1(beta):
    lines.append("\n=== 1B. COMPLETE-CYCLE & HALF-STEP GAINS G(beta) ===")
    cycle_gains = {}
    for label, c_lo, c_hi in [
        ("planar_C1_to_C2", "sq_iso_L2r1_s2", "sq_iso_L2r1_s4"),
        ("planar_C2_to_s5_half", "sq_iso_L2r1_s4", "sq_iso_L2r1_s5"),
        ("torus_C1_to_C2", "rottor_iso_L2r1_s2", "rottor_iso_L2r1_s4"),
        ("planar_3x3_C1_to_s3_half", "sq_L3r1_s2", "sq_L3r1_s3"),
    ]:
        if c_lo not in boot_store or c_hi not in boot_store:
            continue
        cycle_gains[label] = {}
        lines.append(f"  [{label}: {c_lo} -> {c_hi}]")
        for b_v in sorted(set(boot_store[c_lo].keys()) & set(boot_store[c_hi].keys())):
            r_lo = tau_q_table[c_lo][f"{b_v:.2f}"]
            r_hi = tau_q_table[c_hi][f"{b_v:.2f}"]
            g_hat = r_hi["tau_Q"] / r_lo["tau_Q"]
            g_boots = boot_store[c_hi][b_v] / np.maximum(1e-30, boot_store[c_lo][b_v])
            g_se = float(np.std(g_boots, ddof=1))
            g_ci = [float(x) for x in np.percentile(g_boots, [2.5, 97.5])]
            n_factor = r_hi["N"] / r_lo["N"]
            cycle_gains[label][f"{b_v:.2f}"] = {
                "gain": g_hat,
                "se_gain": g_se,
                "ci95_gain": g_ci,
                "qubit_factor": n_factor,
                "tau_Q_lo": r_lo["tau_Q"],
                "tau_Q_hi": r_hi["tau_Q"],
            }
            lines.append(
                f"    b={b_v:.2f}: G={g_hat:.3f}±{g_se:.3f} (95% CI [{g_ci[0]:.3f}, {g_ci[1]:.3f}]), N_ratio={n_factor:.2f}x"
            )

    # 2. Selected-Scale Coupling Intervention H(lambda; U)
    lines.append("\n=== 2. SELECTED-SCALE COUPLING INTERVENTION H(lambda; U) ===")
    interv_data = {}
    inter_raw = {}
    for tid, d in kmc.items():
        m = re.match(r"interv__(.+)_(X_err|Z_err)__(macro_weld|sub_weld|all_c2_weld)__l([0-9]+)__b([0-9.]+)", tid)
        if m:
            cname, sec, tag_u, lam_i, b_v = m.group(1), m.group(2), m.group(3), int(m.group(4)), round(float(m.group(5)), 2)
            inter_raw.setdefault(cname, {}).setdefault(tag_u, {}).setdefault(b_v, {}).setdefault(lam_i, {})[sec] = d

    for cname in sorted(inter_raw.keys()):
        interv_data[cname] = {}
        for tag_u in ("macro_weld", "sub_weld", "all_c2_weld"):
            if tag_u not in inter_raw[cname]:
                continue
            interv_data[cname][tag_u] = {}
            lines.append(f"  [{cname} :: {tag_u}]")
            for b_v in sorted(inter_raw[cname][tag_u].keys()):
                interv_data[cname][tag_u][f"{b_v:.2f}"] = {}
                row_strs = []
                for lam_i in sorted(inter_raw[cname][tag_u][b_v].keys()):
                    pair = inter_raw[cname][tag_u][b_v][lam_i]
                    if "X_err" not in pair or "Z_err" not in pair:
                        continue
                    dx, dz = pair["X_err"], pair["Z_err"]
                    res = exact_two_sector_tau_q(dx["t_list"], dx["capped"], dz["t_list"], dz["capped"], seed=2000 + lam_i)
                    res.pop("_boots")
                    res["lambda"] = lam_i / 10.0
                    res["tau_X"] = dx["mean_t"]
                    res["sem_X"] = dx["sem_t"]
                    res["tau_Z"] = dz["mean_t"]
                    res["sem_Z"] = dz["sem_t"]
                    # Ratio relative to uniform lambda=1.0:
                    unif_q = tau_q_table.get(cname, {}).get(f"{b_v:.2f}", {}).get("tau_Q", float("nan"))
                    res["rel_to_uniform_Q"] = res["tau_Q"] / unif_q if unif_q > 0 else float("nan")
                    interv_data[cname][tag_u][f"{b_v:.2f}"][f"{lam_i/10.0:.1f}"] = res
                    row_strs.append(
                        f"lam={lam_i/10.0:.1f}: Q={res['tau_Q']:.2e}({res['rel_to_uniform_Q']:.3f}x, X={res['tau_X']:.2e}, Z={res['tau_Z']:.2e})"
                    )
                lines.append(f"    b={b_v:.2f} | " + " | ".join(row_strs))

    # 3. Excursion & Escape Diagnostic
    lines.append("\n=== 3. COARSE-SCALE EXCURSION & ESCAPE DIAGNOSTIC ===")
    for tid in sorted(excursions.keys()):
        d = excursions[tid]
        lines.append(
            f"  {tid}: mean_t={d['mean_t']:.3e}±{d['sem_t']:.2e}, t1_frac={d['mean_frac_first_s2']:.3f}, "
            f"n_exc={d['mean_n_exc']:.2f}±{d['sem_n_exc']:.2f}, P_esc={d['escape_prob_per_exc']:.3f}, "
            f"R_att={d['attempt_rate_s2']:.3e}, multi_exc={d['frac_multi_exc']*100:.1f}%"
        )

    # 4. Paired Cadence Audit
    lines.append("\n=== 4. PAIRED READOUT CADENCE AUDIT ===")
    for tid in sorted(paired_cadence.keys()):
        d = paired_cadence[tid]
        r20 = d["ratio_20_over_min"]
        lines.append(
            f"  {tid} (min_eta={d['min_eta']:.2f}, tr={d['trials']}): "
            f"m_min={d['m0']:.3e}, m05={d['m05']:.3e}, m20={d['m20']:.3e}, m50={d['m50']:.3e} | "
            f"r(20/min)={r20['mean_ratio']:.4f} CI95[{r20['ci95_ratio'][0]:.4f},{r20['ci95_ratio'][1]:.4f}] | "
            f"same={d['same_0_20']}, delay_persist={d['delayed_persist']}, missed_heal={d['missed_heal']}"
        )

    # 5. Coarse-Syndrome Flow v14 (Burn-in & i.i.d. reference)
    lines.append("\n=== 5. COARSE-SYNDROME FLOW V14 (BURN-IN & MATCHED-WEIGHT I.I.D. REFERENCE) ===")
    for tid in sorted(rgflow14.keys()):
        d = rgflow14[tid]
        s0 = d["scales"][0]
        s2 = d["scales"][2]
        s4 = d["scales"][4]
        lines.append(
            f"  {tid}: qwt={d['mean_qwt']:.1f} | therm(s4={s4['cnt']:.1f}->s2={s2['cnt']:.2f}->s0={s0['cnt']:.3f}, nz0={d['nz_seed']}/{d['trials']}, lf={d['log_fail']}) "
            f"vs iid_ref(s4={s4['icnt']:.1f}->s2={s2['icnt']:.2f}->s0={s0['icnt']:.3f}, nz0={d['iid_nz_seed']}/{d['trials']}, lf={d['iid_log_fail']})"
        )

    # 6. Exact Matched-beta_eff Wire Dynamical Test & Overlap Residuals
    lines.append("\n=== 6. EXACT MATCHED-BETA_EFF WIRE TEST & OVERLAP RESIDUALS ===")
    wire_beff = {}
    for fam in ("sq", "tor"):
        wire_beff[fam] = {}
        for b0 in (5.00, 5.50, 6.00):
            pts = []
            for w in (1, 2, 4, 8, 16):
                tid = f"wire_beff__{fam}_w{w}__beff{b0:.2f}"
                if tid in kmc:
                    d = kmc[tid]
                    pts.append((w, d["beta"], d["mean_t"], d["sem_t"], d["capped"]))
            if len(pts) >= 2:
                ws = np.array([math.log(t[0]) for t in pts if t[4] == 0])
                ms = [t[2] for t in pts if t[4] == 0]
                ss = [t[3] for t in pts if t[4] == 0]
                gamma, se_gamma, r2 = fit_slope_with_se(ws, ms, ss) if len(ms) >= 2 else (float("nan"), float("nan"), float("nan"))
                t1 = pts[0][2]
                res_w2 = [math.log10((m / (w ** 2)) / t1) for w, _, m, _, c in pts if c == 0]
                rms_dex = float(math.sqrt(np.mean(np.square(res_w2)))) if res_w2 else float("nan")
                max_dex = float(np.max(np.abs(res_w2))) if res_w2 else float("nan")
                wire_beff[fam][f"{b0:.2f}"] = {
                    "gamma_exponent": gamma,
                    "se_gamma": se_gamma,
                    "r2": r2,
                    "rms_log10_resid_vs_w2": rms_dex,
                    "max_log10_resid_vs_w2": max_dex,
                    "points": [
                        {"w": w, "beta_w": bw, "mean_t": m, "sem_t": s, "capped": c, "norm_tau_over_w2_t1": (m / (w ** 2)) / t1}
                        for w, bw, m, s, c in pts
                    ],
                }
                lines.append(
                    f"  [{fam} exact beta_eff={b0:.2f}]: gamma={gamma:.3f}±{se_gamma:.3f} (R2={r2:.4f}), "
                    f"RMS_resid(w^2)={rms_dex:.3f} dex, max_resid={max_dex:.3f} dex | "
                    + ", ".join(f"w={w}:{(m/(w**2))/t1:.3f}x" for w, _, m, _, _ in pts)
                )

    summary_text = "\n".join(lines) + "\n"
    with open(os.path.join(ROOT, "parsed_v14_summary.txt"), "w") as f:
        f.write(summary_text)
    with open(os.path.join(ROOT, "parsed_v14_data.json"), "w") as f:
        json.dump(
            {
                "tau_q": tau_q_table,
                "cycle_gains": cycle_gains,
                "interventions": interv_data,
                "excursions": excursions,
                "paired_cadence": paired_cadence,
                "paired_kmc": paired_kmc,
                "iid_paired": iid_paired,
                "rgflow13": rgflow13,
                "rgflow14": rgflow14,
                "wire_matched_beff": wire_beff,
            },
            f,
            indent=2,
        )
    print(summary_text)


if __name__ == "__main__":
    main()
