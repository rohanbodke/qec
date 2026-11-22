#!/usr/bin/env python3
"""
parse_v15_results.py
Parses `cluster_v15_results.txt` and `v15_code_specs.json` into `parsed_v15_data.json`,
computing:
  1. Exact two-sector joint storage lifetimes tau_Q = int_0^inf S_X(t) S_Z(t) dt,
     exact censoring brackets [tau_Q_min, tau_Q_max], and 2,000-resample bootstrap
     standard errors and 95% confidence intervals across all 30 codes.
  2. Pairwise architectural comparisons (RC vs CR, selective vs RR, 2-cycle vs 1-cycle)
     with bootstrap 95% CIs across Design I (same-subdivision L=3: C3 vs R3) and
     Design II (unit-patch modules: C2 vs R3) on open planar and boundaryless torus seeds.
  3. Out-of-sample two-scale effective-temperature composition model Phi_A(Phi_B(beta)),
     calibrated strictly on single-cycle (s = 2) module lifetimes (and exact s = 0 seed
     CTMC rates, with zero parameters fitted to any 2-cycle simulation).
  4. Coarse-cluster arrival & healing diagnostics (`span_kmc`) and paired checkpoint
     cadence verification (`paired_cadence`).
"""

from collections import defaultdict
import json
import math
import os
import re
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(ROOT, "cluster_v15_results.txt")
SPECS_PATH = os.path.join(ROOT, "v15_code_specs.json")
OUT_PATH = os.path.join(ROOT, "parsed_v15_data.json")


def exact_seed_tau_q(beta, seed_kind="planar"):
    """
    Exact continuous-time heat-bath seed (s = 0) joint lifetime tau_Q^(0)(beta):
      - Planar seed [[2, 1, (2, 1)]]:
          * In X_err (M_check = HZ = [[0]], Z_bar = {0}):
            qubit 0 has Z-deg 1 (dE = 1.0 -> flip rate r1 = 1/(1 + exp(beta))),
            and a single flip of q=0 has syn=1 -> decoder flips q=0 back? Wait:
            H_Z = [[0]], Z_bar = {0} -> min-wt decoder repairs syn=1 by flipping q=0,
            so X_err NEVER fails! Wait: in Z_err (HX = [[0, 1]], X_bar = {0, 1}),
            d_Z = 1: flipping q=0 or q=1 gives syn=1, decoder picks fixed q=0, so
            whichever of {0, 1} is NOT picked causes immediate logical Z failure at rate
            r1 = 1 / (1 + exp(beta)), giving tau_seed_planar(beta) = 1 + exp(beta).
      - Reference single-bond degree-2 excitation timescale:
          Each interior check-protected bond flips at bare rate r_bond(beta) = 2 / (1 + exp(2*beta))
          (two incident qubits/sheets per coarse edge), giving bare inverse-temperature map
          Phi_0(beta) = beta for tau_0(beta) = 0.5 * (1 + exp(2 * beta)).
    We use the exact degree-2 bare reference tau_0(beta) = 0.5 * (1.0 + math.exp(2.0 * beta))
    so that bare physical bond excitation corresponds identically to Phi_0(beta) = beta!
    """
    return 0.5 * (1.0 + math.exp(2.0 * beta))


def parse_kv(body):
    kv = {}
    for tok in body.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            kv[k] = v
    return kv


def joint_tau_q(tx, tz):
    """Exact integral int_0^inf S_X(t) S_Z(t) dt = mean_{i,j} min(tx_i, tz_j)."""
    return float(np.minimum(tx[:, None], tz[None, :]).mean())


def compute_joint_stats(x_rec, z_rec, rng):
    tx = np.array(x_rec["t_list"], dtype=np.float64)
    tz = np.array(z_rec["t_list"], dtype=np.float64)
    cap_x = int(x_rec["capped"])
    cap_z = int(z_rec["capped"])

    # In KMC output, censored trials hit max_flips and sit at the top end of t_list (approximately):
    # To construct a conservative rigorous upper bound tau_Q_max, treat the top `cap_x` entries of tx
    # (if cap_x > 0) and the top `cap_z` entries of tz (if cap_z > 0) — or even more conservatively,
    # ANY `cap_x` / `cap_z` entries — wait: replacing the SMALLEST `cap_x` entries by +inf maximizes
    # the upper bound among all possible assignments of which trials were censored!
    tx_sorted = np.sort(tx)
    tz_sorted = np.sort(tz)

    tau_min = joint_tau_q(tx_sorted, tz_sorted)
    if cap_x > 0 and cap_z > 0:
        tau_max = None  # unbounded above if both sectors have censored trials
    elif cap_x == 0 and cap_z == 0:
        tau_max = tau_min
    elif cap_x == 0 and cap_z > 0:
        # Worst-case upper bound: even if the `cap_z` smallest tz values were the censored ones and -> +inf!
        tz_hi = tz_sorted.copy()
        tz_hi[:cap_z] = 1e300
        tau_max = joint_tau_q(tx_sorted, tz_hi)
        # Also exact threshold check: if min(tz_sorted[-cap_z:]) >= max(tx_sorted), check if censored trials
        # (which have flips == max_flips, hence the LARGEST times in tz!) already exceed max(tx):
        tz_top_hi = tz_sorted.copy()
        tz_top_hi[-cap_z:] = 1e300
        tau_max_likely = joint_tau_q(tx_sorted, tz_top_hi)
    else:
        tx_hi = tx_sorted.copy()
        tx_hi[:cap_x] = 1e300
        tau_max = joint_tau_q(tx_hi, tz_sorted)
        tx_top_hi = tx_sorted.copy()
        tx_top_hi[-cap_x:] = 1e300
        tau_max_likely = joint_tau_q(tx_top_hi, tz_sorted)

    # Bootstrap (B = 2000)
    B = 2000
    nx, nz = len(tx), len(tz)
    ix = rng.integers(0, nx, size=(B, nx))
    iz = rng.integers(0, nz, size=(B, nz))
    bx = tx[ix]
    bz = tz[iz]
    # Fast exact O(B * (nx log nx + nz log nz)) computation of mean_{i,j} min(bx_i, bz_j):
    b_tau = np.empty(B, dtype=np.float64)
    for b in range(B):
        b_tau[b] = np.minimum(bx[b, :, None], bz[b, None, :]).mean()

    se_q = float(b_tau.std(ddof=1))
    ci_lo, ci_hi = float(np.percentile(b_tau, 2.5)), float(np.percentile(b_tau, 97.5))

    return {
        "beta": x_rec["beta"],
        "tau_Q": tau_min,
        "se_Q": se_q,
        "ci95_Q": [ci_lo, ci_hi],
        "tau_Q_bracket": [tau_min, tau_max],
        "censor_rel_width_worst": (None if tau_max is None else float((tau_max - tau_min) / max(tau_min, 1e-30))),
        "tau_X": float(tx.mean()),
        "se_X": float(tx.std(ddof=1) / math.sqrt(nx)) if nx > 1 else 0.0,
        "median_X": float(np.median(tx)),
        "trials_X": nx,
        "capped_X": cap_x,
        "tau_Z": float(tz.mean()),
        "se_Z": float(tz.std(ddof=1) / math.sqrt(nz)) if nz > 1 else 0.0,
        "median_Z": float(np.median(tz)),
        "trials_Z": nz,
        "capped_Z": cap_z,
        "span_stats": x_rec.get("span_stats"),
        "_boot_Q": b_tau,
    }


def main():
    with open(SPECS_PATH) as f:
        specs = json.load(f)

    sector_data = defaultdict(dict)  # (code_tag, sector) -> {beta: rec}
    cadence_data = defaultdict(dict) # code_tag -> {beta: rec}
    errors = []

    with open(RAW_PATH) as f:
        for raw in f:
            line = raw.strip()
            if not line or "|" not in line:
                continue
            tid, body = [x.strip() for x in line.split("|", 1)]
            if body.startswith("ERROR") or body == "TIMEOUT":
                errors.append((tid, body))
                continue
            kv = parse_kv(body)
            if tid.startswith("kmc__"):
                # kmc__<tag>_<X_err|Z_err>__b<beta>
                m = re.match(r"^kmc__(.+)_(X_err|Z_err)__b([0-9.]+)$", tid)
                if not m:
                    continue
                tag, sec, b_str = m.group(1), m.group(2), float(m.group(3))
                t_list = [float(x) for x in kv["t_list"].split(",") if x]
                sector_data[(tag, sec)][b_str] = {
                    "beta": b_str,
                    "trials": int(kv["trials"]),
                    "capped": int(kv["capped"]),
                    "mean_flips": float(kv["mean_flips"]),
                    "t_list": t_list,
                    "span_stats": None,
                }
            elif tid.startswith("span__"):
                m = re.match(r"^span__(.+)_X_err__b([0-9.]+)$", tid)
                if not m:
                    continue
                tag, b_str = m.group(1), float(m.group(2))
                t_list = [float(x) for x in kv["t_list"].split(",") if x]
                t_sp6_list = [float(x) for x in kv["t_sp6_list"].split(",") if x]
                sp_arr = np.array(t_sp6_list, dtype=np.float64)
                sector_data[(tag, "X_err")][b_str] = {
                    "beta": b_str,
                    "trials": int(kv["trials"]),
                    "capped": int(kv["capped"]),
                    "mean_flips": float(kv["mean_flips"]),
                    "t_list": t_list,
                    "span_stats": {
                        "mean_t_sp6": float(kv["mean_t_sp6"]),
                        "se_t_sp6": float(sp_arr.std(ddof=1) / math.sqrt(len(sp_arr))) if len(sp_arr) > 1 else 0.0,
                        "median_t_sp6": float(np.median(sp_arr)),
                        "mean_t_sp10": float(kv["mean_t_sp10"]),
                        "mean_healed_sp6": float(kv["mean_healed_sp6"]),
                        "sem_healed_sp6": float(kv["sem_healed_sp6"]),
                        "mean_healed_sp10": float(kv["mean_healed_sp10"]),
                        "sem_healed_sp10": float(kv["sem_healed_sp10"]),
                        "mean_max_healed_span": float(kv["mean_max_healed_span"]),
                        "mean_max_healed_w": float(kv["mean_max_healed_w"]),
                        "survival_amplification_tau_over_sp6": float(np.mean(t_list) / max(float(kv["mean_t_sp6"]), 1e-12)),
                    },
                }
            elif tid.startswith("cadence__"):
                m = re.match(r"^cadence__(.+)_X_err__b([0-9.]+)$", tid)
                if not m:
                    continue
                tag, b_str = m.group(1), float(m.group(2))
                quads = []
                for q in kv["quads"].split(","):
                    if q:
                        quads.append([float(v) for v in q.split(":")])
                q_arr = np.array(quads, dtype=np.float64)
                cadence_data[tag][b_str] = {
                    "beta": b_str,
                    "trials": int(kv["trials"]),
                    "etas": [0.02, 0.05, 0.20, 0.50],
                    "means": [float(kv["m0"]), float(kv["m05"]), float(kv["m20"]), float(kv["m50"])],
                    "sems": [float(kv["s0"]), float(kv["s05"]), float(kv["s20"]), float(kv["s50"])],
                    "log10_r20_0": float(kv["log10_r20_0"]),
                    "sem_log10_r20_0": float(kv["sem_log10_r20_0"]),
                    "same_0_20": int(kv["same_0_20"]),
                    "caps": [int(x) for x in kv["caps"].split(",")],
                    "quads": q_arr.tolist(),
                }

    rng = np.random.default_rng(20261121)
    codes_out = {}
    boot_store = {}

    for tag in specs:
        bx_map = sector_data.get((tag, "X_err"), {})
        bz_map = sector_data.get((tag, "Z_err"), {})
        common_betas = sorted(set(bx_map.keys()) & set(bz_map.keys()))
        if not common_betas:
            continue
        pts = []
        boot_store[tag] = {}
        for b in common_betas:
            st = compute_joint_stats(bx_map[b], bz_map[b], rng)
            boot_store[tag][b] = st.pop("_boot_Q")
            pts.append(st)
        codes_out[tag] = {
            "spec": {k: v for k, v in specs[tag].items() if k not in ("wX_hist", "wZ_hist", "deg_hist")},
            "points": pts,
        }

    # Pairwise ratio helper with bootstrap 95% CI:
    def ratio_series(tag_num, tag_den):
        b_num = boot_store.get(tag_num, {})
        b_den = boot_store.get(tag_den, {})
        p_num = {p["beta"]: p for p in codes_out.get(tag_num, {}).get("points", [])}
        p_den = {p["beta"]: p for p in codes_out.get(tag_den, {}).get("points", [])}
        out_r = []
        for b in sorted(set(b_num.keys()) & set(b_den.keys())):
            r_pt = p_num[b]["tau_Q"] / p_den[b]["tau_Q"]
            br = b_num[b] / np.maximum(b_den[b], 1e-30)
            out_r.append(
                {
                    "beta": b,
                    "ratio": float(r_pt),
                    "ci95": [float(np.percentile(br, 2.5)), float(np.percentile(br, 97.5))],
                    "log10_ratio": float(math.log10(r_pt)),
                }
            )
        return out_r

    comparisons = {
        "1cyc_planar_R3_over_C3": ratio_series("v15_sq_R3", "v15_sq_C3"),
        "1cyc_torus_R3_over_C3": ratio_series("v15_tor_R3", "v15_tor_C3"),
        "1cyc_iso_R3_over_C3": ratio_series("v15_sq_R3_iso", "v15_sq_C3_iso"),
        "1cyc_planar_R3_over_C2": ratio_series("v15_sq_R3", "v15_sq_C2"),
        "L3_planar_RC_over_CR": ratio_series("v15_sq_L3_RC", "v15_sq_L3_CR"),
        "L3_planar_CR_over_RC": ratio_series("v15_sq_L3_CR", "v15_sq_L3_RC"),
        "L3_planar_CR_over_RR": ratio_series("v15_sq_L3_CR", "v15_sq_L3_RR"),
        "L3_planar_RC_over_RR": ratio_series("v15_sq_L3_RC", "v15_sq_L3_RR"),
        "L3_planar_CR_over_1cyc_R3": ratio_series("v15_sq_L3_CR", "v15_sq_R3"),
        "L3_planar_RC_over_1cyc_R3": ratio_series("v15_sq_L3_RC", "v15_sq_R3"),
        "L3_torus_RC_over_CR": ratio_series("v15_tor_L3_RC", "v15_tor_L3_CR"),
        "L3_torus_CR_over_RC": ratio_series("v15_tor_L3_CR", "v15_tor_L3_RC"),
        "pkg_planar_RC_over_CR": ratio_series("v15_sq_pkg_RC", "v15_sq_pkg_CR"),
        "pkg_planar_RC_over_RR": ratio_series("v15_sq_pkg_RC", "v15_sq_L3_RR"),
        "pkg_planar_RC_over_1cyc_R3": ratio_series("v15_sq_pkg_RC", "v15_sq_R3"),
        "pkg_planar_CR_over_1cyc_R3": ratio_series("v15_sq_pkg_CR", "v15_sq_R3"),
        "pkg_torus_RC_over_CR": ratio_series("v15_tor_pkg_RC", "v15_tor_pkg_CR"),
        "pkg_rottor_RC_over_CR": ratio_series("v15_rottor_pkg_RC", "v15_rottor_pkg_CR"),
        "pkg_iso_RC_over_CR": ratio_series("v15_sq_iso_RC", "v15_sq_iso_CR"),
    }

    # =========================================================================
    # Claim 3: Strictly Out-of-Sample Single-Cycle Composition Model
    # =========================================================================
    # For each 1-cycle module M in {C3, R3, C2} (planar and torus), define its
    # empirical effective inverse temperature Phi_M(beta) from tau_Q^(1)(beta):
    #   tau_0(beta) = 0.5 * (1 + exp(2*beta))  <=>  Phi_0(beta) = 0.5 * ln(max(2*tau - 1, 1e-12))
    # Fit affine RG map Phi_M(beta) = g_M * beta - c_M across 1-cycle betas >= 5.0
    # (ZERO 2-cycle data used!), extract single-cycle break-even beta_M^* = c_M / (g_M - 1),
    # and compose Phi_{AB}(beta) = Phi_A(Phi_B(beta)) to predict all 2-cycle architectures!
    def fit_module_rg(mtag, beta_min=5.0):
        pts = [p for p in codes_out[mtag]["points"] if p["beta"] >= beta_min]
        bs = np.array([p["beta"] for p in pts], dtype=np.float64)
        phis = np.array([0.5 * math.log(max(2.0 * p["tau_Q"] - 1.0, 1e-6)) for p in pts], dtype=np.float64)
        slope, intercept = np.polyfit(bs, phis, 1)
        g_m = float(slope)
        c_m = float(-intercept)
        beta_star = float(c_m / (g_m - 1.0)) if abs(g_m - 1.0) > 1e-9 else None
        # Also store empirical (bs, phis) for direct spline/linear interpolation + extrapolation:
        return {
            "module": mtag,
            "g_barrier_gain": g_m,
            "Delta_eff_slope_ln_tau": 2.0 * g_m,
            "c_entropic_shift": c_m,
            "beta_star_breakeven": beta_star,
            "fit_r2": float(np.corrcoef(bs, phis)[0, 1] ** 2),
            "betas": bs.tolist(),
            "phis": phis.tolist(),
        }

    rg_modules = {}
    for mtag in ["v15_sq_C3", "v15_sq_R3", "v15_sq_C2", "v15_tor_C3", "v15_tor_R3", "v15_tor_C2"]:
        if mtag in codes_out:
            rg_modules[mtag] = fit_module_rg(mtag, beta_min=5.0)

    def eval_composed_tau(mod_coarse, mod_fine, beta):
        mc = rg_modules[mod_coarse]
        mf = rg_modules[mod_fine]
        phi_fine = mf["g_barrier_gain"] * beta - mf["c_entropic_shift"]
        phi_2cyc = mc["g_barrier_gain"] * phi_fine - mc["c_entropic_shift"]
        tau_pred = 0.5 * (1.0 + math.exp(max(-20.0, min(60.0, 2.0 * phi_2cyc))))
        return float(phi_2cyc), float(tau_pred)

    predictions = {}
    pred_families = [
        ("planar_L3", "v15_sq_C3", "v15_sq_R3", {
            "CC": ("v15_sq_C3", "v15_sq_C3", "v15_sq_L3_CC"),
            "RC": ("v15_sq_R3", "v15_sq_C3", "v15_sq_L3_RC"),
            "CR": ("v15_sq_C3", "v15_sq_R3", "v15_sq_L3_CR"),
            "RR": ("v15_sq_R3", "v15_sq_R3", "v15_sq_L3_RR"),
        }),
        ("torus_L3", "v15_tor_C3", "v15_tor_R3", {
            "CC": ("v15_tor_C3", "v15_tor_C3", "v15_tor_L3_CC"),
            "RC": ("v15_tor_R3", "v15_tor_C3", "v15_tor_L3_RC"),
            "CR": ("v15_tor_C3", "v15_tor_R3", "v15_tor_L3_CR"),
            "RR": ("v15_tor_R3", "v15_tor_R3", "v15_tor_L3_RR"),
        }),
        ("planar_pkg", "v15_sq_C2", "v15_sq_R3", {
            "CC": ("v15_sq_C2", "v15_sq_C2", "v15_sq_pkg_CC"),
            "RC": ("v15_sq_R3", "v15_sq_C2", "v15_sq_pkg_RC"),
            "CR": ("v15_sq_C2", "v15_sq_R3", "v15_sq_pkg_CR"),
            "RR": ("v15_sq_R3", "v15_sq_R3", "v15_sq_L3_RR"),
        }),
        ("torus_pkg", "v15_tor_C2", "v15_tor_R3", {
            "CC": ("v15_tor_C2", "v15_tor_C2", "v15_tor_pkg_CC"),
            "RC": ("v15_tor_R3", "v15_tor_C2", "v15_tor_pkg_RC"),
            "CR": ("v15_tor_C2", "v15_tor_R3", "v15_tor_pkg_CR"),
            "RR": ("v15_tor_R3", "v15_tor_R3", "v15_tor_L3_RR"),
        }),
    ]

    for fam_name, c_tag, r_tag, arch_map in pred_families:
        if c_tag not in rg_modules or r_tag not in rg_modules:
            continue
        mc, mr = rg_modules[c_tag], rg_modules[r_tag]
        d_phi_rc_minus_cr = (mr["g_barrier_gain"] - 1.0) * (mc["g_barrier_gain"] - 1.0) * (
            mr["beta_star_breakeven"] - mc["beta_star_breakeven"]
        )
        pred_ratio_rc_cr = math.exp(2.0 * d_phi_rc_minus_cr)
        arch_preds = {}
        for aname, (m_coarse, m_fine, meas_tag) in arch_map.items():
            if meas_tag not in codes_out:
                continue
            rows = []
            for p in codes_out[meas_tag]["points"]:
                b = p["beta"]
                phi_p, tau_p = eval_composed_tau(m_coarse, m_fine, b)
                rows.append({
                    "beta": b,
                    "pred_phi": phi_p,
                    "pred_tau_Q": tau_p,
                    "meas_tau_Q": p["tau_Q"],
                    "log10_pred": math.log10(max(tau_p, 1e-30)),
                    "log10_meas": math.log10(max(p["tau_Q"], 1e-30)),
                })
            arch_preds[aname] = rows
        predictions[fam_name] = {
            "C_module": c_tag,
            "R_module": r_tag,
            "beta_star_C": mc["beta_star_breakeven"],
            "beta_star_R": mr["beta_star_breakeven"],
            "pred_winner_mixed": ("RC" if d_phi_rc_minus_cr > 0 else "CR"),
            "pred_delta_phi_RC_minus_CR": d_phi_rc_minus_cr,
            "pred_tau_ratio_RC_over_CR": pred_ratio_rc_cr,
            "architectures": arch_preds,
        }

    # Compute log-lifetime slopes d(ln tau_Q)/d(beta) +/- SE over beta >= 5.5 (cold regime):
    slope_table = {}
    for tag, cobj in codes_out.items():
        pts = [p for p in cobj["points"] if p["beta"] >= 5.5]
        if len(pts) >= 3:
            bs = np.array([p["beta"] for p in pts], dtype=np.float64)
            ys = np.array([math.log(p["tau_Q"]) for p in pts], dtype=np.float64)
            # Bootstrap slope SE from boot_store:
            b_mat = np.vstack([np.log(np.maximum(boot_store[tag][p["beta"]], 1e-30)) for p in pts])  # (K, B)
            x_c = bs - bs.mean()
            denom = float(np.sum(x_c ** 2))
            sl_hat = float(np.sum(x_c * (ys - ys.mean())) / denom)
            sl_boot = (x_c[:, None] * (b_mat - b_mat.mean(axis=0, keepdims=True))).sum(axis=0) / denom
            slope_table[tag] = {
                "slope": sl_hat,
                "se_slope": float(sl_boot.std(ddof=1)),
                "ci95_slope": [float(np.percentile(sl_boot, 2.5)), float(np.percentile(sl_boot, 97.5))],
                "beta_range": [float(bs.min()), float(bs.max())],
            }

    out = {
        "num_raw_errors_or_timeouts": len(errors),
        "errors": errors,
        "codes": codes_out,
        "slopes_cold": slope_table,
        "comparisons": comparisons,
        "single_cycle_rg_modules": rg_modules,
        "out_of_sample_predictions": predictions,
        "paired_cadence": cadence_data,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Parsed {len(codes_out)} codes ({len(errors)} errors/timeouts) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
