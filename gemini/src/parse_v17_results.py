#!/usr/bin/env python3
"""
parse_v17_results.py
Statistical parser for the V17 campaign (`cluster_v17_results.txt` + `v17_mask_specs.json`),
integrated with `parsed_v16_data.json` into `parsed_v17_data.json`:

  1. Whole-trajectory independent-sector bootstrap (B = 2,000) for passive operational
     recovery P_id(t) = p_X(t) * p_Z(t), no-crossing survival P_nocross(t) = u_X(t) * u_Z(t),
     post-crossing healing P_healed(t) = P_id(t) - P_nocross(t), and threshold times
     t_p(A, beta) = inf { t >= 0 : P_id^A(t, beta) < p } (p = 0.80 primary, p = 0.50 secondary).
  2. Full 3x3 Coarse-by-Fine Operational Matrix at beta = 6.50 (planar K=1 & torus K=2):
     fixed-row contrasts, fixed-column contrasts, standalone vs fine-constituent rankings,
     marginal coarse-bridge benefits G_B(beta) = t_0.80(R3 B; beta) / t_0.80(C3 B; beta),
     separable additive-log fit residuals, and scale-interaction ratio G_{C2}(beta) / G_{C3}(beta).
  3. Cold Operational Readout at beta = 6.75 and beta = 7.00 (R3C2, C3C2, C2C2, C2R3, R3C3, C3C3).
  4. Pre-declared Causal Marginal-Benefit Test G_{C3}(mu) = t_0.80(R3C3; mu) / t_0.80(C3C3; mu)
     (primary) and tau_Q(R3C3; mu) / tau_Q(C3C3; mu) (supporting) across mu in {1.0, 0.3, 0.1, 0.03, 0.01}
     for `fineflat` vs `fineseam` (and `allflat` vs `allseam`), with whole-trajectory 95% CIs.
"""

from collections import defaultdict
import json
import math
import os
import re
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V16_PARSED = os.path.join(ROOT, "parsed_v16_data.json")
V17_RAW = os.path.join(ROOT, "cluster_v17_results.txt")
V17_MASKS = os.path.join(ROOT, "v17_mask_specs.json")
OUT_PATH = os.path.join(ROOT, "parsed_v17_data.json")


def parse_kv(body):
    kv = {}
    for tok in body.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            kv[k] = v
    return kv


def crossing_time_strict(t_grid, p_curve, thresh):
    """
    Computes inf { t >= 0 : p_curve(t) < thresh } with log10-linear interpolation
    between the last grid point >= thresh and the first grid point < thresh.
    Returns (t_val, crossed_bool). If never < thresh, returns (t_grid[-1], False).
    """
    for g in range(len(t_grid)):
        if p_curve[g] < thresh:
            if g == 0:
                return float(t_grid[0]), True
            p0, p1 = float(p_curve[g - 1]), float(p_curve[g])
            if p0 <= p1:
                return float(t_grid[g]), True
            frac = (p0 - thresh) / (p0 - p1)
            lg0, lg1 = math.log10(t_grid[g - 1]), math.log10(t_grid[g])
            return float(10.0 ** (lg0 + frac * (lg1 - lg0))), True
    return float(t_grid[-1]), False


def eval_readout_pair(x_shards, z_shards, rng, B=2000, extra_x_pass=None, extra_z_pass=None):
    xbits, x_tfirst, x_cfirst = [], [], []
    for off in sorted(x_shards):
        sh = x_shards[off]
        xbits.extend(sh["bits"])
        for t, c in sh["tc_first"]:
            x_tfirst.append(t)
            x_cfirst.append(c)

    zbits, z_tfirst, z_cfirst = [], [], []
    for off in sorted(z_shards):
        sh = z_shards[off]
        zbits.extend(sh["bits"])
        for t, c in sh["tc_first"]:
            z_tfirst.append(t)
            z_cfirst.append(c)

    # All grids start at 10^2.00 with 0.10 decade spacing; align to common prefix min_ng:
    lengths = [len(s) for s in xbits] + [len(s) for s in zbits]
    if extra_x_pass:
        lengths.extend(len(s) for s in extra_x_pass)
    if extra_z_pass:
        lengths.extend(len(s) for s in extra_z_pass)
    ng = min(lengths)
    lg0 = 2.00
    lg1 = lg0 + 0.10 * (ng - 1)
    gspec = f"{lg0:.4f}:{lg1:.4f}:{ng}"
    t_grid = np.array([10.0 ** (lg0 + 0.10 * i) for i in range(ng)], dtype=np.float64)
    xbits = [s[:ng] for s in xbits]
    zbits = [s[:ng] for s in zbits]

    nx_v17, nz_v17 = len(xbits), len(zbits)
    all_x_pass_strs = list(xbits) + ([s[:ng] for s in extra_x_pass] if extra_x_pass else [])
    all_z_pass_strs = list(zbits) + ([s[:ng] for s in extra_z_pass] if extra_z_pass else [])
    nx, nz = len(all_x_pass_strs), len(all_z_pass_strs)

    cap_grid_x = sum(1 for s in all_x_pass_strs if "C" in s)
    cap_grid_z = sum(1 for s in all_z_pass_strs if "C" in s)

    # Pooled pass_mat: ch in ('1', '2'); V17 decomposition surv_mat: ch == '2'
    X_pass = np.array([[1.0 if ch in ("1", "2") else 0.0 for ch in s] for s in all_x_pass_strs], dtype=np.float64)
    Z_pass = np.array([[1.0 if ch in ("1", "2") else 0.0 for ch in s] for s in all_z_pass_strs], dtype=np.float64)
    X_pass_v17 = np.array([[1.0 if ch in ("1", "2") else 0.0 for ch in s] for s in xbits], dtype=np.float64)
    Z_pass_v17 = np.array([[1.0 if ch in ("1", "2") else 0.0 for ch in s] for s in zbits], dtype=np.float64)
    X_surv = np.array([[1.0 if ch == "2" else 0.0 for ch in s] for s in xbits], dtype=np.float64)
    Z_surv = np.array([[1.0 if ch == "2" else 0.0 for ch in s] for s in zbits], dtype=np.float64)

    px = X_pass.mean(axis=0)
    pz = Z_pass.mean(axis=0)
    pid = px * pz

    ux = X_surv.mean(axis=0)
    uz = Z_surv.mean(axis=0)
    pid_v17 = X_pass_v17.mean(axis=0) * Z_pass_v17.mean(axis=0)
    pnocross = ux * uz
    phealed = pid_v17 - pnocross

    t80, c80 = crossing_time_strict(t_grid, pid, 0.80)
    t50, c50 = crossing_time_strict(t_grid, pid, 0.50)
    t80_nc, c80_nc = crossing_time_strict(t_grid, pnocross, 0.80)

    # Interpolate P_healed and P_nocross at t = t80:
    idx_hi = int(np.searchsorted(t_grid, t80))
    if idx_hi <= 0:
        pheal_at_t80 = float(phealed[0])
        pnc_at_t80 = float(pnocross[0])
    elif idx_hi >= ng:
        pheal_at_t80 = float(phealed[-1])
        pnc_at_t80 = float(pnocross[-1])
    else:
        w = (math.log10(t80) - math.log10(t_grid[idx_hi - 1])) / (
            math.log10(t_grid[idx_hi]) - math.log10(t_grid[idx_hi - 1])
        )
        pheal_at_t80 = float((1.0 - w) * phealed[idx_hi - 1] + w * phealed[idx_hi])
        pnc_at_t80 = float((1.0 - w) * pnocross[idx_hi - 1] + w * pnocross[idx_hi])

    # Whole-trajectory independent-sector bootstrap (retaining full temporal correlation):
    ix = rng.integers(0, nx, size=(B, nx))
    iz = rng.integers(0, nz, size=(B, nz))
    b_px = X_pass[ix].mean(axis=1)
    b_pz = Z_pass[iz].mean(axis=1)
    b_pid = b_px * b_pz

    ix17 = rng.integers(0, nx_v17, size=(B, nx_v17))
    iz17 = rng.integers(0, nz_v17, size=(B, nz_v17))
    b_ux = X_surv[ix17].mean(axis=1)
    b_uz = Z_surv[iz17].mean(axis=1)
    b_pnc = b_ux * b_uz

    b_t80 = np.empty(B, dtype=np.float64)
    b_c80 = np.empty(B, dtype=bool)
    b_t50 = np.empty(B, dtype=np.float64)
    b_c50 = np.empty(B, dtype=bool)
    b_t80_nc = np.empty(B, dtype=np.float64)
    for k in range(B):
        b_t80[k], b_c80[k] = crossing_time_strict(t_grid, b_pid[k], 0.80)
        b_t50[k], b_c50[k] = crossing_time_strict(t_grid, b_pid[k], 0.50)
        b_t80_nc[k], _ = crossing_time_strict(t_grid, b_pnc[k], 0.80)

    # First-passage tau_Q on the exact same V17 trajectories:
    tx_arr = np.asarray(x_tfirst, dtype=np.float64)
    cx_arr = np.asarray(x_cfirst, dtype=np.int32)
    tz_arr = np.asarray(z_tfirst, dtype=np.float64)
    cz_arr = np.asarray(z_cfirst, dtype=np.int32)
    tau_q_same = float(np.minimum(tx_arr[:, None], tz_arr[None, :]).mean())
    b_tau_same = np.empty(B, dtype=np.float64)
    bx_tf = tx_arr[ix17]
    bz_tf = tz_arr[iz17]
    for k in range(B):
        b_tau_same[k] = np.minimum(bx_tf[k, :, None], bz_tf[k, None, :]).mean()

    zero_censor_first = (int(cx_arr.sum()) == 0) and (
        int(cz_arr.sum()) == 0 or float(tx_arr.max()) < float(tz_arr.min())
    )

    healed_trials_x = sum(1 for s in xbits if "1" in s)

    return {
        "n_X": nx,
        "n_Z": nz,
        "n_X_decomp": nx_v17,
        "n_Z_decomp": nz_v17,
        "capped_grid_X": cap_grid_x,
        "capped_grid_Z": cap_grid_z,
        "healed_trials_X": healed_trials_x,
        "grid_spec": gspec,
        "t_grid": t_grid.tolist(),
        "p_X": px.tolist(),
        "p_Z": pz.tolist(),
        "P_id": pid.tolist(),
        "P_id_ci95_lo": np.percentile(b_pid, 2.5, axis=0).tolist(),
        "P_id_ci95_hi": np.percentile(b_pid, 97.5, axis=0).tolist(),
        "P_nocross": pnocross.tolist(),
        "P_healed": phealed.tolist(),
        "t_80": t80,
        "t_80_crossed": bool(c80),
        "t_80_boot_crossed_frac": float(b_c80.mean()),
        "t_80_ci95": [float(np.percentile(b_t80, 2.5)), float(np.percentile(b_t80, 97.5))],
        "t_80_nocross": t80_nc,
        "t_80_nocross_crossed": bool(c80_nc),
        "t_80_nocross_ci95": [float(np.percentile(b_t80_nc, 2.5)), float(np.percentile(b_t80_nc, 97.5))],
        "healing_gain_t80_over_nocross": float(t80 / max(t80_nc, 1e-30)),
        "P_nocross_at_t80": pnc_at_t80,
        "P_healed_at_t80": pheal_at_t80,
        "t_50": t50,
        "t_50_crossed": bool(c50),
        "t_50_boot_crossed_frac": float(b_c50.mean()),
        "t_50_ci95": [float(np.percentile(b_t50, 2.5)), float(np.percentile(b_t50, 97.5))],
        "same_traj_tau_X": float(tx_arr.mean()),
        "same_traj_cap_first_X": int(cx_arr.sum()),
        "same_traj_tau_Q": tau_q_same,
        "same_traj_tau_Q_ci95": [float(np.percentile(b_tau_same, 2.5)), float(np.percentile(b_tau_same, 97.5))],
        "same_traj_zero_censor_first": bool(zero_censor_first),
        "_boot_t80": b_t80,
        "_boot_c80": b_c80,
        "_boot_t50": b_t50,
        "_boot_tauQ": b_tau_same,
    }


def ratio_ci(pt_num, b_num, c_num, pt_den, b_den, c_den):
    if not c_den:
        return {"status": "denominator_not_crossed", "ratio": None, "ci95": None}
    r = float(pt_num / max(pt_den, 1e-30))
    br = b_num / np.maximum(b_den, 1e-30)
    ci = [float(np.percentile(br, 2.5)), float(np.percentile(br, 97.5))]
    return {
        "status": ("uncensored" if c_num else "lower_bound_numerator_uncrossed"),
        "ratio": r,
        "ci95": ci,
    }


def main():
    with open(V16_PARSED) as f:
        v16 = json.load(f)
    with open(V17_MASKS) as f:
        mask_meta = json.load(f)

    # Key: (prefix, tag, sec, mkind, mu, beta) -> {off: {...}}
    raw_groups = defaultdict(dict)
    v17_errors = []

    with open(V17_RAW) as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            tid, body = [x.strip() for x in line.split("|", 1)]
            if body.startswith("ERROR") or body == "TIMEOUT":
                v17_errors.append((tid, body))
                continue
            kv = parse_kv(body)
            m = re.match(
                r"^(m65|c70|c675|kin)__(.+)_(X_err|Z_err)__(none|fineflat|fineseam|allflat|allseam)__mu([0-9.]+)__b([0-9.]+)__o(\d+)$",
                tid,
            )
            if not m:
                v17_errors.append((tid, "UNPARSED_TID"))
                continue
            pref, tag, sec, mkind, mu_s, b_s, off_s = m.groups()
            mu, beta, off = float(mu_s), float(b_s), int(off_s)
            tcf = []
            for item in kv["tc_first"].split(","):
                if item:
                    ts, cs = item.split(":")
                    tcf.append((float(ts), int(cs)))
            bits = [s for s in kv["bits"].split(",") if s]
            raw_groups[(tag, sec, mkind, mu, beta)][off] = {
                "grid": kv["grid"],
                "tc_first": tcf,
                "bits": bits,
                "masked": int(kv["masked"]),
            }

    # Load independent V16 passive readout bitstrings (`frout__...__b6.50`) to pool at beta = 6.50:
    v16_frout_bits = defaultdict(dict)  # (tag, sec, beta) -> {off: [bits]}
    v16_raw_path = os.path.join(ROOT, "cluster_v16_results.txt")
    if os.path.exists(v16_raw_path):
        with open(v16_raw_path) as f:
            for raw in f:
                line = raw.strip()
                if not line or "|" not in line or not line.startswith("frout__"):
                    continue
                tid, body = [x.strip() for x in line.split("|", 1)]
                if body.startswith("ERROR") or body == "TIMEOUT":
                    continue
                m16 = re.match(r"^frout__(.+)_(X_err|Z_err)__b([0-9.]+)__o(\d+)$", tid)
                if m16:
                    kv16 = parse_kv(body)
                    v16_frout_bits[(m16.group(1), m16.group(2), float(m16.group(3)))][int(m16.group(4))] = [
                        s for s in kv16["bits"].split(",") if s
                    ]

    def get_v16_extra(tag, sec, beta):
        omap = v16_frout_bits.get((tag, sec, beta), {})
        out = []
        for off in sorted(omap):
            out.extend(omap[off])
        return out

    rng = np.random.default_rng(20261123)

    # 1. Standard Operational Readout (mkind == 'none', mu == 1.0) across beta in {6.50, 6.75, 7.00}:
    std_readout = defaultdict(dict)
    std_readout_v17only = defaultdict(dict)
    boot_t80 = {}
    boot_c80 = {}
    boot_t80_v17 = {}
    boot_c80_v17 = {}
    boot_tauQ = {}

    std_keys = sorted({(tag, beta) for (tag, sec, mk, mu, beta) in raw_groups if mk == "none" and abs(mu - 1.0) < 1e-9})
    for tag, beta in std_keys:
        xk = (tag, "X_err", "none", 1.0, beta)
        zk = (tag, "Z_err", "none", 1.0, beta)
        if xk not in raw_groups or zk not in raw_groups:
            continue
        # V17-only cohort (exact matching baseline for V17 causal mu=1.00):
        res17 = eval_readout_pair(raw_groups[xk], raw_groups[zk], rng)
        boot_t80_v17[(tag, beta)] = res17.pop("_boot_t80")
        boot_c80_v17[(tag, beta)] = res17["t_80_crossed"] and (res17["t_80_boot_crossed_frac"] >= 0.95)
        res17.pop("_boot_c80")
        res17.pop("_boot_t50")
        boot_tauQ[(tag, beta)] = res17.pop("_boot_tauQ")
        std_readout_v17only[tag][f"{beta:.2f}"] = res17

        # Pooled V16 + V17 independent passive readout cohort (n_X = 96..176 at beta=6.50):
        ex_x = get_v16_extra(tag, "X_err", beta)
        ex_z = get_v16_extra(tag, "Z_err", beta)
        if ex_x and ex_z:
            res = eval_readout_pair(raw_groups[xk], raw_groups[zk], rng, extra_x_pass=ex_x, extra_z_pass=ex_z)
            boot_t80[(tag, beta)] = res.pop("_boot_t80")
            boot_c80[(tag, beta)] = res["t_80_crossed"] and (res["t_80_boot_crossed_frac"] >= 0.95)
            res.pop("_boot_c80")
            res.pop("_boot_t50")
            res.pop("_boot_tauQ")
            std_readout[tag][f"{beta:.2f}"] = res
        else:
            boot_t80[(tag, beta)] = boot_t80_v17[(tag, beta)]
            boot_c80[(tag, beta)] = boot_c80_v17[(tag, beta)]
            std_readout[tag][f"{beta:.2f}"] = res17

    def t80_comp(tag_num, tag_den, beta):
        rn = std_readout[tag_num][f"{beta:.2f}"]
        rd = std_readout[tag_den][f"{beta:.2f}"]
        return ratio_ci(
            rn["t_80"], boot_t80[(tag_num, beta)], boot_c80[(tag_num, beta)],
            rd["t_80"], boot_t80[(tag_den, beta)], boot_c80[(tag_den, beta)],
        )

    # 2. Full 3x3 Coarse-by-Fine Matrix Analysis at beta = 6.50 (planar & torus):
    MAT_TAGS = {
        "planar": {
            ("C2", "C2"): "v15_sq_pkg_CC",
            ("C2", "C3"): "v16_sq_C2C3",
            ("C2", "R3"): "v15_sq_pkg_CR",
            ("C3", "C2"): "v16_sq_C3C2",
            ("C3", "C3"): "v15_sq_L3_CC",
            ("C3", "R3"): "v15_sq_L3_CR",
            ("R3", "C2"): "v15_sq_pkg_RC",
            ("R3", "C3"): "v15_sq_L3_RC",
            ("R3", "R3"): "v15_sq_L3_RR",
        },
        "torus": {
            ("C2", "C2"): "v15_tor_pkg_CC",
            ("C2", "C3"): "v16_tor_C2C3",
            ("C2", "R3"): "v15_tor_pkg_CR",
            ("C3", "C2"): "v16_tor_C3C2",
            ("C3", "C3"): "v15_tor_L3_CC",
            ("C3", "R3"): "v15_tor_L3_CR",
            ("R3", "C2"): "v15_tor_pkg_RC",
            ("R3", "C3"): "v15_tor_L3_RC",
            ("R3", "R3"): "v15_tor_L3_RR",
        },
    }
    MOD_1CYC = {
        "planar": {"C2": "v15_sq_C2", "C3": "v15_sq_C3", "R3": "v15_sq_R3"},
        "torus": {"C2": "v15_tor_C2", "C3": "v15_tor_C3", "R3": "v15_tor_R3"},
    }

    matrix_summary = {}
    mods = ["C2", "C3", "R3"]
    for seed_name in ("planar", "torus"):
        mtags = MAT_TAGS[seed_name]
        s1 = MOD_1CYC[seed_name]
        cells = {}
        log_mat = np.zeros((3, 3), dtype=np.float64)
        for i, A in enumerate(mods):
            for j, B_m in enumerate(mods):
                tg = mtags[(A, B_m)]
                r = std_readout[tg]["6.50"]
                cells[f"{A}_{B_m}"] = {
                    "tag": tg,
                    "t_80": r["t_80"],
                    "t_80_ci95": r["t_80_ci95"],
                    "t_80_nocross": r["t_80_nocross"],
                    "P_healed_at_t80": r["P_healed_at_t80"],
                    "same_traj_tau_Q": r["same_traj_tau_Q"],
                }
                log_mat[i, j] = math.log(r["t_80"])

        # Additive log fit: log t(A, B) = c + u_A + v_B (sum u_A = sum v_B = 0)
        grand = float(log_mat.mean())
        u_row = log_mat.mean(axis=1) - grand
        v_col = log_mat.mean(axis=0) - grand
        pred = grand + u_row[:, None] + v_col[None, :]
        resid = log_mat - pred
        max_mult_err = float(np.exp(np.max(np.abs(resid))))

        row_contrasts = {}
        for A in mods:
            row_contrasts[f"coarse_{A}_fine_C2_over_C3"] = t80_comp(mtags[(A, "C2")], mtags[(A, "C3")], 6.50)
            row_contrasts[f"coarse_{A}_fine_C2_over_R3"] = t80_comp(mtags[(A, "C2")], mtags[(A, "R3")], 6.50)

        col_contrasts = {}
        for B_m in mods:
            vals = [std_readout[mtags[(A, B_m)]]["6.50"]["t_80"] for A in mods]
            b_stack = np.vstack([boot_t80[(mtags[(A, B_m)], 6.50)] for A in mods])
            b_spread = b_stack.max(axis=0) / np.maximum(b_stack.min(axis=0), 1e-30)
            col_contrasts[f"fine_{B_m}_max_over_min_coarse"] = {
                "ratio": float(max(vals) / min(vals)),
                "ci95": [float(np.percentile(b_spread, 2.5)), float(np.percentile(b_spread, 97.5))],
            }
            col_contrasts[f"G_{B_m}_R3B_over_C3B"] = t80_comp(mtags[("R3", B_m)], mtags[("C3", B_m)], 6.50)

        # Swapped-pair ratios in t_80 at beta = 6.50:
        swaps = {
            "C3R3_over_R3C3": t80_comp(mtags[("C3", "R3")], mtags[("R3", "C3")], 6.50),
            "R3C2_over_C2R3": t80_comp(mtags[("R3", "C2")], mtags[("C2", "R3")], 6.50),
            "C3C2_over_C2C3": t80_comp(mtags[("C3", "C2")], mtags[("C2", "C3")], 6.50),
            "C3C2_over_R3R3": t80_comp(mtags[("C3", "C2")], mtags[("R3", "R3")], 6.50),
            "C3C2_over_1cyc_R3": t80_comp(mtags[("C3", "C2")], s1["R3"], 6.50),
            "R3C2_over_1cyc_R3": t80_comp(mtags[("R3", "C2")], s1["R3"], 6.50),
            "C2R3_over_1cyc_R3": t80_comp(mtags[("C2", "R3")], s1["R3"], 6.50),
            "1cyc_R3_over_C2": t80_comp(s1["R3"], s1["C2"], 6.50),
        }

        # Joint bootstrap probability that C3C2 beats ALL FIVE R3-containing two-cycle codes at beta = 6.50:
        r3_tags = [mtags[("R3", "C2")], mtags[("C2", "R3")], mtags[("C3", "R3")], mtags[("R3", "C3")], mtags[("R3", "R3")]]
        b_c3c2 = boot_t80[(mtags[("C3", "C2")], 6.50)]
        b_max_r3 = np.max(np.vstack([boot_t80[(t, 6.50)] for t in r3_tags]), axis=0)
        joint_prob = float(np.mean(b_c3c2 > b_max_r3))
        min_ratio_vs_r3 = b_c3c2 / np.maximum(b_max_r3, 1e-30)

        matrix_summary[seed_name] = {
            "cells": cells,
            "additive_log_fit_max_mult_error": max_mult_err,
            "row_contrasts": row_contrasts,
            "col_contrasts": col_contrasts,
            "swaps_and_refs": swaps,
            "C3C2_vs_all_R3_frontier": {
                "C3C2_over_max_R3_point": float(
                    std_readout[mtags[("C3", "C2")]]["6.50"]["t_80"]
                    / max(std_readout[t]["6.50"]["t_80"] for t in r3_tags)
                ),
                "C3C2_over_max_R3_ci95": [
                    float(np.percentile(min_ratio_vs_r3, 2.5)),
                    float(np.percentile(min_ratio_vs_r3, 97.5)),
                ],
                "boot_prob_C3C2_beats_all_5_R3": joint_prob,
            },
        }

    # 3. Cold Operational Comparisons & Marginal Coarse-Bridge Gain G_B(beta) across beta in {6.50, 6.75, 7.00}:
    cold_marginal = {}
    for seed_name, b_list in [("planar", [6.50, 6.75, 7.00]), ("torus", [6.50, 7.00])]:
        mtags = MAT_TAGS[seed_name]
        cold_marginal[seed_name] = {}
        for beta in b_list:
            entry = {}
            if (mtags[("R3", "C2")], beta) in boot_t80 and (mtags[("C3", "C2")], beta) in boot_t80:
                entry["G_C2_t80"] = t80_comp(mtags[("R3", "C2")], mtags[("C3", "C2")], beta)
                entry["R3C2_over_C2C2_t80"] = t80_comp(mtags[("R3", "C2")], mtags[("C2", "C2")], beta)
                entry["C3C2_over_C2C2_t80"] = t80_comp(mtags[("C3", "C2")], mtags[("C2", "C2")], beta)
                entry["R3C2_over_C2R3_t80"] = t80_comp(mtags[("R3", "C2")], mtags[("C2", "R3")], beta)
            if (mtags[("R3", "C3")], beta) in boot_t80 and (mtags[("C3", "C3")], beta) in boot_t80:
                entry["G_C3_t80"] = t80_comp(mtags[("R3", "C3")], mtags[("C3", "C3")], beta)
            if "G_C2_t80" in entry and "G_C3_t80" in entry and entry["G_C2_t80"]["ratio"] and entry["G_C3_t80"]["ratio"]:
                b_inter = (
                    boot_t80[(mtags[("R3", "C2")], beta)] * boot_t80[(mtags[("C3", "C3")], beta)]
                ) / np.maximum(
                    boot_t80[(mtags[("C3", "C2")], beta)] * boot_t80[(mtags[("R3", "C3")], beta)], 1e-30
                )
                pt_inter = entry["G_C2_t80"]["ratio"] / entry["G_C3_t80"]["ratio"]
                entry["interaction_GC2_over_GC3_t80"] = {
                    "ratio": float(pt_inter),
                    "ci95": [float(np.percentile(b_inter, 2.5)), float(np.percentile(b_inter, 97.5))],
                }
            cold_marginal[seed_name][f"{beta:.2f}"] = entry

    # 4. Pre-Declared Causal Marginal-Benefit Experiment G_{C3}(mu) at beta = 6.50:
    # Pair each X_err kinetic run (mu in {0.30, 0.10, 0.03, 0.01}) with the exact unperturbed Z_err run at beta = 6.50
    # (and for mu == 1.00, use the first 64 trials of m65 so sample sizes match, PLUS full 128-trial reference):
    causal_arms = defaultdict(lambda: defaultdict(dict))
    kin_b_t80 = {}
    kin_b_tauQ = {}
    kin_c80 = {}

    for (tag, sec, mkind, mu, beta) in sorted(raw_groups):
        if sec != "X_err" or mkind == "none":
            continue
        zk = (tag, "Z_err", "none", 1.0, beta)
        res = eval_readout_pair(raw_groups[(tag, "X_err", mkind, mu, beta)], raw_groups[zk], rng)
        kin_b_t80[(tag, mkind, mu)] = res.pop("_boot_t80")
        kin_c80[(tag, mkind, mu)] = res["t_80_crossed"] and (res["t_80_boot_crossed_frac"] >= 0.95)
        res.pop("_boot_c80")
        res.pop("_boot_t50")
        kin_b_tauQ[(tag, mkind, mu)] = res.pop("_boot_tauQ")
        causal_arms[tag][mkind][f"{mu:.2f}"] = res

    # Add mu = 1.00 point to each arm from the identical V17-only unperturbed generator:
    for tag in list(causal_arms.keys()):
        r1 = std_readout_v17only[tag]["6.50"]
        for mkind in list(causal_arms[tag].keys()):
            causal_arms[tag][mkind]["1.00"] = r1
            kin_b_t80[(tag, mkind, 1.00)] = boot_t80_v17[(tag, 6.50)]
            kin_c80[(tag, mkind, 1.00)] = boot_c80_v17[(tag, 6.50)]
            kin_b_tauQ[(tag, mkind, 1.00)] = boot_tauQ[(tag, 6.50)]
    # On planar R3C3 (`v15_sq_L3_RC`), coarse_desc_flat_count == 0, so allflat == fineflat identically:
    if "fineflat" in causal_arms["v15_sq_L3_RC"]:
        causal_arms["v15_sq_L3_RC"]["allflat"] = causal_arms["v15_sq_L3_RC"]["fineflat"]
        causal_arms["v15_sq_L3_RC"]["allseam"] = causal_arms["v15_sq_L3_RC"]["fineseam"]
        for mu in [1.00, 0.30, 0.10, 0.03, 0.01]:
            if ("v15_sq_L3_RC", "fineflat", mu) in kin_b_t80:
                kin_b_t80[("v15_sq_L3_RC", "allflat", mu)] = kin_b_t80[("v15_sq_L3_RC", "fineflat", mu)]
                kin_c80[("v15_sq_L3_RC", "allflat", mu)] = kin_c80[("v15_sq_L3_RC", "fineflat", mu)]
                kin_b_tauQ[("v15_sq_L3_RC", "allflat", mu)] = kin_b_tauQ[("v15_sq_L3_RC", "fineflat", mu)]
                kin_b_t80[("v15_sq_L3_RC", "allseam", mu)] = kin_b_t80[("v15_sq_L3_RC", "fineseam", mu)]
                kin_c80[("v15_sq_L3_RC", "allseam", mu)] = kin_c80[("v15_sq_L3_RC", "fineseam", mu)]
                kin_b_tauQ[("v15_sq_L3_RC", "allseam", mu)] = kin_b_tauQ[("v15_sq_L3_RC", "fineseam", mu)]

    causal_GC3 = {}
    for seed_name, rc_tag, cc_tag, cr_tag in [
        ("planar", "v15_sq_L3_RC", "v15_sq_L3_CC", "v15_sq_L3_CR"),
        ("torus", "v15_tor_L3_RC", "v15_tor_L3_CC", "v15_tor_L3_CR"),
    ]:
        causal_GC3[seed_name] = {}
        for fkind, skind, mu_list in [
            ("fineflat", "fineseam", [1.00, 0.30, 0.10, 0.03, 0.01]),
            ("allflat", "allseam", [1.00, 0.10, 0.01]),
        ]:
            rows = []
            for mu in mu_list:
                ms = f"{mu:.2f}"
                if ms not in causal_arms[rc_tag].get(fkind, {}) or ms not in causal_arms[cc_tag].get(fkind, {}):
                    continue
                rc_f = causal_arms[rc_tag][fkind][ms]
                cc_f = causal_arms[cc_tag][fkind][ms]
                rc_s = causal_arms[rc_tag][skind][ms]
                cc_s = causal_arms[cc_tag][skind][ms]

                g_flat_t80 = ratio_ci(
                    rc_f["t_80"], kin_b_t80[(rc_tag, fkind, mu)], kin_c80[(rc_tag, fkind, mu)],
                    cc_f["t_80"], kin_b_t80[(cc_tag, fkind, mu)], kin_c80[(cc_tag, fkind, mu)],
                )
                g_seam_t80 = ratio_ci(
                    rc_s["t_80"], kin_b_t80[(rc_tag, skind, mu)], kin_c80[(rc_tag, skind, mu)],
                    cc_s["t_80"], kin_b_t80[(cc_tag, skind, mu)], kin_c80[(cc_tag, skind, mu)],
                )

                b_g_flat = kin_b_t80[(rc_tag, fkind, mu)] / np.maximum(kin_b_t80[(cc_tag, fkind, mu)], 1e-30)
                b_g_seam = kin_b_t80[(rc_tag, skind, mu)] / np.maximum(kin_b_t80[(cc_tag, skind, mu)], 1e-30)
                b_diff_in_diff = b_g_flat / np.maximum(b_g_seam, 1e-30)
                pt_did = g_flat_t80["ratio"] / g_seam_t80["ratio"]

                # First-passage supporting ratios on the exact same trajectories:
                b_tau_flat = kin_b_tauQ[(rc_tag, fkind, mu)] / np.maximum(kin_b_tauQ[(cc_tag, fkind, mu)], 1e-30)
                b_tau_seam = kin_b_tauQ[(rc_tag, skind, mu)] / np.maximum(kin_b_tauQ[(cc_tag, skind, mu)], 1e-30)
                b_tau_did = b_tau_flat / np.maximum(b_tau_seam, 1e-30)
                pt_tau_flat = rc_f["same_traj_tau_Q"] / cc_f["same_traj_tau_Q"]
                pt_tau_seam = rc_s["same_traj_tau_Q"] / cc_s["same_traj_tau_Q"]

                rows.append({
                    "mu": mu,
                    "t80_R3C3_flat": rc_f["t_80"],
                    "t80_R3C3_flat_ci95": rc_f["t_80_ci95"],
                    "t80_C3C3_flat": cc_f["t_80"],
                    "t80_C3C3_flat_ci95": cc_f["t_80_ci95"],
                    "t80_R3C3_seam": rc_s["t_80"],
                    "t80_R3C3_seam_ci95": rc_s["t_80_ci95"],
                    "t80_C3C3_seam": cc_s["t_80"],
                    "t80_C3C3_seam_ci95": cc_s["t_80_ci95"],
                    "G_C3_t80_flat": g_flat_t80,
                    "G_C3_t80_seam": g_seam_t80,
                    "G_C3_t80_flat_over_seam": {
                        "ratio": float(pt_did),
                        "ci95": [float(np.percentile(b_diff_in_diff, 2.5)), float(np.percentile(b_diff_in_diff, 97.5))],
                    },
                    "tauQ_R3C3_flat": rc_f["same_traj_tau_Q"],
                    "tauQ_C3C3_flat": cc_f["same_traj_tau_Q"],
                    "tauQ_R3C3_seam": rc_s["same_traj_tau_Q"],
                    "tauQ_C3C3_seam": cc_s["same_traj_tau_Q"],
                    "G_C3_tauQ_flat": {
                        "ratio": float(pt_tau_flat),
                        "ci95": [float(np.percentile(b_tau_flat, 2.5)), float(np.percentile(b_tau_flat, 97.5))],
                    },
                    "G_C3_tauQ_seam": {
                        "ratio": float(pt_tau_seam),
                        "ci95": [float(np.percentile(b_tau_seam, 2.5)), float(np.percentile(b_tau_seam, 97.5))],
                    },
                    "G_C3_tauQ_flat_over_seam": {
                        "ratio": float(pt_tau_flat / pt_tau_seam),
                        "ci95": [float(np.percentile(b_tau_did, 2.5)), float(np.percentile(b_tau_did, 97.5))],
                    },
                })
            causal_GC3[seed_name][f"{fkind}_vs_{skind}"] = rows

    out_doc = {
        "v17_errors": v17_errors,
        "v17_masks": mask_meta["masks"],
        "std_readout": dict(std_readout),
        "matrix_650": matrix_summary,
        "cold_marginal": cold_marginal,
        "causal_arms": {k: dict(v) for k, v in causal_arms.items()},
        "causal_GC3": causal_GC3,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out_doc, f, indent=2)
    print(f"Wrote {OUT_PATH} (v17_errors={len(v17_errors)}, std_readout_codes={len(std_readout)})")


if __name__ == "__main__":
    main()
