#!/usr/bin/env python3
"""
reanalyze_existing_v16_v17.py
Suite [A]: Reanalysis of existing V16/V17 data with all methodological corrections from the review:
  1. Per-trajectory deduplication by global trajectory ID `g_tr = seed_off + i` (verifying prefix
     agreement when overlapping shards occur, fixing torus standalone C3 at beta=7.0 from 112 -> 96
     unique trajectories).
  2. Single-cohort V17 healing decomposition (`P_id_v17 = P_nocross_v17 + P_healed_v17` with shared
     bootstrap indices on the coherent V17 cohort, fixing the mixed V16+V17 healing curve).
  3. 5,000 whole-trajectory bootstrap draws across p in {0.70, 0.75, 0.80, 0.85, 0.90}:
       - C2C2 vs R3R3 (planar & torus, beta=6.5)
       - C2C2 vs all six fine-C3 / fine-R3 two-cycle architectures (minimum-ratio statistic across p)
       - C2C2 vs C3C2 (planar & torus, beta=6.5) and R3C2 vs C2C2 (planar & torus, beta=7.0)
  4. Explicit null checks:
       - Independent-opportunity 1/N null: (t_80(A B1) / t_80(A B2)) * (N(A B1) / N(A B2))
       - Exact clean-state X departure clock null: (t_80 * Gamma_X^(0)) ratio on C3C2 vs C3C3
  5. Direct temperature contrast G_{C2}(7.0) / G_{C2}(6.5) on planar and torus seeds.
  6. Rare-Z-sector finite-sample 95% one-sided failure bounds: 1 - 0.05^(1 / n_Z).
"""

from collections import defaultdict
import json
import math
import os
import re
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "cluster_v9"))

from build_v7_rescue import seed_planar_11, sheaf_to_css_with_seams
from build_v15_selective_hierarchy import apply_step_with_parent_faces, seed_torus_1face

V16_RAW = os.path.join(ROOT, "cluster_v16_results.txt")
V17_RAW = os.path.join(ROOT, "cluster_v17_results.txt")
V17_MASKS = os.path.join(ROOT, "v17_mask_specs.json")
OUT_PATH = os.path.join(ROOT, "reanalyzed_v16_v17_suiteA.json")


def parse_kv(body):
    kv = {}
    for tok in body.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            kv[k] = v
    return kv


def crossing_time_strict(t_grid, p_curve, thresh):
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


def compute_code_specs_and_hazards():
    C2 = [(2, [1], "X"), (2, [1], "Z")]
    C3 = [(3, [1], "X"), (3, [1], "Z")]
    R3 = [(3, [1, 2], "X"), (3, [1, 2], "Z")]
    table = [
        ("v15_sq_C2",      "planar", C2),
        ("v15_sq_C3",      "planar", C3),
        ("v15_sq_R3",      "planar", R3),
        ("v15_sq_pkg_CC",  "planar", C2 + C2),
        ("v16_sq_C2C3",    "planar", C2 + C3),
        ("v15_sq_pkg_CR",  "planar", C2 + R3),
        ("v16_sq_C3C2",    "planar", C3 + C2),
        ("v15_sq_L3_CC",   "planar", C3 + C3),
        ("v15_sq_L3_CR",   "planar", C3 + R3),
        ("v15_sq_pkg_RC",  "planar", R3 + C2),
        ("v15_sq_L3_RC",   "planar", R3 + C3),
        ("v15_sq_L3_RR",   "planar", R3 + R3),
        ("v15_tor_C2",     "tor1",   C2),
        ("v15_tor_C3",     "tor1",   C3),
        ("v15_tor_R3",     "tor1",   R3),
        ("v15_tor_pkg_CC", "tor1",   C2 + C2),
        ("v16_tor_C2C3",   "tor1",   C2 + C3),
        ("v15_tor_pkg_CR", "tor1",   C2 + R3),
        ("v16_tor_C3C2",   "tor1",   C3 + C2),
        ("v15_tor_L3_CC",  "tor1",   C3 + C3),
        ("v15_tor_L3_CR",  "tor1",   C3 + R3),
        ("v15_tor_pkg_RC", "tor1",   R3 + C2),
        ("v15_tor_L3_RC",  "tor1",   R3 + C3),
        ("v15_tor_L3_RR",  "tor1",   R3 + R3),
    ]
    specs = {}
    for tag, sk, sched in table:
        comp = seed_planar_11() if sk == "planar" else seed_torus_1face()
        cyl = set()
        for L, rb, it in sched:
            comp, cyl, _, _ = apply_step_with_parent_faces(comp, L, rb, it, prev_cyl=cyl)
        N, HX, HZ, *_ = sheaf_to_css_with_seams(comp)
        dz = [0] * N
        for r in HZ:
            for q in r:
                dz[q] += 1
        dx = [0] * N
        for r in HX:
            for q in r:
                dx[q] += 1
        specs[tag] = {
            "N": N,
            "M_X": len(HX),
            "M_Z": len(HZ),
            "Gamma0_X_65": sum(1.0 / (1.0 + math.exp(6.5 * d)) for d in dz),
            "Gamma0_Z_65": sum(1.0 / (1.0 + math.exp(6.5 * d)) for d in dx),
            "Gamma0_X_70": sum(1.0 / (1.0 + math.exp(7.0 * d)) for d in dz),
            "Gamma0_Z_70": sum(1.0 / (1.0 + math.exp(7.0 * d)) for d in dx),
        }
    return specs


def eval_cohort(x_trajs, z_trajs, rng, B=5000, extra_x=None, extra_z=None):
    # x_trajs: list of (bitstr, t_first, c_first) sorted by unique g_tr
    xbits = [t[0] for t in x_trajs]
    zbits = [t[0] for t in z_trajs]
    lengths = [len(s) for s in xbits] + [len(s) for s in zbits]
    if extra_x:
        lengths.extend(len(s) for s in extra_x)
    if extra_z:
        lengths.extend(len(s) for s in extra_z)
    ng = min(lengths)
    lg0 = 2.00
    t_grid = np.array([10.0 ** (lg0 + 0.10 * i) for i in range(ng)], dtype=np.float64)

    xbits_v17 = [s[:ng] for s in xbits]
    zbits_v17 = [s[:ng] for s in zbits]
    all_x = list(xbits_v17) + ([s[:ng] for s in extra_x] if extra_x else [])
    all_z = list(zbits_v17) + ([s[:ng] for s in extra_z] if extra_z else [])
    nx, nz = len(all_x), len(all_z)
    nx17, nz17 = len(xbits_v17), len(zbits_v17)

    X_pass = np.array([[1.0 if ch in ("1", "2") else 0.0 for ch in s] for s in all_x], dtype=np.float64)
    Z_pass = np.array([[1.0 if ch in ("1", "2") else 0.0 for ch in s] for s in all_z], dtype=np.float64)
    X_p17 = np.array([[1.0 if ch in ("1", "2") else 0.0 for ch in s] for s in xbits_v17], dtype=np.float64)
    Z_p17 = np.array([[1.0 if ch in ("1", "2") else 0.0 for ch in s] for s in zbits_v17], dtype=np.float64)
    X_s17 = np.array([[1.0 if ch == "2" else 0.0 for ch in s] for s in xbits_v17], dtype=np.float64)
    Z_s17 = np.array([[1.0 if ch == "2" else 0.0 for ch in s] for s in zbits_v17], dtype=np.float64)

    pid = X_pass.mean(axis=0) * Z_pass.mean(axis=0)
    # Single-cohort V17 decomposition (Review p. 3 & p. 15):
    pid_17 = X_p17.mean(axis=0) * Z_p17.mean(axis=0)
    pnc_17 = X_s17.mean(axis=0) * Z_s17.mean(axis=0)
    pheal_17 = pid_17 - pnc_17

    ps = [0.70, 0.75, 0.80, 0.85, 0.90]
    tp_pts = {}
    tp_crossed = {}
    for p in ps:
        tv, cv = crossing_time_strict(t_grid, pid, p)
        tp_pts[f"{p:.2f}"] = tv
        tp_crossed[f"{p:.2f}"] = cv

    t80_17, c80_17 = crossing_time_strict(t_grid, pid_17, 0.80)
    t80_nc17, c80_nc17 = crossing_time_strict(t_grid, pnc_17, 0.80)

    # Interpolate P_healed_17 at single-cohort t80_17!
    idx_hi = int(np.searchsorted(t_grid, t80_17))
    if idx_hi <= 0:
        pheal_at_t80_17 = float(pheal_17[0])
    elif idx_hi >= ng:
        pheal_at_t80_17 = float(pheal_17[-1])
    else:
        w = (math.log10(t80_17) - math.log10(t_grid[idx_hi - 1])) / (
            math.log10(t_grid[idx_hi]) - math.log10(t_grid[idx_hi - 1])
        )
        pheal_at_t80_17 = float((1.0 - w) * pheal_17[idx_hi - 1] + w * pheal_17[idx_hi])

    ix = rng.integers(0, nx, size=(B, nx))
    iz = rng.integers(0, nz, size=(B, nz))
    b_pid = X_pass[ix].mean(axis=1) * Z_pass[iz].mean(axis=1)
    boot_tp = {}
    boot_cp = {}
    for p in ps:
        arr = np.empty(B, dtype=np.float64)
        carr = np.empty(B, dtype=bool)
        for k in range(B):
            arr[k], carr[k] = crossing_time_strict(t_grid, b_pid[k], p)
        boot_tp[f"{p:.2f}"] = arr
        boot_cp[f"{p:.2f}"] = carr

    # Paired single-cohort V17 healing bootstrap (same resampled V17 trajectories for P_rec and P_nocross!):
    ix17 = rng.integers(0, nx17, size=(B, nx17))
    iz17 = rng.integers(0, nz17, size=(B, nz17))
    b_pid17 = X_p17[ix17].mean(axis=1) * Z_p17[iz17].mean(axis=1)
    b_pnc17 = X_s17[ix17].mean(axis=1) * Z_s17[iz17].mean(axis=1)
    b_t80_17 = np.array([crossing_time_strict(t_grid, b_pid17[k], 0.80)[0] for k in range(B)])
    b_t80_nc17 = np.array([crossing_time_strict(t_grid, b_pnc17[k], 0.80)[0] for k in range(B)])
    b_heal_ratio = b_t80_17 / np.maximum(b_t80_nc17, 1e-30)

    # Check observed Z failures at t_80:
    idx_near = min(range(ng), key=lambda i: abs(t_grid[i] - tp_pts["0.80"]))
    z_fails_at_t80 = int(round((1.0 - float(Z_pass.mean(axis=0)[idx_near])) * nz))
    z_rare_bound_95 = 1.0 - (0.05 ** (1.0 / nz))

    return {
        "n_X": nx,
        "n_Z": nz,
        "n_X_v17": nx17,
        "n_Z_v17": nz17,
        "z_fails_near_t80": z_fails_at_t80,
        "z_zero_fail_95_upper_bound": z_rare_bound_95,
        "t_grid": t_grid.tolist(),
        "P_id": pid.tolist(),
        "v17_single_cohort": {
            "P_id": pid_17.tolist(),
            "P_nocross": pnc_17.tolist(),
            "P_healed": pheal_17.tolist(),
            "t_80": t80_17,
            "t_80_nocross": t80_nc17,
            "P_healed_at_t80": pheal_at_t80_17,
            "t80_over_nocross_ratio": float(t80_17 / t80_nc17),
            "t80_over_nocross_ci95": [float(np.percentile(b_heal_ratio, 2.5)), float(np.percentile(b_heal_ratio, 97.5))],
        },
        "t_p": tp_pts,
        "t_p_crossed": tp_crossed,
        "t_80": tp_pts["0.80"],
        "t_80_ci95": [float(np.percentile(boot_tp["0.80"], 2.5)), float(np.percentile(boot_tp["0.80"], 97.5))],
        "_boot_tp": boot_tp,
        "_boot_cp": boot_cp,
    }


def main():
    specs = compute_code_specs_and_hazards()

    # Deduplicate V17 trajectories by global trajectory ID `g_tr = off + idx`!
    v17_trajs = defaultdict(dict)  # (tag, sec, mkind, mu, beta) -> {g_tr: (bits, tf, cf)}
    dup_overlaps = []
    with open(V17_RAW) as f:
        for raw in f:
            line = raw.strip()
            if not line or "|" not in line:
                continue
            tid, body = [x.strip() for x in line.split("|", 1)]
            if body.startswith("ERROR") or body == "TIMEOUT":
                continue
            m = re.match(
                r"^(m65|c70|c675|kin)__(.+)_(X_err|Z_err)__(none|fineflat|fineseam|allflat|allseam)__mu([0-9.]+)__b([0-9.]+)__o(\d+)$",
                tid,
            )
            if not m:
                continue
            _, tag, sec, mkind, mu_s, b_s, off_s = m.groups()
            mu, beta, off = float(mu_s), float(b_s), int(off_s)
            kv = parse_kv(body)
            tcf = [(float(x.split(":")[0]), int(x.split(":")[1])) for x in kv["tc_first"].split(",") if x]
            bits = [s for s in kv["bits"].split(",") if s]
            key = (tag, sec, mkind, mu, beta)
            tmap = v17_trajs[key]
            for i, bstr in enumerate(bits):
                gtr = off + i
                if gtr in tmap:
                    prev_b, prev_tf, prev_cf = tmap[gtr]
                    ml = min(len(prev_b), len(bstr))
                    assert prev_b[:ml] == bstr[:ml], f"Prefix mismatch on duplicate {key} g_tr={gtr}"
                    dup_overlaps.append((tag, sec, beta, gtr))
                    if len(bstr) > len(prev_b):
                        tmap[gtr] = (bstr, tcf[i][0], tcf[i][1])
                else:
                    tmap[gtr] = (bstr, tcf[i][0], tcf[i][1])

    v16_trajs = defaultdict(dict)  # (tag, sec, beta) -> {g_tr: bitstr}
    with open(V16_RAW) as f:
        for raw in f:
            line = raw.strip()
            if not line or "|" not in line or not line.startswith("frout__"):
                continue
            tid, body = [x.strip() for x in line.split("|", 1)]
            if body.startswith("ERROR") or body == "TIMEOUT":
                continue
            m16 = re.match(r"^frout__(.+)_(X_err|Z_err)__b([0-9.]+)__o(\d+)$", tid)
            if m16:
                tag, sec, b_s, off_s = m16.groups()
                off = int(off_s)
                kv = parse_kv(body)
                bits = [s for s in kv["bits"].split(",") if s]
                for i, bstr in enumerate(bits):
                    v16_trajs[(tag, sec, float(b_s))][off + i] = bstr

    print(f"Deduplicated {len(dup_overlaps)} overlapping trajectory entries (all in v15_tor_C3 X_err beta=7.0).")

    rng = np.random.default_rng(20261122)
    B = 5000
    readouts = defaultdict(dict)
    boot_bank = {}

    std_keys = sorted({(tag, beta) for (tag, sec, mk, mu, beta) in v17_trajs if mk == "none" and abs(mu - 1.0) < 1e-9})
    for tag, beta in std_keys:
        xk = (tag, "X_err", "none", 1.0, beta)
        zk = (tag, "Z_err", "none", 1.0, beta)
        if xk not in v17_trajs or zk not in v17_trajs:
            continue
        x_list = [v17_trajs[xk][k] for k in sorted(v17_trajs[xk])]
        z_list = [v17_trajs[zk][k] for k in sorted(v17_trajs[zk])]
        ex_x = [v16_trajs[(tag, "X_err", beta)][k] for k in sorted(v16_trajs.get((tag, "X_err", beta), {}))]
        ex_z = [v16_trajs[(tag, "Z_err", beta)][k] for k in sorted(v16_trajs.get((tag, "Z_err", beta), {}))]
        res = eval_cohort(x_list, z_list, rng, B=B, extra_x=ex_x or None, extra_z=ex_z or None)
        boot_bank[(tag, f"{beta:.2f}")] = (res.pop("_boot_tp"), res.pop("_boot_cp"))
        if tag in specs:
            res["N"] = specs[tag]["N"]
        readouts[tag][f"{beta:.2f}"] = res

    def ratio_stat(t_num, t_den, bstr, pstr="0.80"):
        pt = readouts[t_num][bstr]["t_p"][pstr] / readouts[t_den][bstr]["t_p"][pstr]
        bn = boot_bank[(t_num, bstr)][0][pstr]
        bd = boot_bank[(t_den, bstr)][0][pstr]
        br = bn / np.maximum(bd, 1e-30)
        return {
            "ratio": float(pt),
            "ci95": [float(np.percentile(br, 2.5)), float(np.percentile(br, 97.5))],
        }

    MAT = {
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

    headlines = {}
    for sname in ("planar", "torus"):
        m = MAT[sname]
        six_non_c2 = [m[("C2", "C3")], m[("C2", "R3")], m[("C3", "C3")], m[("C3", "R3")], m[("R3", "C3")], m[("R3", "R3")]]
        min_ratios_by_p = {}
        for pstr in ["0.70", "0.75", "0.80", "0.85", "0.90"]:
            pt_c2c2 = readouts[m[("C2", "C2")]]["6.50"]["t_p"][pstr]
            pt_best6 = max(readouts[t]["6.50"]["t_p"][pstr] for t in six_non_c2)
            b_c2c2 = boot_bank[(m[("C2", "C2")], "6.50")][0][pstr]
            b_best6 = np.max(np.vstack([boot_bank[(t, "6.50")][0][pstr] for t in six_non_c2]), axis=0)
            br = b_c2c2 / np.maximum(b_best6, 1e-30)
            min_ratios_by_p[pstr] = {
                "ratio": float(pt_c2c2 / pt_best6),
                "ci95": [float(np.percentile(br, 2.5)), float(np.percentile(br, 97.5))],
            }

        # Opportunity null (1/N) and clean-state X departure null:
        opp_null = {}
        for A in ["C2", "C3", "R3"]:
            for B_other in ["C3", "R3"]:
                t1, t2 = m[(A, "C2")], m[(A, B_other)]
                raw_r = ratio_stat(t1, t2, "6.50")
                scale = specs[t1]["N"] / specs[t2]["N"]
                opp_null[f"{A}_C2_vs_{A}_{B_other}"] = {
                    "raw_ratio": raw_r["ratio"],
                    "size_adjusted_ratio": raw_r["ratio"] * scale,
                    "size_adjusted_ci95": [raw_r["ci95"][0] * scale, raw_r["ci95"][1] * scale],
                }

        # Direct temperature contrast G_{C2}(7.0) / G_{C2}(6.5):
        b_rc_70 = boot_bank[(m[("R3", "C2")], "7.00")][0]["0.80"]
        b_cc_70 = boot_bank[(m[("C3", "C2")], "7.00")][0]["0.80"]
        b_rc_65 = boot_bank[(m[("R3", "C2")], "6.50")][0]["0.80"]
        b_cc_65 = boot_bank[(m[("C3", "C2")], "6.50")][0]["0.80"]
        b_temp_contrast = (b_rc_70 / b_cc_70) / (b_rc_65 / b_cc_65)
        pt_temp_contrast = (
            (readouts[m[("R3", "C2")]]["7.00"]["t_80"] / readouts[m[("C3", "C2")]]["7.00"]["t_80"])
            / (readouts[m[("R3", "C2")]]["6.50"]["t_80"] / readouts[m[("C3", "C2")]]["6.50"]["t_80"])
        )

        headlines[sname] = {
            "C2C2_over_R3R3_b65": ratio_stat(m[("C2", "C2")], m[("R3", "R3")], "6.50"),
            "C2C2_over_C3C2_b65": ratio_stat(m[("C2", "C2")], m[("C3", "C2")], "6.50"),
            "R3C2_over_C2C2_b70": ratio_stat(m[("R3", "C2")], m[("C2", "C2")], "7.00"),
            "C2C2_over_best_of_6_nonC2fine_by_p": min_ratios_by_p,
            "opportunity_null_1_over_N": opp_null,
            "GC2_70_over_GC2_65": {
                "ratio": float(pt_temp_contrast),
                "ci95": [float(np.percentile(b_temp_contrast, 2.5)), float(np.percentile(b_temp_contrast, 97.5))],
            },
        }

    # Clean-state X birth-rate null on planar C3C2 vs C3C3 at beta = 6.50:
    r_c3c2_c3c3 = ratio_stat("v16_sq_C3C2", "v15_sq_L3_CC", "6.50")
    g_scale = specs["v16_sq_C3C2"]["Gamma0_X_65"] / specs["v15_sq_L3_CC"]["Gamma0_X_65"]
    headlines["planar"]["clean_birth_X_null_C3C2_vs_C3C3"] = {
        "Gamma0_X_C3C2": specs["v16_sq_C3C2"]["Gamma0_X_65"],
        "Gamma0_X_C3C3": specs["v15_sq_L3_CC"]["Gamma0_X_65"],
        "adjusted_ratio": r_c3c2_c3c3["ratio"] * g_scale,
        "adjusted_ci95": [r_c3c2_c3c3["ci95"][0] * g_scale, r_c3c2_c3c3["ci95"][1] * g_scale],
    }

    print("=== Suite [A] Headline Verification ===")
    for sname in ("planar", "torus"):
        h = headlines[sname]
        print(f"  [{sname}] C2C2 / R3R3 (b=6.5): {h['C2C2_over_R3R3_b65']['ratio']:.2f} {h['C2C2_over_R3R3_b65']['ci95']}")
        print(f"  [{sname}] C2C2 / best-6 (b=6.5, p=0.80): {h['C2C2_over_best_of_6_nonC2fine_by_p']['0.80']['ratio']:.2f} {h['C2C2_over_best_of_6_nonC2fine_by_p']['0.80']['ci95']}")
        print(f"  [{sname}] C2C2 / C3C2 (b=6.5): {h['C2C2_over_C3C2_b65']['ratio']:.2f} {h['C2C2_over_C3C2_b65']['ci95']}")
        print(f"  [{sname}] R3C2 / C2C2 (b=7.0): {h['R3C2_over_C2C2_b70']['ratio']:.2f} {h['R3C2_over_C2C2_b70']['ci95']}")
        print(f"  [{sname}] G_C2(7.0)/G_C2(6.5): {h['GC2_70_over_GC2_65']['ratio']:.2f} {h['GC2_70_over_GC2_65']['ci95']}")
    print("  [planar] Clean-birth X null C3C2/C3C3:", headlines["planar"]["clean_birth_X_null_C3C2_vs_C3C3"])
    sc65 = readouts["v16_sq_C3C2"]["6.50"]["v17_single_cohort"]
    sc70 = readouts["v16_sq_C3C2"]["7.00"]["v17_single_cohort"]
    print(f"  [planar C3C2 single-cohort V17] b=6.5: P_heal={sc65['P_healed_at_t80']:.3f}, ratio={sc65['t80_over_nocross_ratio']:.2f} {sc65['t80_over_nocross_ci95']}")
    print(f"  [planar C3C2 single-cohort V17] b=7.0: P_heal={sc70['P_healed_at_t80']:.3f}, ratio={sc70['t80_over_nocross_ratio']:.2f} {sc70['t80_over_nocross_ci95']}")
    print(f"  [torus C3 b=7.0 unique n_X]: {readouts['v15_tor_C3']['7.00']['n_X']}")

    out_doc = {
        "specs": specs,
        "deduplicated_overlaps_count": len(dup_overlaps),
        "readouts": dict(readouts),
        "headlines": headlines,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out_doc, f, indent=2)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
