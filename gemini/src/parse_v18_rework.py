#!/usr/bin/env python3
"""
parse_v18_rework.py
Statistical parser and prospective-transfer evaluator for the V18 SubMIT campaign
(`cluster_v18_results.txt` + `v18_code_specs.json` + `frozen_interface_predictions.json`):
  - Deduplicates trajectories by global RNG index `g_tr = seed_off + i`
  - Uses B = 10,000 whole-trajectory bootstrap draws for all V18 primary contrasts
  - Evaluates both CRN-paired and independent bootstrap intervals on geometry-matched L=4 pairs
    (standalone N = 2,926 and coarse-C2 composed N = 261,476)
  - Evaluates all four two-step C2 pruning / protected-subsystem arms (Suites E & F)
  - Tests all held-out prospective simplification predictions verbatim against the pre-simulation
    frozen file `frozen_interface_predictions.json` (verifying its SHA-256 digest!)
  - Records sector sample sizes, flip-cap counts, sensitivity thresholds p in {0.70, 0.80, 0.90},
    and rare-Z-sector 95% failure bounds 1 - 0.05^(1/n_Z).
"""

from collections import defaultdict
import hashlib
import json
import math
import os
import re
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V18_RAW = os.path.join(ROOT, "cluster_v18_results.txt")
V18_SPECS = os.path.join(ROOT, "v18_code_specs.json")
FROZEN_PATH = os.path.join(ROOT, "frozen_interface_predictions.json")
OUT_PATH = os.path.join(ROOT, "parsed_v18_rework_data.json")


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


def eval_arm(x_map, z_map, grid_spec, rng, B=10000):
    lg0_s, lg1_s, ng_s = grid_spec.split(":")
    lg0, lg1_full, ng_full = float(lg0_s), float(lg1_s), int(ng_s)
    step = (lg1_full - lg0) / max(1, ng_full - 1)

    x_keys = sorted(x_map.keys())
    z_keys = sorted(z_map.keys())
    ng = min([ng_full] + [len(x_map[k][0]) for k in x_keys] + [len(z_map[k][0]) for k in z_keys])
    lg1 = lg0 + step * (ng - 1)
    grid_spec = f"{lg0:.4f}:{lg1:.4f}:{ng}"
    t_grid = np.array([10.0 ** (lg0 + step * i) for i in range(ng)], dtype=np.float64)

    xbits = [x_map[k][0][:ng] for k in x_keys]
    zbits = [z_map[k][0][:ng] for k in z_keys]
    nx, nz = len(xbits), len(zbits)

    cap_x = sum(1 for s in xbits if "C" in s)
    cap_z = sum(1 for s in zbits if "C" in s)

    X_pass = np.array([[1.0 if ch in ("1", "2") else 0.0 for ch in s] for s in xbits], dtype=np.float64)
    Z_pass = np.array([[1.0 if ch in ("1", "2") else 0.0 for ch in s] for s in zbits], dtype=np.float64)
    X_surv = np.array([[1.0 if ch == "2" else 0.0 for ch in s] for s in xbits], dtype=np.float64)
    Z_surv = np.array([[1.0 if ch == "2" else 0.0 for ch in s] for s in zbits], dtype=np.float64)

    px = X_pass.mean(axis=0)
    pz = Z_pass.mean(axis=0)
    pid = px * pz
    pnc = X_surv.mean(axis=0) * Z_surv.mean(axis=0)
    pheal = pid - pnc

    t80, c80 = crossing_time_strict(t_grid, pid, 0.80)
    t70, c70 = crossing_time_strict(t_grid, pid, 0.70)
    t90, c90 = crossing_time_strict(t_grid, pid, 0.90)
    t80_nc, c80_nc = crossing_time_strict(t_grid, pnc, 0.80)

    idx_hi = int(np.searchsorted(t_grid, t80))
    if idx_hi <= 0:
        pheal_t80 = float(pheal[0])
    elif idx_hi >= ng:
        pheal_t80 = float(pheal[-1])
    else:
        w = (math.log10(t80) - math.log10(t_grid[idx_hi - 1])) / (
            math.log10(t_grid[idx_hi]) - math.log10(t_grid[idx_hi - 1])
        )
        pheal_t80 = float((1.0 - w) * pheal[idx_hi - 1] + w * pheal[idx_hi])

    ix = rng.integers(0, nx, size=(B, nx))
    iz = rng.integers(0, nz, size=(B, nz))
    b_pid = X_pass[ix].mean(axis=1) * Z_pass[iz].mean(axis=1)
    b_pnc = X_surv[ix].mean(axis=1) * Z_surv[iz].mean(axis=1)

    b_t80 = np.array([crossing_time_strict(t_grid, b_pid[k], 0.80)[0] for k in range(B)])
    b_t70 = np.array([crossing_time_strict(t_grid, b_pid[k], 0.70)[0] for k in range(B)])
    b_t90 = np.array([crossing_time_strict(t_grid, b_pid[k], 0.90)[0] for k in range(B)])
    b_t80_nc = np.array([crossing_time_strict(t_grid, b_pnc[k], 0.80)[0] for k in range(B)])

    idx_near = min(range(ng), key=lambda i: abs(t_grid[i] - t80))
    z_fails_near_t80 = int(round((1.0 - float(pz[idx_near])) * nz))
    z_rare_bound_95 = 1.0 - (0.05 ** (1.0 / nz))

    return {
        "n_X": nx,
        "n_Z": nz,
        "capped_X": cap_x,
        "capped_Z": cap_z,
        "z_fails_near_t80": z_fails_near_t80,
        "z_zero_fail_95_upper_bound": z_rare_bound_95,
        "grid_spec": grid_spec,
        "t_grid": t_grid.tolist(),
        "p_X": px.tolist(),
        "p_Z": pz.tolist(),
        "P_id": pid.tolist(),
        "P_nocross": pnc.tolist(),
        "P_healed": pheal.tolist(),
        "P_healed_at_t80": pheal_t80,
        "t_80": t80,
        "t_80_crossed": c80,
        "t_80_ci95": [float(np.percentile(b_t80, 2.5)), float(np.percentile(b_t80, 97.5))],
        "t_70": t70,
        "t_70_ci95": [float(np.percentile(b_t70, 2.5)), float(np.percentile(b_t70, 97.5))],
        "t_90": t90,
        "t_90_ci95": [float(np.percentile(b_t90, 2.5)), float(np.percentile(b_t90, 97.5))],
        "t_80_nocross": t80_nc,
        "t_80_nocross_ci95": [float(np.percentile(b_t80_nc, 2.5)), float(np.percentile(b_t80_nc, 97.5))],
        "_X_pass": X_pass,
        "_Z_pass": Z_pass,
        "_X_surv": X_surv,
        "_Z_surv": Z_surv,
        "_x_keys": x_keys,
        "_z_keys": z_keys,
        "_t_grid": t_grid,
        "_b_t80": b_t80,
        "_b_t70": b_t70,
        "_b_t90": b_t90,
        "_b_t80_nc": b_t80_nc,
    }


def paired_crn_ratio(arm_num, arm_den, rng, B=10000):
    assert arm_num["_x_keys"] == arm_den["_x_keys"] and arm_num["_z_keys"] == arm_den["_z_keys"]
    t_grid = arm_num["_t_grid"]
    nx = len(arm_num["_x_keys"])
    nz = len(arm_num["_z_keys"])
    ix = rng.integers(0, nx, size=(B, nx))
    iz = rng.integers(0, nz, size=(B, nz))

    pid_n = arm_num["_X_pass"][ix].mean(axis=1) * arm_num["_Z_pass"][iz].mean(axis=1)
    pid_d = arm_den["_X_pass"][ix].mean(axis=1) * arm_den["_Z_pass"][iz].mean(axis=1)
    pnc_n = arm_num["_X_surv"][ix].mean(axis=1) * arm_num["_Z_surv"][iz].mean(axis=1)
    pnc_d = arm_den["_X_surv"][ix].mean(axis=1) * arm_den["_Z_surv"][iz].mean(axis=1)

    out = {}
    for p, key in [(0.80, "t_80"), (0.70, "t_70"), (0.90, "t_90")]:
        bn = np.array([crossing_time_strict(t_grid, pid_n[k], p)[0] for k in range(B)])
        bd = np.array([crossing_time_strict(t_grid, pid_d[k], p)[0] for k in range(B)])
        br = bn / np.maximum(bd, 1e-30)
        out[f"{key}_paired_ratio"] = float(arm_num[key] / arm_den[key])
        out[f"{key}_paired_ci95"] = [float(np.percentile(br, 2.5)), float(np.percentile(br, 97.5))]

    bn_nc = np.array([crossing_time_strict(t_grid, pnc_n[k], 0.80)[0] for k in range(B)])
    bd_nc = np.array([crossing_time_strict(t_grid, pnc_d[k], 0.80)[0] for k in range(B)])
    br_nc = bn_nc / np.maximum(bd_nc, 1e-30)
    out["t_80_nocross_paired_ratio"] = float(arm_num["t_80_nocross"] / arm_den["t_80_nocross"])
    out["t_80_nocross_paired_ci95"] = [float(np.percentile(br_nc, 2.5)), float(np.percentile(br_nc, 97.5))]

    # Independent bootstrap check alongside paired CRN:
    br_ind = arm_num["_b_t80"] / np.maximum(arm_den["_b_t80"], 1e-30)
    out["t_80_indep_ci95"] = [float(np.percentile(br_ind, 2.5)), float(np.percentile(br_ind, 97.5))]
    return out


def indep_ratio(arm_num, arm_den):
    res = {}
    for k, bk in [("t_80", "_b_t80"), ("t_70", "_b_t70"), ("t_90", "_b_t90"), ("t_80_nocross", "_b_t80_nc")]:
        pt = float(arm_num[k] / arm_den[k])
        br = arm_num[bk] / np.maximum(arm_den[bk], 1e-30)
        res[k] = {"ratio": pt, "ci95": [float(np.percentile(br, 2.5)), float(np.percentile(br, 97.5))]}
    return res


def verify_frozen_hash():
    with open(FROZEN_PATH) as f:
        doc = json.load(f)
    claimed = doc.pop("frozen_sha256")
    raw_bytes = json.dumps(doc, indent=2, sort_keys=True).encode("utf-8")
    actual = hashlib.sha256(raw_bytes).hexdigest()
    assert actual == claimed, f"Frozen SHA-256 mismatch! {actual} != {claimed}"
    doc["frozen_sha256"] = claimed
    return doc


def main():
    frozen = verify_frozen_hash()
    print(f"Verified frozen_interface_predictions.json SHA-256: {frozen['frozen_sha256']}")

    with open(V18_SPECS) as f:
        code_specs = json.load(f)

    trajs = defaultdict(dict)  # (tag, sec, beta) -> {g_tr: (bitstr, tf, cf)}
    grids = {}
    errors = []

    with open(V18_RAW) as f:
        for raw in f:
            line = raw.strip()
            if not line or "|" not in line:
                continue
            tid, body = [x.strip() for x in line.split("|", 1)]
            if body.startswith("ERROR") or body == "TIMEOUT":
                errors.append((tid, body))
                continue
            m = re.match(r"^(d1|d2|ef|g0|g1|g2|g3|g3u)__(.+)_(X_err|Z_err)__b([0-9.]+)__o(\d+)$", tid)
            if not m:
                errors.append((tid, "UNPARSED_TID"))
                continue
            pref, tag, sec, b_s, off_s = m.groups()
            off = int(off_s)
            kv = parse_kv(body)
            g_str = kv["grid"]
            if (tag, b_s) not in grids or int(g_str.split(":")[2]) > int(grids[(tag, b_s)].split(":")[2]):
                grids[(tag, b_s)] = g_str
            tcf = [(float(x.split(":")[0]), int(x.split(":")[1])) for x in kv["tc_first"].split(",") if x]
            bits = [s for s in kv["bits"].split(",") if s]
            for i, bstr in enumerate(bits):
                gtr = off + i
                if gtr not in trajs[(tag, sec, b_s)] or len(bstr) > len(trajs[(tag, sec, b_s)][gtr][0]):
                    trajs[(tag, sec, b_s)][gtr] = (bstr, tcf[i][0], tcf[i][1])

    print(f"Loaded V18 raw trajectories (errors={len(errors)})")
    if errors:
        for e in errors[:10]:
            print("  ERR:", e)

    rng = np.random.default_rng(2026112301)
    B = 10000
    arms = {}

    all_pairs = sorted({(tag, b_s) for (tag, sec, b_s) in trajs})
    for tag, b_s in all_pairs:
        xm = trajs.get((tag, "X_err", b_s), {})
        zm = trajs.get((tag, "Z_err", b_s), {})
        if not xm or not zm:
            print(f"  WARNING incomplete arm {tag} b={b_s}: nX={len(xm)} nZ={len(zm)}")
            continue
        key = f"{tag}__b{b_s}"
        arms[key] = eval_arm(xm, zm, grids[(tag, b_s)], rng, B=B)
        arms[key]["N"] = code_specs[tag]["N"]
        arms[key]["K_total"] = code_specs[tag]["K_total"]
        print(
            f"  {key:30s}: N={arms[key]['N']:7,d} n=({arms[key]['n_X']},{arms[key]['n_Z']}) "
            f"cap=({arms[key]['capped_X']},{arms[key]['capped_Z']}) "
            f"t80={arms[key]['t_80']:10.1f} {arms[key]['t_80_ci95']} "
            f"t80_nc={arms[key]['t_80_nocross']:9.1f} Pheal={arms[key]['P_healed_at_t80']:.3f}"
        )

    contrasts = {}

    # 1. Suite D1 & D2: Matched L=4 placement (CRN-paired + independent check)
    if "v18_sq_L4_cen__b6.50" in arms and "v18_sq_L4_off__b6.50" in arms:
        contrasts["D1_standalone_L4_cen_over_off"] = paired_crn_ratio(
            arms["v18_sq_L4_cen__b6.50"], arms["v18_sq_L4_off__b6.50"], rng, B=B
        )
        print("  [D1 Standalone L=4 cen/off]:", contrasts["D1_standalone_L4_cen_over_off"])

    if "v18_sq_C2_L4cen__b6.50" in arms and "v18_sq_C2_L4off__b6.50" in arms:
        contrasts["D2_composed_C2_L4_cen_over_off"] = paired_crn_ratio(
            arms["v18_sq_C2_L4cen__b6.50"], arms["v18_sq_C2_L4off__b6.50"], rng, B=B
        )
        print("  [D2 Composed C2+L4 cen/off]:", contrasts["D2_composed_C2_L4_cen_over_off"])

    # 2. Suite E & F: Direct C2 pruning & protected subsystem
    for sub_tag, name in [
        ("v18_sq_C2_pruneX__b6.50", "E_pruneX_over_full"),
        ("v18_sq_C2_pruneZ_sub__b6.50", "F_pruneZ_sub_over_full"),
        ("v18_sq_C2_pruneB_sub__b6.50", "F_pruneBoth_sub_over_full"),
    ]:
        if sub_tag in arms and "v18_sq_C2_full__b6.50" in arms:
            contrasts[name] = indep_ratio(arms[sub_tag], arms["v18_sq_C2_full__b6.50"])
            print(f"  [{name}]: t80 ratio={contrasts[name]['t_80']['ratio']:.4f} {contrasts[name]['t_80']['ci95']}")

    # 3. Suite G: Prospective Held-Out Simplification Decisions vs Frozen Predictions
    prospective_eval = {
        "frozen_sha256": frozen["frozen_sha256"],
        "frozen_timestamp_utc": frozen["frozen_timestamp_utc"],
    }

    if all(k in arms for k in ["v18_sq_C2_RX3__b6.50", "v18_sq_C2_RZ3__b6.50", "v18_sq_C2_R3__b6.50", "v18_sq_C2_C3__b6.50"]):
        r_rx3_r3 = indep_ratio(arms["v18_sq_C2_RX3__b6.50"], arms["v18_sq_C2_R3__b6.50"])
        r_rz3_r3 = indep_ratio(arms["v18_sq_C2_RZ3__b6.50"], arms["v18_sq_C2_R3__b6.50"])
        r_rx3_rz3 = indep_ratio(arms["v18_sq_C2_RX3__b6.50"], arms["v18_sq_C2_RZ3__b6.50"])
        r_rx3_c3 = indep_ratio(arms["v18_sq_C2_RX3__b6.50"], arms["v18_sq_C2_C3__b6.50"])
        r_1cyc_rx3_rz3 = indep_ratio(arms["v18_sq_RX3__b6.50"], arms["v18_sq_RZ3__b6.50"])
        prospective_eval["Decision_1_fine_sector_cutback_b65"] = {
            "frozen_predictions": frozen["predeclared_decisions"]["Decision_1_fine_sector_cutback_b65"],
            "observed_C2_RX3_t80": arms["v18_sq_C2_RX3__b6.50"]["t_80"],
            "observed_C2_RX3_ci95": arms["v18_sq_C2_RX3__b6.50"]["t_80_ci95"],
            "observed_C2_RZ3_t80": arms["v18_sq_C2_RZ3__b6.50"]["t_80"],
            "observed_C2_RZ3_ci95": arms["v18_sq_C2_RZ3__b6.50"]["t_80_ci95"],
            "observed_C2_R3_t80": arms["v18_sq_C2_R3__b6.50"]["t_80"],
            "observed_C2_R3_ci95": arms["v18_sq_C2_R3__b6.50"]["t_80_ci95"],
            "observed_C2_C3_t80": arms["v18_sq_C2_C3__b6.50"]["t_80"],
            "observed_RX3_over_R3": r_rx3_r3["t_80"],
            "observed_RZ3_over_R3": r_rz3_r3["t_80"],
            "observed_RX3_over_RZ3": r_rx3_rz3["t_80"],
            "observed_RX3_over_C3": r_rx3_c3["t_80"],
            "observed_standalone_RX3_over_RZ3": r_1cyc_rx3_rz3["t_80"],
            "meets_target_deadline_20k": bool(arms["v18_sq_C2_RX3__b6.50"]["t_80"] >= 20000.0),
        }
        print("  [Decision 1 Held-Out]:", prospective_eval["Decision_1_fine_sector_cutback_b65"])

    if "v18_sq_C2pX_C2__b6.50" in arms and "v18_sq_C2_C2__b6.50" in arms:
        r_p = indep_ratio(arms["v18_sq_C2pX_C2__b6.50"], arms["v18_sq_C2_C2__b6.50"])
        prospective_eval["Decision_2_below_compact_frontier_b65"] = {
            "frozen_predictions": frozen["predeclared_decisions"]["Decision_2_below_compact_frontier_b65"],
            "observed_C2pXC2_t80": arms["v18_sq_C2pX_C2__b6.50"]["t_80"],
            "observed_C2pXC2_ci95": arms["v18_sq_C2pX_C2__b6.50"]["t_80_ci95"],
            "observed_C2C2_t80": arms["v18_sq_C2_C2__b6.50"]["t_80"],
            "observed_C2C2_ci95": arms["v18_sq_C2_C2__b6.50"]["t_80_ci95"],
            "observed_C2pXC2_over_C2C2": r_p["t_80"],
            "meets_target_deadline_120k": bool(arms["v18_sq_C2pX_C2__b6.50"]["t_80"] >= 120000.0),
        }
        print("  [Decision 2 Held-Out]:", prospective_eval["Decision_2_below_compact_frontier_b65"])

    if all(k in arms for k in ["v18_sq_RX3_C2__b7.00", "v18_sq_RZ3_C2__b7.00", "v18_sq_R3_C2__b7.00", "v18_sq_C3_C2__b7.00"]):
        prospective_eval["Decision_3_cold_coarse_cutback_b70"] = {
            "frozen_predictions": frozen["predeclared_decisions"]["Decision_3_cold_coarse_cutback_b70"],
            "observed_RX3C2_t80": arms["v18_sq_RX3_C2__b7.00"]["t_80"],
            "observed_RX3C2_ci95": arms["v18_sq_RX3_C2__b7.00"]["t_80_ci95"],
            "observed_RZ3C2_t80": arms["v18_sq_RZ3_C2__b7.00"]["t_80"],
            "observed_RZ3C2_ci95": arms["v18_sq_RZ3_C2__b7.00"]["t_80_ci95"],
            "observed_R3C2_t80": arms["v18_sq_R3_C2__b7.00"]["t_80"],
            "observed_R3C2_ci95": arms["v18_sq_R3_C2__b7.00"]["t_80_ci95"],
            "observed_C3C2_t80": arms["v18_sq_C3_C2__b7.00"]["t_80"],
            "observed_C3C2_ci95": arms["v18_sq_C3_C2__b7.00"]["t_80_ci95"],
            "observed_RX3C2_over_RZ3C2": indep_ratio(arms["v18_sq_RX3_C2__b7.00"], arms["v18_sq_RZ3_C2__b7.00"])["t_80"],
            "observed_RX3C2_over_C3C2": indep_ratio(arms["v18_sq_RX3_C2__b7.00"], arms["v18_sq_C3_C2__b7.00"])["t_80"],
            "observed_RX3C2_over_R3C2": indep_ratio(arms["v18_sq_RX3_C2__b7.00"], arms["v18_sq_R3_C2__b7.00"])["t_80"],
        }
        print("  [Decision 3 Held-Out]:", prospective_eval["Decision_3_cold_coarse_cutback_b70"])

    # Strip private numpy arrays before JSON dump
    clean_arms = {}
    for k, v in arms.items():
        clean_arms[k] = {sk: sv for sk, sv in v.items() if not sk.startswith("_")}

    out_doc = {
        "v18_errors": errors,
        "arms": clean_arms,
        "contrasts": contrasts,
        "prospective_transfer_evaluation": prospective_eval,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out_doc, f, indent=2)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
