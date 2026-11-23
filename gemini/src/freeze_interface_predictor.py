#!/usr/bin/env python3
"""
freeze_interface_predictor.py
Zero-fitted-parameter Collared Fine-to-Parent Interface Predictor & Pre-Simulation Freeze.

Strictly executed BEFORE any whole-memory thermal simulation of the seven held-out two-cycle
architectures (C2+RX3, C2+RZ3, C2pX+C2, C2+L4_off, C2+L4_cen at beta=6.5; RX3+C2, RZ3+C2 at beta=7.0).

Physical Formulation (zero fitted parameters):
  1. Open 1x1 cell (`patch_1x1(B)`, d_coarse = (1, 1)): isolated module without neighbor seams.
  2. Collared 3x3 interface patch (`patch_3x3(B)`, d_coarse = (3, 3), N_base = 18): surrounds an
     interior fine cell on all four sides with restoring parent seams and permits weight-1 coarse
     error recovery under Union-Find.
  3. Exact monotone inversion on `base_3x3` (via `src/eval_coarse_uf`, reusing `FastDecoderV7`
     verbatim from `kmc_engine_v17.cpp`) extracts the per-coarse-edge interface error probability
     p_collar^(B, sigma)(t) both WITH returns ('1' or '2') and WITHOUT returns ('2' only), and
     evaluates coarse architecture A's Union-Find recovery P_rec^{AB}(t):
       - Primary frozen predictor: Collared 3x3 Interface + Return
       - Ablation 1 (No Return):   Collared 3x3 Interface, Absorbing First-Exit ('2' only)
       - Ablation 2 (No Collar):   Open 1x1 Interface + Return (inverted on `base_1x1`)
     compared against explicit scalar baselines (1/N, clean birth hazard 1/Gamma_0, standalone t_0.80).
"""

import hashlib
import json
import math
import os
import subprocess
import sys
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "cluster_v9"))

from verify_local_quotients_and_algebra import (
    GeneralCubical2D,
    apply_step_general,
    build_multistep_with_seed_logicals,
    clean_departure_hazard,
    gf2_rank_sparse,
    seed_planar_11,
    sheaf_to_css_with_seams,
)

ENGINE_LOCAL = os.path.join(ROOT, "src", "kmc_engine_v17_local")
EVAL_UF = os.path.join(ROOT, "src", "eval_coarse_uf")
PATCH_DIR = os.path.join(ROOT, "interface_patches")
FROZEN_OUT = os.path.join(ROOT, "frozen_interface_predictions.json")


def seed_patch_1x1():
    return seed_planar_11(), set()


def seed_patch_3x3():
    def vid(x, y):
        return 4 * y + x

    edges, cols = [], []
    he = {}
    for y in range(4):
        for x in range(3):
            he[(x, y)] = len(edges)
            edges.append((vid(x, y), vid(x + 1, y)))
            cols.append("blue" if y in (0, 3) else "none")
    ve = {}
    for x in range(4):
        for y in range(3):
            ve[(x, y)] = len(edges)
            edges.append((vid(x, y), vid(x, y + 1)))
            cols.append("red" if x in (0, 3) else "none")
    faces = []
    for y in range(3):
        for x in range(3):
            faces.append(
                (
                    vid(x, y),
                    vid(x + 1, y),
                    vid(x + 1, y + 1),
                    vid(x, y + 1),
                    he[(x, y)],
                    he[(x, y + 1)],
                    ve[(x, y)],
                    ve[(x + 1, y)],
                )
            )
    return GeneralCubical2D(16, edges, cols, faces), set()


def write_code_and_roles(comp, tag_prefix):
    N, HX, HZ, seam_qs, _, cyl_qs = sheaf_to_css_with_seams(comp)
    junc_set = set(seam_qs) | set(cyl_qs)
    roles = [1 if q in junc_set else 0 for q in range(N)]

    for sec, checks, stabs in [("X_err", HZ, HX), ("Z_err", HX, HZ)]:
        raw_p = os.path.join(PATCH_DIR, f"{tag_prefix}_{sec}.raw")
        code_p = os.path.join(PATCH_DIR, f"{tag_prefix}_{sec}.code")
        role_p = os.path.join(PATCH_DIR, f"{tag_prefix}_{sec}.roles")
        with open(raw_p, "w") as f:
            f.write(f"{N} {len(checks)} {len(stabs)}\n")
            for r in checks:
                f.write(str(len(r)) + " " + " ".join(map(str, r)) + "\n")
            for r in stabs:
                f.write(str(len(r)) + " " + " ".join(map(str, r)) + "\n")
        subprocess.run([ENGINE_LOCAL, "prep_logicals", raw_p, code_p], check=True)
        os.remove(raw_p)
        subprocess.run([ENGINE_LOCAL, "verify", code_p], check=True, stdout=subprocess.DEVNULL)
        with open(role_p, "w") as f:
            f.write(f"{N}\n" + " ".join(map(str, roles)) + "\n0\n")
    return N, sum(roles), 0, 0


class CoarseUFEvaluator:
    def __init__(self, code_path, roles_path):
        self.proc = subprocess.Popen(
            [EVAL_UF, code_path, roles_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def eval_fail(self, p_flat, p_junc, p_span=0.0, shots=1200, seed=1234567):
        self.proc.stdin.write(f"{p_flat:.8f} {p_junc:.8f} {p_span:.8f} {shots} {seed}\n")
        self.proc.stdin.flush()
        return float(self.proc.stdout.readline().strip())

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait()


def bisect_monotone(fn, target, lo=0.0, hi=0.49, iters=16):
    if target <= 1e-6:
        return 0.0
    if target >= fn(hi):
        return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if fn(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def run_local_patch_kmc(code_path, beta, trials, lg0, lg1, ng, max_flips=450_000_000):
    res = subprocess.run(
        [
            ENGINE_LOCAL,
            "kinetic_readout",
            code_path,
            "none",
            "1.0000",
            f"{beta:.2f}",
            str(trials),
            "0",
            f"{lg0:.2f}",
            f"{lg1:.2f}",
            str(ng),
            str(max_flips),
            "8",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    line = res.stdout.strip().splitlines()[-1]
    kv = dict(tok.split("=", 1) for tok in line.split() if "=" in tok)
    assert int(kv["capped"]) == 0, f"Hit flip cap on {code_path}: {line}"
    bits = kv["bits"].split(",")
    ret_mat = np.array([[1.0 if ch in ("1", "2") else 0.0 for ch in s] for s in bits], dtype=np.float64)
    abs_mat = np.array([[1.0 if ch == "2" else 0.0 for ch in s] for s in bits], dtype=np.float64)
    return ret_mat, abs_mat


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


def patch_stats(t_grid, x_ret, z_ret, x_abs, z_abs, B=4000, seed=99):
    pid = x_ret.mean(axis=0) * z_ret.mean(axis=0)
    pnc = x_abs.mean(axis=0) * z_abs.mean(axis=0)
    t80, c80 = crossing_time_strict(t_grid, pid, 0.80)
    t80_nc, _ = crossing_time_strict(t_grid, pnc, 0.80)
    rng = np.random.default_rng(seed)
    nx, nz = x_ret.shape[0], z_ret.shape[0]
    ix = rng.integers(0, nx, size=(B, nx))
    iz = rng.integers(0, nz, size=(B, nz))
    b_pid = x_ret[ix].mean(axis=1) * z_ret[iz].mean(axis=1)
    b_pnc = x_abs[ix].mean(axis=1) * z_abs[iz].mean(axis=1)
    bt80 = np.array([crossing_time_strict(t_grid, b_pid[k], 0.80)[0] for k in range(B)])
    bt80_nc = np.array([crossing_time_strict(t_grid, b_pnc[k], 0.80)[0] for k in range(B)])
    return {
        "t_80": t80,
        "t_80_crossed": c80,
        "t_80_ci95": [float(np.percentile(bt80, 2.5)), float(np.percentile(bt80, 97.5))],
        "t_80_nocross": t80_nc,
        "t_80_nocross_ci95": [float(np.percentile(bt80_nc, 2.5)), float(np.percentile(bt80_nc, 97.5))],
    }


def main():
    t0 = time.time()
    os.makedirs(PATCH_DIR, exist_ok=True)

    for btag, sfn in [("base_1x1", seed_patch_1x1), ("base_3x3", seed_patch_3x3)]:
        c0, _ = sfn()
        write_code_and_roles(c0, btag)

    MODULE_SPECS = {
        "C2":     [(2, [1], "X", "full"),   (2, [1], "Z", "full")],
        "C2pX":   [(2, [1], "X", "cb3rib"), (2, [1], "Z", "full")],
        "C3":     [(3, [1], "X", "full"),   (3, [1], "Z", "full")],
        "RX3":    [(3, [1, 2], "X", "full"), (3, [1], "Z", "full")],
        "RZ3":    [(3, [1], "X", "full"),   (3, [1, 2], "Z", "full")],
        "R3":     [(3, [1, 2], "X", "full"), (3, [1, 2], "Z", "full")],
        "L4_off": [(4, [1], "X", "full"),   (4, [1], "Z", "full")],
        "L4_cen": [(4, [2], "X", "full"),   (4, [2], "Z", "full")],
    }

    coarse_meta = {}
    for aname in ["C2", "C2pX", "C3", "RX3", "RZ3", "R3"]:
        comp, cyl = seed_patch_1x1()
        for L, rb, it, gm in MODULE_SPECS[aname]:
            comp, cyl, _ = apply_step_general(comp, L, rb, it, cyl, gm)
        nA, nJunc, *_ = write_code_and_roles(comp, f"coarse_{aname}")
        coarse_meta[aname] = {"N": nA, "n_junc": nJunc, "n_flat": nA - nJunc}

    FINE_MODULES_65 = ["C2", "C3", "RX3", "RZ3", "R3", "L4_off", "L4_cen"]
    patch_sizes = {}
    for bname in FINE_MODULES_65:
        patch_sizes[bname] = {}
        for pkind, sfn in [("1x1", seed_patch_1x1), ("3x3", seed_patch_3x3)]:
            comp, cyl = sfn()
            for L, rb, it, gm in MODULE_SPECS[bname]:
                comp, cyl, _ = apply_step_general(comp, L, rb, it, cyl, gm)
            nP, *_ = write_code_and_roles(comp, f"patch_{pkind}_{bname}")
            patch_sizes[bname][pkind] = nP
        print(f"  Built patches {bname:6s}: N_1x1={patch_sizes[bname]['1x1']:5,d}, N_3x3={patch_sizes[bname]['3x3']:6,d}", flush=True)

    ev_1x1 = {sec: CoarseUFEvaluator(f"{PATCH_DIR}/base_1x1_{sec}.code", f"{PATCH_DIR}/base_1x1_{sec}.roles") for sec in ("X_err", "Z_err")}
    ev_3x3 = {sec: CoarseUFEvaluator(f"{PATCH_DIR}/base_3x3_{sec}.code", f"{PATCH_DIR}/base_3x3_{sec}.roles") for sec in ("X_err", "Z_err")}

    lg0_65, lg1_65, ng_65 = 2.00, 5.80, 39
    t_grid_65 = np.array([10.0 ** (lg0_65 + 0.10 * i) for i in range(ng_65)])

    theta_bank = {"6.50": {}, "7.00": {}}
    standalone_1x1_65 = {}
    collared_3x3_65 = {}

    for bname in FINE_MODULES_65:
        lg1_use = 5.80 if bname == "C2" else (4.40 if "L4" in bname else 5.40)
        ng_use = int(round((lg1_use - lg0_65) / 0.10)) + 1
        t_sub = t_grid_65[:ng_use]
        n_tr_1 = 96 if bname in ("RX3", "RZ3") else 64
        n_tr_3 = 48 if "L4" not in bname else 40

        print(f"Running local 1x1 & collared 3x3 KMC for B={bname} at beta=6.50...", flush=True)
        s1_mats, s3_mats = {}, {}
        th_sec = {}
        for sec in ("X_err", "Z_err"):
            r1, a1 = run_local_patch_kmc(f"{PATCH_DIR}/patch_1x1_{bname}_{sec}.code", 6.50, n_tr_1, lg0_65, lg1_use, ng_use)
            r3, a3 = run_local_patch_kmc(f"{PATCH_DIR}/patch_3x3_{bname}_{sec}.code", 6.50, n_tr_3, lg0_65, lg1_use, ng_use)
            s1_mats[sec] = (r1, a1)
            s3_mats[sec] = (r3, a3)

            def pad(v):
                return v if len(v) == ng_65 else np.concatenate([v, np.full(ng_65 - len(v), v[-1])])

            f1_ret = pad(1.0 - r1.mean(axis=0))
            f3_ret = pad(1.0 - r3.mean(axis=0))
            f3_abs = pad(1.0 - a3.mean(axis=0))

            p1_r = np.array([bisect_monotone(lambda x: ev_1x1[sec].eval_fail(x, x, 0.0, 1500, 61001 + g), float(f1_ret[g])) for g in range(ng_65)])
            pc_r = np.array([bisect_monotone(lambda x: ev_3x3[sec].eval_fail(x, x, 0.0, 1500, 71001 + g), float(f3_ret[g])) for g in range(ng_65)])
            pc_a = np.array([bisect_monotone(lambda x: ev_3x3[sec].eval_fail(x, x, 0.0, 1500, 81001 + g), float(f3_abs[g])) for g in range(ng_65)])
            th_sec[sec] = {"p_open_ret": p1_r, "p_collar_ret": pc_r, "p_collar_abs": pc_a}

        standalone_1x1_65[bname] = patch_stats(t_sub, s1_mats["X_err"][0], s1_mats["Z_err"][0], s1_mats["X_err"][1], s1_mats["Z_err"][1])
        collared_3x3_65[bname] = patch_stats(t_sub, s3_mats["X_err"][0], s3_mats["Z_err"][0], s3_mats["X_err"][1], s3_mats["Z_err"][1])
        theta_bank["6.50"][bname] = th_sec
        print(
            f"  B={bname:6s}: 1x1 t80={standalone_1x1_65[bname]['t_80']:8.1f} -> "
            f"3x3_collar t80={collared_3x3_65[bname]['t_80']:8.1f} (nocross={collared_3x3_65[bname]['t_80_nocross']:8.1f})",
            flush=True,
        )

    # Beta = 7.00 local patches for B = C2 (Decision 3):
    lg0_70, lg1_70, ng_70 = 3.00, 7.00, 41
    t_grid_70 = np.array([10.0 ** (lg0_70 + 0.10 * i) for i in range(ng_70)])
    print("Running local 1x1 & collared 3x3 KMC for B=C2 at beta=7.00...", flush=True)
    th_c2_70 = {}
    for sec in ("X_err", "Z_err"):
        r1, _ = run_local_patch_kmc(f"{PATCH_DIR}/patch_1x1_C2_{sec}.code", 7.00, 48, lg0_70, lg1_70, ng_70, max_flips=900_000_000)
        r3, a3 = run_local_patch_kmc(f"{PATCH_DIR}/patch_3x3_C2_{sec}.code", 7.00, 32, lg0_70, lg1_70, ng_70, max_flips=1_200_000_000)
        f1_ret = 1.0 - r1.mean(axis=0)
        f3_ret = 1.0 - r3.mean(axis=0)
        f3_abs = 1.0 - a3.mean(axis=0)
        p1_r = np.array([bisect_monotone(lambda x: ev_1x1[sec].eval_fail(x, x, 0.0, 1500, 62001 + g), float(f1_ret[g])) for g in range(ng_70)])
        pc_r = np.array([bisect_monotone(lambda x: ev_3x3[sec].eval_fail(x, x, 0.0, 1500, 72001 + g), float(f3_ret[g])) for g in range(ng_70)])
        pc_a = np.array([bisect_monotone(lambda x: ev_3x3[sec].eval_fail(x, x, 0.0, 1500, 82001 + g), float(f3_abs[g])) for g in range(ng_70)])
        th_c2_70[sec] = {"p_open_ret": p1_r, "p_collar_ret": pc_r, "p_collar_abs": pc_a}
    theta_bank["7.00"]["C2"] = th_c2_70

    for sec in ("X_err", "Z_err"):
        ev_1x1[sec].close()
        ev_3x3[sec].close()

    def predict_pair(aname, bname, bstr):
        t_grid = t_grid_65 if bstr == "6.50" else t_grid_70
        th = theta_bank[bstr][bname]
        ng = len(t_grid)
        evX = CoarseUFEvaluator(f"{PATCH_DIR}/coarse_{aname}_X_err.code", f"{PATCH_DIR}/coarse_{aname}_X_err.roles")
        evZ = CoarseUFEvaluator(f"{PATCH_DIR}/coarse_{aname}_Z_err.code", f"{PATCH_DIR}/coarse_{aname}_Z_err.roles")
        res = {}
        for vkey, pkey in [
            ("collared_3x3_ret", "p_collar_ret"),
            ("ablation_no_return", "p_collar_abs"),
            ("ablation_open_1x1", "p_open_ret"),
        ]:
            px = np.array([1.0 - evX.eval_fail(th["X_err"][pkey][g], th["X_err"][pkey][g], 0.0, 1200, 90001 + 37 * g) for g in range(ng)])
            pz = np.array([1.0 - evZ.eval_fail(th["Z_err"][pkey][g], th["Z_err"][pkey][g], 0.0, 1200, 95001 + 37 * g) for g in range(ng)])
            pid = px * pz
            res[vkey] = {
                "t_80_pred": crossing_time_strict(t_grid, pid, 0.80)[0],
                "t_70_pred": crossing_time_strict(t_grid, pid, 0.70)[0],
                "t_90_pred": crossing_time_strict(t_grid, pid, 0.90)[0],
            }
        evX.close()
        evZ.close()
        return res

    print("\n=== Exploratory Check on Known 3x3 Matrix (beta = 6.50, zero fitted parameters) ===")
    exploratory_val = {}
    for aname in ["C2", "C3", "R3"]:
        for bname in ["C2", "C3", "R3"]:
            k = f"{aname}_{bname}"
            exploratory_val[k] = predict_pair(aname, bname, "6.50")
            print(
                f"  {k:6s}: Collared+Ret={exploratory_val[k]['collared_3x3_ret']['t_80_pred']:8.1f} | "
                f"NoRet={exploratory_val[k]['ablation_no_return']['t_80_pred']:8.1f} | "
                f"Open1x1={exploratory_val[k]['ablation_open_1x1']['t_80_pred']:8.1f}"
            )

    held_out_specs = {
        "C2_RX3_b65":   ("C2",   "RX3",    "6.50", MODULE_SPECS["C2"]   + MODULE_SPECS["RX3"]),
        "C2_RZ3_b65":   ("C2",   "RZ3",    "6.50", MODULE_SPECS["C2"]   + MODULE_SPECS["RZ3"]),
        "C2pX_C2_b65":  ("C2pX", "C2",     "6.50", MODULE_SPECS["C2pX"] + MODULE_SPECS["C2"]),
        "C2_L4off_b65": ("C2",   "L4_off", "6.50", MODULE_SPECS["C2"]   + MODULE_SPECS["L4_off"]),
        "C2_L4cen_b65": ("C2",   "L4_cen", "6.50", MODULE_SPECS["C2"]   + MODULE_SPECS["L4_cen"]),
        "RX3_C2_b70":   ("RX3",  "C2",     "7.00", MODULE_SPECS["RX3"]  + MODULE_SPECS["C2"]),
        "RZ3_C2_b70":   ("RZ3",  "C2",     "7.00", MODULE_SPECS["RZ3"]  + MODULE_SPECS["C2"]),
    }
    ref_b70 = {
        "C2_C2_b70": predict_pair("C2", "C2", "7.00"),
        "C3_C2_b70": predict_pair("C3", "C2", "7.00"),
        "R3_C2_b70": predict_pair("R3", "C2", "7.00"),
    }

    print("\n=== Freezing Prospective Predictions on 7 Unsimulated Held-Out Two-Cycle Targets ===")
    held_out_frozen = {}
    for hkey, (aname, bname, bstr, steps) in held_out_specs.items():
        N, HX, HZ, lx, lz = build_multistep_with_seed_logicals(steps)
        rX, _ = gf2_rank_sparse(HX, N)
        rZ, _ = gf2_rank_sparse(HZ, N)
        K = N - rX - rZ
        assert K == 1
        bval = float(bstr)
        hazX, _ = clean_departure_hazard(N, HZ, bval)
        hazZ, _ = clean_departure_hazard(N, HX, bval)
        preds = predict_pair(aname, bname, bstr)
        held_out_frozen[hkey] = {
            "coarse": aname,
            "fine": bname,
            "beta": bstr,
            "N": N,
            "M_X": len(HX),
            "M_Z": len(HZ),
            "rank_HX": rX,
            "rank_HZ": rZ,
            "K": K,
            "clean_hazard_X": hazX,
            "clean_hazard_Z": hazZ,
            "clean_hazard_tot": hazX + hazZ,
            "predictions": preds,
        }
        print(
            f"  FROZEN {hkey:14s} (N={N:7,d}, K={K}): "
            f"Collared+Ret={preds['collared_3x3_ret']['t_80_pred']:10.1f} | "
            f"NoRet={preds['ablation_no_return']['t_80_pred']:10.1f} | "
            f"Open1x1={preds['ablation_open_1x1']['t_80_pred']:10.1f}"
        )

    decisions = {
        "Decision_1_fine_sector_cutback_b65": {
            "description": "Under coarse C2 at beta=6.5, delete the second parallel rib from R3's nonlimiting Z step (C2+RX3, N=99,580, -27.1% qubits vs C2+R3=136,536) vs from R3's limiting X step (C2+RZ3, N=131,820).",
            "target_deadline_t80": 20000.0,
            "pred_RX3_over_R3_collared_ret": held_out_frozen["C2_RX3_b65"]["predictions"]["collared_3x3_ret"]["t_80_pred"] / exploratory_val["C2_R3"]["collared_3x3_ret"]["t_80_pred"],
            "pred_RZ3_over_R3_collared_ret": held_out_frozen["C2_RZ3_b65"]["predictions"]["collared_3x3_ret"]["t_80_pred"] / exploratory_val["C2_R3"]["collared_3x3_ret"]["t_80_pred"],
            "pred_RX3_over_RZ3_collared_ret": held_out_frozen["C2_RX3_b65"]["predictions"]["collared_3x3_ret"]["t_80_pred"] / held_out_frozen["C2_RZ3_b65"]["predictions"]["collared_3x3_ret"]["t_80_pred"],
            "direct_3x3_patch_ratio_RX3_over_RZ3": collared_3x3_65["RX3"]["t_80"] / collared_3x3_65["RZ3"]["t_80"],
            "baseline_1_over_N_RX3_over_RZ3": held_out_frozen["C2_RZ3_b65"]["N"] / held_out_frozen["C2_RX3_b65"]["N"],
            "baseline_clean_birth_tot_RX3_over_RZ3": held_out_frozen["C2_RZ3_b65"]["clean_hazard_tot"] / held_out_frozen["C2_RX3_b65"]["clean_hazard_tot"],
            "baseline_standalone_1x1_RX3_over_RZ3": standalone_1x1_65["RX3"]["t_80"] / standalone_1x1_65["RZ3"]["t_80"],
        },
        "Decision_2_below_compact_frontier_b65": {
            "description": "Prune one of four bridge plaquettes in the coarse X step of all-compact C2C2 (C2pX+C2, N=24,086, K=1, -4.22% qubits below C2C2=25,148) at beta=6.5.",
            "target_deadline_t80": 120000.0,
            "pred_C2pXC2_over_C2C2_collared_ret": held_out_frozen["C2pX_C2_b65"]["predictions"]["collared_3x3_ret"]["t_80_pred"] / exploratory_val["C2_C2"]["collared_3x3_ret"]["t_80_pred"],
            "baseline_standalone_C2pX_over_C2": 3487.0 / 5027.0,
            "baseline_1_over_N": 25148.0 / 24086.0,
        },
        "Decision_3_cold_coarse_cutback_b70": {
            "description": "At planar beta=7.0, test whether RX3+C2 (N=116,164, removing 66.2% of R3+C2's extra qubits over C3+C2) retains R3+C2's cold coarse-bridge gain over C3+C2 and RZ3+C2 (note: flat 2D coarse i.i.d. model explicitly tests whether coarse size penalty or 3D coarse X-bridge cleaning dominates!).",
            "pred_RX3C2_over_C3C2_collared_ret": held_out_frozen["RX3_C2_b70"]["predictions"]["collared_3x3_ret"]["t_80_pred"] / ref_b70["C3_C2_b70"]["collared_3x3_ret"]["t_80_pred"],
            "pred_RX3C2_over_RZ3C2_collared_ret": held_out_frozen["RX3_C2_b70"]["predictions"]["collared_3x3_ret"]["t_80_pred"] / held_out_frozen["RZ3_C2_b70"]["predictions"]["collared_3x3_ret"]["t_80_pred"],
            "pred_R3C2_over_C3C2_collared_ret": ref_b70["R3_C2_b70"]["collared_3x3_ret"]["t_80_pred"] / ref_b70["C3_C2_b70"]["collared_3x3_ret"]["t_80_pred"],
        },
        "Matched_L4_composed_transfer_b65": {
            "description": "Resource-matched L=4 placement under coarse C2 (N=261,476, K=1): centered (2+2) vs offset (1+3).",
            "pred_cen_over_off_collared_ret": held_out_frozen["C2_L4cen_b65"]["predictions"]["collared_3x3_ret"]["t_80_pred"] / held_out_frozen["C2_L4off_b65"]["predictions"]["collared_3x3_ret"]["t_80_pred"],
            "pred_cen_over_off_no_return_ablation": held_out_frozen["C2_L4cen_b65"]["predictions"]["ablation_no_return"]["t_80_pred"] / held_out_frozen["C2_L4off_b65"]["predictions"]["ablation_no_return"]["t_80_pred"],
            "direct_3x3_patch_ratio_ret": collared_3x3_65["L4_cen"]["t_80"] / collared_3x3_65["L4_off"]["t_80"],
            "direct_3x3_patch_ratio_nocross": collared_3x3_65["L4_cen"]["t_80_nocross"] / collared_3x3_65["L4_off"]["t_80_nocross"],
        },
    }

    payload = {
        "frozen_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "zero_fitted_parameters": True,
        "coarse_module_roles": coarse_meta,
        "local_patch_qubit_counts": patch_sizes,
        "standalone_1x1_b65": standalone_1x1_65,
        "collared_3x3_b65": collared_3x3_65,
        "exploratory_3x3_b65": exploratory_val,
        "reference_b70": ref_b70,
        "held_out_frozen_predictions": held_out_frozen,
        "predeclared_decisions": decisions,
    }
    raw_bytes = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    sha256_hex = hashlib.sha256(raw_bytes).hexdigest()
    payload["frozen_sha256"] = sha256_hex
    with open(FROZEN_OUT, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nFROZEN predictions written to {FROZEN_OUT}")
    print(f"FROZEN SHA-256: {sha256_hex}")
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
