#!/usr/bin/env python3
"""
eval_L3_collared_predictions.py
Evaluates zero-fitted-parameter collared 3x3 interface predictions across pure-L=3 two-cycle
architectures ({C3, RX3, R3} x {C3, R3, RX3, RZ3}) at beta = 6.50 and beta = 7.00 using the
unchanged `CoarseUFEvaluator` and `interface_patches/` codes from `freeze_interface_predictor.py`
(leaving `frozen_interface_predictions.json` cryptographically untouched).
"""

import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from freeze_interface_predictor import (
    CoarseUFEvaluator,
    bisect_monotone,
    crossing_time_strict,
    run_local_patch_kmc,
)


def main():
    ev_1x1 = {
        sec: CoarseUFEvaluator(f"{ROOT}/interface_patches/base_1x1_{sec}.code", f"{ROOT}/interface_patches/base_1x1_{sec}.roles")
        for sec in ("X_err", "Z_err")
    }
    ev_3x3 = {
        sec: CoarseUFEvaluator(f"{ROOT}/interface_patches/base_3x3_{sec}.code", f"{ROOT}/interface_patches/base_3x3_{sec}.roles")
        for sec in ("X_err", "Z_err")
    }

    lg0, lg1, ng = 2.00, 5.40, 35
    t_grid = np.array([10.0 ** (lg0 + 0.10 * i) for i in range(ng)])

    theta = {}
    for bname in ["C3", "R3", "RX3", "RZ3"]:
        theta[bname] = {}
        for sec in ("X_err", "Z_err"):
            r1, _ = run_local_patch_kmc(f"{ROOT}/interface_patches/patch_1x1_{bname}_{sec}.code", 6.50, 64, lg0, lg1, ng)
            r3, a3 = run_local_patch_kmc(f"{ROOT}/interface_patches/patch_3x3_{bname}_{sec}.code", 6.50, 48, lg0, lg1, ng)
            p1_r = np.array([bisect_monotone(lambda x: ev_1x1[sec].eval_fail(x, x, 0.0, 1500, 61001 + g), float(1.0 - r1.mean(axis=0)[g])) for g in range(ng)])
            pc_r = np.array([bisect_monotone(lambda x: ev_3x3[sec].eval_fail(x, x, 0.0, 1500, 71001 + g), float(1.0 - r3.mean(axis=0)[g])) for g in range(ng)])
            pc_a = np.array([bisect_monotone(lambda x: ev_3x3[sec].eval_fail(x, x, 0.0, 1500, 81001 + g), float(1.0 - a3.mean(axis=0)[g])) for g in range(ng)])
            theta[bname][sec] = {"collar_ret": pc_r, "collar_abs": pc_a, "open_1x1": p1_r}

    for sec in ("X_err", "Z_err"):
        ev_1x1[sec].close()
        ev_3x3[sec].close()

    out_preds = {}
    for aname in ["C3", "RX3", "RZ3", "R3"]:
        evX = CoarseUFEvaluator(f"{ROOT}/interface_patches/coarse_{aname}_X_err.code", f"{ROOT}/interface_patches/coarse_{aname}_X_err.roles")
        evZ = CoarseUFEvaluator(f"{ROOT}/interface_patches/coarse_{aname}_Z_err.code", f"{ROOT}/interface_patches/coarse_{aname}_Z_err.roles")
        for bname in ["C3", "R3", "RX3", "RZ3"]:
            k = f"{aname}_{bname}"
            out_preds[k] = {}
            for mode in ("collar_ret", "collar_abs", "open_1x1"):
                px = np.array([1.0 - evX.eval_fail(theta[bname]["X_err"][mode][g], theta[bname]["X_err"][mode][g], 0.0, 1200, 90001 + 37 * g) for g in range(ng)])
                pz = np.array([1.0 - evZ.eval_fail(theta[bname]["Z_err"][mode][g], theta[bname]["Z_err"][mode][g], 0.0, 1200, 95001 + 37 * g) for g in range(ng)])
                t80, _ = crossing_time_strict(t_grid, px * pz, 0.80)
                out_preds[k][mode] = t80
            cr = out_preds[k]["collar_ret"]
            ca = out_preds[k]["collar_abs"]
            op = out_preds[k]["open_1x1"]
            print(f"  3D L=3 Zero-Param Pred {k:8s}: Collar+Ret={cr:9.1f} | NoRet={ca:9.1f} | Open1x1={op:9.1f}")
        evX.close()
        evZ.close()

    ratios = {
        "C3_RX3_over_C3_R3":  out_preds["C3_RX3"]["collar_ret"]  / out_preds["C3_R3"]["collar_ret"],
        "C3_RZ3_over_C3_R3":  out_preds["C3_RZ3"]["collar_ret"]  / out_preds["C3_R3"]["collar_ret"],
        "C3_RX3_over_C3_RZ3": out_preds["C3_RX3"]["collar_ret"]  / out_preds["C3_RZ3"]["collar_ret"],
        "RX3_RX3_over_R3_R3": out_preds["RX3_RX3"]["collar_ret"] / out_preds["R3_R3"]["collar_ret"],
        "C3_RX3_over_R3_R3":  out_preds["C3_RX3"]["collar_ret"]  / out_preds["R3_R3"]["collar_ret"],
    }
    print("\n=== Pure-L=3 Zero-Parameter Collared Ratios (beta = 6.50) ===")
    for rk, rv in ratios.items():
        print(f"  {rk:24s}: {rv:8.3f}x")

    geom_p = ROOT / "verified_3d_local_L3_geometry.json"
    with open(geom_p) as f:
        d = json.load(f)
    d["collared_L3_predictions_b65"] = out_preds
    d["collared_L3_ratios_b65"] = ratios
    with open(geom_p, "w") as f:
        json.dump(d, f, indent=2)
    print(f"Updated {geom_p}")


if __name__ == "__main__":
    main()
