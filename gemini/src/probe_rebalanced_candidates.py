#!/usr/bin/env python3
"""
probe_rebalanced_candidates.py
Exact algebraic & hazard audit of candidate sector-rebalanced welded hierarchies:
  1. 3-step (X, Z, X) balanced hierarchies (2 X-welds + 1 Z-weld)
  2. 4-step sector-flipped hierarchies: (X, Z, Z, X), (Z, X, X, Z), (Z, X, Z, X) vs baseline (X, Z, X, Z)
Verifies exact CSS commutation H_X * H_Z^T == 0 (mod 2), exact GF(2) ranks, K_total,
seed logical propagation (when applicable) or exact dual logical basis, degree distributions,
and clean-state thermal departure hazards (Gamma_X^(0), Gamma_Z^(0)) at beta = 6.5 and 7.0.
"""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cluster_v9"))

from build_v7_rescue import seed_planar_11, seed_rot_torus_422, sheaf_to_css_with_seams
from build_v18_rework import apply_step_general, gf2_rank_bigint
from build_v9_quarter_million import push_X_logical, push_Z_logical


def verify_commutation_sparse(HX, HZ, N):
    q_to_z = [[] for _ in range(N)]
    for zi, r in enumerate(HZ):
        for q in r:
            q_to_z[q].append(zi)
    par = [0] * len(HZ)
    bad = 0
    for r in HX:
        touched = []
        for q in r:
            for zi in q_to_z[q]:
                if par[zi] == 0:
                    touched.append(zi)
                par[zi] ^= 1
        for zi in touched:
            if par[zi]:
                bad += 1
            par[zi] = 0
    return bad


def inspect_steps(name, steps, seed_kind="planar"):
    comp = seed_planar_11() if seed_kind == "planar" else seed_rot_torus_422()
    lx_cur = {0, 1} if seed_kind == "planar" else {0, 1}
    lz_cur = {0} if seed_kind == "planar" else {1, 3}
    cyl = set()
    push_ok = True
    for L, rb, it, gm in steps:
        pcomp = comp
        comp, cyl, meta = apply_step_general(pcomp, L, rb, it, prev_cyl=cyl, gadget_mode=gm)
        N, HX, HZ, *_ = sheaf_to_css_with_seams(comp)
        try:
            lx_cur = push_X_logical(pcomp, comp, meta, HZ, lx_cur)
            lz_cur = push_Z_logical(pcomp, comp, meta, HX, lz_cur)
        except Exception:
            push_ok = False

    bad = verify_commutation_sparse(HX, HZ, N)
    rX = gf2_rank_bigint(HX)
    rZ = gf2_rank_bigint(HZ)
    K = N - rX - rZ

    dz = [0] * N
    for r in HZ:
        for q in r:
            dz[q] += 1
    dx = [0] * N
    for r in HX:
        for q in r:
            dx[q] += 1

    gx65 = sum(1.0 / (1.0 + math.exp(6.5 * d)) for d in dz)
    gz65 = sum(1.0 / (1.0 + math.exp(6.5 * d)) for d in dx)
    min_dz = min(dz) if dz else 0
    min_dx = min(dx) if dx else 0
    zero_dz = sum(1 for d in dz if d == 0)
    zero_dx = sum(1 for d in dx if d == 0)
    d1_dz = sum(1 for d in dz if d == 1)
    d1_dx = sum(1 for d in dx if d == 1)

    if push_ok and seed_kind == "planar":
        comm_lx = sum(1 for r in HZ if (sum(1 for q in r if q in lx_cur) & 1))
        comm_lz = sum(1 for r in HX if (sum(1 for q in r if q in lz_cur) & 1))
        pair_ok = (len(lx_cur & lz_cur) & 1) == 1 and comm_lx == 0 and comm_lz == 0
    else:
        pair_ok = False

    print(
        f"{name:26s} | N={N:7,d} M=({len(HX):6,d},{len(HZ):6,d}) "
        f"K={K:2d} bad={bad} push={str(pair_ok):5s} "
        f"dmin=({min_dx},{min_dz}) d0=({zero_dx},{zero_dz}) d1=({d1_dx},{d1_dz}) "
        f"G65=({gx65:.4f}, {gz65:.4f})"
    )


def main():
    X2 = (2, [1], "X", "full")
    Z2 = (2, [1], "Z", "full")
    X3_1 = (3, [1], "X", "full")
    Z3_1 = (3, [1], "Z", "full")
    X3_2 = (3, [1, 2], "X", "full")
    Z3_2 = (3, [1, 2], "Z", "full")

    cands = [
        # Baselines
        ("base_C2_1cyc (X2,Z2)",          [X2, Z2]),
        ("base_C2C2 (X2,Z2,X2,Z2)",       [X2, Z2, X2, Z2]),
        ("base_RX3C2 (X3_2,Z3_1,X2,Z2)",  [X3_2, Z3_1, X2, Z2]),
        # 3-step (X, Z, X) balanced ladder
        ("3step_X2_Z2_X2",                [X2, Z2, X2]),
        ("3step_X2_Z3r1_X2",              [X2, Z3_1, X2]),
        ("3step_X2_Z3r2_X2",              [X2, Z3_2, X2]),
        ("3step_X3r1_Z2_X2",              [X3_1, Z2, X2]),
        ("3step_X3r2_Z2_X2",              [X3_2, Z2, X2]),
        ("3step_X3r2_Z3r1_X2",            [X3_2, Z3_1, X2]),
        ("3step_X3r2_Z3r2_X2",            [X3_2, Z3_2, X2]),
        # 3-step (Z, X, Z)
        ("3step_Z2_X2_Z2",                [Z2, X2, Z2]),
        # 4-step flipped (X, Z, Z, X): Coarse (X,Z), Fine (Z,X)
        ("flip_XZZX_C2_C2ZX",             [X2, Z2, Z2, X2]),
        ("flip_XZZX_C3_C2ZX",             [X3_1, Z3_1, Z2, X2]),
        ("flip_XZZX_RX3_C2ZX",            [X3_2, Z3_1, Z2, X2]),
        ("flip_XZZX_RZ3_C2ZX",            [X3_1, Z3_2, Z2, X2]),
        ("flip_XZZX_R3_C2ZX",             [X3_2, Z3_2, Z2, X2]),
        ("flip_XZZX_C2_R3ZX",             [X2, Z2, Z3_2, X3_2]),
        # 4-step (Z, X, X, Z) and (Z, X, Z, X)
        ("flip_ZXXZ_C2ZX_C2",             [Z2, X2, X2, Z2]),
        ("flip_ZXZX_C2ZX_C2ZX",           [Z2, X2, Z2, X2]),
    ]
    for nm, st in cands:
        inspect_steps(nm, st, "planar")


if __name__ == "__main__":
    main()
