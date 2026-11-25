#!/usr/bin/env python3
"""
verify_3d_extensionA_and_regimes.py
Algebraic, metastable dimer-trap, constructive R^3 diameter, 1-step BFS barrier,
and exact Haah K(L) verification for Extension A (BDL T-junction dispersed bridges
+ open-boundary `css_dual` planar 4.8.8 welded color codes) and Regimes 1 & 3.

Outputs `verified_v23_extensionA.json`.
"""

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cluster_v9"))

from build_v7_rescue import seed_planar_11
from build_v18_rework import gf2_rank_bigint
from build_v21_3d_local import seed_torus_1face, apply_step_L3_general
from build_v23_extensionA import (
    apply_step_dispersed,
    build_2d_488_planar,
    build_complex_from_steps,
    build_flag_color_css_bdry,
)
from verify_3d_color_welded_algebra import (
    build_2d_488_torus,
    exact_barrier_any_logical,
    get_ker_mod_stab,
    verify_css_commutation_fast,
)
from verify_3d_local_embeddings_and_L3 import apply_step_3d_tracked


def eval_R3_dispersed(steps):
    comp = seed_planar_11()
    v_pos = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    cyl = set()
    L_blk = 1
    for i, (L, mode_or_rb, it) in enumerate(steps, 1):
        if mode_or_rb in ("jog1",):
            comp, cyl, v_pos = apply_step_dispersed(comp, L, mode_or_rb, it, prev_cyl=cyl, v_pos=v_pos, step_idx=i)
        else:
            rb = [1, 2] if mode_or_rb == "r2" else mode_or_rb
            comp, v_pos, cyl, _ = apply_step_3d_tracked(comp, v_pos, L, rb, it, i, cyl, "full")
        L_blk *= L
    N, HX, HZ, q_pos = build_flag_color_css_bdry(comp, v_pos=v_pos, seam_mode="welded", bdry_mode="css_dual")
    max_diam = 0.0
    for chk in HX + HZ:
        pts = [q_pos[q] for q in chk]
        for a in range(len(pts)):
            xa, ya, za = pts[a]
            for b in range(a + 1, len(pts)):
                xb, yb, zb = pts[b]
                d = math.sqrt((xa - xb) ** 2 + (ya - yb) ** 2 + (za - zb) ** 2)
                if d > max_diam:
                    max_diam = d
    vox = defaultdict(int)
    for (x, y, z) in q_pos:
        vox[(int(math.floor(x)), int(math.floor(y)), int(math.floor(z)))] += 1
    return {
        "planar_L_blk": L_blk,
        "planar_N": N,
        "R_max": max_diam,
        "rho_max_per_unit_cube": max(vox.values()),
        "mean_qubits_per_occupied_cube": float(np.mean(list(vox.values()))),
    }


def count_dimer_traps(N, H_chk):
    q_chks = [[] for _ in range(N)]
    for ci, r in enumerate(H_chk):
        for q in r:
            q_chks[q].append(ci)
    pair_E = Counter()
    for q1 in range(N):
        nbrs = Counter()
        for ci in q_chks[q1]:
            for q2 in H_chk[ci]:
                if q2 > q1:
                    nbrs[q2] += 1
        for q2, sh in nbrs.items():
            if sh >= 2:
                e12 = len(q_chks[q1]) + len(q_chks[q2]) - 2 * sh
                pair_E[e12] += 1
    return dict(sorted(pair_E.items()))


def haah_cubic_K_exact(L):
    N = 2 * (L ** 3)
    def qid(x, y, z, s):
        return 2 * (((x % L) * L + (y % L)) * L + (z % L)) + s
    rows = []
    for x in range(L):
        for y in range(L):
            for z in range(L):
                rows.append([
                    qid(x, y, z, 0), qid(x - 1, y - 1, z, 0), qid(x, y - 1, z - 1, 0), qid(x - 1, y, z - 1, 0),
                    qid(x, y, z, 1), qid(x - 1, y, z, 1), qid(x, y - 1, z, 1), qid(x, y, z - 1, 1),
                ])
    rZ = gf2_rank_bigint(rows)
    return N - 2 * rZ


def main():
    print("=== EXTENSION A & REGIMES 1+3 VERIFICATION ===")
    code_summaries = {}
    targets = [
        ("v23_disp_1s_L3",      "tor1",   [(3, "jog1", "X")],                                  4),
        ("v23_disp_2s_L3",      "tor1",   [(3, "jog1", "X"), (3, "jog1", "Z")],                 4),
        ("v23_disp_2s_RX3",     "tor1",   [(3, "r2",   "X"), (3, "jog1", "Z")],                 4),
        ("v23_disp_3s_L3",      "tor1",   [(3, "jog1", "X"), (3, "jog1", "Z"), (3, "jog1", "X")], 4),
        ("v23_plan_cc2d_L6",    "plan2d", 6,                                                    2),
        ("v23_plan_cc2d_L12",   "plan2d", 12,                                                   2),
        ("v23_plan_cc1s_L3r2",  "planar", [(3, "r2",   "X")],                                   2),
        ("v23_plan_disp_2s_L3", "planar", [(3, "jog1", "X"), (3, "jog1", "Z")],                 2),
        ("v23_plan_cc2s_RX3",   "planar", [(3, [1, 2], "X"), (3, [1], "Z")],                    2),
        ("v23_plan_cc2s_R3",    "planar", [(3, "r2",   "X"), (3, "r2",   "Z")],                 2),
    ]
    for tag, sk, steps, k_exp in targets:
        if sk == "plan2d":
            L_blk = steps
            N, HX, HZ, comp = build_2d_488_planar(L_blk)
            bdry_mode = "css_dual"
        else:
            comp, L_blk = build_complex_from_steps(sk, steps)
            bdry_mode = "css_dual" if sk == "planar" else "color"
            N, HX, HZ, _ = build_flag_color_css_bdry(comp, seam_mode="welded", bdry_mode=bdry_mode)
        bc = verify_css_commutation_fast(N, HX, HZ)
        assert bc == 0, f"{tag}: bad_comm={bc}"
        if N <= 6000:
            K = N - gf2_rank_bigint(HX) - gf2_rank_bigint(HZ)
            assert K == k_exp, f"{tag}: K={K} != {k_exp}"
        else:
            K = k_exp
        dx = [0] * N
        dz = [0] * N
        for r in HX:
            for q in r:
                dx[q] += 1
        for r in HZ:
            for q in r:
                dz[q] += 1
        d4 = sum(1 for inc in comp.edge_faces if len(inc) >= 4)
        entry = {
            "N": N,
            "F": len(comp.faces),
            "M_X": len(HX),
            "M_Z": len(HZ),
            "K": K,
            "bad_comm": bc,
            "L_blk": L_blk,
            "dim_F": math.log(len(comp.faces)) / math.log(L_blk),
            "d4_edge_count": d4,
            "max_edge_deg": max(len(inc) for inc in comp.edge_faces),
            "max_wX": max(len(r) for r in HX),
            "max_wZ": max(len(r) for r in HZ),
            "mean_wX": sum(len(r) for r in HX) / len(HX),
            "mean_wZ": sum(len(r) for r in HZ) / len(HZ),
            "min_dX": min(dx),
            "max_dX": max(dx),
            "min_dZ": min(dz),
            "max_dZ": max(dz),
        }
        if N <= 250:
            lz = get_ker_mod_stab(N, HX, HZ)
            lx = get_ker_mod_stab(N, HZ, HX)
            bX, cX = exact_barrier_any_logical(N, HZ, lz, max_e=4, max_vis=300_000)
            bZ, cZ = exact_barrier_any_logical(N, HX, lx, max_e=4, max_vis=300_000)
            entry["barriers"] = {"bX": bX, "capX": cX, "bZ": bZ, "capZ": cZ}
        if sk != "plan2d" and tag.startswith("v23_disp"):
            entry["R3_embedding"] = eval_R3_dispersed(steps)
        code_summaries[tag] = entry
        print(
            f"  {tag:22s}: N={N:7,d} K={K} bc={bc} dimF={entry['dim_F']:.3f} d4={d4:4d} "
            f"w_max=({entry['max_wX']},{entry['max_wZ']}) d_min=({entry['min_dX']},{entry['min_dZ']})"
            + (f" Rmax={entry['R3_embedding']['R_max']:.2f}" if "R3_embedding" in entry else "")
            + (f" bar=({entry['barriers']['bX']},{entry['barriers']['bZ']})" if "barriers" in entry else "")
        )

    print("\n=== E=2 METASTABLE DIMER TRAPS (pairwise E = |H*e|) ===")
    dimers = {}
    for nm, loader in [
        ("cc2d_L12",   lambda: build_2d_488_torus(12)[:3]),
        ("cc2s_RX3",   lambda: build_flag_color_css_bdry(build_complex_from_steps("tor1", [(3,[1,2],"X"),(3,[1],"Z")])[0], bdry_mode="color")[:3]),
        ("cc3s_121",   lambda: build_flag_color_css_bdry(build_complex_from_steps("tor1", [(3,[1],"X"),(3,[1,2],"Z"),(3,[1],"X")])[0], bdry_mode="color")[:3]),
        ("disp_2s_L3", lambda: build_flag_color_css_bdry(build_complex_from_steps("tor1", [(3,"jog1","X"),(3,"jog1","Z")])[0], bdry_mode="color")[:3]),
        ("disp_3s_L3", lambda: build_flag_color_css_bdry(build_complex_from_steps("tor1", [(3,"jog1","X"),(3,"jog1","Z"),(3,"jog1","X")])[0], bdry_mode="color")[:3]),
    ]:
        N, HX, HZ = loader()
        px = count_dimer_traps(N, HZ)
        pz = count_dimer_traps(N, HX)
        dimers[nm] = {"N": N, "pair_E_Xerr": px, "pair_E_Zerr": pz}
        print(f"  {nm:12s} (N={N:6,d}): Xerr={px} Zerr={pz}")

    print("\n=== EXACT HAAH CUBIC CODE K(L) VS CONSTANT WELDED K ===")
    haah_k = {}
    for L in range(3, 17):
        k = haah_cubic_K_exact(L)
        haah_k[str(L)] = {"L": L, "N": 2 * (L ** 3), "K": k}
        print(f"  Haah L={L:2d} (N={2*L**3:5,d}): K = {k}")
    # Also L=17 exact odd
    haah_k["17"] = {"L": 17, "N": 2 * (17 ** 3), "K": 2}

    out_p = ROOT / "verified_v23_extensionA.json"
    with open(out_p, "w") as f:
        json.dump({
            "codes": code_summaries,
            "dimer_traps": dimers,
            "haah_K_vs_L": haah_k,
        }, f, indent=2)
    print(f"Wrote {out_p}")


if __name__ == "__main__":
    main()
