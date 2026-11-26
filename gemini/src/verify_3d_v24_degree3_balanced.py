#!/usr/bin/env python3
"""
verify_3d_v24_degree3_balanced.py
Algebraic, packing-dimension, E=2 dimer-trap, constructive R^3 diameter,
and 1-step BFS barrier verification for Campaign V24 (degree-3 X/Z-balanced
3D-local welded 4.8.8 color codes).

Outputs `verified_v24_degree3_balanced.json`.
"""

import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cluster_v9"))

from build_v7_rescue import GeneralCubical2D, seed_planar_11
from build_v18_rework import gf2_rank_bigint
from build_v21_3d_local import seed_torus_1face, apply_step_L3_general
from build_v22_color_welded import build_flag_color_code
from build_v23_extensionA import (
    apply_step_dispersed,
    build_complex_from_steps,
    build_flag_color_css_bdry,
)
from verify_3d_color_welded_algebra import (
    exact_barrier_any_logical,
    get_ker_mod_stab,
    verify_css_commutation_fast,
)
from verify_3d_extensionA_and_regimes import count_dimer_traps, eval_R3_dispersed


def compute_packing_dimensions():
    """Exact asymptotic packing dimension d_pack(L, r) = log_L(rho(T_{L,r}))."""
    out = {}
    for L in (3, 4, 5, 6):
        for r in range(1, L):
            mat = np.array([[2 * L * L, 2 * L], [2 * r * L, r]], dtype=float)
            rho = float(max(abs(np.linalg.eigvals(mat))))
            dpack = float(math.log(rho) / math.log(L))
            F1 = 2 * L * L + 2 * r * L
            assert dpack < 3.0, f"(L={L}, r={r}) d_pack={dpack} >= 3!"
            out[f"L{L}_r{r}"] = {
                "L": L,
                "r": r,
                "F1": F1,
                "N1": 8 * F1,
                "rho": rho,
                "d_pack": dpack,
            }
    return out


def _apply_step_copy_split(comp, L, rib_mode, pillar_col="blue"):
    prev_cyl = set()
    if rib_mode == "jog1":
        res, new_cyl, _ = apply_step_dispersed(comp, L, "jog1", "X", prev_cyl=prev_cyl)
        resZ, _, _ = apply_step_dispersed(comp, L, "jog1", "Z", prev_cyl=prev_cyl)
    else:
        rb = [1, 2] if rib_mode == "r2" else [1]
        res, new_cyl, _ = apply_step_L3_general(comp, L, rb, "X", prev_cyl=prev_cyl)
        resZ, _, _ = apply_step_L3_general(comp, L, rb, "Z", prev_cyl=prev_cyl)
    n_cyl = len(new_cyl)
    n_xe = (len(res.edges) - n_cyl) // 2
    bridge_e0 = {f[4] for f in res.faces if f[4] < n_xe and f[5] == f[4] + n_xe}
    fresh_cyl = [ce for ce in new_cyl if res.edge_colors[ce] != resZ.edge_colors[ce]]
    for e0 in bridge_e0:
        res.edge_colors[e0] = "blue"
        res.edge_colors[e0 + n_xe] = "red"
    for ce in fresh_cyl:
        res.edge_colors[ce] = pillar_col
    return res


def _build_1s_rib_split(L, blue_ribs, red_ribs):
    xp_v = L * L
    def vid(x, y):
        return (x % L) * L + (y % L)
    xp_edges, xp_rib_col = [], []
    he = [[None] * L for _ in range(L)]
    ve = [[None] * L for _ in range(L)]
    for x in range(L):
        for y in range(L):
            he[x][y] = len(xp_edges)
            xp_edges.append((vid(x, y), vid(x + 1, y)))
            xp_rib_col.append("blue" if y in blue_ribs else ("red" if y in red_ribs else "none"))
            ve[x][y] = len(xp_edges)
            xp_edges.append((vid(x, y), vid(x, y + 1)))
            xp_rib_col.append("blue" if x in blue_ribs else ("red" if x in red_ribs else "none"))
    xp_faces = [
        (vid(x, y), vid(x + 1, y), vid(x + 1, y + 1), vid(x, y + 1),
         he[x][y], he[x][(y + 1) % L], ve[x][y], ve[(x + 1) % L][y])
        for x in range(L) for y in range(L)
    ]
    n_xe = len(xp_edges)
    out_e, out_c, out_f = [], [], []
    def vc(v, c):
        return v + c * xp_v
    for copy in (0, 1):
        for eid, (u, w) in enumerate(xp_edges):
            out_e.append((vc(u, copy), vc(w, copy)))
            out_c.append(xp_rib_col[eid])
    for copy in (0, 1):
        off = copy * n_xe
        for v00, v10, v11, v01, eb, et, el, er in xp_faces:
            out_f.append((vc(v00, copy), vc(v10, copy), vc(v11, copy), vc(v01, copy),
                          eb + off, et + off, el + off, er + off))
    rib_vs = {u for eid, col in enumerate(xp_rib_col) if col != "none" for u in xp_edges[eid]}
    cyl = {}
    for v in sorted(rib_vs):
        x, y = v // L, v % L
        ce = len(out_e)
        out_e.append((vc(v, 0), vc(v, 1)))
        cx = "blue" if x in blue_ribs else ("red" if x in red_ribs else "none")
        cy = "blue" if y in blue_ribs else ("red" if y in red_ribs else "none")
        if cx == cy:
            out_c.append(cx)
        elif cx == "none":
            out_c.append(cy)
        elif cy == "none":
            out_c.append(cx)
        else:
            out_c.append("blue")
        cyl[v] = ce
    for eid, col in enumerate(xp_rib_col):
        if col != "none":
            u, w = xp_edges[eid]
            out_f.append((vc(u, 0), vc(w, 0), vc(w, 1), vc(u, 1), eid, eid + n_xe, cyl[u], cyl[w]))
    res = GeneralCubical2D(2 * xp_v, out_e, out_c, out_f)
    for eid in range(len(res.edges)):
        if len(res.edge_faces[eid]) == 2:
            res.edge_colors[eid] = "none"
    return res


def _build_1s_3sheet(L, r1, r2):
    xp_v = L * L
    def vid(x, y):
        return (x % L) * L + (y % L)
    xp_edges, is_r1, is_r2 = [], [], []
    he = [[None] * L for _ in range(L)]
    ve = [[None] * L for _ in range(L)]
    for x in range(L):
        for y in range(L):
            he[x][y] = len(xp_edges)
            xp_edges.append((vid(x, y), vid(x + 1, y)))
            is_r1.append(y in r1)
            is_r2.append(y in r2)
            ve[x][y] = len(xp_edges)
            xp_edges.append((vid(x, y), vid(x, y + 1)))
            is_r1.append(x in r1)
            is_r2.append(x in r2)
    xp_faces = [
        (vid(x, y), vid(x + 1, y), vid(x + 1, y + 1), vid(x, y + 1),
         he[x][y], he[x][(y + 1) % L], ve[x][y], ve[(x + 1) % L][y])
        for x in range(L) for y in range(L)
    ]
    n_xe = len(xp_edges)
    out_e, out_c, out_f = [], [], []
    def vc(v, c):
        return v + c * xp_v
    for c in (0, 1, 2):
        for eid, (u, w) in enumerate(xp_edges):
            out_e.append((vc(u, c), vc(w, c)))
            if c == 0:
                out_c.append("blue" if is_r1[eid] else "none")
            elif c == 1:
                if is_r1[eid]:
                    out_c.append("blue")
                elif is_r2[eid]:
                    out_c.append("red")
                else:
                    out_c.append("none")
            else:
                out_c.append("red" if is_r2[eid] else "none")
    for c in (0, 1, 2):
        off = c * n_xe
        for v00, v10, v11, v01, eb, et, el, er in xp_faces:
            out_f.append((vc(v00, c), vc(v10, c), vc(v11, c), vc(v01, c),
                          eb + off, et + off, el + off, er + off))
    for (cA, cB, mask, col) in [(0, 1, is_r1, "blue"), (1, 2, is_r2, "red")]:
        rvs = {u for eid, fl in enumerate(mask) if fl for u in xp_edges[eid]}
        cyl = {}
        for v in sorted(rvs):
            ce = len(out_e)
            out_e.append((vc(v, cA), vc(v, cB)))
            out_c.append(col)
            cyl[v] = ce
        for eid, fl in enumerate(mask):
            if fl:
                u, w = xp_edges[eid]
                out_f.append((vc(u, cA), vc(w, cA), vc(w, cB), vc(u, cB),
                              eid + cA * n_xe, eid + cB * n_xe, cyl[u], cyl[w]))
    res = GeneralCubical2D(3 * xp_v, out_e, out_c, out_f)
    for eid in range(len(res.edges)):
        if len(res.edge_faces[eid]) == 2:
            res.edge_colors[eid] = "none"
    return res


def check_single_step_mixing_obstructions():
    """Verify algebraic obstructions when mixing blue and red seams at a single scale."""
    configs = [
        ("1s_r1_copySplit_pB",   _apply_step_copy_split(seed_torus_1face(), 3, "r1",   "blue")),
        ("1s_r2_copySplit_pB",   _apply_step_copy_split(seed_torus_1face(), 3, "r2",   "blue")),
        ("1s_jog1_copySplit_pB", _apply_step_copy_split(seed_torus_1face(), 3, "jog1", "blue")),
        ("ribSplit_L3_B1_R2_pB", _build_1s_rib_split(3, [1], [2])),
        ("3sheet_L3_B1_R2",      _build_1s_3sheet(3, [1], [2])),
        ("3sheet_L3_B1_R1",      _build_1s_3sheet(3, [1], [1])),
    ]
    out = {}
    for name, comp in configs:
        N, HX, HZ = build_flag_color_code(comp, seam_mode="welded")
        bc = verify_css_commutation_fast(N, HX, HZ)
        assert bc == 0, f"{name}: bad_comm={bc}"
        K = N - gf2_rank_bigint(HX) - gf2_rank_bigint(HZ)
        dx = [0] * N
        dz = [0] * N
        for r in HX:
            for q in r:
                dx[q] += 1
        for r in HZ:
            for q in r:
                dz[q] += 1
        lz = get_ker_mod_stab(N, HX, HZ)
        lx = get_ker_mod_stab(N, HZ, HX)
        bX, cX = exact_barrier_any_logical(N, HZ, lz, max_e=5, max_vis=200_000)
        bZ, cZ = exact_barrier_any_logical(N, HX, lx, max_e=5, max_vis=200_000)
        out[name] = {
            "N": N,
            "K": K,
            "bad_comm": bc,
            "min_dX": min(dx),
            "min_dZ": min(dz),
            "bX": bX,
            "capX": cX,
            "bZ": bZ,
            "capZ": cZ,
        }
        print(
            f"  [OBS] {name:24s}: N={N:4d} K={K:2d} bc={bc} dmin=({min(dx)},{min(dz)}) "
            f"bar=({bX}{'+' if cX else ''},{bZ}{'+' if cZ else ''})"
        )
    return out


def verify_v24_suite():
    targets = [
        ("v24_1s_L4r3",            "tor1",   [(4, [1, 2, 3], "X")],                                   4),
        ("v24_1s_L5r4",            "tor1",   [(5, [1, 2, 3, 4], "X")],                                4),
        ("v24_2s_X4r3_Z3r1",       "tor1",   [(4, [1, 2, 3], "X"), (3, [1], "Z")],                    4),
        ("v24_2s_X4r3_Z3jog1",     "tor1",   [(4, [1, 2, 3], "X"), (3, "jog1", "Z")],                 4),
        ("v24_2s_X4r3_Z3r2",       "tor1",   [(4, [1, 2, 3], "X"), (3, [1, 2], "Z")],                 4),
        ("v24_2s_X5r4_Z3r1",       "tor1",   [(5, [1, 2, 3, 4], "X"), (3, [1], "Z")],                 4),
        ("v24_2s_X5r4_Z3r2",       "tor1",   [(5, [1, 2, 3, 4], "X"), (3, [1, 2], "Z")],              4),
        ("v24_3s_X1_X2_Z1",        "tor1",   [(3, [1], "X"), (3, [1, 2], "X"), (3, [1], "Z")],        4),
        ("v24_3s_X2_X1_Z1",        "tor1",   [(3, [1, 2], "X"), (3, [1], "X"), (3, [1], "Z")],        4),
        ("v24_3s_X2_X2_Z1",        "tor1",   [(3, [1, 2], "X"), (3, [1, 2], "X"), (3, [1], "Z")],     4),
        ("v24_plan_2s_X4r3_Z3r1",  "planar", [(4, [1, 2, 3], "X"), (3, [1], "Z")],                    2),
        ("v24_plan_2s_X5r4_Z3r1",  "planar", [(5, [1, 2, 3, 4], "X"), (3, [1], "Z")],                 2),
    ]

    summaries = {}
    for tag, sk, steps, k_exp in targets:
        comp, L_blk = build_complex_from_steps(sk, steps)
        bdry_mode = "css_dual" if sk == "planar" else "color"
        N, HX, HZ, _ = build_flag_color_css_bdry(comp, seam_mode="welded", bdry_mode=bdry_mode)
        bc = verify_css_commutation_fast(N, HX, HZ)
        assert bc == 0, f"{tag}: bad_comm={bc}"

        if N <= 8000:
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
        if sk == "tor1":
            assert min(dx) == 3 and min(dz) == 3, f"{tag}: min_deg=({min(dx)},{min(dz)})"

        dim_F = math.log(len(comp.faces)) / math.log(L_blk)
        assert dim_F < 3.0, f"{tag}: dim_F={dim_F} >= 3"
        d4 = sum(1 for inc in comp.edge_faces if len(inc) >= 4)

        entry = {
            "seed_kind": sk,
            "N": N,
            "F": len(comp.faces),
            "M_X": len(HX),
            "M_Z": len(HZ),
            "K": K,
            "K_exact_checked_locally": (N <= 8000),
            "bad_comm": bc,
            "L_blk": L_blk,
            "dim_F": dim_F,
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

        if N <= 16000:
            entry["pair_E_Xerr"] = count_dimer_traps(N, HZ)
            entry["pair_E_Zerr"] = count_dimer_traps(N, HX)

        if tag in ("v24_1s_L4r3", "v24_1s_L5r4"):
            lz = get_ker_mod_stab(N, HX, HZ)
            lx = get_ker_mod_stab(N, HZ, HX)
            bX, cX = exact_barrier_any_logical(N, HZ, lz, max_e=6, max_vis=250_000)
            bZ, cZ = exact_barrier_any_logical(N, HX, lx, max_e=6, max_vis=250_000)
            entry["barriers"] = {"bX": bX, "capX": cX, "bZ": bZ, "capZ": cZ}

        if len(steps) <= 2:
            emb = eval_R3_dispersed(steps)
            assert emb["R_max"] <= 3.25, f"{tag}: R_max={emb['R_max']:.4f} > 3.25"
            entry["R3_embedding"] = emb

        summaries[tag] = entry
        extra = ""
        if "R3_embedding" in entry:
            extra += f" Rmax={entry['R3_embedding']['R_max']:.2f}"
        if "barriers" in entry:
            b = entry["barriers"]
            extra += f" bar=({b['bX']}{'+' if b['capX'] else ''},{b['bZ']}{'+' if b['capZ'] else ''})"
        if "pair_E_Xerr" in entry:
            extra += f" E2=({entry['pair_E_Xerr'].get(2,0)},{entry['pair_E_Zerr'].get(2,0)})"
        print(
            f"  [V24] {tag:24s}: N={N:6,d} K={K} bc={bc} L={L_blk:2d} dimF={dim_F:.3f} "
            f"dmin=({min(dx)},{min(dz)}) wmax=({entry['max_wX']:2d},{entry['max_wZ']:2d}){extra}"
        )

    return summaries


def main():
    print("=== 1. PACKING DIMENSION d_pack(L, r) ===")
    dpack = compute_packing_dimensions()
    for k, v in dpack.items():
        print(f"  {k:8s}: F1={v['F1']:3d} N1={v['N1']:4d} rho={v['rho']:7.3f} d_pack={v['d_pack']:.4f}")

    print("\n=== 2. SINGLE-STEP SEAM-COLOR MIXING OBSTRUCTIONS ===")
    obs = check_single_step_mixing_obstructions()

    print("\n=== 3. CAMPAIGN V24 SUITE VERIFICATION ===")
    v24 = verify_v24_suite()

    out_p = ROOT / "verified_v24_degree3_balanced.json"
    with open(out_p, "w") as f:
        json.dump({
            "packing_dimensions": dpack,
            "single_step_mixing_obstructions": obs,
            "v24_codes": v24,
        }, f, indent=2)
    print(f"\nWrote {out_p}")


if __name__ == "__main__":
    main()
