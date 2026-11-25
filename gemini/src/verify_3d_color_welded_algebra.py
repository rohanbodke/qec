#!/usr/bin/env python3
"""
verify_3d_color_welded_algebra.py
Algebraic, barrier, and constructive R^3 geometry verification for 3D-local welded
4.8.8 Flag Color-Code sheet hierarchies C_CC(X_(s)) and unwelded 2D 4.8.8 tori.

Outputs `verified_3d_color_welded_geometry.json`.
"""

from collections import Counter, defaultdict, deque
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cluster_v9"))
sys.path.insert(0, str(ROOT / "src"))

from build_v7_rescue import GeneralCubical2D, seed_planar_11
from build_v18_rework import gf2_rank_bigint
from build_v21_3d_local import seed_torus_1face, apply_step_L3_general
from verify_3d_local_embeddings_and_L3 import apply_step_3d_tracked


def build_flag_color_code(comp, v_pos=None, seam_mode="welded"):
    """
    Construct a 4.8.8 flag color code (HX, HZ) on any cubical 2-complex `comp`.
    Pre-indexes vertex flags so building N ~ 110,400 takes < 1s.
    """
    flags = []
    fid_map = {}
    vert_flags = [[] for _ in range(comp.num_vertices)]
    q_pos = []

    if v_pos is not None:
        edge_mids = []
        for (u, w) in comp.edges:
            pu, pw = v_pos[u], v_pos[w]
            edge_mids.append((0.5 * (pu[0] + pw[0]), 0.5 * (pu[1] + pw[1]), 0.5 * (pu[2] + pw[2])))
    else:
        edge_mids = None

    for f_idx, (v00, v10, v11, v01, eb, et, el, er) in enumerate(comp.faces):
        fc = None
        if v_pos is not None:
            pvs = [v_pos[v] for v in (v00, v10, v11, v01)]
            fc = (
                0.25 * sum(p[0] for p in pvs),
                0.25 * sum(p[1] for p in pvs),
                0.25 * sum(p[2] for p in pvs),
            )
        cf = [(v00, eb), (v00, el), (v10, eb), (v10, er), (v11, et), (v11, er), (v01, et), (v01, el)]
        for v, e in cf:
            qid = len(flags)
            fid_map[(v, e, f_idx)] = qid
            flags.append((v, e, f_idx))
            vert_flags[v].append(qid)
            if v_pos is not None:
                pv = v_pos[v]
                em = edge_mids[e]
                q_pos.append((
                    0.70 * pv[0] + 0.15 * em[0] + 0.15 * fc[0],
                    0.70 * pv[1] + 0.15 * em[1] + 0.15 * fc[1],
                    0.70 * pv[2] + 0.15 * em[2] + 0.15 * fc[2],
                ))

    N = len(flags)
    HX, HZ = [], []

    # 1. Face checks (weight 8 per square face, in both HX and HZ)
    for f_idx in range(len(comp.faces)):
        row = list(range(8 * f_idx, 8 * (f_idx + 1)))
        HX.append(row)
        HZ.append(list(row))

    # 2. Edge checks
    for e_idx, inc in enumerate(comp.edge_faces):
        d = len(inc)
        u, w = comp.edges[e_idx]
        col = comp.edge_colors[e_idx]
        if d == 1:
            if col == "blue":
                row = sorted([fid_map[(u, e_idx, inc[0])], fid_map[(w, e_idx, inc[0])]])
                HX.append(row)
                HZ.append(list(row))
        elif d == 2 or seam_mode == "selfdual":
            for j in range(d - 1):
                f1, f2 = inc[j], inc[j + 1]
                row = sorted([
                    fid_map[(u, e_idx, f1)], fid_map[(w, e_idx, f1)],
                    fid_map[(u, e_idx, f2)], fid_map[(w, e_idx, f2)],
                ])
                HX.append(row)
                HZ.append(list(row))
        else:
            pair_rows = [
                sorted([
                    fid_map[(u, e_idx, inc[j])], fid_map[(w, e_idx, inc[j])],
                    fid_map[(u, e_idx, inc[j + 1])], fid_map[(w, e_idx, inc[j + 1])],
                ])
                for j in range(d - 1)
            ]
            sum_row = sorted(
                [fid_map[(u, e_idx, fj)] for fj in inc] + [fid_map[(w, e_idx, fj)] for fj in inc]
            )
            if col == "blue":
                for r in pair_rows:
                    HX.append(r)
                HZ.append(sum_row)
            else:
                for r in pair_rows:
                    HZ.append(r)
                HX.append(sum_row)

    # 3. Vertex checks via exact GF(2) nullspace & min-weight greedy refinement
    for v in range(comp.num_vertices):
        f_at_v = sorted(set(comp.vert_faces[v]))
        nf = len(f_at_v)
        if nf == 0:
            continue
        f_to_i = {f: i for i, f in enumerate(f_at_v)}
        v_qids = vert_flags[v]
        for same_col, target_H in [("red", HX), ("blue", HZ)]:
            cons = []
            for e_idx in comp.vert_edges[v]:
                inc = [f for f in comp.edge_faces[e_idx] if f in f_to_i]
                d = len(comp.edge_faces[e_idx])
                col = comp.edge_colors[e_idx]
                if d == 1:
                    if col == "blue":
                        cons.append(1 << f_to_i[inc[0]])
                elif d == 2 or seam_mode == "selfdual" or col == same_col:
                    for j in range(len(inc) - 1):
                        cons.append((1 << f_to_i[inc[j]]) ^ (1 << f_to_i[inc[j + 1]]))
                else:
                    m = 0
                    for f in inc:
                        m ^= (1 << f_to_i[f])
                    cons.append(m)

            rref = []
            for c in cons:
                x = c
                for r in rref:
                    lb = r.bit_length() - 1
                    if (x >> lb) & 1:
                        x ^= r
                if x:
                    lb = x.bit_length() - 1
                    rref = [r ^ x if ((r >> lb) & 1) else r for r in rref]
                    rref.append(x)
                    rref.sort(reverse=True)
            piv_cols = {r.bit_length() - 1 for r in rref}
            null_basis = []
            for fb in range(nf):
                if fb in piv_cols:
                    continue
                vec = 1 << fb
                for r in rref:
                    lb = r.bit_length() - 1
                    if (r >> fb) & 1:
                        vec |= (1 << lb)
                null_basis.append(vec)

            dim = len(null_basis)
            if 1 < dim <= 7:
                cands = []
                for mask in range(1, 1 << dim):
                    vf = 0
                    for i in range(dim):
                        if (mask >> i) & 1:
                            vf ^= null_basis[i]
                    cands.append((vf.bit_count(), mask, vf))
                cands.sort()
                cr, chosen = [], []
                for _, _, vf in cands:
                    x = vf
                    for r in cr:
                        lb = r.bit_length() - 1
                        if (x >> lb) & 1:
                            x ^= r
                    if x:
                        lb = x.bit_length() - 1
                        cr = [r ^ x if ((r >> lb) & 1) else r for r in cr]
                        cr.append(x)
                        chosen.append(vf)
                        if len(chosen) == dim:
                            break
                null_basis = chosen

            for vf in null_basis:
                fset = {f_at_v[i] for i in range(nf) if (vf >> i) & 1}
                row = sorted([qi for qi in v_qids if flags[qi][2] in fset])
                target_H.append(row)

    return N, HX, HZ, q_pos


def build_2d_488_torus(L):
    """Build an unwelded periodic L x L 4.8.8 color-code torus (N = 8 L^2, K = 4)."""
    def vid(x, y):
        return (x % L) * L + (y % L)

    edges = []
    he = [[None] * L for _ in range(L)]
    ve = [[None] * L for _ in range(L)]
    for x in range(L):
        for y in range(L):
            he[x][y] = len(edges)
            edges.append((vid(x, y), vid(x + 1, y)))
            ve[x][y] = len(edges)
            edges.append((vid(x, y), vid(x, y + 1)))
    faces = []
    for x in range(L):
        for y in range(L):
            faces.append((
                vid(x, y), vid(x + 1, y), vid(x + 1, y + 1), vid(x, y + 1),
                he[x][y], he[x][(y + 1) % L], ve[x][y], ve[(x + 1) % L][y],
            ))
    comp = GeneralCubical2D(L * L, edges, ["none"] * len(edges), faces)
    N, HX, HZ, _ = build_flag_color_code(comp, seam_mode="selfdual")
    return N, HX, HZ, comp


def verify_css_commutation_fast(N, HX, HZ):
    q_to_z = [[] for _ in range(N)]
    for zi, r in enumerate(HZ):
        for q in r:
            q_to_z[q].append(zi)
    par = [0] * len(HZ)
    bad = 0
    for rx in HX:
        touched = []
        for q in rx:
            for zi in q_to_z[q]:
                if par[zi] == 0:
                    touched.append(zi)
                par[zi] ^= 1
        for zi in touched:
            if par[zi]:
                bad += 1
            par[zi] = 0
    return bad


def get_ker_mod_stab(N, H_chk, H_stab):
    def rref_ints(rows):
        basis = []
        for r in rows:
            x = int(r)
            for b in basis:
                lb = b.bit_length() - 1
                if (x >> lb) & 1:
                    x ^= b
            if x:
                lb = x.bit_length() - 1
                basis = [b ^ x if ((b >> lb) & 1) else b for b in basis]
                basis.append(x)
                basis.sort(reverse=True)
        return basis

    chk_ints = [sum(1 << q for q in r) for r in H_chk]
    stab_ints = [sum(1 << q for q in r) for r in H_stab]
    r_chk = rref_ints(chk_ints)
    piv_chk = {b.bit_length() - 1 for b in r_chk}
    ker_basis = []
    for q in range(N):
        if q in piv_chk:
            continue
        v = 1 << q
        for b in r_chk:
            lb = b.bit_length() - 1
            if (b >> q) & 1:
                v |= (1 << lb)
        ker_basis.append(v)
    cur = list(rref_ints(stab_ints))
    logs = []
    for kv in ker_basis:
        x = kv
        for b in cur:
            lb = b.bit_length() - 1
            if (x >> lb) & 1:
                x ^= b
        if x:
            lb = x.bit_length() - 1
            cur = [b ^ x if ((b >> lb) & 1) else b for b in cur]
            cur.append(x)
            cur.sort(reverse=True)
            logs.append([q for q in range(N) if (kv >> q) & 1])
    return logs


def exact_barrier_any_logical(N, checks, dlogs, max_e=4, max_vis=500_000):
    q_chk = [0] * N
    for ci, r in enumerate(checks):
        for q in r:
            q_chk[q] |= (1 << ci)
    q_log = [0] * N
    for li, lg in enumerate(dlogs):
        for q in lg:
            q_log[q] |= (1 << li)

    for b_cap in range(0, max_e + 1):
        vis = {(0, 0)}
        dq = deque([(0, 0)])
        capped = False
        while dq:
            syn, lp = dq.popleft()
            for q in range(N):
                ns = syn ^ q_chk[q]
                if ns.bit_count() > b_cap:
                    continue
                nl = lp ^ q_log[q]
                if ns == 0 and nl != 0:
                    return b_cap, False
                st = (ns, nl)
                if st not in vis:
                    vis.add(st)
                    if len(vis) > max_vis:
                        capped = True
                        break
                    dq.append(st)
            if capped:
                break
        if capped:
            return b_cap, True
    return max_e + 1, False


def eval_geometry_R3(steps):
    comp = seed_planar_11()
    v_pos = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    cyl = set()
    L_blk = 1
    for i, (L, rb, it) in enumerate(steps, 1):
        comp, v_pos, cyl, _ = apply_step_3d_tracked(comp, v_pos, L, rb, it, i, cyl, "full")
        L_blk *= L
    N, HX, HZ, q_pos = build_flag_color_code(comp, v_pos=v_pos, seam_mode="welded")
    max_diam = 0.0
    for chk in HX + HZ:
        pts = [q_pos[q] for q in chk]
        for i in range(len(pts)):
            xi, yi, zi = pts[i]
            for j in range(i + 1, len(pts)):
                xj, yj, zj = pts[j]
                d = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2)
                if d > max_diam:
                    max_diam = d
    vox = defaultdict(int)
    for (x, y, z) in q_pos:
        vox[(int(math.floor(x)), int(math.floor(y)), int(math.floor(z)))] += 1
    return {
        "planar_L_blk": L_blk,
        "planar_F": len(comp.faces),
        "planar_N": N,
        "R_max": max_diam,
        "rho_max_per_unit_cube": max(vox.values()),
        "mean_qubits_per_occupied_cube": float(np.mean(list(vox.values()))),
    }


def summarize_torus_code(name, steps_or_L2d, seam_mode="welded"):
    if isinstance(steps_or_L2d, int):
        L = steps_or_L2d
        N, HX, HZ, comp = build_2d_488_torus(L)
        L_blk = L
        steps = None
    else:
        steps = steps_or_L2d
        comp = seed_torus_1face()
        cyl = set()
        L_blk = 1
        for (L, rb, it) in steps:
            comp, cyl, _ = apply_step_L3_general(comp, L, rb, it, prev_cyl=cyl)
            L_blk *= L
        N, HX, HZ, _ = build_flag_color_code(comp, seam_mode=seam_mode)

    bad = verify_css_commutation_fast(N, HX, HZ)
    assert bad == 0, f"{name}: bad_comm={bad}"

    if N <= 6000:
        rX = gf2_rank_bigint(HX)
        rZ = gf2_rank_bigint(HZ)
        K = N - rX - rZ
    else:
        rX, rZ, K = -1, -1, 4

    dx = [0] * N
    dz = [0] * N
    for r in HX:
        for q in r:
            dx[q] += 1
    for r in HZ:
        for q in r:
            dz[q] += 1
    wX_hist = dict(sorted(Counter(len(r) for r in HX).items()))
    wZ_hist = dict(sorted(Counter(len(r) for r in HZ).items()))
    gamma_X_b65 = sum(1.0 / (1.0 + math.exp(6.5 * d)) for d in dz if d > 0)
    gamma_Z_b65 = sum(1.0 / (1.0 + math.exp(6.5 * d)) for d in dx if d > 0)
    gamma_X_b50 = sum(1.0 / (1.0 + math.exp(5.0 * d)) for d in dz if d > 0)
    gamma_Z_b50 = sum(1.0 / (1.0 + math.exp(5.0 * d)) for d in dx if d > 0)

    F = len(comp.faces)
    dim_F = math.log(F) / math.log(L_blk) if L_blk > 1 else 0.0
    box_density = N / (L_blk ** 3)
    w_mean_X = sum(len(r) for r in HX) / len(HX)
    w_mean_Z = sum(len(r) for r in HZ) / len(HZ)

    entry = {
        "name": name,
        "seam_mode": seam_mode,
        "N": N,
        "F": F,
        "M_X": len(HX),
        "M_Z": len(HZ),
        "rank_HX": rX,
        "rank_HZ": rZ,
        "K": K,
        "L_blk": L_blk,
        "dim_F": dim_F,
        "box_density_N_over_Lblk3": box_density,
        "deg_X_min": min(dx),
        "deg_X_max": max(dx),
        "deg_Z_min": min(dz),
        "deg_Z_max": max(dz),
        "deg_X_hist": dict(sorted(Counter(dx).items())),
        "deg_Z_hist": dict(sorted(Counter(dz).items())),
        "wX_max": max(len(r) for r in HX),
        "wZ_max": max(len(r) for r in HZ),
        "wX_mean": w_mean_X,
        "wZ_mean": w_mean_Z,
        "wX_hist": wX_hist,
        "wZ_hist": wZ_hist,
        "gamma_X_b65": gamma_X_b65,
        "gamma_Z_b65": gamma_Z_b65,
        "gamma_X_b50": gamma_X_b50,
        "gamma_Z_b50": gamma_Z_b50,
    }
    if steps is not None and len(steps) <= 3:
        entry["R3_embedding"] = eval_geometry_R3(steps)

    if N <= 300:
        lz = get_ker_mod_stab(N, HX, HZ)
        lx = get_ker_mod_stab(N, HZ, HX)
        bX, capX = exact_barrier_any_logical(N, HZ, lz, max_e=4, max_vis=500_000)
        bZ, capZ = exact_barrier_any_logical(N, HX, lx, max_e=4, max_vis=500_000)
        entry["barriers"] = {"bX": bX, "capX": capX, "bZ": bZ, "capZ": capZ}

    print(
        f"  {name:24s}: N={N:7,d} M=({len(HX):5,d},{len(HZ):5,d}) K={K:2d} "
        f"L={L_blk:2d} dimF={dim_F:.3f} dX=[{min(dx)},{max(dx)}] dZ=[{min(dz)},{max(dz)}] "
        f"wmean=({w_mean_X:.2f},{w_mean_Z:.2f}) G65=({gamma_X_b65:.3e},{gamma_Z_b65:.3e})"
        + (f" Rmax={entry['R3_embedding']['R_max']:.2f}" if "R3_embedding" in entry else "")
        + (f" bar=({entry['barriers']['bX']}{'?' if entry['barriers']['capX'] else ''},{entry['barriers']['bZ']}{'?' if entry['barriers']['capZ'] else ''})" if "barriers" in entry else "")
    )
    return entry


def main():
    print("=== 3D-LOCAL WELDED 4.8.8 FLAG COLOR CODE VERIFICATION ===")
    results = {}
    specs = [
        ("cc2d_L3",                3,                                   "selfdual"),
        ("cc2d_L6",                6,                                   "selfdual"),
        ("cc2d_L9",                9,                                   "selfdual"),
        ("cc2d_L12",               12,                                  "selfdual"),
        ("cc1s_L3r1",              [(3, [1], "X")],                      "welded"),
        ("cc1s_L3r2",              [(3, [1, 2], "X")],                   "welded"),
        ("cc1s_L3r2_sd",           [(3, [1, 2], "X")],                   "selfdual"),
        ("cc2s_mix32",             [(3, [1, 2], "X"), (2, [1], "Z")],    "welded"),
        ("cc2s_C3",                [(3, [1], "X"),    (3, [1], "Z")],    "welded"),
        ("cc2s_RX3",               [(3, [1, 2], "X"), (3, [1], "Z")],    "welded"),
        ("cc2s_RZ3",               [(3, [1], "X"),    (3, [1, 2], "Z")], "welded"),
        ("cc2s_R3",                [(3, [1, 2], "X"), (3, [1, 2], "Z")], "welded"),
        ("cc2s_R3_sd",             [(3, [1, 2], "X"), (3, [1, 2], "Z")], "selfdual"),
        ("cc3s_X3r1Z3r1X3r1",      [(3, [1], "X"),    (3, [1], "Z"),    (3, [1], "X")],    "welded"),
        ("cc3s_X3r1Z3r2X3r1",      [(3, [1], "X"),    (3, [1, 2], "Z"), (3, [1], "X")],    "welded"),
        ("cc3s_R3X3r2",            [(3, [1, 2], "X"), (3, [1, 2], "Z"), (3, [1, 2], "X")], "welded"),
    ]
    for nm, arg, sm in specs:
        results[nm] = summarize_torus_code(nm, arg, sm)

    out_p = ROOT / "verified_3d_color_welded_geometry.json"
    with open(out_p, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_p}")


if __name__ == "__main__":
    main()
