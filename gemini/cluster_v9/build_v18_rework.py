#!/usr/bin/env python3
"""
build_v18_rework.py
Builds and mathematically verifies all V18 `.code` files using the unchanged generator rules
(`H_X * H_Z^T == 0`, propagated seed logical pairing `L_X . L_Z^T == 1`, and `kmc_engine_v17 verify`),
records exact code metadata in `v18_code_specs.json`, and writes two load-balanced SLURM manifests:
  - `manifest_v18_std.txt`    (80 shards, 8 threads, 12G RAM): Suites D1, E, F, and G (Decisions 1, 2, 3)
  - `manifest_v18_bigmem.txt` (32 shards, 4 threads, 24G RAM): Suite D2 (`C2 + L4_off` & `C2 + L4_cen`, N = 261,476)
"""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import heapq
import json
import math
import os
import subprocess
import sys

WORK_DIR = "/work/submit/harshila/6s980_qmem_v9"
DATA_DIR = os.path.join(WORK_DIR, "data")
ENGINE = os.path.join(WORK_DIR, "kmc_engine_v17")
sys.path.insert(0, WORK_DIR)

from build_v7_rescue import GeneralCubical2D, seed_planar_11, sheaf_to_css_with_seams
from build_v9_quarter_million import push_X_logical, push_Z_logical


def apply_step_general(comp, L, rib_coords, iter_type, prev_cyl=None, gadget_mode="full"):
    new_col = "blue" if iter_type == "X" else "red"
    rib_set = set(rib_coords)
    prev_cyl = prev_cyl or set()
    xp_num_v = 0

    def add_v():
        nonlocal xp_num_v
        r = xp_num_v
        xp_num_v += 1
        return r

    orig_v = [add_v() for _ in range(comp.num_vertices)]
    xp_edges, xp_colors, xp_is_rib = [], [], []
    edge_sub_verts, edge_sub_eids, edge_L = [], [], []
    xp_vert_orig_edge = {}

    for e_idx, (a, b) in enumerate(comp.edges):
        Le = 1 if (e_idx in prev_cyl) else L
        edge_L.append(Le)
        pts = [orig_v[a]] + [add_v() for _ in range(Le - 1)] + [orig_v[b]]
        for v in pts[1:-1]:
            xp_vert_orig_edge[v] = e_idx
        edge_sub_verts.append(pts)
        fdeg = len(comp.edge_faces[e_idx])
        inh_col = comp.edge_colors[e_idx] if fdeg != 2 else "none"
        eids = []
        for k in range(Le):
            eid = len(xp_edges)
            xp_edges.append((pts[k], pts[k + 1]))
            xp_colors.append(inh_col)
            xp_is_rib.append(False)
            eids.append(eid)
        edge_sub_eids.append(eids)

    def oriented(e_idx, sv, ev):
        a, b = comp.edges[e_idx]
        pts, eids, Le = edge_sub_verts[e_idx], edge_sub_eids[e_idx], edge_L[e_idx]
        if a == sv and b == ev:
            return pts, eids, Le
        elif b == sv and a == ev:
            if Le > 1:
                assert all((Le - r) in rib_set for r in rib_set)
            return list(reversed(pts)), list(reversed(eids)), Le
        raise ValueError("Endpoint mismatch")

    xp_faces, face_grids = [], []
    for f_idx, (v00, v10, v11, v01, eb, et, el, er) in enumerate(comp.faces):
        pb, eb_ids, Lx = oriented(eb, v00, v10)
        pt, et_ids, Lx2 = oriented(et, v01, v11)
        pl, el_ids, Ly = oriented(el, v00, v01)
        pr, er_ids, Ly2 = oriented(er, v10, v11)
        assert Lx == Lx2 and Ly == Ly2
        grid = [[None] * (Ly + 1) for _ in range(Lx + 1)]
        for s in range(Lx + 1):
            grid[s][0] = pb[s]
            grid[s][Ly] = pt[s]
        for t in range(Ly + 1):
            grid[0][t] = pl[t]
            grid[Lx][t] = pr[t]
        for s in range(1, Lx):
            for t in range(1, Ly):
                grid[s][t] = add_v()
        he = [[None] * (Ly + 1) for _ in range(Lx)]
        ve = [[None] * Ly for _ in range(Lx + 1)]
        for s in range(Lx):
            he[s][0] = eb_ids[s]
            he[s][Ly] = et_ids[s]
        for t in range(Ly):
            ve[0][t] = el_ids[t]
            ve[Lx][t] = er_ids[t]

        par = (f_idx + v00 + v11) & 1
        for t in range(1, Ly):
            for s in range(Lx):
                isr = t in rib_set
                if isr and Lx == 2 and Ly == 2 and gadget_mode == "cb2rib" and s == 1:
                    isr = False
                eid = len(xp_edges)
                xp_edges.append((grid[s][t], grid[s + 1][t]))
                xp_colors.append("none")
                xp_is_rib.append(isr)
                he[s][t] = eid
        for s in range(1, Lx):
            for t in range(Ly):
                isr = s in rib_set
                if isr and Lx == 2 and Ly == 2:
                    if gadget_mode == "cb3rib" and t == par:
                        isr = False
                    elif gadget_mode == "cb2rib" and t == 1:
                        isr = False
                eid = len(xp_edges)
                xp_edges.append((grid[s][t], grid[s][t + 1]))
                xp_colors.append("none")
                xp_is_rib.append(isr)
                ve[s][t] = eid
        f_start = len(xp_faces)
        for s in range(Lx):
            for t in range(Ly):
                xp_faces.append(
                    (
                        grid[s][t],
                        grid[s + 1][t],
                        grid[s + 1][t + 1],
                        grid[s][t + 1],
                        he[s][t],
                        he[s][t + 1],
                        ve[s][t],
                        ve[s + 1][t],
                    )
                )
        face_grids.append((Lx, Ly, he, ve, f_start))

    tot_v = 2 * xp_num_v

    def vc(v, c):
        return v + c * xp_num_v

    out_e, out_c, out_f = [], [], []
    n_xe = len(xp_edges)
    for c in (0, 1):
        for eid, (u, w) in enumerate(xp_edges):
            out_e.append((vc(u, c), vc(w, c)))
            out_c.append(new_col if xp_is_rib[eid] else xp_colors[eid])
    n_xf = len(xp_faces)
    for c in (0, 1):
        for v00, v10, v11, v01, eb, et, el, er in xp_faces:
            off = c * n_xe
            out_f.append((vc(v00, c), vc(v10, c), vc(v11, c), vc(v01, c), eb + off, et + off, el + off, er + off))
    rib_vs = {x for eid, isr in enumerate(xp_is_rib) if isr for x in xp_edges[eid]}
    cyl_eid, new_cyl = {}, set()
    for v in sorted(rib_vs):
        ceid = len(out_e)
        new_cyl.add(ceid)
        out_e.append((vc(v, 0), vc(v, 1)))
        if v in xp_vert_orig_edge:
            oe = xp_vert_orig_edge[v]
            fdeg = len(comp.edge_faces[oe])
            out_c.append(comp.edge_colors[oe] if fdeg != 2 else "none")
        else:
            out_c.append(new_col)
        cyl_eid[v] = ceid
    for eid, isr in enumerate(xp_is_rib):
        if isr:
            u, w = xp_edges[eid]
            out_f.append((vc(u, 0), vc(w, 0), vc(w, 1), vc(u, 1), eid, eid + n_xe, cyl_eid[u], cyl_eid[w]))

    res = GeneralCubical2D(tot_v, out_e, out_c, out_f)
    res.cyl_edges = new_cyl
    for eid in range(len(res.edges)):
        fd = len(res.edge_faces[eid])
        if fd == 2:
            res.edge_colors[eid] = "none"
        elif res.edge_colors[eid] == "none":
            res.edge_colors[eid] = new_col

    meta = {
        "iter_type": iter_type,
        "n_xe": n_xe,
        "n_xf": n_xf,
        "edge_sub_eids": edge_sub_eids,
        "face_grids": face_grids,
    }
    return res, new_cyl, meta


def gf2_rank_bigint(rows):
    piv = {}
    for r in rows:
        cur = 0
        for q in r:
            cur |= 1 << q
        while cur:
            p = (cur & -cur).bit_length() - 1
            if p in piv:
                cur ^= piv[p]
            else:
                piv[p] = cur
                break
    return len(piv)


def build_one_code(spec):
    tag, steps, expected_k_total = spec
    comp = seed_planar_11()
    # Planar [[2,1,1]] seed: HZ0 = [[0, 1]], HX0 = [] -> L_X = {0, 1} in Ker(HZ0), L_Z = {0} in Ker(HX0)
    lx_cur, lz_cur = {0, 1}, {0}
    cyl = set()
    for L, rb, it, gm in steps:
        pcomp = comp
        comp, cyl, meta = apply_step_general(pcomp, L, rb, it, prev_cyl=cyl, gadget_mode=gm)
        N, HX, HZ, *_ = sheaf_to_css_with_seams(comp)
        lx_cur = push_X_logical(pcomp, comp, meta, HZ, lx_cur)
        lz_cur = push_Z_logical(pcomp, comp, meta, HX, lz_cur)

    # Verify CSS commutation and seed logical pairing
    assert (len(lx_cur & lz_cur) % 2) == 1
    rX = gf2_rank_bigint(HX)
    rZ = gf2_rank_bigint(HZ)
    k_total = N - rX - rZ
    assert k_total == expected_k_total, f"{tag}: got K_total={k_total}, expected {expected_k_total}"

    lx_sorted = sorted(lx_cur)
    lz_sorted = sorted(lz_cur)

    for sec, checks, stabs, dual_l in [
        ("X_err", HZ, HX, lz_sorted),
        ("Z_err", HX, HZ, lx_sorted),
    ]:
        cpath = os.path.join(DATA_DIR, f"{tag}_{sec}.code")
        with open(cpath, "w") as out:
            out.write(f"{N} {len(checks)} {len(stabs)} 1\n")
            for r in checks:
                out.write(str(len(r)) + " " + " ".join(map(str, r)) + "\n")
            for r in stabs:
                out.write(str(len(r)) + " " + " ".join(map(str, r)) + "\n")
            out.write(str(len(dual_l)) + " " + " ".join(map(str, dual_l)) + "\n")
        subprocess.run([ENGINE, "verify", cpath], check=True, stdout=subprocess.DEVNULL)

    dz = [0] * N
    for r in HZ:
        for q in r:
            dz[q] += 1
    dx = [0] * N
    for r in HX:
        for q in r:
            dx[q] += 1

    return {
        "tag": tag,
        "N": N,
        "M_X": len(HX),
        "M_Z": len(HZ),
        "rank_HX": rX,
        "rank_HZ": rZ,
        "K_total": k_total,
        "K_target": 1,
        "K_unused": k_total - 1,
        "clean_hazard_X_65": sum(1.0 / (1.0 + math.exp(6.5 * d)) for d in dz),
        "clean_hazard_Z_65": sum(1.0 / (1.0 + math.exp(6.5 * d)) for d in dx),
    }


def pack_shards(weighted_tasks, num_shards, out_path):
    weighted_tasks.sort(key=lambda x: -x[0])
    heap = [(0.0, s) for s in range(num_shards)]
    heapq.heapify(heap)
    assigned = [[] for _ in range(num_shards)]
    for cost, tstr in weighted_tasks:
        c_tot, sh = heapq.heappop(heap)
        assigned[sh].append(tstr)
        heapq.heappush(heap, (c_tot + cost, sh))
    with open(out_path, "w") as f:
        for sh in range(num_shards):
            for tstr in assigned[sh]:
                f.write(f"{sh}|{tstr}\n")
    loads = [c for (c, _) in heap]
    print(f"Wrote {out_path}: tasks={len(weighted_tasks)}, shards={num_shards}, load=[{min(loads):.1f}, {max(loads):.1f}]", flush=True)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    C2 = [(2, [1], "X", "full"), (2, [1], "Z", "full")]
    C2pX = [(2, [1], "X", "cb3rib"), (2, [1], "Z", "full")]
    C3 = [(3, [1], "X", "full"), (3, [1], "Z", "full")]
    RX3 = [(3, [1, 2], "X", "full"), (3, [1], "Z", "full")]
    RZ3 = [(3, [1], "X", "full"), (3, [1, 2], "Z", "full")]
    R3 = [(3, [1, 2], "X", "full"), (3, [1, 2], "Z", "full")]
    L4_off = [(4, [1], "X", "full"), (4, [1], "Z", "full")]
    L4_cen = [(4, [2], "X", "full"), (4, [2], "Z", "full")]

    code_specs = [
        # Suite D1 & D2: Resource-matched L=4 offset vs centered (standalone & under coarse C2)
        ("v18_sq_L4_off",        L4_off,      1),
        ("v18_sq_L4_cen",        L4_cen,      1),
        ("v18_sq_C2_L4off",      C2 + L4_off, 1),
        ("v18_sq_C2_L4cen",      C2 + L4_cen, 1),
        # Suite E & F: Two-step C2 direct pruning & protected-subsystem candidates
        ("v18_sq_C2_full",       C2,          1),
        ("v18_sq_C2_pruneX",     C2pX,        1),
        ("v18_sq_C2_pruneZ_sub", [(2, [1], "X", "full"),   (2, [1], "Z", "cb3rib")], 9),
        ("v18_sq_C2_pruneB_sub", [(2, [1], "X", "cb3rib"), (2, [1], "Z", "cb3rib")], 9),
        # Suite G: Held-out prospective simplifications + matched reference cohorts
        ("v18_sq_RX3",           RX3,         1),
        ("v18_sq_RZ3",           RZ3,         1),
        ("v18_sq_C2_RX3",        C2 + RX3,    1),
        ("v18_sq_C2_RZ3",        C2 + RZ3,    1),
        ("v18_sq_C2_R3",         C2 + R3,     1),
        ("v18_sq_C2_C3",         C2 + C3,     1),
        ("v18_sq_C2pX_C2",       C2pX + C2,   1),
        ("v18_sq_C2_C2",         C2 + C2,     1),
        ("v18_sq_RX3_C2",        RX3 + C2,    1),
        ("v18_sq_RZ3_C2",        RZ3 + C2,    1),
        ("v18_sq_R3_C2",         R3 + C2,     1),
        ("v18_sq_C3_C2",         C3 + C2,     1),
    ]

    meta_out = {}
    with ProcessPoolExecutor(max_workers=6) as ex:
        for info in ex.map(build_one_code, code_specs):
            meta_out[info["tag"]] = info
            print(
                f"Verified {info['tag']:22s}: N={info['N']:7,d} K_tot={info['K_total']} "
                f"(rX={info['rank_HX']:6,d}, rZ={info['rank_HZ']:6,d})",
                flush=True,
            )

    with open(os.path.join(WORK_DIR, "v18_code_specs.json"), "w") as f:
        json.dump(meta_out, f, indent=2)

    std_tasks = []
    big_tasks = []

    def add_task(dest, prefix, tag, sec, b, trials, off, lg0, lg1, ng, cap, cost):
        cp = os.path.join(DATA_DIR, f"{tag}_{sec}.code")
        tid = f"{prefix}__{tag}_{sec}__b{b:.2f}__o{off}"
        dest.append((cost, f"{tid}|kinetic_readout|{cp}|none|1.0000|{b:.2f}|{trials}|{off}|{lg0:.2f}|{lg1:.2f}|{ng}|{cap}"))

    # 1. Suite D1: Standalone L=4 offset vs centered (256 X, 128 Z, 41-pt grid 10^1..10^5, beta=6.50)
    for tag in ["v18_sq_L4_off", "v18_sq_L4_cen"]:
        for sh in range(4):
            add_task(std_tasks, "d1", tag, "X_err", 6.50, 64, 64 * sh, 1.00, 5.00, 41, 600_000_000, 12.0)
        for sh in range(2):
            add_task(std_tasks, "d1", tag, "Z_err", 6.50, 64, 64 * sh, 1.00, 5.00, 41, 600_000_000, 12.0)

    # 2. Suite D2 (BIGMEM 24G, 4 threads): Composed C2 + L4_off vs C2 + L4_cen (128 X, 128 Z, grid 10^1..10^4, 31 pts)
    for tag in ["v18_sq_C2_L4off", "v18_sq_C2_L4cen"]:
        for sh in range(16):
            add_task(big_tasks, "d2", tag, "X_err", 6.50, 8, 8 * sh, 1.00, 4.00, 31, 600_000_000, 40.0)
        for sh in range(8):
            add_task(big_tasks, "d2", tag, "Z_err", 6.50, 16, 16 * sh, 1.00, 4.00, 31, 400_000_000, 18.0)

    # 3. Suite E & F: Two-step C2 direct pruning & protected-subsystem arms (256 X, 128 Z, 41-pt grid 10^1..10^5)
    for tag in ["v18_sq_C2_full", "v18_sq_C2_pruneX", "v18_sq_C2_pruneZ_sub", "v18_sq_C2_pruneB_sub"]:
        for sh in range(2):
            add_task(std_tasks, "ef", tag, "X_err", 6.50, 128, 128 * sh, 1.00, 5.00, 41, 400_000_000, 6.0)
        add_task(std_tasks, "ef", tag, "Z_err", 6.50, 128, 0, 1.00, 5.00, 41, 400_000_000, 5.0)

    # 4. Suite G0: Standalone RX3 & RZ3 (256 X, 128 Z, grid 10^2..10^5.6, 37 pts)
    for tag in ["v18_sq_RX3", "v18_sq_RZ3"]:
        for sh in range(4):
            add_task(std_tasks, "g0", tag, "X_err", 6.50, 64, 64 * sh, 2.00, 5.60, 37, 600_000_000, 10.0)
        for sh in range(2):
            add_task(std_tasks, "g0", tag, "Z_err", 6.50, 64, 64 * sh, 2.00, 5.60, 37, 600_000_000, 9.0)

    # 5. Suite G1 (Decision 1, beta=6.50, 128 X, 128 Z, grid 10^2..10^5.8, 39 pts):
    for tag in ["v18_sq_C2_RX3", "v18_sq_C2_RZ3", "v18_sq_C2_R3", "v18_sq_C2_C3"]:
        for sh in range(16):
            add_task(std_tasks, "g1", tag, "X_err", 6.50, 8, 8 * sh, 2.00, 5.80, 39, 900_000_000, 26.0)
        for sh in range(8):
            add_task(std_tasks, "g1", tag, "Z_err", 6.50, 16, 16 * sh, 2.00, 5.80, 39, 500_000_000, 14.0)

    # 6. Suite G2 (Decision 2, beta=6.50, 128 X, 128 Z, grid 10^2..10^6.4, 45 pts):
    for tag in ["v18_sq_C2pX_C2", "v18_sq_C2_C2"]:
        for sh in range(16):
            add_task(std_tasks, "g2", tag, "X_err", 6.50, 8, 8 * sh, 2.00, 6.40, 45, 1_200_000_000, 24.0)
        for sh in range(8):
            add_task(std_tasks, "g2", tag, "Z_err", 6.50, 16, 16 * sh, 2.00, 6.40, 45, 600_000_000, 12.0)

    # 7. Suite G3 (Decision 3, beta=7.00, 96 X, 64 Z, grid 10^3..10^7.4, 45 pts):
    for tag in ["v18_sq_RX3_C2", "v18_sq_RZ3_C2", "v18_sq_R3_C2", "v18_sq_C3_C2"]:
        for sh in range(12):
            add_task(std_tasks, "g3", tag, "X_err", 7.00, 8, 8 * sh, 3.00, 7.40, 45, 2_200_000_000, 52.0)
        for sh in range(4):
            add_task(std_tasks, "g3", tag, "Z_err", 7.00, 16, 16 * sh, 3.00, 7.40, 45, 800_000_000, 16.0)

    pack_shards(std_tasks, 80, os.path.join(WORK_DIR, "manifest_v18_std.txt"))
    pack_shards(big_tasks, 32, os.path.join(WORK_DIR, "manifest_v18_bigmem.txt"))


if __name__ == "__main__":
    main()
