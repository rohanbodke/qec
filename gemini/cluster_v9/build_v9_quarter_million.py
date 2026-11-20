#!/usr/bin/env python3
"""
build_v9_quarter_million.py
Builds and verifies the V9 Quarter-Million-Qubit (up to 379,976 qubits!) & Live 6-Level RG Flow Suite
using O(N) exact recursive logical operator propagation (verified by H_Z*L_X^T == 0, H_X*L_Z^T == 0,
L_X*L_Z^T == I_K, AND C++ `kmc_engine_v9 verify` `bad_comm == 0`!):

  1. 6-Level (s = 0..5) 3D 2x2 X-Welded Scaling Families:
     - `sq_rx_iso_s0..s5` (Uniform 2x2 Planar, K=1): N = 2 -> 19 -> 203 -> 2,311 -> 27,107 -> 321,383
     - `rottor_rx_s0..s5` (Thin-Bridge 2x2 Torus, K=2): N = 4 -> 38 -> 332 -> 3,100 -> 29,704 -> 288,344
     - `sq_rx_s0..s5`     (Thin-Bridge 2x2 Planar, K=1): N = 2 -> 19 -> 166 -> 1,550 -> 14,852 -> 144,172
  2. 5-Step Full Two-Sector Quantum Welded Memories (3 X-scales + 2 Z-scales!):
     - `sq_L2r1_s1, s3, s5`     (Planar K=1): N = 19 -> 1,942 -> 189,988 qubits!
     - `rottor_L2r1_s1, s3, s5` (Torus  K=2): N = 38 -> 3,884 -> 379,976 qubits!
  3. 6-Level `.rg` Hierarchies (`sq_rx_iso_s5.rg`, N=321,383; `rottor_rx_s5.rg`, N=288,344)
     for live syndrome-only thermal RG defect-density flow across 10 temperatures!
"""

from concurrent.futures import ProcessPoolExecutor
import os
import subprocess
import sys
import time

WORK_DIR = "/work/submit/harshila/6s980_qmem_v9"
DATA_DIR = os.path.join(WORK_DIR, "data")
RES_DIR = os.path.join(WORK_DIR, "results")
ENGINE = os.path.join(WORK_DIR, "kmc_engine_v9")

sys.path.insert(0, WORK_DIR)
from build_v7_rescue import (
    GeneralCubical2D,
    seed_planar_11,
    seed_rot_torus_422,
    sheaf_to_css_with_seams,
)


def apply_sheaf_step_with_maps(comp, L, rib_coords, iter_type, prev_cyl=None):
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

    xp_faces = []
    face_grids = []
    for f_idx, (v00, v10, v11, v01, eb, et, el, er) in enumerate(comp.faces):
        pb, eb_ids, Lx = oriented(eb, v00, v10)
        pt, et_ids, Lx2 = oriented(et, v01, v11)
        pl, el_ids, Ly = oriented(el, v00, v01)
        pr, er_ids, Ly2 = oriented(er, v10, v11)
        assert Lx == Lx2 and Ly == Ly2
        grid = [[None] * (Ly + 1) for _ in range(Lx + 1)]
        for s in range(Lx + 1):
            grid[s][0] = pb[s]; grid[s][Ly] = pt[s]
        for t in range(Ly + 1):
            grid[0][t] = pl[t]; grid[Lx][t] = pr[t]
        for s in range(1, Lx):
            for t in range(1, Ly):
                grid[s][t] = add_v()
        he = [[None] * (Ly + 1) for _ in range(Lx)]
        ve = [[None] * Ly for _ in range(Lx + 1)]
        for s in range(Lx):
            he[s][0] = eb_ids[s]; he[s][Ly] = et_ids[s]
        for t in range(Ly):
            ve[0][t] = el_ids[t]; ve[Lx][t] = er_ids[t]
        for t in range(1, Ly):
            for s in range(Lx):
                isr = (t in rib_set)
                eid = len(xp_edges)
                xp_edges.append((grid[s][t], grid[s + 1][t]))
                xp_colors.append("none")
                xp_is_rib.append(isr)
                he[s][t] = eid
        for s in range(1, Lx):
            for t in range(Ly):
                isr = (s in rib_set)
                eid = len(xp_edges)
                xp_edges.append((grid[s][t], grid[s][t + 1]))
                xp_colors.append("none")
                xp_is_rib.append(isr)
                ve[s][t] = eid
        f_start = len(xp_faces)
        for s in range(Lx):
            for t in range(Ly):
                xp_faces.append(
                    (grid[s][t], grid[s + 1][t], grid[s + 1][t + 1], grid[s][t + 1], he[s][t], he[s][t + 1], ve[s][t], ve[s + 1][t])
                )
        face_grids.append((Lx, Ly, he, ve, f_start))

    tot_v = 2 * xp_num_v
    def vc(v, c): return v + c * xp_num_v

    out_e, out_c, out_f = [], [], []
    n_xe = len(xp_edges)
    for c in (0, 1):
        for eid, (u, w) in enumerate(xp_edges):
            out_e.append((vc(u, c), vc(w, c)))
            out_c.append(new_col if xp_is_rib[eid] else xp_colors[eid])
    n_xf = len(xp_faces)
    for c in (0, 1):
        for (v00, v10, v11, v01, eb, et, el, er) in xp_faces:
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


def edge_qstarts(comp):
    qs = []
    tot = 0
    for e_idx, inc in enumerate(comp.edge_faces):
        qs.append(tot)
        d = len(inc)
        col = comp.edge_colors[e_idx]
        if d == 0:
            pass
        elif d == 2 or col == "blue":
            tot += 1
        elif col == "red":
            tot += (d - 1)
    return qs, tot


def push_X_logical(parent_comp, child_comp, meta, HZ_child, lx_set):
    p_qs, _ = edge_qstarts(parent_comp)
    c_qs, _ = edge_qstarts(child_comp)
    n_xe = meta["n_xe"]
    itype = meta["iter_type"]
    edge_sub = meta["edge_sub_eids"]
    fgrids = meta["face_grids"]
    copies = (0, 1) if itype == "X" else (0,)
    out_q = set()

    for e_idx, inc in enumerate(parent_comp.edge_faces):
        d = len(inc)
        col = parent_comp.edge_colors[e_idx]
        nq_e = 0 if d == 0 else (1 if (d == 2 or col == "blue") else d - 1)
        pq0 = p_qs[e_idx]
        sub0_eid = edge_sub[e_idx][0]
        for j in range(nq_e):
            if (pq0 + j) in lx_set:
                for c in copies:
                    out_q.symmetric_difference_update([c_qs[sub0_eid + c * n_xe] + j])

    syn0 = [0] * meta["n_xf"]
    for f_idx in range(meta["n_xf"]):
        p = 0
        for q in HZ_child[f_idx]:
            if q in out_q:
                p ^= 1
        syn0[f_idx] = p

    for pf_idx, (Lx, Ly, he, ve, f_start) in enumerate(fgrids):
        g = [[syn0[f_start + s * Ly + t] for t in range(Ly)] for s in range(Lx)]
        for s in range(Lx):
            for t in range(Ly - 1):
                if g[s][t]:
                    eid = he[s][t + 1]
                    for c in copies:
                        out_q.symmetric_difference_update([c_qs[eid + c * n_xe]])
                    g[s][t] ^= 1; g[s][t + 1] ^= 1
        for s in range(Lx - 1):
            if g[s][Ly - 1]:
                eid = ve[s + 1][Ly - 1]
                for c in copies:
                    out_q.symmetric_difference_update([c_qs[eid + c * n_xe]])
                g[s][Ly - 1] ^= 1; g[s + 1][Ly - 1] ^= 1
        assert g[Lx - 1][Ly - 1] == 0
    return out_q


def push_Z_logical(parent_comp, child_comp, meta, HX_child, lz_set):
    p_qs, _ = edge_qstarts(parent_comp)
    c_qs, _ = edge_qstarts(child_comp)
    n_xe = meta["n_xe"]
    itype = meta["iter_type"]
    edge_sub = meta["edge_sub_eids"]
    copies = (0,) if itype == "X" else (0, 1)
    out_q = set()

    for e_idx, inc in enumerate(parent_comp.edge_faces):
        d = len(inc)
        col = parent_comp.edge_colors[e_idx]
        nq_e = 0 if d == 0 else (1 if (d == 2 or col == "blue") else d - 1)
        pq0 = p_qs[e_idx]
        for j in range(nq_e):
            if (pq0 + j) in lz_set:
                for sub_eid in edge_sub[e_idx]:
                    for c in copies:
                        out_q.symmetric_difference_update([c_qs[sub_eid + c * n_xe] + j])

    if itype == "Z":
        viol = any(sum(1 for q in r if q in out_q) & 1 for r in HX_child)
        if viol:
            # Map midpoint vertices to cyl_edges in O(N):
            v_to_cyl = {}
            for ceid in child_comp.cyl_edges:
                u0, _ = child_comp.edges[ceid]
                v_to_cyl[u0] = ceid
            for e_idx, inc in enumerate(parent_comp.edge_faces):
                d = len(inc)
                col = parent_comp.edge_colors[e_idx]
                nq_e = 0 if d == 0 else (1 if (d == 2 or col == "blue") else d - 1)
                pq0 = p_qs[e_idx]
                if len(edge_sub[e_idx]) > 1:
                    for seid in edge_sub[e_idx][:-1]:
                        mv = child_comp.edges[seid][1]
                        if mv in v_to_cyl:
                            ceid = v_to_cyl[mv]
                            for j in range(nq_e):
                                if (pq0 + j) in lz_set:
                                    out_q.symmetric_difference_update([c_qs[ceid] + j])
    return out_q


def verify_and_write_code(N, HX, HZ, lx_list, lz_list, tag, sectors):
    # Hard mathematical gate in Python: L_X in Ker(HZ), L_Z in Ker(HX), L_X . L_Z^T == I_K!
    K = len(lx_list)
    for a in range(K):
        assert sum(1 for r in HZ if sum(1 for q in r if q in lx_list[a]) & 1) == 0
        assert sum(1 for r in HX if sum(1 for q in r if q in lz_list[a]) & 1) == 0
        for b in range(K):
            assert (len(lx_list[a].intersection(lz_list[b])) & 1) == (1 if a == b else 0)

    for sec in sectors:
        # For X_err: checks = HZ, stabs = HX, dual_logicals = L_Z!
        # For Z_err: checks = HX, stabs = HZ, dual_logicals = L_X!
        checks, stabs, dlogs = (HZ, HX, lz_list) if sec == "X_err" else (HX, HZ, lx_list)
        if len(checks) == 0:
            continue
        fin_p = os.path.join(DATA_DIR, f"{tag}_{sec}.code")
        if not os.path.exists(fin_p):
            with open(fin_p, "w") as f:
                f.write(f"{N} {len(checks)} {len(stabs)} {K}\n")
                for r in checks:
                    f.write(str(len(r)) + " " + " ".join(map(str, r)) + "\n")
                for r in stabs:
                    f.write(str(len(r)) + " " + " ".join(map(str, r)) + "\n")
                for lg in dlogs:
                    slg = sorted(lg)
                    f.write(str(len(slg)) + " " + " ".join(map(str, slg)) + "\n")
        vout = subprocess.run([ENGINE, "verify", fin_p], capture_output=True, text=True, check=True).stdout.strip()
        print(f"  Verified: {vout}", flush=True)


def build_chain_job(spec):
    name, seed_kind, step_types, export_steps, aniso, sectors, write_rg = spec
    t0 = time.time()
    if seed_kind == "planar":
        comp = seed_planar_11()
        lx, lz = [{0, 1}], [{0}]
    else:
        comp = seed_rot_torus_422()
        lx, lz = [{0, 1}, {0, 2}], [{1, 3}, {2, 3}]

    cyl = set()
    N, HX, HZ, *_ = sheaf_to_css_with_seams(comp)
    rg_levels = [(N, HZ, 0, [])] if write_rg else None

    if 0 in export_steps:
        verify_and_write_code(N, HX, HZ, lx, lz, f"{name}_s0", sectors)

    for s, itype in enumerate(step_types, 1):
        comp_prev = comp
        comp, new_cyl, meta = apply_sheaf_step_with_maps(comp_prev, 2, [1], itype, prev_cyl=(cyl if aniso else None))
        cyl = new_cyl
        N, HX, HZ, *_ = sheaf_to_css_with_seams(comp)
        lx = [push_X_logical(comp_prev, comp, meta, HZ, x) for x in lx]
        lz = [push_Z_logical(comp_prev, comp, meta, HX, z) for z in lz]
        if write_rg:
            spans = [(fs, Lx * Ly) for (Lx, Ly, _, _, fs) in meta["face_grids"]]
            rg_levels.append((N, HZ, meta["n_xf"], spans))
        if s in export_steps:
            verify_and_write_code(N, HX, HZ, lx, lz, f"{name}_s{s}", sectors)

    if write_rg:
        rg_p = os.path.join(DATA_DIR, f"{name}_s{len(step_types)}.rg")
        with open(rg_p, "w") as f:
            f.write(f"{len(step_types)}\n")
            for nq, hz, n_xf, spans in rg_levels:
                f.write(f"{nq} {len(hz)} {n_xf} {len(spans)}\n")
                for r in hz:
                    f.write(str(len(r)) + " " + " ".join(map(str, r)) + "\n")
                for fs, cnt in spans:
                    f.write(f"{fs} {cnt}\n")
        print(f"  Wrote RG hierarchy {rg_p} ({time.time()-t0:.1f}s)", flush=True)
    return name


def kmc_line(tag, b, trials, cap):
    cp = os.path.join(DATA_DIR, f"{tag}.code")
    return f"kmc__{tag}__uniform__uf__b{b:.2f}|kmc|{cp}|uniform|uf|{b:.2f}|{trials}|{cap}"


def rg_line(rg_name, b, t_eq, trials):
    rp = os.path.join(DATA_DIR, f"{rg_name}.rg")
    return f"rg__{rg_name}__b{b:.2f}|rg_flow|{rp}|{b:.2f}|{t_eq:.1f}|{trials}"


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)

    specs = [
        ("sq_rx_iso",   "planar", ["X"]*5,             [0, 1, 2, 3, 4, 5], False, ["X_err"], True),
        ("rottor_rx",   "torus",  ["X"]*5,             [0, 1, 2, 3, 4, 5], True,  ["X_err"], True),
        ("sq_rx",       "planar", ["X"]*5,             [0, 1, 2, 3, 4, 5], True,  ["X_err"], False),
        ("sq_L2r1",     "planar", ["X","Z","X","Z","X"], [1, 3, 5],        True,  ["X_err", "Z_err"], False),
        ("rottor_L2r1", "torus",  ["X","Z","X","Z","X"], [1, 3, 5],        True,  ["X_err", "Z_err"], False),
    ]

    with ProcessPoolExecutor(max_workers=5) as ex:
        for done_name in ex.map(build_chain_job, specs):
            print(f"Finished building chain: {done_name}", flush=True)

    tasks = []

    # 1. Intermediate fine-grid betas (1.90, 2.10, 2.15) on s=0..4 PLUS full s=5 (up to 321,383 qubits!)
    for fam in ("sq_rx_iso", "sq_rx", "rottor_rx"):
        # s = 0..4 fine betas [1.90, 2.10, 2.15] so all 6 levels s=0..5 share identical matched beta grid!
        for s in range(5):
            tag = f"{fam}_s{s}_X_err"
            tr = 160 if s <= 2 else (96 if s == 3 else 36)
            for b in [1.90, 2.10, 2.15]:
                tasks.append(kmc_line(tag, b, tr, 150000000))
        # s = 5 (N = 321,383 / 144,172 / 288,344 qubits!) across [1.50, 1.75, 1.90, 2.00, 2.10, 2.15, 2.25]
        tag5 = f"{fam}_s5_X_err"
        for b in [1.50, 1.75, 1.90, 2.00, 2.10, 2.15, 2.25]:
            tasks.append(kmc_line(tag5, b, 20, 320000000))

    # 2. Live 6-Level Syndrome-Only Thermal RG Flow (`s = 5 -> 4 -> 3 -> 2 -> 1 -> 0`) on 321,383 & 288,344 qubits!
    for rg_nm in ("sq_rx_iso_s5", "rottor_rx_s5"):
        for b in [0.80, 1.10, 1.35, 1.60, 1.75, 1.86, 2.00, 2.15, 2.35, 2.65]:
            tasks.append(rg_line(rg_nm, b, 25.0, 48))

    # 3. 5-Step Full Two-Sector Quantum Welded Ladder (`s = 1 -> 3 -> 5`, up to 189,988 planar & 379,976 torus qubits!)
    # Notice: on odd steps (`X -> Z -> X -> Z -> X`), `Z_err` is the bottleneck sector (2 Z-steps vs 3 X-steps),
    # just as `X_err` was the bottleneck on even steps! We run BOTH `Z_err` AND `X_err`!
    for fam in ("sq_L2r1", "rottor_L2r1"):
        for s, tr, cap in [(1, 160, 50000000), (3, 64, 150000000), (5, 20, 320000000)]:
            for sec in ("Z_err", "X_err"):
                if s == 5 and sec == "X_err":
                    # Run X_err at 5.0, 5.5 to verify t_X >> t_Z on s=5 (since s=5 just added a 3rd X-bridge step!)
                    for b in [4.50, 5.00, 5.50]:
                        tasks.append(kmc_line(f"{fam}_s5_X_err", b, 16, 250000000))
                    continue
                for b in [4.50, 5.00, 5.50, 5.75, 6.00, 6.25, 6.50]:
                    tasks.append(kmc_line(f"{fam}_s{s}_{sec}", b, tr, cap))

    man_p = os.path.join(WORK_DIR, "manifest_v9.txt")
    with open(man_p, "w") as f:
        for t in tasks:
            f.write(t + "\n")
    print(f"Wrote {len(tasks)} V9 tasks to {man_p}", flush=True)


if __name__ == "__main__":
    main()
