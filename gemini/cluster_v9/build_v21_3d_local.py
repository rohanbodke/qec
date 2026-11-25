#!/usr/bin/env python3
"""
build_v21_3d_local.py
Builds and verifies the V21 Strictly 3D-Local (L = 3 at every step, dim_eff < 3) welded code suite
using `kmc_engine_v19` (paired 4-decoder readout: `UF`, `Cool-UF`, `Hier-UF`, `BP-OSD`), covering:
  1. 2-step L = 3 bridge-arm pruning & protected-subsystem test (`C3` full vs prune-X, prune-Z,
     prune-both) + standalone reversed L = 3 modules (`C3^ZX`, `RX3^ZX`, `RZ3^ZX`, `R3^ZX`).
  2. Pure-L = 3 three-step `(X, Z, X)` and `(Z, X, Z)` hierarchies (`N = 16,798..22,160`,
     `dim_eff = 2.78..2.89 < 3`), testing bottleneck reversal and selective bridge placement.
  3. Pure-L = 3 prospective interface-guided sector-selective bridge simplification in `XZXZ`:
     `C3 R_X3` (`N = 431,188`, -27.1% vs `C3 R_3`, -39.2% vs `R3 R_3`), `C3 R_Z3` (`570,476`),
     `C3 R_3` (`591,208`), `R_X3 R_X3` (`470,332`), and `R_3 R_3` (`709,408`).
  4. Pure-L = 3 symmetric `XZZX` welding-order quartet on the identical 298,896-face 3D complex
     (`R3 R3` XZXZ, `R3^ZX R3^ZX` ZXZX, `R3^ZX R3` ZXXZ, `R3 R3^ZX` XZZX on planar & torus)
     plus simplified pure-L = 3 `XZZX` hierarchies (`C3 C3^ZX`, `C3 RX3^ZX`, `C3 RZ3^ZX`,
     `C3 R3^ZX`, `RX3 R3^ZX`, `RZ3 R3^ZX`).
  5. Paired 4-decoder unrelaxed i.i.d. single-shot bit-flip benchmarks on pure-L = 3 hierarchies.
"""

from concurrent.futures import ProcessPoolExecutor
import heapq
import json
import math
import os
import subprocess
import sys

WORK_DIR = "/work/submit/harshila/6s980_qmem_v9"
DATA_DIR = os.path.join(WORK_DIR, "data_v21")
ENGINE = os.path.join(WORK_DIR, "kmc_engine_v19")
sys.path.insert(0, WORK_DIR)

from build_v7_rescue import GeneralCubical2D, seed_planar_11, sheaf_to_css_with_seams
from build_v18_rework import gf2_rank_bigint
from build_v9_quarter_million import push_X_logical, push_Z_logical


def seed_torus_1face():
    return GeneralCubical2D(1, [(0, 0), (0, 0)], ["none", "none"], [(0, 0, 0, 0, 0, 0, 1, 1)])


def apply_step_L3_general(comp, L, rib_coords, iter_type, prev_cyl=None, prune_mode="full"):
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
            if Le > 1 and prune_mode == "full":
                assert all((Le - r) in rib_set for r in rib_set)
            return list(reversed(pts)), list(reversed(eids)), Le
        raise ValueError("Endpoint mismatch")

    xp_faces, face_grids = [], []
    first_rib = min(rib_set)
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
                eid = len(xp_edges)
                xp_edges.append((grid[s][t], grid[s + 1][t]))
                xp_colors.append("none")
                xp_is_rib.append(isr)
                he[s][t] = eid
        for s in range(1, Lx):
            for t in range(Ly):
                isr = s in rib_set
                if isr and Lx == L and Ly == L and prune_mode == "prune_1arm":
                    drop_t = 0 if par == 0 else (Ly - 1)
                    if s == first_rib and t == drop_t:
                        isr = False
                eid = len(xp_edges)
                xp_edges.append((grid[s][t], grid[s][t + 1]))
                xp_colors.append("none")
                xp_is_rib.append(isr)
                ve[s][t] = eid
        f_start = len(xp_faces)
        for s in range(Lx):
            for t in range(Ly):
                xp_faces.append((
                    grid[s][t], grid[s + 1][t], grid[s + 1][t + 1], grid[s][t + 1],
                    he[s][t], he[s][t + 1], ve[s][t], ve[s + 1][t],
                ))
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


def build_code_job(job):
    tag, seed_kind, steps = job
    if seed_kind == "planar":
        comp = seed_planar_11()
        lx_list, lz_list = [{0, 1}], [{0}]
    elif seed_kind == "tor1":
        comp = seed_torus_1face()
        lx_list, lz_list = [{0}, {1}], [{0}, {1}]
    else:
        raise ValueError(seed_kind)

    cyl = set()
    Lprod = 1
    for st in steps:
        L, rb, it = st[:3]
        pm = st[3] if len(st) > 3 else "full"
        pcomp = comp
        comp, cyl, meta = apply_step_L3_general(pcomp, L, rb, it, prev_cyl=cyl, prune_mode=pm)
        N, HX, HZ, *_ = sheaf_to_css_with_seams(comp)
        lx_list = [push_X_logical(pcomp, comp, meta, HZ, x) for x in lx_list]
        lz_list = [push_Z_logical(pcomp, comp, meta, HX, z) for z in lz_list]
        Lprod *= L

    # Verify anticommutation L_X . L_Z^T == I_{K_target}
    k_tgt = len(lx_list)
    for i in range(k_tgt):
        for j in range(k_tgt):
            ov = len(lx_list[i] & lz_list[j]) & 1
            assert ov == (1 if i == j else 0), f"{tag}: bad logical pairing ({i},{j})={ov}"

    if N <= 30000:
        rX = gf2_rank_bigint(HX)
        rZ = gf2_rank_bigint(HZ)
        k_tot = N - rX - rZ
    else:
        rX, rZ, k_tot = -1, -1, k_tgt

    for sec, checks, stabs, dlogs in [
        ("X_err", HZ, HX, lz_list),
        ("Z_err", HX, HZ, lx_list),
    ]:
        cpath = os.path.join(DATA_DIR, f"{tag}_{sec}.code")
        with open(cpath, "w") as out:
            out.write(f"{N} {len(checks)} {len(stabs)} {k_tgt}\n")
            for r in checks:
                out.write(str(len(r)) + " " + " ".join(map(str, r)) + "\n")
            for r in stabs:
                out.write(str(len(r)) + " " + " ".join(map(str, r)) + "\n")
            for lg in dlogs:
                slg = sorted(lg)
                out.write(str(len(slg)) + " " + " ".join(map(str, slg)) + "\n")
        subprocess.run([ENGINE, "verify", cpath], check=True, stdout=subprocess.DEVNULL)

    dz = [0] * N
    for r in HZ:
        for q in r:
            dz[q] += 1
    dx = [0] * N
    for r in HX:
        for q in r:
            dx[q] += 1
    gamma_X_65 = sum(1.0 / (1.0 + math.exp(6.5 * d)) for d in dz if d > 0)
    gamma_Z_65 = sum(1.0 / (1.0 + math.exp(6.5 * d)) for d in dx if d > 0)
    dim_eff = math.log(len(HZ)) / math.log(Lprod)

    return {
        "tag": tag,
        "seed": seed_kind,
        "N": N,
        "M_X": len(HX),
        "M_Z": len(HZ),
        "rank_HX": rX,
        "rank_HZ": rZ,
        "K_tot": k_tot,
        "K_target": k_tgt,
        "L_blk": Lprod,
        "dim_eff": dim_eff,
        "max_wX": max(len(r) for r in HX),
        "max_wZ": max(len(r) for r in HZ),
        "max_dX": max(dx),
        "max_dZ": max(dz),
        "gamma_X_b65": gamma_X_65,
        "gamma_Z_b65": gamma_Z_65,
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
            for ln in assigned[sh]:
                f.write(f"{sh}|{ln}\n")
    loads = [c for c, _ in heap]
    print(f"Packed {len(weighted_tasks)} tasks across {num_shards} shards (load min={min(loads):.1f}, max={max(loads):.1f})")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    C3    = [(3, [1],    "X"), (3, [1],    "Z")]
    RX3   = [(3, [1, 2], "X"), (3, [1],    "Z")]
    RZ3   = [(3, [1],    "X"), (3, [1, 2], "Z")]
    R3    = [(3, [1, 2], "X"), (3, [1, 2], "Z")]

    C3ZX  = [(3, [1],    "Z"), (3, [1],    "X")]
    RX3ZX = [(3, [1],    "Z"), (3, [1, 2], "X")]
    RZ3ZX = [(3, [1, 2], "Z"), (3, [1],    "X")]
    R3ZX  = [(3, [1, 2], "Z"), (3, [1, 2], "X")]

    jobs = [
        # 1. 2-step L=3 pruning & standalone reversed L=3 modules
        ("v21_L3_C3_full",      "planar", [(3, [1], "X", "full"),       (3, [1], "Z", "full")]),
        ("v21_L3_C3_pruneX",    "planar", [(3, [1], "X", "prune_1arm"), (3, [1], "Z", "full")]),
        ("v21_L3_C3_pruneZ",    "planar", [(3, [1], "X", "full"),       (3, [1], "Z", "prune_1arm")]),
        ("v21_L3_C3_pruneBoth", "planar", [(3, [1], "X", "prune_1arm"), (3, [1], "Z", "prune_1arm")]),
        ("v21_L3_RX3",          "planar", RX3),
        ("v21_L3_RZ3",          "planar", RZ3),
        ("v21_L3_R3",           "planar", R3),
        ("v21_L3_C3ZX",         "planar", C3ZX),
        ("v21_L3_RX3ZX",        "planar", RX3ZX),
        ("v21_L3_RZ3ZX",        "planar", RZ3ZX),
        ("v21_L3_R3ZX",         "planar", R3ZX),

        # 2. Pure-L=3 three-step (X, Z, X) and (Z, X, Z) hierarchies
        ("v21_3s_Z3r1_X3r1_Z3r1",     "planar", [(3, [1], "Z"),    (3, [1], "X"),    (3, [1], "Z")]),
        ("v21_3s_X3r1_Z3r1_X3r1",     "planar", [(3, [1], "X"),    (3, [1], "Z"),    (3, [1], "X")]),
        ("v21_3s_X3r2_Z3r1_X3r1",     "planar", [(3, [1, 2], "X"), (3, [1], "Z"),    (3, [1], "X")]),
        ("v21_3s_X3r1_Z3r2_X3r1",     "planar", [(3, [1], "X"),    (3, [1, 2], "Z"), (3, [1], "X")]),
        ("v21_3s_X3r2_Z3r2_X3r2",     "planar", [(3, [1, 2], "X"), (3, [1, 2], "Z"), (3, [1, 2], "X")]),
        ("v21_tor_3s_X3r2_Z3r2_X3r2", "tor1",   [(3, [1, 2], "X"), (3, [1, 2], "Z"), (3, [1, 2], "X")]),

        # 3. Pure-L=3 4-step XZXZ selective bridge cuts + full two-bridge welding-order quartet
        ("v21_4s_C3_R3",        "planar", C3  + R3),
        ("v21_4s_C3_RX3",       "planar", C3  + RX3),
        ("v21_4s_C3_RZ3",       "planar", C3  + RZ3),
        ("v21_4s_RX3_RX3",      "planar", RX3 + RX3),
        ("v21_4s_R3_R3",        "planar", R3  + R3),
        ("v21_tor_4s_C3_RX3",   "tor1",   C3  + RX3),
        ("v21_tor_4s_R3_R3",    "tor1",   R3  + R3),

        # 4. Pure-L=3 symmetric XZZX quartet & simplified frontier
        ("v21_4s_R3ZX_R3ZX",    "planar", R3ZX + R3ZX),
        ("v21_4s_R3ZX_R3",      "planar", R3ZX + R3),
        ("v21_4s_R3_R3ZX",      "planar", R3   + R3ZX),
        ("v21_tor_4s_R3_R3ZX",  "tor1",   R3   + R3ZX),
        ("v21_4s_C3_C3ZX",      "planar", C3   + C3ZX),
        ("v21_4s_C3_RX3ZX",     "planar", C3   + RX3ZX),
        ("v21_4s_C3_RZ3ZX",     "planar", C3   + RZ3ZX),
        ("v21_4s_C3_R3ZX",      "planar", C3   + R3ZX),
        ("v21_4s_RX3_R3ZX",     "planar", RX3  + R3ZX),
        ("v21_4s_RZ3_R3ZX",     "planar", RZ3  + R3ZX),
    ]

    specs = {}
    with ProcessPoolExecutor(max_workers=4) as ex:
        for meta in ex.map(build_code_job, jobs):
            specs[meta["tag"]] = meta
            print(
                f"Built {meta['tag']:26s}: N={meta['N']:7,d} M=({meta['M_X']:7,d},{meta['M_Z']:7,d}) "
                f"K={meta['K_tot']}/{meta['K_target']} dim={meta['dim_eff']:.3f}"
            )

    with open(os.path.join(WORK_DIR, "v21_code_specs.json"), "w") as f:
        json.dump(specs, f, indent=2)

    arms = []
    # Suite 1: 2-step L=3 pruning (n=512) & standalone L=3 modules (n=256)
    arms.append(("v21_L3_C3_full",      6.50, 1.5, 5.0, 36, 512, 40000000, 64))
    arms.append(("v21_L3_C3_pruneX",    6.50, 1.5, 5.0, 36, 512, 40000000, 64))
    arms.append(("v21_L3_C3_pruneZ",    6.50, 0.8, 4.2, 35, 512, 30000000, 64))
    arms.append(("v21_L3_C3_pruneBoth", 6.50, 0.8, 4.2, 35, 512, 30000000, 64))
    for tag in ("v21_L3_RX3", "v21_L3_RZ3", "v21_L3_R3", "v21_L3_C3ZX", "v21_L3_RX3ZX", "v21_L3_RZ3ZX", "v21_L3_R3ZX"):
        arms.append((tag, 6.50, 2.0, 6.2, 36, 256, 60000000, 64))

    # Suite 2: Pure-L=3 3-step hierarchies (n=128 at b=6.5, n=96 at b=7.0)
    for tag in (
        "v21_3s_Z3r1_X3r1_Z3r1",
        "v21_3s_X3r1_Z3r1_X3r1",
        "v21_3s_X3r2_Z3r1_X3r1",
        "v21_3s_X3r1_Z3r2_X3r1",
        "v21_3s_X3r2_Z3r2_X3r2",
        "v21_tor_3s_X3r2_Z3r2_X3r2",
    ):
        arms.append((tag, 6.50, 2.0, 6.2, 36, 128, 120000000, 32))
        arms.append((tag, 7.00, 2.5, 6.8, 36,  96, 160000000, 32))

    # Suite 3: Pure-L=3 4-step XZXZ selective bridge cuts
    for tag, lg0_65, lg1_65 in [
        ("v21_4s_C3_R3",      2.5, 5.6),
        ("v21_4s_C3_RX3",     2.5, 6.0),
        ("v21_4s_C3_RZ3",     2.0, 5.0),
        ("v21_4s_RX3_RX3",    2.5, 6.0),
        ("v21_tor_4s_C3_RX3", 2.5, 6.0),
        ("v21_tor_4s_R3_R3",  2.5, 5.6),
    ]:
        arms.append((tag, 6.50, lg0_65, lg1_65, 31, 96, 350000000, 16))
    for tag in ("v21_4s_C3_R3", "v21_4s_C3_RX3", "v21_4s_RX3_RX3", "v21_4s_R3_R3"):
        arms.append((tag, 7.00, 3.0, 6.8, 31, 64, 550000000, 16))
    # Temperature curve on C3_RX3 & R3_R3 for Fig. 7a
    for b, lg0, lg1 in [(4.50, 1.0, 3.5), (5.00, 1.2, 4.0), (5.50, 1.6, 4.6), (6.00, 2.0, 5.2)]:
        arms.append(("v21_4s_C3_RX3", b, lg0, lg1, 26, 64, 250000000, 16))

    # Suite 4: Pure-L=3 4-step Welding-Order Quartet (`R3` quartet across b=5.5, 6.0, 6.5, 7.0) + Simplified `XZZX`
    quartet_grids = {
        5.50: (1.5, 5.2, 28, 350000000),
        6.00: (2.0, 6.4, 29, 450000000),
        6.50: (2.5, 7.4, 31, 650000000),
        7.00: (3.2, 7.8, 29, 800000000),
    }
    for qtag in ("v21_4s_R3_R3", "v21_4s_R3ZX_R3ZX", "v21_4s_R3ZX_R3", "v21_4s_R3_R3ZX"):
        for b, (lg0, lg1, ng, mf) in quartet_grids.items():
            if qtag == "v21_4s_R3_R3" and b == 7.00:
                continue  # already added above
            ntr = 96 if b == 6.50 else 64
            arms.append((qtag, b, lg0, lg1, ng, ntr, mf, 16))
    for b in (5.50, 6.00, 6.50):
        lg0, lg1, ng, mf = quartet_grids[b]
        arms.append(("v21_tor_4s_R3_R3ZX", b, lg0, lg1 + 0.2, ng, 64, mf, 16))

    # Simplified pure-L=3 XZZX frontier (`C3 C3^ZX`, `C3 RX3^ZX`, `C3 RZ3^ZX`, `C3 R3^ZX`, `RX3 R3^ZX`, `RZ3 R3^ZX`)
    arms.append(("v21_4s_C3_C3ZX",  6.50, 2.2, 6.8, 30, 64, 500000000, 16))
    arms.append(("v21_4s_C3_RX3ZX", 6.50, 2.5, 7.2, 30, 64, 600000000, 16))
    arms.append(("v21_4s_C3_RZ3ZX", 6.50, 2.5, 7.2, 30, 64, 600000000, 16))

    for b, lg0, lg1, mf in [
        (4.50, 1.0, 4.0, 250000000),
        (5.00, 1.5, 4.8, 300000000),
        (5.50, 2.0, 5.6, 400000000),
        (6.00, 2.4, 6.6, 500000000),
        (6.50, 2.8, 7.4, 650000000),
        (7.00, 3.4, 7.8, 800000000),
    ]:
        ntr = 96 if b == 6.50 else 64
        arms.append(("v21_4s_C3_R3ZX", b, lg0, lg1, 30, ntr, mf, 16))

    for tag in ("v21_4s_RX3_R3ZX", "v21_4s_RZ3_R3ZX"):
        arms.append((tag, 6.00, 2.4, 6.6, 30, 64, 500000000, 16))
        arms.append((tag, 6.50, 2.8, 7.4, 30, 96, 650000000, 16))

    weighted = []
    for tag, beta_val, lg0, lg1, ng, tot_trials, maxf, chunk in arms:
        n_val = specs[tag]["N"]
        for sec in ("X_err", "Z_err"):
            cpath = os.path.join(DATA_DIR, f"{tag}_{sec}.code")
            for off in range(0, tot_trials, chunk):
                ntr = min(chunk, tot_trials - off)
                tid = f"v21__{tag}_{sec}__b{beta_val:.2f}__o{off}"
                cost = max(1.0, (n_val / 15000.0) * (ntr / 16.0) * max(1.0, lg1 - 3.8))
                cmd_s = f"{tid}|paired_readout|{cpath}|{beta_val:.4f}|{ntr}|{off + 1}|{lg0:.4f}|{lg1:.4f}|{ng}|{maxf}"
                weighted.append((cost, cmd_s))

    # Suite 5: Paired 4-decoder single-shot i.i.d. noise on pure-L=3 codes
    for tag in ("v21_3s_X3r2_Z3r2_X3r2", "v21_4s_C3_RX3", "v21_4s_C3_R3ZX", "v21_4s_R3_R3ZX"):
        n_val = specs[tag]["N"]
        for sec in ("X_err", "Z_err"):
            cpath = os.path.join(DATA_DIR, f"{tag}_{sec}.code")
            for off in range(0, 64, 16):
                tid = f"v21_iid__{tag}_{sec}__o{off}"
                cost = max(2.0, (n_val / 15000.0) * 1.5)
                cmd_s = f"{tid}|iid_readout|{cpath}|16|{off + 1}|0.00500|0.08500|17"
                weighted.append((cost, cmd_s))

    pack_shards(weighted, 96, os.path.join(WORK_DIR, "manifest_v21.txt"))


if __name__ == "__main__":
    main()
