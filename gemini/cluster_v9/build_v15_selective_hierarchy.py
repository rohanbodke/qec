#!/usr/bin/env python3
"""
build_v15_selective_hierarchy.py
Builds and verifies all codes, exact Cycle-1 parent-cell check maps (.c1map),
architectural specification metadata (`v15_code_specs.json`), and SLURM manifest
(`manifest_v15.txt`) for the V15 Selective Multi-Scale Hierarchy campaign:

  Suite 1 (Claim 1 — Single-Cycle Module Characterization, s = 2):
    Matched-temperature two-sector comparison across beta in [4.50, 7.25]:
    - Same-subdivision L=3 pair (aniso=True):
        `v15_sq_C3` (1,036q, 1-rib) vs `v15_sq_R3` (1,480q, 2-rib) on Planar (K=1)
        `v15_tor_C3` (1,036q, 1-rib) vs `v15_tor_R3` (1,480q, 2-rib) on Periodic Torus (K=2)
    - Same-subdivision L=3 pair (aniso=False, isotropic bridges):
        `v15_sq_C3_iso` (1,271q, 1-rib) vs `v15_sq_R3_iso` (2,152q, 2-rib) on Planar (K=1)
    - Unit-patch L=2 1-rib vs L=3 2-rib pair (aniso=True):
        `v15_sq_C2` (254q) vs `v15_sq_R3` (1,480q) on Planar (K=1)
        `v15_rottor_C2` (508q) vs `v15_rottor_R3` (2,960q) on Rotated Torus (K=2)

  Suite 2A (Claim 2 — Pure Rib Placement at Fixed L=3: CC, RC, CR, RR, s = 4):
    - Planar (K=1): `v15_sq_L3_CC` (415,004q), `v15_sq_L3_RC` (499,088q),
      `v15_sq_L3_CR` (591,208q), `v15_sq_L3_RR` (709,408q)
    - Boundaryless Periodic Torus (K=2): `v15_tor_L3_CC` (415,004q), `v15_tor_L3_RC` (499,088q),
      `v15_tor_L3_CR` (591,208q), `v15_tor_L3_RR` (709,408q)
    Evaluated with `span_kmc` in X_err (recording both failure times and coarse Cycle-1
    excursion arrivals & healed excursions) and `kmc` in Z_err across beta in [5.00, 7.00].

  Suite 2B (Claim 2 — Unit-Patch Module Placement C2=(L=2,1-rib) vs R3=(L=3,2-rib): CC, RC, CR, RR):
    - Planar (aniso=True, K=1): `v15_sq_pkg_CC` (25,148q), `v15_sq_pkg_RC` (133,864q),
      `v15_sq_pkg_CR` (136,536q) [with RR = `v15_sq_L3_RR` 709,408q]
    - Boundaryless Rotated Torus (aniso=True, K=2): `v15_rottor_pkg_CC` (50,296q),
      `v15_rottor_pkg_RC` (267,728q), `v15_rottor_pkg_CR` (273,072q)
    - Isotropic Planar (aniso=False, K=1, exact MZ=129,600 match):
      `sq_iso_L2r1_s4` (CC, 47,387q), `v15_sq_iso_RC` (299,500q), `v15_sq_iso_CR` (312,128q)

  Suite 3 (Reliability Check — Paired Checkpoint Cadence Audit on RC vs CR):
    - `paired_cadence` across eta in {0.02, 0.05, 0.20, 0.50} at beta in {6.00, 6.50}.
"""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import os
import subprocess
import sys
import time

WORK_DIR = "/work/submit/harshila/6s980_qmem_v9"
DATA_DIR = os.path.join(WORK_DIR, "data")
ENGINE = os.path.join(WORK_DIR, "kmc_engine_v15")

sys.path.insert(0, WORK_DIR)
from build_v7_rescue import (
    GeneralCubical2D,
    seed_planar_11,
    seed_rot_torus_422,
    sheaf_to_css_with_seams,
)
from build_v9_quarter_million import (
    push_X_logical,
    push_Z_logical,
    verify_and_write_code,
)
import build_v9_quarter_million as b9
b9.ENGINE = ENGINE


def seed_torus_1face():
    return GeneralCubical2D(1, [(0, 0), (0, 0)], ["none", "none"], [(0, 0, 0, 0, 0, 0, 1, 1)])


def apply_step_with_parent_faces(comp, L, rib_coords, iter_type, prev_cyl=None):
    """Exact replica of apply_sheaf_step_with_maps PLUS exact child_face -> parent_face map."""
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
    xp_edges, xp_colors, xp_is_rib, xp_edge_pf = [], [], [], []
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
            xp_edge_pf.append(-1)
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

    xp_faces, xp_face_pf, face_grids = [], [], []
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
                xp_edge_pf.append(f_idx)
                he[s][t] = eid
        for s in range(1, Lx):
            for t in range(Ly):
                isr = (s in rib_set)
                eid = len(xp_edges)
                xp_edges.append((grid[s][t], grid[s][t + 1]))
                xp_colors.append("none")
                xp_is_rib.append(isr)
                xp_edge_pf.append(f_idx)
                ve[s][t] = eid
        f_start = len(xp_faces)
        for s in range(Lx):
            for t in range(Ly):
                xp_faces.append(
                    (grid[s][t], grid[s + 1][t], grid[s + 1][t + 1], grid[s][t + 1], he[s][t], he[s][t + 1], ve[s][t], ve[s + 1][t])
                )
                xp_face_pf.append(f_idx)
        face_grids.append((Lx, Ly, he, ve, f_start))

    tot_v = 2 * xp_num_v
    def vc(v, c): return v + c * xp_num_v

    out_e, out_c, out_f, child_to_pf = [], [], [], []
    n_xe = len(xp_edges)
    for c in (0, 1):
        for eid, (u, w) in enumerate(xp_edges):
            out_e.append((vc(u, c), vc(w, c)))
            out_c.append(new_col if xp_is_rib[eid] else xp_colors[eid])
    n_xf = len(xp_faces)
    for c in (0, 1):
        for k, (v00, v10, v11, v01, eb, et, el, er) in enumerate(xp_faces):
            off = c * n_xe
            out_f.append((vc(v00, c), vc(v10, c), vc(v11, c), vc(v01, c), eb + off, et + off, el + off, er + off))
            child_to_pf.append(xp_face_pf[k])
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
            child_to_pf.append(xp_edge_pf[eid])

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
    return res, new_cyl, meta, child_to_pf


def build_and_characterize(spec):
    import build_v9_quarter_million as b9
    b9.ENGINE = ENGINE
    name, seed_kind, sched, aniso = spec
    t0 = time.time()
    if seed_kind == "planar":
        comp = seed_planar_11()
        lx, lz = [{0, 1}], [{0}]
    elif seed_kind == "rottor":
        comp = seed_rot_torus_422()
        lx, lz = [{0, 1}, {0, 2}], [{1, 3}, {2, 3}]
    elif seed_kind == "tor1":
        comp = seed_torus_1face()
        lx, lz = [{0}, {1}], [{0}, {1}]
    else:
        raise ValueError(seed_kind)

    cyl = set()
    pf_maps = []
    f_c1 = 0
    for step_idx, (L, rb, it) in enumerate(sched, 1):
        pcomp = comp
        comp, cyl, meta, c2pf = apply_step_with_parent_faces(pcomp, L, rb, it, prev_cyl=(cyl if aniso else None))
        pf_maps.append(c2pf)
        N, HX, HZ, seam_qs, *_ = sheaf_to_css_with_seams(comp)
        lx = [push_X_logical(pcomp, comp, meta, HZ, x) for x in lx]
        lz = [push_Z_logical(pcomp, comp, meta, HX, z) for z in lz]
        if step_idx == 2:
            f_c1 = len(comp.faces)

    verify_and_write_code(N, HX, HZ, lx, lz, name, ["X_err", "Z_err"])

    # If 2-cycle (len(sched) == 4), export exact .c1map for X_err (M_check = len(HZ)):
    if len(sched) == 4:
        m3 = pf_maps[2]  # f3 -> f2 (in 0..f_c1-1)
        m4 = pf_maps[3]  # f4 -> f3
        c1_map = [m3[m4[f4]] for f4 in range(len(HZ))]
        assert len(c1_map) == len(HZ)
        assert max(c1_map) == f_c1 - 1 and min(c1_map) == 0
        c1_p = os.path.join(DATA_DIR, f"{name}_X_err.c1map")
        with open(c1_p, "w") as f:
            f.write(f"{len(c1_map)}\n")
            f.write(" ".join(map(str, c1_map)) + "\n")

    wx_hist = {str(k): v for k, v in sorted(Counter(len(r) for r in HX).items())}
    wz_hist = {str(k): v for k, v in sorted(Counter(len(r) for r in HZ).items())}
    qdeg = [0] * N
    for r in HX:
        for q in r:
            qdeg[q] += 1
    for r in HZ:
        for q in r:
            qdeg[q] += 1
    deg_hist = {str(k): v for k, v in sorted(Counter(qdeg).items())}

    info = {
        "name": name,
        "seed": seed_kind,
        "sched": [(L, list(rb), it) for (L, rb, it) in sched],
        "aniso": aniso,
        "N": N,
        "K": len(lx),
        "M_X": len(HX),
        "M_Z": len(HZ),
        "F_C1": f_c1 if len(sched) == 4 else len(HZ),
        "seam_qubit_fraction": len(seam_qs) / float(N),
        "max_wX": max((len(r) for r in HX), default=0),
        "max_wZ": max((len(r) for r in HZ), default=0),
        "max_deg": max(qdeg, default=0),
        "pushed_wLX": [len(x) for x in lx],
        "pushed_wLZ": [len(z) for z in lz],
        "wX_hist": wx_hist,
        "wZ_hist": wz_hist,
        "deg_hist": deg_hist,
        "build_sec": round(time.time() - t0, 2),
    }
    return info


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    C3 = [(3, [1], "X"), (3, [1], "Z")]
    R3 = [(3, [1, 2], "X"), (3, [1, 2], "Z")]
    C2 = [(2, [1], "X"), (2, [1], "Z")]

    specs = [
        # Suite 1: 1-cycle modules
        ("v15_sq_C3",         "planar", C3, True),
        ("v15_sq_R3",         "planar", R3, True),
        ("v15_tor_C3",        "tor1",   C3, True),
        ("v15_tor_R3",        "tor1",   R3, True),
        ("v15_sq_C3_iso",     "planar", C3, False),
        ("v15_sq_R3_iso",     "planar", R3, False),
        ("v15_sq_C2",         "planar", C2, True),
        ("v15_tor_C2",        "tor1",   C2, True),
        ("v15_rottor_C2",     "rottor", C2, True),
        ("v15_rottor_R3",     "rottor", R3, True),

        # Suite 2A: Pure L=3 2-cycle (CC, RC, CR, RR) on Planar & Torus
        ("v15_sq_L3_CC",      "planar", C3 + C3, True),
        ("v15_sq_L3_RC",      "planar", R3 + C3, True),
        ("v15_sq_L3_CR",      "planar", C3 + R3, True),
        ("v15_sq_L3_RR",      "planar", R3 + R3, True),
        ("v15_tor_L3_CC",     "tor1",   C3 + C3, True),
        ("v15_tor_L3_RC",     "tor1",   R3 + C3, True),
        ("v15_tor_L3_CR",     "tor1",   C3 + R3, True),
        ("v15_tor_L3_RR",     "tor1",   R3 + R3, True),

        # Suite 2B: Unit-patch package 2-cycle (CC, RC, CR) on Planar, 1-Face Torus, Rotated Torus, & Isotropic Planar
        ("v15_sq_pkg_CC",     "planar", C2 + C2, True),
        ("v15_sq_pkg_RC",     "planar", R3 + C2, True),
        ("v15_sq_pkg_CR",     "planar", C2 + R3, True),
        ("v15_tor_pkg_CC",    "tor1",   C2 + C2, True),
        ("v15_tor_pkg_RC",    "tor1",   R3 + C2, True),
        ("v15_tor_pkg_CR",    "tor1",   C2 + R3, True),
        ("v15_rottor_pkg_CC", "rottor", C2 + C2, True),
        ("v15_rottor_pkg_RC", "rottor", R3 + C2, True),
        ("v15_rottor_pkg_CR", "rottor", C2 + R3, True),
        ("sq_iso_L2r1_s4_v15","planar", C2 + C2, False),
        ("v15_sq_iso_RC",     "planar", R3 + C2, False),
        ("v15_sq_iso_CR",     "planar", C2 + R3, False),
    ]

    all_info = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for info in ex.map(build_and_characterize, specs):
            all_info[info["name"]] = info
            print(f"Ready: {info['name']:20s} N={info['N']:7,d} K={info['K']} F_C1={info['F_C1']:5,d} ({info['build_sec']}s)", flush=True)

    meta_path = os.path.join(WORK_DIR, "v15_code_specs.json")
    with open(meta_path, "w") as f:
        json.dump(all_info, f, indent=2)
    print(f"Saved {meta_path}")

    # Build load-balanced manifest_v15.txt (weighted task cost assignment across 96 shards):
    weighted_tasks = []  # (est_cost, task_str)

    def add_kmc(tag, sec, b, trials, cap, cost):
        cp = os.path.join(DATA_DIR, f"{tag}_{sec}.code")
        weighted_tasks.append((cost, f"kmc__{tag}_{sec}__b{b:.2f}|kmc|{cp}|uniform|uf|{b:.2f}|{trials}|{cap}"))

    def add_span(tag, b, trials, cap, cost):
        cp = os.path.join(DATA_DIR, f"{tag}_X_err.code")
        mp = os.path.join(DATA_DIR, f"{tag}_X_err.c1map")
        weighted_tasks.append((cost, f"span__{tag}_X_err__b{b:.2f}|span_kmc|{cp}|uniform|{mp}|{b:.2f}|{trials}|{cap}"))

    def add_cadence(tag, b, trials, cap, min_eta, cost):
        cp = os.path.join(DATA_DIR, f"{tag}_X_err.code")
        weighted_tasks.append((cost, f"cadence__{tag}_X_err__b{b:.2f}|paired_cadence|{cp}|{b:.2f}|{trials}|{cap}|{min_eta:.2f}"))

    # 1. Suite 1: All 1-cycle modules across matched beta grid [4.50..7.25] (128 trials, fast!)
    betas_1cyc = [4.50, 5.00, 5.25, 5.50, 5.75, 6.00, 6.25, 6.50, 6.75, 7.00, 7.25]
    for mtag in [
        "v15_sq_C3", "v15_sq_R3", "v15_tor_C3", "v15_tor_R3",
        "v15_sq_C3_iso", "v15_sq_R3_iso",
        "v15_sq_C2", "v15_tor_C2", "v15_rottor_C2", "v15_rottor_R3",
    ]:
        for b in betas_1cyc:
            add_kmc(mtag, "X_err", b, 128, 80000000, 2.0)
            add_kmc(mtag, "Z_err", b, 64, 60000000, 2.0)

    # 2. Suite 2A: Pure L=3 2-cycle (CC, RC, CR, RR) on Planar & Periodic Torus
    # Use 24 trials up to beta=6.25, 16 trials at 6.50..7.00 so every task finishes cleanly within shard walltime!
    betas_L3_2cyc = [4.50, 5.00, 5.50, 5.75, 6.00, 6.25, 6.50, 6.75, 7.00]
    for ltag in [
        "v15_sq_L3_CC", "v15_sq_L3_RC", "v15_sq_L3_CR", "v15_sq_L3_RR",
        "v15_tor_L3_CC", "v15_tor_L3_RC", "v15_tor_L3_CR", "v15_tor_L3_RR",
    ]:
        for b in betas_L3_2cyc:
            tr_x = 24 if b <= 6.25 else 16
            cap_x = 160000000 if b <= 6.25 else 240000000
            est_x = 12.0 if b <= 5.75 else (25.0 if b <= 6.25 else (55.0 if b <= 6.50 else 95.0))
            add_span(ltag, b, tr_x, cap_x, est_x)
            add_kmc(ltag, "Z_err", b, 12, 40000000, 20.0)

    # 3. Suite 2B: Unit-patch package 2-cycle (CC, RC, CR) on Planar, 1-Face Torus, Rotated Torus, & Isotropic Planar
    betas_pkg = [4.50, 5.00, 5.50, 5.75, 6.00, 6.25, 6.50, 6.75, 7.00]
    for ptag in [
        "v15_sq_pkg_CC", "v15_sq_pkg_RC", "v15_sq_pkg_CR",
        "v15_tor_pkg_CC", "v15_tor_pkg_RC", "v15_tor_pkg_CR",
        "v15_rottor_pkg_CC", "v15_rottor_pkg_RC", "v15_rottor_pkg_CR",
    ]:
        for b in betas_pkg:
            tr_x = 32 if b <= 6.25 else 20
            cap_x = 160000000 if b <= 6.25 else 240000000
            est_x = 10.0 if b <= 5.75 else (22.0 if b <= 6.25 else (45.0 if b <= 6.50 else 75.0))
            add_span(ptag, b, tr_x, cap_x, est_x)
            add_kmc(ptag, "Z_err", b, 16, 40000000, 14.0)

    for itag in ["sq_iso_L2r1_s4_v15", "v15_sq_iso_RC", "v15_sq_iso_CR"]:
        for b in [4.50, 5.00, 5.50, 6.00, 6.25, 6.50]:
            add_span(itag, b, 24, 180000000, 25.0)
            add_kmc(itag, "Z_err", b, 12, 40000000, 16.0)

    # 4. Suite 3: Paired cadence audit on RC vs CR (eta in {0.02, 0.05, 0.20, 0.50})
    for ctag in ["v15_sq_pkg_RC", "v15_sq_pkg_CR", "v15_sq_L3_RC", "v15_sq_L3_CR"]:
        for b in [6.00, 6.50]:
            add_cadence(ctag, b, 20, 180000000, 0.02, 45.0)

    # Greedy LPT bin-packing across N_SHARDS = 112 shards:
    N_SHARDS = 112
    shard_loads = [0.0] * N_SHARDS
    shard_bins = [[] for _ in range(N_SHARDS)]
    for cost, tstr in sorted(weighted_tasks, key=lambda x: -x[0]):
        s_min = min(range(N_SHARDS), key=lambda i: shard_loads[i])
        shard_loads[s_min] += cost
        shard_bins[s_min].append(tstr)

    man_p = os.path.join(WORK_DIR, "manifest_v15.txt")
    with open(man_p, "w") as f:
        for sid in range(N_SHARDS):
            for tstr in shard_bins[sid]:
                f.write(f"{sid}|{tstr}\n")

    print(f"Wrote {len(weighted_tasks)} tasks across {N_SHARDS} balanced shards to {man_p} "
          f"(load range: {min(shard_loads):.1f} .. {max(shard_loads):.1f})")


if __name__ == "__main__":
    main()
