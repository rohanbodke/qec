#!/usr/bin/env python3
"""
build_v23_extensionA.py
Builds and verifies the V23 Extension A + Regime 1/3 campaign using `kmc_engine_v19`
(`prep_logicals`, `verify`, and 2-trial `self_test` gate) and emits a 64-shard SLURM
manifest `manifest_v23.txt` + `v23_code_specs.json`.

Two strictly 3D-local families:
  A1. BDL T-junction dispersed (`jog1`, zero d=4 edge pillars) 4.8.8 welded tori (K = 4):
      - `v23_disp_1s_L3` (N=200), `v23_disp_2s_L3` (N=3,992), `v23_disp_2s_RX3` (N=4,272),
        and `v23_disp_3s_L3` (N=80,648) + beta=7.0 comparison on `v22_cc3s_X3r1Z3r2X3r1`.
  A2. Open-boundary planar `css_dual` 4.8.8 color codes (constant K = 2, Regime 3):
      - `v23_plan_cc2d_L6` (288), `v23_plan_cc2d_L12` (1,152), `v23_plan_cc1s_L3r2` (240),
        `v23_plan_disp_2s_L3` (3,992), `v23_plan_cc2s_RX3` (4,128), `v23_plan_cc2s_R3` (5,088).
"""

from collections import defaultdict
import heapq
import json
import math
import os
import subprocess
import sys

WORK_DIR = "/work/submit/harshila/6s980_qmem_v9"
DATA_DIR = os.path.join(WORK_DIR, "data_v23")
DATA_V22_DIR = os.path.join(WORK_DIR, "data_v22")
ENGINE = os.path.join(WORK_DIR, "kmc_engine_v19")
sys.path.insert(0, WORK_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_v7_rescue import GeneralCubical2D, seed_planar_11
from build_v18_rework import gf2_rank_bigint
from build_v21_3d_local import seed_torus_1face, apply_step_L3_general


def _step_3d_positions(comp, v_pos, L, step_idx):
    """Apply golden-lattice sub-voxel dispersion and Lx spatial scaling to vertex coordinates."""
    voxel_buckets = defaultdict(list)
    for vid, p in enumerate(v_pos):
        vox = (int(math.floor(p[0] + 1e-9)), int(math.floor(p[1] + 1e-9)), int(math.floor(p[2] + 1e-9)))
        voxel_buckets[vox].append(vid)
    scaled = [None] * comp.num_vertices
    for vids in voxel_buckets.values():
        m = len(vids)
        for idx, vid in enumerate(vids):
            px, py, pz = v_pos[vid]
            if m == 1:
                eta = (0.0, 0.0, 0.0)
            else:
                u1 = ((idx + 1) * 0.61803398875) % 1.0 - 0.5
                u2 = ((idx + 1) * 0.38196601125) % 1.0 - 0.5
                u3 = ((idx + 1) * 0.75487766625) % 1.0 - 0.5
                eta = (0.60 * u1, 0.60 * u2, 0.60 * u3)
            scaled[vid] = (L * px + eta[0], L * py + eta[1], L * pz + eta[2])
    return scaled


def _shift_vec(step_idx):
    if step_idx == 1:
        return (0.0, 0.0, 1.0)
    if step_idx == 2:
        return (0.35, 0.25, 0.90)
    return (-0.25, 0.35, 0.85)


def apply_step_dispersed(comp, L, rib_mode, iter_type, prev_cyl=None, v_pos=None, step_idx=1):
    """
    BDL T-junction dispersed step (`rib_mode='jog1'` for L=3) or standard `(1,2)` step (`rib_mode='r2'`).
    When `rib_mode=='jog1'`, splits every 4-way interior cross into two degree-3 T-junctions so no edge
    ever exceeds face-degree d = 3, capping check weights while keeping O(1) R^3 check diameter.
    """
    if rib_mode == "r2":
        if v_pos is not None:
            raise ValueError("Use verify_3d_local_embeddings_and_L3.apply_step_3d_tracked for r2 3D tracking")
        res, new_cyl, _ = apply_step_L3_general(comp, L, [1, 2], iter_type, prev_cyl=prev_cyl)
        return res, new_cyl, None

    assert rib_mode == "jog1" and L == 3, f"Unsupported rib_mode={rib_mode}, L={L}"
    new_col = "blue" if iter_type == "X" else "red"
    prev_cyl = prev_cyl or set()
    xp_num_v = 0
    xp_pos = [] if v_pos is not None else None

    def add_v(pos=None):
        nonlocal xp_num_v
        r = xp_num_v
        xp_num_v += 1
        if xp_pos is not None:
            xp_pos.append(pos)
        return r

    scaled = _step_3d_positions(comp, v_pos, L, step_idx) if v_pos is not None else [None] * comp.num_vertices
    orig_v = [add_v(scaled[v]) for v in range(comp.num_vertices)]

    xp_edges, xp_colors, xp_is_rib = [], [], []
    edge_sub_verts, edge_sub_eids, edge_L = [], [], []
    xp_vert_orig_edge = {}

    for e_idx, (a, b) in enumerate(comp.edges):
        Le = 1 if (e_idx in prev_cyl) else L
        edge_L.append(Le)
        mids = []
        if v_pos is not None:
            pa, pb = xp_pos[orig_v[a]], xp_pos[orig_v[b]]
            for k in range(1, Le):
                fr = k / Le
                mids.append(add_v((pa[0] + fr * (pb[0] - pa[0]), pa[1] + fr * (pb[1] - pa[1]), pa[2] + fr * (pb[2] - pa[2]))))
        else:
            mids = [add_v() for _ in range(Le - 1)]
        pts = [orig_v[a]] + mids + [orig_v[b]]
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
        if b == sv and a == ev:
            assert Le == 1, f"Asymmetric jog1 cannot flip edge {e_idx} with Le={Le}"
            return list(reversed(pts)), list(reversed(eids)), Le
        raise ValueError("Endpoint mismatch")

    xp_faces = []
    for (v00, v10, v11, v01, eb, et, el, er) in comp.faces:
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
        if v_pos is not None:
            p00, p10, p11, p01 = xp_pos[pb[0]], xp_pos[pb[Lx]], xp_pos[pt[Lx]], xp_pos[pt[0]]
        for s in range(1, Lx):
            for t in range(1, Ly):
                pos = None
                if v_pos is not None:
                    u, v = s / Lx, t / Ly
                    pos = tuple(
                        (1 - u) * (1 - v) * p00[k] + u * (1 - v) * p10[k] + u * v * p11[k] + (1 - u) * v * p01[k]
                        for k in range(3)
                    )
                grid[s][t] = add_v(pos)

        he = [[None] * (Ly + 1) for _ in range(Lx)]
        ve = [[None] * Ly for _ in range(Lx + 1)]
        for s in range(Lx):
            he[s][0] = eb_ids[s]
            he[s][Ly] = et_ids[s]
        for t in range(Ly):
            ve[0][t] = el_ids[t]
            ve[Lx][t] = er_ids[t]

        h_rib_set, v_rib_set = set(), set()
        if Lx == L and Ly == L:
            for s in range(Lx):
                h_rib_set.add((s, 1))
            v_rib_set.add((1, 0))
            v_rib_set.add((2, 1))
            h_rib_set.add((1, 2))
            for t in range(2, Ly):
                v_rib_set.add((1, t))
        elif Lx == L and Ly == 1:
            v_rib_set.add((1, 0))
        elif Lx == 1 and Ly == L:
            h_rib_set.add((0, 1))

        for t in range(1, Ly):
            for s in range(Lx):
                eid = len(xp_edges)
                xp_edges.append((grid[s][t], grid[s + 1][t]))
                xp_colors.append("none")
                xp_is_rib.append((s, t) in h_rib_set)
                he[s][t] = eid
        for s in range(1, Lx):
            for t in range(Ly):
                eid = len(xp_edges)
                xp_edges.append((grid[s][t], grid[s][t + 1]))
                xp_colors.append("none")
                xp_is_rib.append((s, t) in v_rib_set)
                ve[s][t] = eid
        for s in range(Lx):
            for t in range(Ly):
                xp_faces.append((
                    grid[s][t], grid[s + 1][t], grid[s + 1][t + 1], grid[s][t + 1],
                    he[s][t], he[s][t + 1], ve[s][t], ve[s + 1][t],
                ))

    tot_v = 2 * xp_num_v
    def vc(v, c):
        return v + c * xp_num_v

    out_pos = None
    if xp_pos is not None:
        sv = _shift_vec(step_idx)
        out_pos = list(xp_pos) + [(p[0] + sv[0], p[1] + sv[1], p[2] + sv[2]) for p in xp_pos]

    out_e, out_c, out_f = [], [], []
    n_xe = len(xp_edges)
    for c in (0, 1):
        for eid, (u, w) in enumerate(xp_edges):
            out_e.append((vc(u, c), vc(w, c)))
            out_c.append(new_col if xp_is_rib[eid] else xp_colors[eid])
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
    return res, new_cyl, out_pos


def build_flag_color_css_bdry(comp, v_pos=None, seam_mode="welded", bdry_mode="css_dual"):
    """
    Build 4.8.8 flag color code with `css_dual` open boundaries (blue d=1 rough boundaries
    carry weight-2 X-checks, red d=1 smooth boundaries carry weight-2 Z-checks) so planar
    seeds preserve constant K = 2 and higher boundary barriers than color boundaries.
    """
    flags = []
    fid_map = {}
    vert_flags = [[] for _ in range(comp.num_vertices)]
    q_pos = []
    edge_mids = None
    if v_pos is not None:
        edge_mids = [
            (0.5 * (v_pos[u][0] + v_pos[w][0]), 0.5 * (v_pos[u][1] + v_pos[w][1]), 0.5 * (v_pos[u][2] + v_pos[w][2]))
            for (u, w) in comp.edges
        ]

    for f_idx, (v00, v10, v11, v01, eb, et, el, er) in enumerate(comp.faces):
        fc = None
        if v_pos is not None:
            pvs = [v_pos[v] for v in (v00, v10, v11, v01)]
            fc = tuple(0.25 * sum(p[k] for p in pvs) for k in range(3))
        for v, e in [(v00, eb), (v00, el), (v10, eb), (v10, er), (v11, et), (v11, er), (v01, et), (v01, el)]:
            qid = len(flags)
            fid_map[(v, e, f_idx)] = qid
            flags.append((v, e, f_idx))
            vert_flags[v].append(qid)
            if v_pos is not None:
                pv = v_pos[v]
                em = edge_mids[e]
                q_pos.append(tuple(0.70 * pv[k] + 0.15 * em[k] + 0.15 * fc[k] for k in range(3)))

    N = len(flags)
    HX, HZ = [], []
    for f_idx in range(len(comp.faces)):
        row = list(range(8 * f_idx, 8 * (f_idx + 1)))
        HX.append(row)
        HZ.append(list(row))

    for e_idx, inc in enumerate(comp.edge_faces):
        d = len(inc)
        u, w = comp.edges[e_idx]
        col = comp.edge_colors[e_idx]
        if d == 1:
            row = sorted([fid_map[(u, e_idx, inc[0])], fid_map[(w, e_idx, inc[0])]])
            if bdry_mode == "color":
                if col == "blue":
                    HX.append(row)
                    HZ.append(list(row))
            elif bdry_mode == "css_dual":
                if col == "blue":
                    HX.append(row)
                else:
                    HZ.append(row)
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
                HX.extend(pair_rows)
                HZ.append(sum_row)
            else:
                HZ.extend(pair_rows)
                HX.append(sum_row)

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
                    if bdry_mode == "color":
                        if col == "blue":
                            cons.append(1 << f_to_i[inc[0]])
                    elif bdry_mode == "css_dual":
                        if (same_col == "red" and col == "red") or (same_col == "blue" and col == "blue"):
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
                target_H.append(sorted([qi for qi in v_qids if flags[qi][2] in fset]))

    return N, HX, HZ, q_pos


def build_2d_488_planar(L):
    """Build an open-boundary L x L 4.8.8 color-code patch (N = 8 L^2, K = 2)."""
    def vid(x, y):
        return y * (L + 1) + x

    edges, cols = [], []
    he, ve = {}, {}
    for y in range(L + 1):
        for x in range(L):
            he[(x, y)] = len(edges)
            edges.append((vid(x, y), vid(x + 1, y)))
            cols.append("blue" if y in (0, L) else "none")
    for x in range(L + 1):
        for y in range(L):
            ve[(x, y)] = len(edges)
            edges.append((vid(x, y), vid(x, y + 1)))
            cols.append("red" if x in (0, L) else "none")
    faces = []
    for y in range(L):
        for x in range(L):
            faces.append((
                vid(x, y), vid(x + 1, y), vid(x + 1, y + 1), vid(x, y + 1),
                he[(x, y)], he[(x, y + 1)], ve[(x, y)], ve[(x + 1, y)],
            ))
    comp = GeneralCubical2D((L + 1) * (L + 1), edges, cols, faces)
    N, HX, HZ, _ = build_flag_color_css_bdry(comp, seam_mode="selfdual", bdry_mode="css_dual")
    return N, HX, HZ, comp


def build_complex_from_steps(seed_kind, steps):
    comp = seed_torus_1face() if seed_kind == "tor1" else seed_planar_11()
    cyl = set()
    L_blk = 1
    for (L, mode_or_rb, it) in steps:
        if mode_or_rb in ("jog1", "r2"):
            comp, cyl, _ = apply_step_dispersed(comp, L, mode_or_rb, it, prev_cyl=cyl)
        else:
            comp, cyl, _ = apply_step_L3_general(comp, L, mode_or_rb, it, prev_cyl=cyl)
        L_blk *= L
    return comp, L_blk


def build_code_job(job):
    tag, seed_kind, steps, k_exp = job
    if seed_kind == "plan2d":
        L_blk = steps
        N, HX, HZ, comp = build_2d_488_planar(L_blk)
        bdry_mode = "css_dual"
    else:
        comp, L_blk = build_complex_from_steps(seed_kind, steps)
        bdry_mode = "css_dual" if seed_kind == "planar" else "color"
        N, HX, HZ, _ = build_flag_color_css_bdry(comp, seam_mode="welded", bdry_mode=bdry_mode)

    if N <= 6000:
        rX = gf2_rank_bigint(HX)
        rZ = gf2_rank_bigint(HZ)
        k_tot = N - rX - rZ
        assert k_tot == k_exp, f"{tag}: got K={k_tot} != {k_exp}"
    else:
        rX, rZ = -1, -1

    for sec, checks, stabs in [("X_err", HZ, HX), ("Z_err", HX, HZ)]:
        raw_p = os.path.join(DATA_DIR, f"{tag}_{sec}.raw")
        cpath = os.path.join(DATA_DIR, f"{tag}_{sec}.code")
        with open(raw_p, "w") as out:
            out.write(f"{N} {len(checks)} {len(stabs)}\n")
            for r in checks:
                out.write(str(len(r)) + " " + " ".join(map(str, r)) + "\n")
            for r in stabs:
                out.write(str(len(r)) + " " + " ".join(map(str, r)) + "\n")
        subprocess.run([ENGINE, "prep_logicals", raw_p, cpath], check=True, stdout=subprocess.DEVNULL)
        os.remove(raw_p)
        vout = subprocess.check_output([ENGINE, "verify", cpath], text=True).strip()
        assert f"K={k_exp}" in vout and "bad_comm=0" in vout, f"{tag}_{sec}: {vout}"
        subprocess.run([ENGINE, "self_test", cpath, "4.5", "2", "1", "1.0", "2.0", "2", "50000", "2"],
                       check=True, stdout=subprocess.DEVNULL)

    F = len(comp.faces)
    dz = [0] * N
    for r in HZ:
        for q in r:
            dz[q] += 1
    dx = [0] * N
    for r in HX:
        for q in r:
            dx[q] += 1
    d4_edges = sum(1 for inc in comp.edge_faces if len(inc) >= 4)
    return tag, {
        "tag": tag,
        "seed_kind": seed_kind,
        "N": N,
        "F": F,
        "M_X": len(HX),
        "M_Z": len(HZ),
        "rank_HX": rX,
        "rank_HZ": rZ,
        "K": k_exp,
        "L_blk": L_blk,
        "dim_F": (math.log(F) / math.log(L_blk)) if L_blk > 1 else 0.0,
        "d4_edge_count": d4_edges,
        "max_edge_deg": max(len(inc) for inc in comp.edge_faces),
        "max_wX": max(len(r) for r in HX),
        "max_wZ": max(len(r) for r in HZ),
        "mean_wX": sum(len(r) for r in HX) / len(HX),
        "mean_wZ": sum(len(r) for r in HZ) / len(HZ),
        "min_dX": min(dx),
        "max_dX": max(dx),
        "min_dZ": min(dz),
        "max_dZ": max(dz),
        "gamma_X_b50": sum(1.0 / (1.0 + math.exp(5.0 * d)) for d in dz if d > 0),
        "gamma_Z_b50": sum(1.0 / (1.0 + math.exp(5.0 * d)) for d in dx if d > 0),
        "gamma_X_b65": sum(1.0 / (1.0 + math.exp(6.5 * d)) for d in dz if d > 0),
        "gamma_Z_b65": sum(1.0 / (1.0 + math.exp(6.5 * d)) for d in dx if d > 0),
    }


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(WORK_DIR, "results_v23"), exist_ok=True)
    os.makedirs(os.path.join(WORK_DIR, "logs_v23"), exist_ok=True)

    jobs = [
        # Extension A1: BDL T-junction dispersed (`d <= 3` everywhere on edge-faces) torus K = 4
        ("v23_disp_1s_L3",      "tor1",   [(3, "jog1", "X")],                                  4),
        ("v23_disp_2s_L3",      "tor1",   [(3, "jog1", "X"), (3, "jog1", "Z")],                 4),
        ("v23_disp_2s_RX3",     "tor1",   [(3, "r2",   "X"), (3, "jog1", "Z")],                 4),
        ("v23_disp_3s_L3",      "tor1",   [(3, "jog1", "X"), (3, "jog1", "Z"), (3, "jog1", "X")], 4),
        # Extension A2: Open-boundary planar `css_dual` 4.8.8 color codes (Regime 3: constant K = 2)
        ("v23_plan_cc2d_L6",    "plan2d", 6,                                                    2),
        ("v23_plan_cc2d_L12",   "plan2d", 12,                                                   2),
        ("v23_plan_cc1s_L3r2",  "planar", [(3, "r2",   "X")],                                   2),
        ("v23_plan_disp_2s_L3", "planar", [(3, "jog1", "X"), (3, "jog1", "Z")],                 2),
        ("v23_plan_cc2s_RX3",   "planar", [(3, "r2",   "X"), (3, "jog1", "Z")],                 2),  # wait -> matches RX3 using r2+1:
        ("v23_plan_cc2s_R3",    "planar", [(3, "r2",   "X"), (3, "r2",   "Z")],                 2),
    ]
    # Keep `v23_plan_cc2s_RX3` exact standard RX3 = [(3, [1, 2], 'X'), (3, [1], 'Z')] so torus vs planar matches!
    jobs[8] = ("v23_plan_cc2s_RX3", "planar", [(3, [1, 2], "X"), (3, [1], "Z")], 2)

    specs = {}
    for job in jobs:
        tag, info = build_code_job(job)
        specs[tag] = info
        print(
            f"  [BUILD] {tag:22s}: N={info['N']:7,d} K={info['K']} d4_edges={info['d4_edge_count']:4d} "
            f"w_max=({info['max_wX']},{info['max_wZ']}) d_min=({info['min_dX']},{info['min_dZ']}) dimF={info['dim_F']:.3f}",
            flush=True,
        )

    # Also add metadata entry for existing V22 cc3s_121 beta=7.0 comparison run
    specs["v23_cc3s_121_ref"] = {"tag": "v23_cc3s_121_ref", "seed_kind": "tor1", "N": 81888, "K": 4, "L_blk": 27, "dim_F": 2.8016}

    with open(os.path.join(WORK_DIR, "v23_code_specs.json"), "w") as f:
        json.dump(specs, f, indent=2)

    tasks = []

    def add_th(tag, code_dir, code_stem, beta, n_tot, per_task, lg0, lg1, ng, max_fl, w_mult=1.0):
        N = specs[tag]["N"]
        base_w = (max(1.0, N / 1000.0) ** 1.05) * (max_fl / 4e7) * (per_task / 16.0) * w_mult
        for sec in ("X_err", "Z_err"):
            cpath = os.path.join(code_dir, f"{code_stem}_{sec}.code")
            for off in range(0, n_tot, per_task):
                nt = min(per_task, n_tot - off)
                tid = f"{tag}_{sec}__b{beta:.2f}__o{off}"
                tasks.append((base_w, f"{tid}|paired_readout|{cpath}|{beta:.5f}|{nt}|{off}|{lg0:.4f}|{lg1:.4f}|{ng}|{int(max_fl)}"))

    def add_iid(tag, n_tot=64, per_task=16, p0=0.005, p1=0.115, ng=23):
        N = specs[tag]["N"]
        base_w = max(5.0, N / 400.0)
        for sec in ("X_err", "Z_err"):
            cpath = os.path.join(DATA_DIR, f"{tag}_{sec}.code")
            for off in range(0, n_tot, per_task):
                nt = min(per_task, n_tot - off)
                tid = f"v23_iid__{tag}_{sec}__o{off}"
                tasks.append((base_w, f"{tid}|iid_readout|{cpath}|{nt}|{off}|{p0:.5f}|{p1:.5f}|{ng}"))

    # 1. Small 1s & planar 2D/1s controls (include beta=7.0 on tiny N=200 disp_1s for Regime 1 same-parity b=7.0!)
    grid_sm = [(4.5, 96, 32, 1.5, 7.5, 25, 50_000_000),
               (5.0, 96, 32, 2.0, 8.0, 25, 50_000_000),
               (5.5, 96, 32, 3.0, 8.5, 23, 50_000_000),
               (6.0, 96, 32, 3.5, 9.0, 23, 50_000_000),
               (6.5, 96, 32, 4.0, 9.5, 23, 50_000_000),
               (7.0, 96, 32, 4.5, 10.5, 25, 50_000_000)]
    for b, ntot, pt, lg0, lg1, ng, mfl in grid_sm:
        add_th("v23_disp_1s_L3", DATA_DIR, "v23_disp_1s_L3", b, ntot, pt, lg0, lg1, ng, mfl, 0.5)
    for tag in ("v23_plan_cc2d_L6", "v23_plan_cc2d_L12", "v23_plan_cc1s_L3r2"):
        for b, ntot, pt, lg0, lg1, ng, mfl in [grid_sm[1], grid_sm[3], grid_sm[4]]:
            add_th(tag, DATA_DIR, tag, b, ntot, pt, lg0, lg1, ng, mfl, 0.5)

    # 2. 2-step dispersed & open-boundary planar 2-step codes
    grid_2s_tor = [
        (4.5, 64, 32, 2.0, 7.0, 21, 60_000_000),
        (5.0, 96, 32, 2.5, 8.0, 23, 60_000_000),
        (5.5, 64, 32, 3.5, 8.5, 21, 60_000_000),
        (6.0, 64, 32, 4.5, 9.0, 19, 60_000_000),
        (6.5, 96, 32, 5.0, 9.5, 19, 80_000_000),
        (7.0, 64, 16, 5.5, 10.5, 21, 100_000_000),
    ]
    for b, ntot, pt, lg0, lg1, ng, mfl in grid_2s_tor:
        add_th("v23_disp_2s_L3", DATA_DIR, "v23_disp_2s_L3", b, ntot, pt, lg0, lg1, ng, mfl, 1.5)
    for b, ntot, pt, lg0, lg1, ng, mfl in grid_2s_tor[1:5]:
        add_th("v23_disp_2s_RX3", DATA_DIR, "v23_disp_2s_RX3", b, ntot, pt, lg0, lg1, ng, mfl, 1.5)

    grid_2s_plan = [
        (4.5, 64, 32, 1.5, 6.5, 21, 60_000_000),
        (5.0, 96, 32, 2.0, 7.5, 23, 60_000_000),
        (5.5, 64, 32, 3.0, 8.5, 23, 60_000_000),
        (6.0, 64, 32, 3.5, 9.0, 23, 60_000_000),
        (6.5, 96, 32, 4.0, 9.5, 23, 80_000_000),
    ]
    for b, ntot, pt, lg0, lg1, ng, mfl in grid_2s_plan:
        add_th("v23_plan_disp_2s_L3", DATA_DIR, "v23_plan_disp_2s_L3", b, ntot, pt, lg0, lg1, ng, mfl, 1.5)
        add_th("v23_plan_cc2s_RX3",   DATA_DIR, "v23_plan_cc2s_RX3",   b, ntot, pt, lg0, lg1, ng, mfl, 1.5)
    for b, ntot, pt, lg0, lg1, ng, mfl in [grid_2s_plan[1], grid_2s_plan[3], grid_2s_plan[4]]:
        add_th("v23_plan_cc2s_R3", DATA_DIR, "v23_plan_cc2s_R3", b, ntot, pt, lg0, lg1, ng, mfl, 1.5)

    # 3. 3-step dispersed L=3 (N=80,648) & existing V22 cc3s_121 at beta=7.0 comparison
    grid_3s = [
        (4.5, 64, 16, 1.5, 5.5, 17, 60_000_000),
        (5.0, 64, 16, 2.0, 6.8, 20, 60_000_000),
        (5.5, 64, 16, 3.0, 7.8, 20, 60_000_000),
        (6.0, 64, 16, 4.0, 8.8, 20, 60_000_000),
        (6.5, 64, 16, 5.0, 9.5, 19, 80_000_000),
        (7.0, 32, 16, 5.5, 10.5, 21, 100_000_000),
    ]
    for b, ntot, pt, lg0, lg1, ng, mfl in grid_3s:
        add_th("v23_disp_3s_L3", DATA_DIR, "v23_disp_3s_L3", b, ntot, pt, lg0, lg1, ng, mfl, 2.5)
    add_th("v23_cc3s_121_ref", DATA_V22_DIR, "v22_cc3s_X3r1Z3r2X3r1", 7.0, 32, 16, 5.5, 10.5, 21, 100_000_000, 2.5)

    # 4. Paired i.i.d. threshold scans
    add_iid("v23_disp_2s_L3")
    add_iid("v23_plan_cc2s_RX3")

    n_shards = 64
    bins = [(0.0, i) for i in range(n_shards)]
    heapq.heapify(bins)
    assigned = [[] for _ in range(n_shards)]
    for w, line in sorted(tasks, key=lambda x: -x[0]):
        load, sh = heapq.heappop(bins)
        assigned[sh].append(line)
        heapq.heappush(bins, (load + w, sh))

    mpath = os.path.join(WORK_DIR, "manifest_v23.txt")
    with open(mpath, "w") as out:
        for sh in range(n_shards):
            for line in assigned[sh]:
                out.write(f"{sh}|{line}\n")
    print(f"Wrote {len(tasks)} tasks across {n_shards} shards to {mpath}")


if __name__ == "__main__":
    main()
