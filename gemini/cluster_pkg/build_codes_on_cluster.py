#!/usr/bin/env python3
"""
Runs on MIT SubMIT cluster (inside Slurm prep job) to generate all benchmark codes,
exact dual logicals, RG unit hierarchies (.rg), and the task manifest for the Slurm array!
"""

import os
import random
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


class OrientedCubical2D:
    def __init__(self, num_vertices, edges, edge_colors, faces, parent_face=None):
        self.num_vertices = num_vertices
        self.edges = edges
        self.edge_colors = edge_colors
        self.faces = faces
        self.parent_face = parent_face

        self.edge_faces = [[] for _ in range(len(edges))]
        self.vert_faces = [[] for _ in range(num_vertices)]
        self.vert_edges = [[] for _ in range(num_vertices)]

        for e_idx, (u, v) in enumerate(edges):
            self.vert_edges[u].append(e_idx)
            if v != u:
                self.vert_edges[v].append(e_idx)

        for f_idx, fdata in enumerate(faces):
            v00, v10, v11, v01, eb, et, el, er = fdata
            for e_idx in (eb, et, el, er):
                self.edge_faces[e_idx].append(f_idx)
            for v in {v00, v10, v11, v01}:
                self.vert_faces[v].append(f_idx)


def seed_planar_surface(dx: int, dz: int) -> OrientedCubical2D:
    def vid(ix, iz):
        return iz * (dx + 1) + ix
    num_v = (dx + 1) * (dz + 1)
    edges, colors = [], []
    h_edge, v_edge = {}, {}
    for iz in range(dz + 1):
        for ix in range(dx):
            h_edge[(ix, iz)] = len(edges)
            edges.append((vid(ix, iz), vid(ix + 1, iz)))
            colors.append('blue' if (iz == 0 or iz == dz) else 'none')
    for iz in range(dz):
        for ix in range(dx + 1):
            v_edge[(ix, iz)] = len(edges)
            edges.append((vid(ix, iz), vid(ix, iz + 1)))
            colors.append('red' if (ix == 0 or ix == dx) else 'none')
    faces = []
    for iz in range(dz):
        for ix in range(dx):
            faces.append((
                vid(ix, iz), vid(ix + 1, iz), vid(ix + 1, iz + 1), vid(ix, iz + 1),
                h_edge[(ix, iz)], h_edge[(ix, iz + 1)],
                v_edge[(ix, iz)], v_edge[(ix + 1, iz)],
            ))
    return OrientedCubical2D(num_v, edges, colors, faces)


def seed_torus(Lx: int, Lz: int) -> OrientedCubical2D:
    def vid(ix, iz):
        return (iz % Lz) * Lx + (ix % Lx)
    num_v = Lx * Lz
    edges, colors = [], []
    h_edge, v_edge = {}, {}
    for iz in range(Lz):
        for ix in range(Lx):
            h_edge[(ix, iz)] = len(edges)
            edges.append((vid(ix, iz), vid(ix + 1, iz)))
            colors.append('none')
    for iz in range(Lz):
        for ix in range(Lx):
            v_edge[(ix, iz)] = len(edges)
            edges.append((vid(ix, iz), vid(ix, iz + 1)))
            colors.append('none')
    faces = []
    for iz in range(Lz):
        for ix in range(Lx):
            faces.append((
                vid(ix, iz), vid(ix + 1, iz), vid(ix + 1, iz + 1), vid(ix, iz + 1),
                h_edge[(ix, iz)], h_edge[(ix, (iz + 1) % Lz)],
                v_edge[(ix, iz)], v_edge[((ix + 1) % Lx, iz)],
            ))
    return OrientedCubical2D(num_v, edges, colors, faces)


def apply_sheaf_step(comp: OrientedCubical2D, L: int, rib_coords: list, iter_type: str) -> OrientedCubical2D:
    new_col = 'blue' if iter_type == 'X' else 'red'
    rib_set = set(rib_coords)
    xp_num_v = 0
    def add_xp_v():
        nonlocal xp_num_v
        r = xp_num_v
        xp_num_v += 1
        return r

    orig_v = [add_xp_v() for _ in range(comp.num_vertices)]
    xp_edges, xp_colors, xp_is_rib = [], [], []
    edge_sub_verts, edge_sub_eids = [], []
    xp_vert_orig_edge = {}

    for e_idx, (a, b) in enumerate(comp.edges):
        pts = [orig_v[a]] + [add_xp_v() for _ in range(L - 1)] + [orig_v[b]]
        for v in pts[1:-1]:
            xp_vert_orig_edge[v] = e_idx
        edge_sub_verts.append(pts)
        fdeg = len(comp.edge_faces[e_idx])
        inh_col = comp.edge_colors[e_idx] if fdeg != 2 else 'none'
        eids = []
        for k in range(L):
            eid = len(xp_edges)
            xp_edges.append((pts[k], pts[k + 1]))
            xp_colors.append(inh_col)
            xp_is_rib.append(False)
            eids.append(eid)
        edge_sub_eids.append(eids)

    xp_faces = []
    xp_edge_parent_face = {}
    for f_idx, (v00, v10, v11, v01, eb, et, el, er) in enumerate(comp.faces):
        grid = [[None] * (L + 1) for _ in range(L + 1)]
        for s in range(L + 1):
            grid[s][0] = edge_sub_verts[eb][s]
            grid[s][L] = edge_sub_verts[et][s]
        for t in range(L + 1):
            grid[0][t] = edge_sub_verts[el][t]
            grid[L][t] = edge_sub_verts[er][t]
        for s in range(1, L):
            for t in range(1, L):
                grid[s][t] = add_xp_v()

        h_eids = [[None] * (L + 1) for _ in range(L)]
        v_eids = [[None] * L for _ in range(L + 1)]
        for s in range(L):
            h_eids[s][0] = edge_sub_eids[eb][s]
            h_eids[s][L] = edge_sub_eids[et][s]
        for t in range(L):
            v_eids[0][t] = edge_sub_eids[el][t]
            v_eids[L][t] = edge_sub_eids[er][t]

        for t in range(1, L):
            isr = (t in rib_set)
            for s in range(L):
                eid = len(xp_edges)
                xp_edges.append((grid[s][t], grid[s + 1][t]))
                xp_colors.append('none')
                xp_is_rib.append(isr)
                h_eids[s][t] = eid
                xp_edge_parent_face[eid] = f_idx

        for s in range(1, L):
            isr = (s in rib_set)
            for t in range(L):
                eid = len(xp_edges)
                xp_edges.append((grid[s][t], grid[s][t + 1]))
                xp_colors.append('none')
                xp_is_rib.append(isr)
                v_eids[s][t] = eid
                xp_edge_parent_face[eid] = f_idx

        for s in range(L):
            for t in range(L):
                xp_faces.append((
                    grid[s][t], grid[s + 1][t], grid[s + 1][t + 1], grid[s][t + 1],
                    h_eids[s][t], h_eids[s][t + 1], v_eids[s][t], v_eids[s + 1][t],
                    f_idx
                ))

    total_v = 2 * xp_num_v
    def vc(v, c):
        return v + c * xp_num_v

    out_edges, out_colors, out_faces, parent_face = [], [], [], []
    n_xp_e = len(xp_edges)
    for c in (0, 1):
        for eid, (u, w) in enumerate(xp_edges):
            out_edges.append((vc(u, c), vc(w, c)))
            out_colors.append(new_col if xp_is_rib[eid] else xp_colors[eid])

    for c in (0, 1):
        for (v00, v10, v11, v01, eb, et, el, er, orig_f) in xp_faces:
            off = c * n_xp_e
            out_faces.append((
                vc(v00, c), vc(v10, c), vc(v11, c), vc(v01, c),
                eb + off, et + off, el + off, er + off
            ))
            parent_face.append(orig_f)

    rib_verts = set()
    for eid, isr in enumerate(xp_is_rib):
        if isr:
            u, w = xp_edges[eid]
            rib_verts.add(u); rib_verts.add(w)

    cyl_vert_eid = {}
    for v in sorted(rib_verts):
        ceid = len(out_edges)
        out_edges.append((vc(v, 0), vc(v, 1)))
        if v in xp_vert_orig_edge:
            orig_e = xp_vert_orig_edge[v]
            fdeg = len(comp.edge_faces[orig_e])
            out_colors.append(comp.edge_colors[orig_e] if fdeg != 2 else 'none')
        else:
            out_colors.append(new_col)
        cyl_vert_eid[v] = ceid

    for eid, isr in enumerate(xp_is_rib):
        if isr:
            u, w = xp_edges[eid]
            orig_f = xp_edge_parent_face[eid]
            out_faces.append((
                vc(u, 0), vc(w, 0), vc(w, 1), vc(u, 1),
                eid, eid + n_xp_e, cyl_vert_eid[u], cyl_vert_eid[w]
            ))
            parent_face.append(orig_f)

    res = OrientedCubical2D(total_v, out_edges, out_colors, out_faces, parent_face=parent_face)
    for eid in range(len(res.edges)):
        if len(res.edge_faces[eid]) == 2:
            res.edge_colors[eid] = 'none'
    return res


def sheaf_to_css(comp: OrientedCubical2D):
    num_faces = len(comp.faces)
    edge_qstart, qubits = [], []
    for e_idx, inc_faces in enumerate(comp.edge_faces):
        d = len(inc_faces)
        col = comp.edge_colors[e_idx]
        edge_qstart.append(len(qubits))
        if d == 0:
            pass
        elif d == 2 or col == 'blue':
            qubits.append((e_idx, {f: 1 for f in inc_faces}))
        elif col == 'red':
            for j in range(d - 1):
                qubits.append((e_idx, {inc_faces[j]: 1, inc_faces[j + 1]: 1}))

    num_q = len(qubits)
    H_Z = [[] for _ in range(num_faces)]
    for q_idx, (_, vec) in enumerate(qubits):
        for f_idx in vec:
            H_Z[f_idx].append(q_idx)

    H_X = []
    hx_vert = []
    for v in range(comp.num_vertices):
        v_faces = sorted(set(comp.vert_faces[v]))
        if not v_faces:
            continue
        m = len(v_faces)
        f_to_loc = {f: i for i, f in enumerate(v_faces)}
        constraints = []
        for e_idx in comp.vert_edges[v]:
            inc = comp.edge_faces[e_idx]
            d = len(inc)
            col = comp.edge_colors[e_idx]
            if d == 2 or col == 'blue':
                for j in range(1, d):
                    constraints.append((1 << f_to_loc[inc[0]]) ^ (1 << f_to_loc[inc[j]]))
            elif col == 'red':
                row = 0
                for f in inc:
                    row ^= (1 << f_to_loc[f])
                constraints.append(row)

        pivots = {}
        for r in constraints:
            x = r
            while x:
                lsb = (x & -x).bit_length() - 1
                if lsb in pivots:
                    x ^= pivots[lsb]
                else:
                    pivots[lsb] = x
                    break
        free_vars = [i for i in range(m) if i not in pivots]
        if not free_vars:
            continue
        for p in sorted(pivots.keys(), reverse=True):
            for p2 in pivots:
                if p2 != p and ((pivots[p2] >> p) & 1):
                    pivots[p2] ^= pivots[p]

        for fv in free_vars:
            alpha = (1 << fv)
            for p, prow in pivots.items():
                if (prow >> fv) & 1:
                    alpha |= (1 << p)
            x_supp = []
            for e_idx in comp.vert_edges[v]:
                inc = comp.edge_faces[e_idx]
                d = len(inc)
                col = comp.edge_colors[e_idx]
                q0 = edge_qstart[e_idx]
                if d == 0:
                    continue
                elif d == 2 or col == 'blue':
                    if (alpha >> f_to_loc[inc[0]]) & 1:
                        x_supp.append(q0)
                elif col == 'red':
                    pref = 0
                    for j in range(d - 1):
                        pref ^= ((alpha >> f_to_loc[inc[j]]) & 1)
                        if pref:
                            x_supp.append(q0 + j)
            if x_supp:
                H_X.append(sorted(set(x_supp)))
                hx_vert.append(v)

    return num_q, H_X, H_Z, hx_vert


def gf2_rref_pivots(nq, rows):
    pivots = {}
    for r in rows:
        x = 0
        for c in r:
            x ^= (1 << c)
        while x:
            lsb = (x & -x).bit_length() - 1
            if lsb in pivots:
                x ^= pivots[lsb]
            else:
                pivots[lsb] = x
                break
    for p in sorted(pivots.keys(), reverse=True):
        for p2 in pivots:
            if p2 != p and ((pivots[p2] >> p) & 1):
                pivots[p2] ^= pivots[p]
    return pivots


def compute_dual_logicals(nq, H_check, H_stab):
    stab_pivs = gf2_rref_pivots(nq, H_stab)
    free_cols = [c for c in range(nq) if c not in stab_pivs]
    ker_stab_basis = []
    for fc in free_cols:
        v = (1 << fc)
        for p, prow in stab_pivs.items():
            if (prow >> fc) & 1:
                v |= (1 << p)
        ker_stab_basis.append(v)

    check_pivs = gf2_rref_pivots(nq, H_check)
    quot_pivs = dict(check_pivs)
    dual_logicals = []
    for v in ker_stab_basis:
        x = v
        while x:
            lsb = (x & -x).bit_length() - 1
            if lsb in quot_pivs:
                x ^= quot_pivs[lsb]
            else:
                quot_pivs[lsb] = x
                supp = []
                tmp = v
                while tmp:
                    b = (tmp & -tmp).bit_length() - 1
                    supp.append(b)
                    tmp ^= (1 << b)
                dual_logicals.append(supp)
                break
    return dual_logicals


def write_code_and_rg(path, nq, H_check, H_stab, dual_logicals, rg_levels=None):
    with open(path, "w") as f:
        f.write(f"{nq} {len(H_check)} {len(H_stab)} {len(dual_logicals)}\n")
        for r in H_check:
            f.write(f"{len(r)} " + " ".join(map(str, r)) + "\n")
        for r in H_stab:
            f.write(f"{len(r)} " + " ".join(map(str, r)) + "\n")
        for r in dual_logicals:
            f.write(f"{len(r)} " + " ".join(map(str, r)) + "\n")

    if rg_levels:
        with open(path + ".rg", "w") as f:
            f.write(f"{len(rg_levels)}\n")
            for units in rg_levels:
                nonempty = [u for u in units if u]
                f.write(f"{len(nonempty)}\n")
                for u in nonempty:
                    f.write(f"{len(u)} " + " ".join(map(str, u)) + "\n")


def build_rg_levels_from_history(history_comps, hx_vert):
    """
    Given history [comp_0, comp_1, ..., comp_k] with k >= 1:
      Build RG unit partitions from finest (grouping by comp_{k-1} faces) to coarsest (comp_0 faces)!
    """
    k = len(history_comps) - 1
    if k <= 0:
        return [], []
    comp_k = history_comps[-1]
    n_faces_k = len(comp_k.faces)

    # face_anc[m][f] = ancestor face index in comp_{k - 1 - m} for m = 0 .. k-1
    face_anc = []
    cur = list(comp_k.parent_face)
    face_anc.append(cur)
    for step_back in range(k - 1, 0, -1):
        p_map = history_comps[step_back].parent_face
        cur = [p_map[u] for u in cur]
        face_anc.append(cur)

    rg_Xerr = []
    rg_Zerr = []
    for m, anc in enumerate(face_anc):
        num_u = len(history_comps[k - 1 - m].faces)
        if num_u <= 1 and m == len(face_anc) - 1:
            # A single unit containing the entire code is just global decode (handled at top-level!)
            continue
        ux = [[] for _ in range(num_u)]
        for f_idx, u in enumerate(anc):
            ux[u].append(f_idx)
        rg_Xerr.append(ux)

        uz = [[] for _ in range(num_u)]
        for hx_idx, v in enumerate(hx_vert):
            f0 = min(comp_k.vert_faces[v])
            uz[anc[f0]].append(hx_idx)
        rg_Zerr.append(uz)

    return rg_Xerr, rg_Zerr


def export_sheaf_chain(prefix, seed_comp, schedule):
    """
    schedule: list of (L, ribs, iter_type) steps!
    """
    history = [seed_comp]
    c = seed_comp
    for st in range(len(schedule) + 1):
        if st > 0:
            L, ribs, itype = schedule[st - 1]
            c = apply_sheaf_step(c, L, ribs, itype)
            history.append(c)
        nq, HX, HZ, hx_vert = sheaf_to_css(c)
        if nq > 6000:
            print(f"  [skip RREF >6000] {prefix}_s{st}: n={nq}")
            continue
        logX = compute_dual_logicals(nq, HZ, HX)
        logZ = compute_dual_logicals(nq, HX, HZ)
        rgX, rgZ = build_rg_levels_from_history(history, hx_vert)
        if HZ:
            write_code_and_rg(os.path.join(DATA_DIR, f"{prefix}_s{st}_Xerr.code"), nq, HZ, HX, logX, rgX)
        if HX:
            write_code_and_rg(os.path.join(DATA_DIR, f"{prefix}_s{st}_Zerr.code"), nq, HX, HZ, logZ, rgZ)
        print(f"  Exported {prefix}_s{st}: n={nq}, |HX|={len(HX)}, |HZ|={len(HZ)}, k={len(logX)}, rg_lv={len(rgX)}")


def make_3d_toric_code(L: int):
    def eid_x(x, y, z): return 0 * (L**3) + ((z % L) * L + (y % L)) * L + (x % L)
    def eid_y(x, y, z): return 1 * (L**3) + ((z % L) * L + (y % L)) * L + (x % L)
    def eid_z(x, y, z): return 2 * (L**3) + ((z % L) * L + (y % L)) * L + (x % L)
    nq = 3 * (L ** 3)
    H_X, H_Z = [], []
    for z in range(L):
        for y in range(L):
            for x in range(L):
                H_X.append(sorted([
                    eid_x(x, y, z), eid_x(x - 1, y, z),
                    eid_y(x, y, z), eid_y(x, y - 1, z),
                    eid_z(x, y, z), eid_z(x, y, z - 1),
                ]))
                H_Z.append(sorted([eid_x(x, y, z), eid_y(x + 1, y, z), eid_x(x, y + 1, z), eid_y(x, y, z)]))
                H_Z.append(sorted([eid_x(x, y, z), eid_z(x + 1, y, z), eid_x(x, y, z + 1), eid_z(x, y, z)]))
                H_Z.append(sorted([eid_y(x, y, z), eid_z(x, y + 1, z), eid_y(x, y, z + 1), eid_z(x, y, z)]))
    return nq, H_X, H_Z


def make_layer_code_algebraic(n_in: int, HX_in: list, HZ_in: list, K_sep: int = 1):
    nX, nZ = len(HX_in), len(HZ_in)
    Mx = (nZ + 1) * K_sep + 1
    My = (n_in + 1) * K_sep + 1
    Mz = (nX + 1) * K_sep + 1
    if Mx % 2 == 0: Mx += 1
    if My % 2 == 0: My += 1
    if Mz % 2 == 0: Mz += 1

    px = [(b + 1) * K_sep for b in range(nZ)]
    py = [(j + 1) * K_sep for j in range(n_in)]
    pz = [(a + 1) * K_sep for a in range(nX)]

    c0_count = c1_count = c2_count = 0
    X0_off, X1h_off, X1v_off, X2_off = [], [], [], []
    for a in range(nX):
        X0_off.append(c0_count); c0_count += Mx * My
        X1h_off.append(c1_count); c1_count += (Mx - 1) * My
        X1v_off.append(c1_count); c1_count += Mx * (My - 1)
        X2_off.append(c2_count); c2_count += (Mx - 1) * (My - 1)

    Q0_off, Q1a_off, Q1b_off, Q2_off = [], [], [], []
    for j in range(n_in):
        Q0_off.append(c0_count); c0_count += Mx * (Mz - 1)
        Q1a_off.append(c1_count); c1_count += (Mx - 1) * (Mz - 1)
        Q1b_off.append(c1_count); c1_count += Mx * Mz
        Q2_off.append(c2_count); c2_count += (Mx - 1) * Mz

    Z0_off, Z1a_off, Z1b_off, Z2_off = [], [], [], []
    for b in range(nZ):
        Z0_off.append(c0_count); c0_count += (My - 1) * (Mz - 1)
        Z1a_off.append(c1_count); c1_count += My * (Mz - 1)
        Z1b_off.append(c1_count); c1_count += (My - 1) * Mz
        Z2_off.append(c2_count); c2_count += My * Mz

    d0_of_c0 = [set() for _ in range(c0_count)]
    d1_of_c1 = [set() for _ in range(c1_count)]
    def toggle(s, idx):
        if idx in s: s.remove(idx)
        else: s.add(idx)

    for a in range(nX):
        for ix in range(Mx):
            for iy in range(My):
                u = X0_off[a] + ix * My + iy
                if ix > 0: toggle(d0_of_c0[u], X1h_off[a] + (ix - 1) * My + iy)
                if ix < Mx - 1: toggle(d0_of_c0[u], X1h_off[a] + ix * My + iy)
                if iy > 0: toggle(d0_of_c0[u], X1v_off[a] + ix * (My - 1) + (iy - 1))
                if iy < My - 1: toggle(d0_of_c0[u], X1v_off[a] + ix * (My - 1) + iy)
        for ix in range(Mx - 1):
            for iy in range(My):
                q = X1h_off[a] + ix * My + iy
                if iy > 0: toggle(d1_of_c1[q], X2_off[a] + ix * (My - 1) + (iy - 1))
                if iy < My - 1: toggle(d1_of_c1[q], X2_off[a] + ix * (My - 1) + iy)
        for ix in range(Mx):
            for iy in range(My - 1):
                q = X1v_off[a] + ix * (My - 1) + iy
                if ix > 0: toggle(d1_of_c1[q], X2_off[a] + (ix - 1) * (My - 1) + iy)
                if ix < Mx - 1: toggle(d1_of_c1[q], X2_off[a] + ix * (My - 1) + iy)

    for j in range(n_in):
        for ix in range(Mx):
            for iz in range(Mz - 1):
                u = Q0_off[j] + ix * (Mz - 1) + iz
                if ix > 0: toggle(d0_of_c0[u], Q1a_off[j] + (ix - 1) * (Mz - 1) + iz)
                if ix < Mx - 1: toggle(d0_of_c0[u], Q1a_off[j] + ix * (Mz - 1) + iz)
                toggle(d0_of_c0[u], Q1b_off[j] + ix * Mz + iz)
                toggle(d0_of_c0[u], Q1b_off[j] + ix * Mz + (iz + 1))
        for ix in range(Mx - 1):
            for iz in range(Mz - 1):
                q = Q1a_off[j] + ix * (Mz - 1) + iz
                toggle(d1_of_c1[q], Q2_off[j] + ix * Mz + iz)
                toggle(d1_of_c1[q], Q2_off[j] + ix * Mz + (iz + 1))
        for ix in range(Mx):
            for iz in range(Mz):
                q = Q1b_off[j] + ix * Mz + iz
                if ix > 0: toggle(d1_of_c1[q], Q2_off[j] + (ix - 1) * Mz + iz)
                if ix < Mx - 1: toggle(d1_of_c1[q], Q2_off[j] + ix * Mz + iz)

    for b in range(nZ):
        for iy in range(My - 1):
            for iz in range(Mz - 1):
                u = Z0_off[b] + iy * (Mz - 1) + iz
                toggle(d0_of_c0[u], Z1a_off[b] + iy * (Mz - 1) + iz)
                toggle(d0_of_c0[u], Z1a_off[b] + (iy + 1) * (Mz - 1) + iz)
                toggle(d0_of_c0[u], Z1b_off[b] + iy * Mz + iz)
                toggle(d0_of_c0[u], Z1b_off[b] + iy * Mz + (iz + 1))
        for iy in range(My):
            for iz in range(Mz - 1):
                q = Z1a_off[b] + iy * (Mz - 1) + iz
                toggle(d1_of_c1[q], Z2_off[b] + iy * Mz + iz)
                toggle(d1_of_c1[q], Z2_off[b] + iy * Mz + (iz + 1))
        for iy in range(My - 1):
            for iz in range(Mz):
                q = Z1b_off[b] + iy * Mz + iz
                toggle(d1_of_c1[q], Z2_off[b] + iy * Mz + iz)
                toggle(d1_of_c1[q], Z2_off[b] + (iy + 1) * Mz + iz)

    for a, supp_x in enumerate(HX_in):
        za = pz[a]
        for j in supp_x:
            yj = py[j]
            for ix in range(Mx):
                toggle(d0_of_c0[X0_off[a] + ix * My + yj], Q1b_off[j] + ix * Mz + za)
            for ix in range(Mx - 1):
                toggle(d1_of_c1[X1h_off[a] + ix * My + yj], Q2_off[j] + ix * Mz + za)

    for b, supp_z in enumerate(HZ_in):
        xb = px[b]
        for j in supp_z:
            yj = py[j]
            for iz in range(Mz - 1):
                toggle(d0_of_c0[Q0_off[j] + xb * (Mz - 1) + iz], Z1a_off[b] + yj * (Mz - 1) + iz)
            for iz in range(Mz):
                toggle(d1_of_c1[Q1b_off[j] + xb * Mz + iz], Z2_off[b] + yj * Mz + iz)

    set_x = [set(r) for r in HX_in]
    set_z = [set(r) for r in HZ_in]
    for a in range(nX):
        za = pz[a]
        for b in range(nZ):
            xb = px[b]
            common = sorted(set_x[a] & set_z[b])
            for t in range(0, len(common), 2):
                y1, y2 = py[common[t]], py[common[t + 1]]
                for y in range(y1, y2):
                    toggle(d1_of_c1[X1v_off[a] + xb * (My - 1) + y], Z2_off[b] + y * Mz + za)
                    toggle(d0_of_c0[X0_off[a] + xb * My + (y + 1)], Z1b_off[b] + y * Mz + za)

    HX_out = [sorted(s) for s in d0_of_c0 if s]
    HZ_rows = [[] for _ in range(c2_count)]
    for q, s in enumerate(d1_of_c1):
        for zc in s:
            HZ_rows[zc].append(q)
    HZ_out = [sorted(r) for r in HZ_rows if r]
    return c1_count, HX_out, HZ_out


# Classical 2D Dual-Self-Correcting BDL code (up to step 4: n=9604 for ell=3 and n=4000 for ell=4!)
class ClassicalSheaf1D:
    def __init__(self, v_coords, edges, v_colors, edge_parent=None):
        self.v_coords = v_coords
        self.edges = edges
        self.v_colors = v_colors
        self.edge_parent = edge_parent
        self.vert_edges = [[] for _ in range(len(v_coords))]
        for e_idx, (u, v) in enumerate(edges):
            self.vert_edges[u].append(e_idx)
            self.vert_edges[v].append(e_idx)


def export_classical_family(prefix, ell: int, max_step: int):
    comp = ClassicalSheaf1D([(0, 0), (1, 0), (1, 1), (0, 1)], [(0, 1), (1, 2), (2, 3), (3, 0)], ['none'] * 4)
    history = [comp]
    macro_L = [0]

    for step in range(max_step + 1):
        if step > 0:
            itype = 'Rep' if step % 2 == 1 else 'Par'
            attach_col = 'black' if itype == 'Rep' else 'white'
            E_prev = len(comp.edges)
            n_xp_e = E_prev * ell

            xp_coords, xp_colors, xp_edges = [], [], []
            orig_map = {}
            for v_idx, (x, y) in enumerate(comp.v_coords):
                deg = len(comp.vert_edges[v_idx])
                orig_map[v_idx] = len(xp_coords)
                xp_coords.append((ell * x, ell * y))
                xp_colors.append(comp.v_colors[v_idx] if deg != 2 else 'none')

            edge_sub_verts = []
            eparent = []
            for e_idx, (u, v) in enumerate(comp.edges):
                p1, p2 = comp.v_coords[u], comp.v_coords[v]
                if p1 > p2:
                    u, v = v, u
                    p1, p2 = p2, p1
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                pts = [orig_map[u]]
                for m in range(1, ell):
                    pts.append(len(xp_coords))
                    xp_coords.append((ell * p1[0] + m * dx, ell * p1[1] + m * dy))
                    xp_colors.append('none')
                pts.append(orig_map[v])
                edge_sub_verts.append(pts)
                for m in range(ell):
                    xp_edges.append((pts[m], pts[m + 1]))
                    eparent.append(e_idx)

            n_xp = len(xp_coords)
            out_coords = list(xp_coords) + [(x + 1, y + 1) for (x, y) in xp_coords]
            out_colors = list(xp_colors) + list(xp_colors)
            out_edges = list(xp_edges) + [(u + n_xp, v + n_xp) for (u, v) in xp_edges]
            eparent = list(eparent) + list(eparent)

            for e_idx, pts in enumerate(edge_sub_verts):
                for m in range(2, ell):
                    u0 = pts[m]
                    u1 = pts[m - 1] + n_xp
                    out_edges.append((u0, u1))
                    eparent.append(e_idx)
                    out_colors[u0] = attach_col
                    out_colors[u1] = attach_col

            comp = ClassicalSheaf1D(out_coords, out_edges, out_colors, edge_parent=eparent)
            history.append(comp)

            if itype == 'Rep':
                macro_L = [e * ell for e in macro_L]
            else:
                macro_L = [e * ell for e in macro_L] + [n_xp_e + e * ell for e in macro_L]

        # Build parity checks H and vertex -> parent-edge RG hierarchy!
        H = []
        check_edge_owner = []
        for v_idx, inc in enumerate(comp.vert_edges):
            d = len(inc)
            col = comp.v_colors[v_idx]
            if d == 2 or col == 'black':
                for j in range(1, d):
                    H.append([inc[0], inc[j]])
                    check_edge_owner.append(min(inc))
            elif col == 'white':
                H.append(list(inc))
                check_edge_owner.append(min(inc))

        # Build RG levels for checks using edge_parent chain!
        rg_levels = []
        if step >= 1:
            cur_e = list(range(len(comp.edges)))
            for s_back in range(step, 0, -1):
                pm = history[s_back].edge_parent
                cur_e = [pm[e] for e in cur_e]
                num_u = len(history[s_back - 1].edges)
                units = [[] for _ in range(num_u)]
                for ci, e0 in enumerate(check_edge_owner):
                    units[cur_e[e0]].append(ci)
                rg_levels.append(units)

        # Notice: for KMC and threshold modes, `kmc_engine` ONLY uses `H` and `macro_L`!
        # `macro_L` ALREADY vanishes identically on ALL gauge codewords of `C_{(k)}` (as proven & verified!).
        # We only need `stabs` if running `stats` (ISD distance modulo gauge), which we compute for n <= 4200!
        gauge_cycles = []
        n = len(comp.edges)
        if n <= 4200:
            pivs = gf2_rref_pivots(n, H)
            fcols = [c for c in range(n) if c not in pivs]
            ker_vecs = []
            for fc in fcols:
                v = (1 << fc)
                for p, prow in pivs.items():
                    if (prow >> fc) & 1:
                        v |= (1 << p)
                ker_vecs.append(v)
            L_mask = 0
            for q in macro_L:
                L_mask |= (1 << q)
            ones = [v for v in ker_vecs if (bin(v & L_mask).count("1") & 1) == 1]
            zeros = [v for v in ker_vecs if (bin(v & L_mask).count("1") & 1) == 0]
            assert len(ones) >= 1
            base_one = ones[0]
            for gm in zeros + [v ^ base_one for v in ones[1:]]:
                supp = []
                tmp = gm
                while tmp:
                    b = (tmp & -tmp).bit_length() - 1
                    supp.append(b)
                    tmp ^= (1 << b)
                gauge_cycles.append(supp)

        out_path = os.path.join(DATA_DIR, f"{prefix}_step{step}.code")
        write_code_and_rg(out_path, n, H, gauge_cycles, [sorted(macro_L)], rg_levels)
        print(f"  Exported {prefix}_step{step}: n={n}, |H|={len(H)}, rg_lv={len(rg_levels)}")


if __name__ == "__main__":
    print("=== Building All Codes on SubMIT Cluster ===")
    # 1. 3D Quantum BDL Sheaf Families
    export_sheaf_chain("bdl_sq_L2r1",    seed_planar_surface(1, 1), [(2, [1], 'X'), (2, [1], 'Z'), (2, [1], 'X')])
    export_sheaf_chain("bdl_sq_L3r1",    seed_planar_surface(1, 1), [(3, [1], 'X'), (3, [1], 'Z')])
    export_sheaf_chain("bdl_sq_L3r2",    seed_planar_surface(1, 1), [(3, [1, 2], 'X'), (3, [1, 2], 'Z')])
    export_sheaf_chain("bdl_surf2_L2r1", seed_planar_surface(2, 2), [(2, [1], 'X'), (2, [1], 'Z')])
    export_sheaf_chain("bdl_tor2_L2r1",  seed_torus(2, 2),          [(2, [1], 'X'), (2, [1], 'Z')])
    export_sheaf_chain("bdl_tor2_L3r1",  seed_torus(2, 2),          [(3, [1], 'X'), (3, [1], 'Z')])
    export_sheaf_chain("bdl_tor2_L3r2",  seed_torus(2, 2),          [(3, [1, 2], 'X'), (3, [1, 2], 'Z')])
    # Novel Hybrid Rib Schedule: R_X with 2 ribs (L=3,r=[1,2]) then R_Z with 1 rib (L=2,r=[1])!
    export_sheaf_chain("bdl_tor2_hyb21", seed_torus(2, 2),          [(3, [1, 2], 'X'), (2, [1], 'Z')])
    export_sheaf_chain("bdl_tor3_L2r1",  seed_torus(3, 3),          [(2, [1], 'X'), (2, [1], 'Z')])

    # 2. Layer Codes ([[5,1,2]], Steane [[7,1,3]], and Shor [[9,1,3]]!)
    hx_512 = [[0, 1, 2], [2, 3, 4]]
    hz_512 = [[0, 2, 3], [1, 2, 4]]
    steane = [[0, 2, 4, 6], [1, 2, 5, 6], [3, 4, 5, 6]]
    # [[9, 1, 3]] rotated surface code (4 X-checks, 4 Z-checks, d=(3,3)!)
    hx_913 = [[0, 1, 3, 4], [4, 5, 7, 8], [1, 2], [6, 7]]
    hz_913 = [[1, 2, 4, 5], [3, 4, 6, 7], [0, 3], [5, 8]]
    for lc_name, nin, hxi, hzi in [
        ("layer_512",    5, hx_512, hz_512),
        ("layer_steane", 7, steane, steane),
        ("layer_913",    9, hx_913, hz_913),
    ]:
        nqL, HXL, HZL = make_layer_code_algebraic(nin, hxi, hzi, K_sep=1)
        logX = compute_dual_logicals(nqL, HZL, HXL)
        logZ = compute_dual_logicals(nqL, HXL, HZL)
        write_code_and_rg(os.path.join(DATA_DIR, f"{lc_name}_Xerr.code"), nqL, HZL, HXL, logX)
        write_code_and_rg(os.path.join(DATA_DIR, f"{lc_name}_Zerr.code"), nqL, HXL, HZL, logZ)
        print(f"  Exported {lc_name}: n={nqL}, k={len(logX)}")

    # 3. 2D & 3D Toric Baselines
    for L in (3, 6, 12, 20, 30):
        c = seed_torus(L, L)
        nq, HX, HZ, _ = sheaf_to_css(c)
        logX = compute_dual_logicals(nq, HZ, HX)
        logZ = compute_dual_logicals(nq, HX, HZ)
        write_code_and_rg(os.path.join(DATA_DIR, f"toric2d_L{L}_Xerr.code"), nq, HZ, HX, logX)
        write_code_and_rg(os.path.join(DATA_DIR, f"toric2d_L{L}_Zerr.code"), nq, HX, HZ, logZ)
        print(f"  Exported toric2d_L{L}: n={nq}")

    for L in (3, 5, 7, 9):
        nq, HX, HZ = make_3d_toric_code(L)
        logX = compute_dual_logicals(nq, HZ, HX)
        logZ = compute_dual_logicals(nq, HX, HZ)
        write_code_and_rg(os.path.join(DATA_DIR, f"toric3d_L{L}_Xerr.code"), nq, HZ, HX, logX)
        write_code_and_rg(os.path.join(DATA_DIR, f"toric3d_L{L}_Zerr.code"), nq, HX, HZ, logZ)
        print(f"  Exported toric3d_L{L}: n={nq}")

    # 4. 2D Classical Dual-Self-Correcting BDL Codes (up to step 4: n=9604 for 1-rib, n=40000 for 2-rib!)
    export_classical_family("cls_l3_r1", 3, max_step=4)
    export_classical_family("cls_l4_r2", 4, max_step=4)

    # 5. Generate Slurm Task List (`tasks.txt`): each line is one independent multi-threaded C++ run!
    tasks = []
    # Suite A: Exact stats on new codes (bdl_tor2_L3r1_s2, bdl_tor2_L3r2_s2, bdl_tor2_hyb21_s2, layer_913, cls_l4_r2_step3)
    for cname in [
        "bdl_tor2_L3r1_s2_Xerr.code", "bdl_tor2_L3r1_s2_Zerr.code",
        "bdl_tor2_hyb21_s2_Xerr.code", "bdl_tor2_hyb21_s2_Zerr.code",
        "layer_913_Xerr.code", "layer_913_Zerr.code",
        "cls_l4_r2_step3.code",
    ]:
        if os.path.exists(os.path.join(DATA_DIR, cname)):
            tasks.append(f"STATS|{cname}|stats|{cname}")

    # Suite B: 3D Quantum Memory KMC Scaling across beta in [3.0, 4.0, 5.0, 6.0, 7.0]
    q_kmc_codes = [
        # 1. Planar 1x1 square 1-rib L=2 steps 1, 2, 3 (sawtooth + recovery!)
        "bdl_sq_L2r1_s1_Xerr.code", "bdl_sq_L2r1_s1_Zerr.code",
        "bdl_sq_L2r1_s2_Xerr.code", "bdl_sq_L2r1_s2_Zerr.code",
        "bdl_sq_L2r1_s3_Xerr.code", "bdl_sq_L2r1_s3_Zerr.code",
        # 2. Controlled 1-rib vs 2-rib at fixed L=3 (planar square & 2x2 torus!)
        "bdl_sq_L3r1_s1_Xerr.code", "bdl_sq_L3r1_s2_Xerr.code", "bdl_sq_L3r1_s2_Zerr.code",
        "bdl_sq_L3r2_s1_Xerr.code", "bdl_sq_L3r2_s2_Xerr.code", "bdl_sq_L3r2_s2_Zerr.code",
        # 3. Toroidal vs Planar Seeds & Hybrid 2->1 Rib Schedule!
        "bdl_surf2_L2r1_s0_Xerr.code", "bdl_surf2_L2r1_s1_Xerr.code",
        "bdl_surf2_L2r1_s2_Xerr.code", "bdl_surf2_L2r1_s2_Zerr.code",
        "bdl_tor2_L2r1_s0_Xerr.code", "bdl_tor2_L2r1_s1_Xerr.code", "bdl_tor2_L2r1_s1_Zerr.code",
        "bdl_tor2_L2r1_s2_Xerr.code", "bdl_tor2_L2r1_s2_Zerr.code",
        "bdl_tor2_L3r1_s1_Xerr.code", "bdl_tor2_L3r2_s1_Xerr.code",
        "bdl_tor2_hyb21_s2_Xerr.code", "bdl_tor2_hyb21_s2_Zerr.code",
        "bdl_tor3_L2r1_s0_Xerr.code", "bdl_tor3_L2r1_s1_Xerr.code", "bdl_tor3_L2r1_s1_Zerr.code",
        "bdl_tor3_L2r1_s2_Xerr.code", "bdl_tor3_L2r1_s2_Zerr.code",
        # 4. Layer Codes & 2D/3D Toric Baselines!
        "layer_512_Xerr.code", "layer_steane_Xerr.code", "layer_913_Xerr.code",
        "toric2d_L3_Xerr.code", "toric2d_L6_Xerr.code", "toric2d_L12_Xerr.code",
        "toric2d_L20_Xerr.code", "toric2d_L30_Xerr.code",
        "toric3d_L3_Xerr.code", "toric3d_L3_Zerr.code",
        "toric3d_L5_Xerr.code", "toric3d_L5_Zerr.code",
        "toric3d_L7_Xerr.code", "toric3d_L7_Zerr.code",
    ]
    for cname in q_kmc_codes:
        if not os.path.exists(os.path.join(DATA_DIR, cname)):
            continue
        for beta in (3.0, 4.0, 5.0, 6.0, 7.0):
            # Generous flip cap (15,000,000 flips!) and 120 trials on cluster!
            tasks.append(f"KMC_Q|{cname}|kmc|{cname}|{beta}|uf|120|15000000|8")

    # Suite C: 5-Level Classical Dual-Self-Correcting Scaling (k=0,1,2,3,4 for 1-rib ell=3 vs 2-rib ell=4!)
    for prefix in ("cls_l3_r1", "cls_l4_r2"):
        for st in range(5):
            cname = f"{prefix}_step{st}.code"
            if not os.path.exists(os.path.join(DATA_DIR, cname)):
                continue
            trials = 200 if st <= 2 else (100 if st == 3 else 40)
            max_fl = 20000000 if st <= 3 else 8000000
            for beta in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
                tasks.append(f"KMC_CLS|{cname}|kmc|{cname}|{beta}|uf|{trials}|{max_fl}|8")

    # Suite D: Decoder Showdown (rg_strict vs rg_relaxed vs uf vs bposd) on 1-rib vs 2-rib!
    dec_test_codes = [
        "bdl_sq_L3r1_s2_Xerr.code", "bdl_sq_L3r2_s2_Xerr.code",
        "bdl_tor2_L2r1_s2_Xerr.code", "bdl_tor2_L2r1_s2_Zerr.code",
        "bdl_tor2_hyb21_s2_Xerr.code",
        "cls_l3_r1_step2.code", "cls_l4_r2_step2.code",
        "cls_l3_r1_step3.code", "cls_l4_r2_step3.code",
    ]
    for cname in dec_test_codes:
        if not os.path.exists(os.path.join(DATA_DIR, cname)):
            continue
        # 1) Threshold curves P_L(p)
        for p in (0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.12):
            for dec in ("rg_strict", "rg_relaxed", "uf", "bposd"):
                tr = 400 if dec != "bposd" else 200
                tasks.append(f"THRESH|{cname}|threshold|{cname}|{p}|{dec}|{tr}|8")
        # 2) Thermal KMC at beta=3.0, 4.0, 5.0 across all 4 decoders
        betas = (2.5, 3.0, 3.5) if cname.startswith("cls_") else (3.5, 4.5, 5.5)
        for beta in betas:
            for dec in ("rg_strict", "rg_relaxed", "uf", "bposd"):
                tr = 80 if dec != "bposd" else 40
                tasks.append(f"KMC_DEC|{cname}|kmc|{cname}|{beta}|{dec}|{tr}|5000000|8")

    # Suite E: Passive Thermal Cooling / Initialization from T=inf (p_init=0.5) & Noisy State (p_init=0.15)
    cool_codes = [
        "bdl_sq_L3r1_s2_Xerr.code",
        "bdl_sq_L3r2_s2_Xerr.code",
        "bdl_tor3_L2r1_s2_Xerr.code",
        "toric2d_L20_Xerr.code",
        "toric3d_L7_Xerr.code",
        "cls_l3_r1_step3.code",
        "cls_l4_r2_step3.code",
    ]
    for cname in cool_codes:
        if not os.path.exists(os.path.join(DATA_DIR, cname)):
            continue
        for beta in (2.0, 4.0, 6.0):
            tasks.append(f"COOL|{cname}|cool|{cname}|{beta}|0.5|64|5000000|8")

    # Suite F: Hybrid Lazy Active QEC + Passive Thermal Bath (100 QEC cycles at varying intervals tau_ec!)
    hyb_codes = [
        "bdl_tor3_L2r1_s2_Xerr.code",
        "bdl_tor3_L2r1_s2_Zerr.code",
        "bdl_sq_L3r2_s2_Xerr.code",
        "layer_913_Xerr.code",
        "toric2d_L20_Xerr.code",
    ]
    for cname in hyb_codes:
        if not os.path.exists(os.path.join(DATA_DIR, cname)):
            continue
        for tau_ec in (10.0, 50.0, 200.0, 1000.0, 5000.0):
            tasks.append(f"HYBRID|{cname}|hybrid|{cname}|5.0|{tau_ec}|20|uf|120|8")

    random.Random(2026).shuffle(tasks)  # Shuffle for even load balancing across Slurm array shards!
    manifest_path = os.path.join(BASE_DIR, "tasks.txt")
    with open(manifest_path, "w") as f:
        for line in tasks:
            f.write(line + "\n")
    print(f"Total tasks written to {manifest_path}: {len(tasks)}")
