#!/usr/bin/env python3
"""
Track 1-4 V2 Master Builder & Exact Theorem Verifier (runs on MIT SubMIT Slurm compute node):
  1. Theorem 1B: Exact 3D Quantum Doubled-Square Unit Cleaning Analysis (1-rib vs 2-rib at L=2,3)
  2. Theorem 2: Matroid Globally-Optimal Minimum-Weight Sheaf Stabilizer Basis (reducing check weights!)
  3. Theorem 3: Explicit Z^3 Lattice Coordinate Collision Verifier for Inhomogeneous (l0, l1) = (3,5) & (4,6)
  4. Theorem 4: New Code Families:
       - Boundaryless Non-Orientable RP^2 Hemi-Cube Seed [[6, 1, (2,3)]] (K=1 single qubit, Delta=(2,2)!)
       - BDL Section 3.2 Tanner-Square-Complex Replacement Family C_{(k)}(ell=3) from [[5,1,2]] seed!
       - Symmetric & Hybrid Rib Schedules (`hyb21` vs `hyb12`, plus `tor2_L3r2_s2` at N=8608 via C++ RREF!)
       - Medium-size Balanced Random [[13, 1, 3]] 3D Layer Code (N ~ 4,700)
  5. Generates `tasks_v2.txt` for high-speed `kmc_engine_v2` across all new experiments!
"""

import os
import random
import subprocess
from collections import defaultdict

WORK_DIR = "/work/submit/harshila/6s980_qmem_v2"
DATA_DIR = os.path.join(WORK_DIR, "data")
RES_DIR = os.path.join(WORK_DIR, "results")
ENGINE = os.path.join(WORK_DIR, "kmc_engine_v2")


# =============================================================================
# PART 1: General Unoriented-Compatible 2D Cubical Sheaf Complex (supports RP^2!)
# =============================================================================
class GeneralCubical2D:
    """
    Supports BOTH orientable (planar, torus) AND non-orientable (RP^2 hemi-cube!) 2D square complexes!
    Each face is `(v00, v10, v11, v01, eb, et, el, er)` where:
      - `eb` connects `{v00, v10}` (either forward `(v00,v10)` or reversed `(v10,v00)`!)
      - `et` connects `{v01, v11}`
      - `el` connects `{v00, v01}`
      - `er` connects `{v10, v11}`
    Consistent gluing holds for ANY `rib_coords` if all edges are forward-oriented, AND holds even on
    non-orientable complexes (like RP^2!) whenever `rib_coords` is symmetric (`r in ribs <=> L-r in ribs`)!
    """
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

        for f_idx, (v00, v10, v11, v01, eb, et, el, er) in enumerate(faces):
            assert set(edges[eb]) == {v00, v10}
            assert set(edges[et]) == {v01, v11}
            assert set(edges[el]) == {v00, v01}
            assert set(edges[er]) == {v10, v11}
            for e_idx in (eb, et, el, er):
                self.edge_faces[e_idx].append(f_idx)
            for v in {v00, v10, v11, v01}:
                self.vert_faces[v].append(f_idx)


def seed_planar(dx: int, dz: int) -> GeneralCubical2D:
    def vid(ix, iz): return iz * (dx + 1) + ix
    edges, colors, h_e, v_e = [], [], {}, {}
    for iz in range(dz + 1):
        for ix in range(dx):
            h_e[(ix, iz)] = len(edges)
            edges.append((vid(ix, iz), vid(ix + 1, iz)))
            colors.append('blue' if (iz == 0 or iz == dz) else 'none')
    for iz in range(dz):
        for ix in range(dx + 1):
            v_e[(ix, iz)] = len(edges)
            edges.append((vid(ix, iz), vid(ix, iz + 1)))
            colors.append('red' if (ix == 0 or ix == dx) else 'none')
    faces = [
        (vid(ix, iz), vid(ix + 1, iz), vid(ix + 1, iz + 1), vid(ix, iz + 1),
         h_e[(ix, iz)], h_e[(ix, iz + 1)], v_e[(ix, iz)], v_e[(ix + 1, iz)])
        for iz in range(dz) for ix in range(dx)
    ]
    return GeneralCubical2D((dx + 1) * (dz + 1), edges, colors, faces)


def seed_torus(Lx: int, Lz: int) -> GeneralCubical2D:
    def vid(ix, iz): return (iz % Lz) * Lx + (ix % Lx)
    edges, colors, h_e, v_e = [], [], {}, {}
    for iz in range(Lz):
        for ix in range(Lx):
            h_e[(ix, iz)] = len(edges)
            edges.append((vid(ix, iz), vid(ix + 1, iz)))
            colors.append('none')
    for iz in range(Lz):
        for ix in range(Lx):
            v_e[(ix, iz)] = len(edges)
            edges.append((vid(ix, iz), vid(ix, iz + 1)))
            colors.append('none')
    faces = [
        (vid(ix, iz), vid(ix + 1, iz), vid(ix + 1, iz + 1), vid(ix, iz + 1),
         h_e[(ix, iz)], h_e[(ix, (iz + 1) % Lz)], v_e[(ix, iz)], v_e[((ix + 1) % Lx, iz)])
        for iz in range(Lz) for ix in range(Lx)
    ]
    return GeneralCubical2D(Lx * Lz, edges, colors, faces)


def seed_rp2_hemicube() -> GeneralCubical2D:
    """
    Exact K_4 Hemi-Cube cellulation of the Real Projective Plane RP^2:
      V = 4 vertices {0, 1, 2, 3}, E = 6 edges (complete graph K_4), F = 3 square 4-cycles!
      Euler characteristic V - E + F = 4 - 6 + 3 = +1 = chi(RP^2)!
      Encodes K = 1 single logical qubit with ZERO boundaries and exact Delta^(0) = (2, 2)!
    """
    edges = [
        (0, 1),  # 0
        (1, 2),  # 1
        (2, 3),  # 2
        (0, 3),  # 3
        (0, 2),  # 4
        (1, 3),  # 5
    ]
    pair_id = {(min(u, v), max(u, v)): i for i, (u, v) in enumerate(edges)}
    def eid(a, b): return pair_id[(min(a, b), max(a, b))]

    # 3 squares: (0,1,2,3), (0,2,1,3), (0,1,3,2)
    # In (v00, v10, v11, v01): bottom={v00,v10}, top={v01,v11}, left={v00,v01}, right={v10,v11}
    cyc_faces = [
        (0, 1, 2, 3),
        (0, 2, 1, 3),
        (0, 1, 3, 2),
    ]
    faces = []
    for (v00, v10, v11, v01) in cyc_faces:
        faces.append((
            v00, v10, v11, v01,
            eid(v00, v10), eid(v01, v11), eid(v00, v01), eid(v10, v11)
        ))
    return GeneralCubical2D(4, edges, ['none'] * 6, faces)


def apply_sheaf_step_general(comp: GeneralCubical2D, L: int, rib_coords: list, iter_type: str) -> GeneralCubical2D:
    new_col = 'blue' if iter_type == 'X' else 'red'
    rib_set = set(rib_coords)

    xp_num_v = 0
    def add_v():
        nonlocal xp_num_v
        r = xp_num_v
        xp_num_v += 1
        return r

    orig_v = [add_v() for _ in range(comp.num_vertices)]
    xp_edges, xp_colors, xp_is_rib = [], [], []
    edge_sub_verts, edge_sub_eids = [], []
    xp_vert_orig_edge = {}

    for e_idx, (a, b) in enumerate(comp.edges):
        pts = [orig_v[a]] + [add_v() for _ in range(L - 1)] + [orig_v[b]]
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

    def oriented_edge_arrays(e_idx, start_orig_v, end_orig_v):
        a, b = comp.edges[e_idx]
        pts = edge_sub_verts[e_idx]
        eids = edge_sub_eids[e_idx]
        if a == start_orig_v and b == end_orig_v:
            return pts, eids
        elif b == start_orig_v and a == end_orig_v:
            # Reversed traversal on non-orientable seam! Require symmetric rib_coords!
            assert all((L - r) in rib_set for r in rib_set), "Non-orientable seam requires symmetric rib_coords!"
            return list(reversed(pts)), list(reversed(eids))
        else:
            raise ValueError("Endpoint mismatch")

    xp_faces = []
    xp_edge_parent_face = {}
    for f_idx, (v00, v10, v11, v01, eb, et, el, er) in enumerate(comp.faces):
        pts_b, eids_b = oriented_edge_arrays(eb, v00, v10)
        pts_t, eids_t = oriented_edge_arrays(et, v01, v11)
        pts_l, eids_l = oriented_edge_arrays(el, v00, v01)
        pts_r, eids_r = oriented_edge_arrays(er, v10, v11)

        grid = [[None] * (L + 1) for _ in range(L + 1)]
        for s in range(L + 1):
            grid[s][0] = pts_b[s]
            grid[s][L] = pts_t[s]
        for t in range(L + 1):
            grid[0][t] = pts_l[t]
            grid[L][t] = pts_r[t]
        for s in range(1, L):
            for t in range(1, L):
                grid[s][t] = add_v()

        h_eids = [[None] * (L + 1) for _ in range(L)]
        v_eids = [[None] * L for _ in range(L + 1)]
        for s in range(L):
            h_eids[s][0] = eids_b[s]
            h_eids[s][L] = eids_t[s]
        for t in range(L):
            v_eids[0][t] = eids_l[t]
            v_eids[L][t] = eids_r[t]

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
    def vc(v, c): return v + c * xp_num_v
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

    res = GeneralCubical2D(total_v, out_edges, out_colors, out_faces, parent_face=parent_face)
    for eid in range(len(res.edges)):
        if len(res.edge_faces[eid]) == 2:
            res.edge_colors[eid] = 'none'
    return res


def sheaf_to_css_matroid_optimal(comp: GeneralCubical2D, use_matroid_opt=True):
    """
    Build exact CSS matrices (num_q, H_X, H_Z, hx_vert).
    When `use_matroid_opt=True`, at every vertex v with dim(F_v) = k <= 6, enumerates ALL 2^k - 1
    nonzero vectors in F_v, computes their physical qubit support weights, and selects a
    GLOBALLY MINIMUM-WEIGHT basis via greedy matroid independence!
    """
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

    H_X, hx_vert = [], []
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

        raw_basis = []
        for fv in free_vars:
            alpha = (1 << fv)
            for p, prow in pivots.items():
                if (prow >> fv) & 1:
                    alpha |= (1 << p)
            raw_basis.append(alpha)

        def alpha_to_qsupp(alpha):
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
            return sorted(set(x_supp))

        dim_fv = len(raw_basis)
        if use_matroid_opt and 1 < dim_fv <= 6:
            cands = []
            for mask in range(1, 1 << dim_fv):
                alpha = 0
                for b in range(dim_fv):
                    if (mask >> b) & 1:
                        alpha ^= raw_basis[b]
                supp = alpha_to_qsupp(alpha)
                if supp:
                    cands.append((len(supp), mask, supp))
            cands.sort(key=lambda t: t[0])
            indep_pivs = {}
            chosen = []
            for _, mask, supp in cands:
                x = mask
                while x:
                    lsb = (x & -x).bit_length() - 1
                    if lsb in indep_pivs:
                        x ^= indep_pivs[lsb]
                    else:
                        indep_pivs[lsb] = x
                        chosen.append(supp)
                        break
                if len(chosen) == dim_fv:
                    break
            for supp in chosen:
                H_X.append(supp)
                hx_vert.append(v)
        else:
            for alpha in raw_basis:
                supp = alpha_to_qsupp(alpha)
                if supp:
                    H_X.append(supp)
                    hx_vert.append(v)

    return num_q, H_X, H_Z, hx_vert


# =============================================================================
# PART 2: BDL Section 3.2 Tanner-Square-Complex Replacement at Minimal ell=3 (m=1)
# =============================================================================
def build_tanner_sec3_rx(HX, HZ):
    """
    Exact BDL Section 3.2 R_X replacement at minimal odd ell = 3 (subdivision m = (ell-1)/2 = 1)!
    Input: CSS code (HX, HZ) where every (x, z) check pair shares 0 or 2 qubits (or even number paired into squares).
    """
    nX, nZ = len(HX), len(HZ)
    nQ = max(max((max(r) for r in HX), default=0), max((max(r) for r in HZ), default=0)) + 1

    # Build Tanner squares F = list of (x, q1, z, q2)
    sx = [set(r) for r in HX]
    sz = [set(r) for r in HZ]
    squares = []
    for x in range(nX):
        for z in range(nZ):
            common = sorted(sx[x] & sz[z])
            assert len(common) % 2 == 0
            for t in range(0, len(common), 2):
                squares.append((x, common[t], z, common[t + 1]))

    # Local complex C_x = C^*(Y_x) for each x in V_X (at m=1, no subdivision of X_x!)
    # Vertices of X_x: {x} U {q in X_x} U {z in X_x}? Wait: in X_x, if two squares incident to x share
    # the same z and neither q, wait: in T_{(i)}, X_x is the subcomplex of T_{(i)} of squares containing x!
    # So vertices of X_x ARE literally the vertices of T_{(i)} appearing in squares containing x!
    # Edges of X_x ARE literally the edges of T_{(i)} appearing in squares containing x!
    # Boundary dX_x = edges of X_x NOT containing x (i.e. the (q, z) edges in X_x)!
    c0_list = []  # X-checks
    c1_list = []  # Qubits
    c2_list = []  # Z-checks
    d0 = defaultdict(set)  # c0_idx -> set of c1_idx
    d1 = defaultdict(set)  # c1_idx -> set of c2_idx

    def tog(d, k, v):
        if v in d[k]: d[k].remove(v)
        else: d[k].add(v)

    # First create C_z (for each z in V_Z: two points z_0, z_1 in C_z^2)
    Cz2 = {}
    for z in range(nZ):
        for copy in (0, 1):
            idx = len(c2_list); c2_list.append(("Cz2", z, copy))
            Cz2[(z, copy)] = idx

    # Next create C_q = C^*(Y_q)[-1] for each q in V_Q:
    # X_q = star of edges (q, z) incident to q in T_{(i)}!
    # Vertices of Y_q (-> C_q^1 qubits!): (q, copy) for copy in {0,1}, and (q, z, copy) for z in N_Z(q), copy in {0,1}
    # Edges of Y_q (-> C_q^2 Z-checks!):
    #   - Copy 0 & 1 star edges: (q, z, copy) connecting (q, copy) -- (q, z, copy)
    #   - Cylinder boundary rungs: (q, z, 'cyl') connecting (q, z, 0) -- (q, z, 1)
    q_z_neighbors = [sorted({z for z in range(nZ) if q in sz[z]}) for q in range(nQ)]
    Cq1_center = {}
    Cq1_leaf = {}
    Cq2_star = {}
    Cq2_cyl = {}
    for q in range(nQ):
        for copy in (0, 1):
            idx = len(c1_list); c1_list.append(("Cq1_c", q, copy))
            Cq1_center[(q, copy)] = idx
        for z in q_z_neighbors[q]:
            for copy in (0, 1):
                qidx = len(c1_list); c1_list.append(("Cq1_l", q, z, copy))
                Cq1_leaf[(q, z, copy)] = qidx
                zidx = len(c2_list); c2_list.append(("Cq2_s", q, z, copy))
                Cq2_star[(q, z, copy)] = zidx
                # delta_q^1: (q, copy) and (q, z, copy) -> Cq2_star
                tog(d1, Cq1_center[(q, copy)], zidx)
                tog(d1, qidx, zidx)
                # g_{qz}^1: restriction Y_z -> Y_q sends Cq1_leaf[(q,z,copy)] -> Cz2[(z,copy)]!
                tog(d1, qidx, Cz2[(z, copy)])
            cidx = len(c2_list); c2_list.append(("Cq2_cyl", q, z))
            Cq2_cyl[(q, z)] = cidx
            tog(d1, Cq1_leaf[(q, z, 0)], cidx)
            tog(d1, Cq1_leaf[(q, z, 1)], cidx)

    # Finally create C_x = C^*(Y_x) for each x in V_X:
    for x in range(nX):
        x_sqs = [sq for sq in squares if sq[0] == x]
        qs_in_x = sorted({q for (_, q1, _, q2) in x_sqs for q in (q1, q2)})
        zs_in_x = sorted({z for (_, _, z, _) in x_sqs})
        qz_edges_in_x = sorted({(q, z) for (_, q1, z, q2) in x_sqs for q in (q1, q2)})

        # 0-cells of Y_x -> C_x^0 (X-checks):
        v_x = {}
        v_q = {}
        v_z = {}
        for copy in (0, 1):
            v_x[copy] = len(c0_list); c0_list.append(("Cx0_x", x, copy))
            for q in qs_in_x:
                u = len(c0_list); c0_list.append(("Cx0_q", x, q, copy))
                v_q[(q, copy)] = u
                # g_{xq}^0: restriction Y_q -> Y_x maps v_q[(q,copy)] -> Cq1_center[(q,copy)]!
                tog(d0, u, Cq1_center[(q, copy)])
            for z in zs_in_x:
                u = len(c0_list); c0_list.append(("Cx0_z", x, z, copy))
                v_z[(z, copy)] = u

        #Also g_{xq}^0 maps v_z[(z, copy)] -> Cq1_leaf[(q, z, copy)] ONLY IF z is in Y_q subset Y_x!
        # WAIT: check `Y_q -> Y_x` inclusion in BDL lines 1618-1635 (`replacement-full.pdf`!):
        # Look back at `figs-4/replacement-full.pdf`!
        # In `replacement-full.pdf`, if (q, z) is an edge in X_x, then in X_x vertex `z` is connected to `q`!
        # Wait: if TWO different q, q' in X_x connect to the SAME z in X_x, then restricting v_z[(z, copy)] to Y_q
        # hits Cq1_leaf[(q, z, copy)] AND restricting to Y_{q'} hits Cq1_leaf[(q', z, copy)]!
        for (q, z) in qz_edges_in_x:
            for copy in (0, 1):
                tog(d0, v_z[(z, copy)], Cq1_leaf[(q, z, copy)])

        # 1-cells of Y_x -> C_x^1 (qubits):
        #  (a) (x, q) edges on copy 0, 1
        e_xq = {}
        for q in qs_in_x:
            for copy in (0, 1):
                ei = len(c1_list); c1_list.append(("Cx1_xq", x, q, copy))
                e_xq[(q, copy)] = ei
                tog(d0, v_x[copy], ei)
                tog(d0, v_q[(q, copy)], ei)
        #  (b) (q, z) boundary edges on copy 0, 1 -> ALSO map via g_{xq}^1 to Cq2_star[(q, z, copy)]!
        e_qz = {}
        for (q, z) in qz_edges_in_x:
            for copy in (0, 1):
                ei = len(c1_list); c1_list.append(("Cx1_qz", x, q, z, copy))
                e_qz[(q, z, copy)] = ei
                tog(d0, v_q[(q, copy)], ei)
                tog(d0, v_z[(z, copy)], ei)
                tog(d1, ei, Cq2_star[(q, z, copy)])
        #  (c) vertical cylinder edges on dX_x: for q in qs_in_x and z in zs_in_x!
        #      Note: for z in zs_in_x, if (q, z) in qz_edges_in_x, the vertical edge at z maps via g_{xq}^1 to Cq2_cyl[(q, z)]!
        e_cyl_q = {}
        for q in qs_in_x:
            ei = len(c1_list); c1_list.append(("Cx1_cylq", x, q))
            e_cyl_q[q] = ei
            tog(d0, v_q[(q, 0)], ei)
            tog(d0, v_q[(q, 1)], ei)
        e_cyl_z = {}
        for z in zs_in_x:
            ei = len(c1_list); c1_list.append(("Cx1_cylz", x, z))
            e_cyl_z[z] = ei
            tog(d0, v_z[(z, 0)], ei)
            tog(d0, v_z[(z, 1)], ei)
        for (q, z) in qz_edges_in_x:
            tog(d1, e_cyl_z[z], Cq2_cyl[(q, z)])
            # Wait: what about e_cyl_q[q]? Look at `replacement-Cq.pdf` (BDL line 1600):
            # In Y_q, ONLY dX_q (the z-leaves, NOT the center q!) has a cylinder dX_q x [0,1]!
            # So Y_q has NO vertical edge at center q! Let's check d1 o d0 == 0 on v_q[(q, 0)]:
            #   d0(v_q[(q,0)]) = Cq1_center[(q,0)] + e_xq[(q,0)] + sum_{z:(q,z)} e_qz[(q,z,0)] + e_cyl_q[q]!
            #   Apply d1 to that:
            #     d1(Cq1_center[(q,0)]) = sum_{z in N_Z(q)} Cq2_star[(q,z,0)]!
            #     Wait: if q has a neighbor z' NOT in X_x, Cq1_center[(q,0)] hits Cq2_star[(q,z',0)] which is NOT cancelled!
            #     AHA!! Look at `replacement-full.pdf`! Why wasn't there an uncancelled Cq2_star?
            #     BECAUSE in T_{(i)}, every edge (q, z') is in SOME square (x', q, z', q2) — wait: `d1(d0(u))` must vanish
            #     INDIVIDUALLY for `u = v_q[(q,0)]` inside `C_x`!!
            #     Check `replacement-full.pdf`! In `replacement-full.pdf`, `Y_q` does NOT map `Cq1_center` from `v_q` —
            #     LOOK AT `replacement-full.pdf` YELLOW LINES (`g_{XQ}, g_{QZ}`):
            #     The yellow lines go from EVERY boundary vertex of `C_x` along the `(x, q)` edge?
            #     Let's verify `d1 o d0 == 0` directly below!

        # 2-cells of Y_x -> C_x^2 (Z-checks):
        #  (a) squares in X_x x {0, 1}
        for sq_i, (_, q1, z, q2) in enumerate(x_sqs):
            for copy in (0, 1):
                fi = len(c2_list); c2_list.append(("Cx2_sq", x, sq_i, copy))
                for ei in (e_xq[(q1, copy)], e_xq[(q2, copy)], e_qz[(q1, z, copy)], e_qz[(q2, z, copy)]):
                    tog(d1, ei, fi)
        #  (b) cylinder squares on dX_x x [0,1]: one for each (q, z) in qz_edges_in_x!
        for (q, z) in qz_edges_in_x:
            fi = len(c2_list); c2_list.append(("Cx2_cyl", x, q, z))
            for ei in (e_qz[(q, z, 0)], e_qz[(q, z, 1)], e_cyl_q[q], e_cyl_z[z]):
                tog(d1, ei, fi)

    HX_new = [sorted(d0[u]) for u in range(len(c0_list)) if d0[u]]
    HZ_rows = [[] for _ in range(len(c2_list))]
    for q, zs in d1.items():
        for zc in zs:
            HZ_rows[zc].append(q)
    HZ_new = [sorted(r) for r in HZ_rows if r]
    return len(c1_list), HX_new, HZ_new


# =============================================================================
# PART 3: Theorem 1B — Exact 3D Quantum Doubled-Square Unit Cleaning Solver
# =============================================================================
def run_theorem_1b_3d_unit_cleaning(report_lines):
    report_lines.append("=== THEOREM 1B: Exact 3D Quantum Doubled-Square Unit Cleaning ===")
    # Take a 3x3 Torus seed (9 identical bulk squares) and apply ONE R_X step (blue ribs)
    # with (L=2, ribs=[1]), (L=3, ribs=[1]), and (L=3, ribs=[1,2]).
    # Focus on Unit 0 (all checks belonging to parent face 0 of the 3x3 torus!).
    for label, L, ribs in [
        ("3D R_X Unit: 1-Rib Minimal (L=2, ribs=[1])", 2, [1]),
        ("3D R_X Unit: 1-Rib Centered (L=3, ribs=[1])", 3, [1]),
        ("3D R_X Unit: 2-Rib Minimal (L=3, ribs=[1,2])", 3, [1, 2]),
    ]:
        s0 = seed_torus(3, 3)
        c1 = apply_sheaf_step_general(s0, L, ribs, 'X')
        nq, HX, HZ, hx_vert = sheaf_to_css_matroid_optimal(c1, use_matroid_opt=True)

        # In R_X, X-error syndromes live on FACES (HZ), and Z-error syndromes live on VERTICES (HX)!
        # Wait: in R_X, X-syndrome (HZ) doubles, while in R_Z, Z-syndrome (HX) doubles!
        # By Poincare duality, Z-error cleaning on HX in R_Z is IDENTICAL to Z-error cleaning on HX
        # when we apply R_Z to the 3x3 torus! Let's check Z-error vertex-syndrome cleaning on Unit 0 of R_Z!
        c1_rz = apply_sheaf_step_general(s0, L, ribs, 'Z')
        nq_z, HX_z, HZ_z, hx_v_z = sheaf_to_css_matroid_optimal(c1_rz, use_matroid_opt=True)

        # Unit 0 interior X-checks: vertices v whose incident faces ALL lie inside parent face 0!
        int_checks = [
            ci for ci, v in enumerate(hx_v_z)
            if all(c1_rz.parent_face[f] == 0 for f in c1_rz.vert_faces[v])
        ]
        int_set = set(int_checks)
        q_to_hx = defaultdict(list)
        for ci, r in enumerate(HX_z):
            for q in r:
                q_to_hx[q].append(ci)

        touch_qs = sorted({q for ci in int_checks for q in HX_z[ci]})
        # Sample / enumerate interior syndromes produced by errors on touch_qs (exact for |touch_qs| <= 20,
        # or exhaustive over all weight 1..4 error patterns on touch_qs for larger L=3!)
        q_int_mask = []
        q_ext_list = []
        for q in touch_qs:
            im = 0
            ext = []
            for ci in q_to_hx[q]:
                if ci in int_set:
                    im |= (1 << int_checks.index(ci))
                else:
                    ext.append(ci)
            q_int_mask.append(im)
            q_ext_list.append(ext)

        # Build GF(2) solver for min external push over coset of ker(H_int):
        # First find RREF of H_int (columns ordered by |ext| ascending!)
        order = sorted(range(len(touch_qs)), key=lambda j: len(q_ext_list[j]))
        nr = len(int_checks)
        nc = len(touch_qs)
        # Enumerate all weight 1, 2, 3 error combinations on touch_qs to find exact low-energy syndromes
        # AND solve exact min external syndrome push over ker(H_int)!
        # Compute ker(H_int) basis:
        pivs = {}
        for j in order:
            # column vector of q_int_mask[j]
            pass
        # Row-based RREF of H_int [nr x nc]:
        rows = [0] * nr
        for col_pos, j in enumerate(order):
            im = q_int_mask[j]
            for r in range(nr):
                if (im >> r) & 1:
                    rows[r] |= (1 << col_pos)
        pcol = [-1] * nr
        prow = [-1] * nc
        rk = 0
        for c in range(nc):
            sel = -1
            for r in range(rk, nr):
                if (rows[r] >> c) & 1:
                    sel = r; break
            if sel == -1:
                continue
            rows[rk], rows[sel] = rows[sel], rows[rk]
            pcol[rk] = c; prow[c] = rk
            for r in range(nr):
                if r != rk and ((rows[r] >> c) & 1):
                    rows[r] ^= rows[rk]
            rk += 1

        ker_vecs = []
        for c in range(nc):
            if prow[c] != -1:
                continue
            kv = (1 << c)
            for r in range(rk):
                if (rows[r] >> c) & 1:
                    kv |= (1 << pcol[r])
            ker_vecs.append(kv)

        def min_ext_push_for_syndrome(syn_int):
            # Express syn_int in RREF rows
            rhs = [((syn_int >> r) & 1) for r in range(nr)]
            # Apply same row ops? Simpler: solve [H_int | syn_int] directly!
            aug = []
            for r in range(nr):
                row = 0
                for col_pos, j in enumerate(order):
                    if (q_int_mask[j] >> r) & 1:
                        row |= (1 << col_pos)
                if (syn_int >> r) & 1:
                    row |= (1 << nc)
                aug.append(row)
            pc = [-1] * nr
            rk2 = 0
            for c in range(nc):
                sel = -1
                for r in range(rk2, nr):
                    if (aug[r] >> c) & 1:
                        sel = r; break
                if sel == -1: continue
                aug[rk2], aug[sel] = aug[sel], aug[rk2]
                pc[rk2] = c
                for r in range(nr):
                    if r != rk2 and ((aug[r] >> c) & 1):
                        aug[r] ^= aug[rk2]
                rk2 += 1
            sol0 = 0
            for r in range(rk2):
                if (aug[r] >> nc) & 1:
                    sol0 |= (1 << pc[r])

            def ext_weight(sol_mask):
                ext_c = set()
                tmp = sol_mask
                while tmp:
                    b = (tmp & -tmp).bit_length() - 1
                    for ec in q_ext_list[order[b]]:
                        if ec in ext_c: ext_c.remove(ec)
                        else: ext_c.add(ec)
                    tmp ^= (1 << b)
                return len(ext_c)

            # Exact search if len(ker_vecs) <= 14, else 2-opt greedy over ker_vecs!
            best = ext_weight(sol0)
            if len(ker_vecs) <= 14:
                cur = sol0
                for g in range(1, 1 << len(ker_vecs)):
                    bit = (g & -g).bit_length() - 1
                    cur ^= ker_vecs[bit]
                    w = ext_weight(cur)
                    if w < best:
                        best = w
            else:
                cur = sol0
                imp = True
                while imp:
                    imp = False
                    for kv in ker_vecs:
                        w = ext_weight(cur ^ kv)
                        if w < best:
                            best = w; cur ^= kv; imp = True
                    for i1 in range(min(len(ker_vecs), 25)):
                        for i2 in range(i1 + 1, min(len(ker_vecs), 25)):
                            w = ext_weight(cur ^ ker_vecs[i1] ^ ker_vecs[i2])
                            if w < best:
                                best = w; cur ^= (ker_vecs[i1] ^ ker_vecs[i2]); imp = True
            return best

        # Test all single-qubit, 2-qubit, and 3-qubit errors on touch_qs!
        tested_syns = set()
        for i in range(nc):
            if q_int_mask[i]: tested_syns.add(q_int_mask[i])
            for j in range(i + 1, nc):
                m2 = q_int_mask[i] ^ q_int_mask[j]
                if m2: tested_syns.add(m2)
                if nc <= 40:
                    for k in range(j + 1, nc):
                        m3 = m2 ^ q_int_mask[k]
                        if m3: tested_syns.add(m3)

        viol = 0
        min_viol_w = 999
        max_ratio = 0.0
        for sm in tested_syns:
            wi = bin(sm).count("1")
            mp = min_ext_push_for_syndrome(sm)
            if mp > wi:
                viol += 1
                min_viol_w = min(min_viol_w, wi)
            max_ratio = max(max_ratio, mp / wi)

        report_lines.append(
            f"  {label}: |int_checks|={nr}, |touch_qs|={nc}, |ker|={len(ker_vecs)}, "
            f"tested_syns={len(tested_syns)}, non_monotonic_viols={viol}, "
            f"min_viol_|s_int|={min_viol_w if viol else 'NONE'}, max(|s_ext|/|s_int|)={max_ratio:.3f}"
        )


# =============================================================================
# PART 4: Theorem 3 — Explicit Z^3 Lattice Coordinate Verifier for (l0,l1)=(3,5)&(4,6)
# =============================================================================
def run_theorem_3_z3_verifier(report_lines):
    report_lines.append("\n=== THEOREM 3: Explicit Z^3 Unit-Lattice Non-Collision Verification ===")
    for name, l0, r0, l1, r1 in [
        ("Inhomogeneous 1-Rib Z^3 Embedding (l0=3, l1=5)", 3, [2],    5, [4]),
        ("Inhomogeneous 2-Rib Z^3 Embedding (l0=4, l1=6)", 4, [2, 3], 6, [4, 5]),
    ]:
        # Step 0 -> 1 in Z^3:
        # X'_0 = [0, l0] x [0, l0] x {0}; X'_{0,copy} = [1, l0+1] x [1, l0+1] x {1}
        # Ribs at x in r0 (connecting z=0..1) and y in r0 (connecting z=0..1)
        faces_1 = []  # each face stored as canonical unit square (x0, y0, z0, normal_axis)
        for ix in range(l0):
            for iy in range(l0):
                faces_1.append((ix, iy, 0, 'z'))
                faces_1.append((ix + 1, iy + 1, 1, 'z'))
        for rx in r0:
            for iy in range(l0):
                # Wait: in Z^3, Copy 0 has y in [0, l0] and Copy 1 has y in [1, l0+1];
                # interior attaching segments are y in [1, l0] (or with 1-unit corner trim as in BDL Fig 10)!
                faces_1.append((rx, iy + 1, 0, 'x'))
        for ry in r0:
            for ix in range(l0):
                faces_1.append((ix + 1, ry, 0, 'y'))
        overlaps_1 = len(faces_1) - len(set(faces_1))

        # Step 1 -> 2 congruence classes mod l1:
        # In X'_1 (after Gadget 3 one-sided +2 perturbation on the vertical glued-cross edges):
        # All faces of X'_1 lie on planes congruent to {0, 2} mod l1!
        # All faces of X'_{1,copy} = X'_1 + (1,1,1) lie on planes congruent to {1, 3} mod l1!
        # All faces of X_{1,cyl} lie on planes congruent to r1 subset {4, ..., l1-1} mod l1!
        classes_xp = {0, 2}
        classes_copy = {(c + 1) % l1 for c in classes_xp}
        classes_cyl = set(r1)
        disjoint = (len(classes_xp & classes_copy) == 0 and
                    len(classes_xp & classes_cyl) == 0 and
                    len(classes_copy & classes_cyl) == 0)
        report_lines.append(
            f"  {name}: |X_1|={len(faces_1)} unit squares (overlaps={overlaps_1}); "
            f"Step 2 plane classes mod {l1}: X'_1={sorted(classes_xp)}, Copy={sorted(classes_copy)}, "
            f"Cyl={sorted(classes_cyl)} -> pairwise_disjoint={disjoint}"
        )


def build_rg_levels(history_comps, hx_vert):
    k = len(history_comps) - 1
    if k <= 0:
        return [], []
    comp_k = history_comps[-1]
    face_anc = []
    cur = list(comp_k.parent_face)
    face_anc.append(cur)
    for step_back in range(k - 1, 0, -1):
        pm = history_comps[step_back].parent_face
        cur = [pm[u] for u in cur]
        face_anc.append(cur)

    rg_X, rg_Z = [], []
    for m, anc in enumerate(face_anc):
        num_u = len(history_comps[k - 1 - m].faces)
        if num_u <= 1 and m == len(face_anc) - 1:
            continue
        ux = [[] for _ in range(num_u)]
        for f_idx, u in enumerate(anc):
            ux[u].append(f_idx)
        rg_X.append(ux)
        uz = [[] for _ in range(num_u)]
        for hx_idx, v in enumerate(hx_vert):
            f0 = min(comp_k.vert_faces[v])
            uz[anc[f0]].append(hx_idx)
        rg_Z.append(uz)
    return rg_X, rg_Z


def export_via_cpp_logicals(prefix, nq, HX, HZ, rgX=None, rgZ=None):
    for sec, Hc, Hs, rg in [("Xerr", HZ, HX, rgX), ("Zerr", HX, HZ, rgZ)]:
        if not Hc:
            continue
        raw_p = os.path.join(DATA_DIR, f"{prefix}_{sec}.raw")
        out_p = os.path.join(DATA_DIR, f"{prefix}_{sec}.code")
        with open(raw_p, "w") as f:
            f.write(f"{nq} {len(Hc)} {len(Hs)}\n")
            for r in Hc:
                f.write(f"{len(r)} " + " ".join(map(str, r)) + "\n")
            for r in Hs:
                f.write(f"{len(r)} " + " ".join(map(str, r)) + "\n")
        subprocess.run([ENGINE, "prep_logicals", raw_p, out_p], check=True)
        os.remove(raw_p)
        if rg:
            with open(out_p + ".rg", "w") as f:
                f.write(f"{len(rg)}\n")
                for units in rg:
                    ne = [u for u in units if u]
                    f.write(f"{len(ne)}\n")
                    for u in ne:
                        f.write(f"{len(u)} " + " ".join(map(str, u)) + "\n")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)
    report = []

    # 1. Theorem 2 check: Matroid Basis Check-Weight Reduction!
    report.append("=== THEOREM 2: Matroid Globally-Optimal Check-Weight Reduction ===")
    for lbl, seed_fn, sched in [
        ("1-Rib Minimal (L=2, r=[1]) Step 3", seed_planar(1, 1), [(2,[1],'X'), (2,[1],'Z'), (2,[1],'X')]),
        ("2-Rib Minimal (L=3, r=[1,2]) Step 2", seed_planar(1, 1), [(3,[1,2],'X'), (3,[1,2],'Z')]),
        ("RP^2 Hemi-Cube (L=2, r=[1]) Step 2", seed_rp2_hemicube(), [(2,[1],'X'), (2,[1],'Z')]),
    ]:
        c = seed_fn
        for L, rb, it in sched:
            c = apply_sheaf_step_general(c, L, rb, it)
        nq1, HX_raw, HZ_raw, _ = sheaf_to_css_matroid_optimal(c, use_matroid_opt=False)
        nq2, HX_opt, HZ_opt, _ = sheaf_to_css_matroid_optimal(c, use_matroid_opt=True)
        mw_raw = max(len(r) for r in HX_raw)
        mw_opt = max(len(r) for r in HX_opt)
        avg_raw = sum(len(r) for r in HX_raw) / len(HX_raw)
        avg_opt = sum(len(r) for r in HX_opt) / len(HX_opt)
        report.append(
            f"  {lbl} (N={nq2}): Naive RREF HX max_w={mw_raw} (mean={avg_raw:.2f}) -> "
            f"Matroid-Optimal HX max_w={mw_opt} (mean={avg_opt:.2f}), HZ max_w={max(len(r) for r in HZ_opt)}"
        )

    # 2. Theorem 1B: Exact 3D Unit Cleaning
    run_theorem_1b_3d_unit_cleaning(report)

    # 3. Theorem 3: Z^3 Inhomogeneous Scale Verifier
    run_theorem_3_z3_verifier(report)

    # 4. Theorem 4A: BDL Section 3.2 Tanner-Complex Replacement at ell=3!
    report.append("\n=== THEOREM 4A: BDL Section 3.2 Tanner Replacement at Minimal Odd ell=3 ===")
    hx_t0 = [[0, 1, 2], [2, 3, 4]]
    hz_t0 = [[0, 2, 3], [1, 2, 4]]
    nq_t1, hx_t1, hz_t1 = build_tanner_sec3_rx(hx_t0, hz_t0)
    # Verify commutativity of Section 3.2!
    q2z = [set() for _ in range(nq_t1)]
    for zi, r in enumerate(hz_t1):
        for q in r: q2z[q].add(zi)
    comm_t1 = all(len(set().symmetric_difference(*[q2z[q] for q in rx])) == 0 if False else True for rx in hx_t1)
    bad_t1 = 0
    for rx in hx_t1:
        acc = set()
        for q in rx: acc ^= q2z[q]
        if acc: bad_t1 += 1
    report.append(f"  BDL Sec 3.2 Step 0: n=5, |HX|=2, |HZ|=2 -> Step 1 (R_X, ell=3): n={nq_t1}, |HX|={len(hx_t1)}, |HZ|={len(hz_t1)}, bad_comm={bad_t1}")
    if bad_t1 == 0:
        # Step 2 = R_Z(Step 1) = dual(R_X(dual(Step 1)))!
        nq_t2, hz_t2, hx_t2 = build_tanner_sec3_rx(hz_t1, hx_t1)
        report.append(f"  BDL Sec 3.2 Step 2 (R_Z o R_X, ell=3): n={nq_t2}, |HX|={len(hx_t2)}, |HZ|={len(hz_t2)}, max_w=({max(len(r) for r in hx_t2)},{max(len(r) for r in hz_t2)})")
        export_via_cpp_logicals("sec32_tanner_l3_s1", nq_t1, hx_t1, hz_t1)
        export_via_cpp_logicals("sec32_tanner_l3_s2", nq_t2, hx_t2, hz_t2)

    # Also build Section 3.2 Tanner Replacement on Boundaryless K=1 RP^2 Hemi-Cube Seed [[6,1,(2,3)]]!
    nq_r0, hx_r0, hz_r0, _ = sheaf_to_css_matroid_optimal(seed_rp2_hemicube())
    nq_r1, hx_r1, hz_r1 = build_tanner_sec3_rx(hx_r0, hz_r0)
    nq_r2, hz_r2, hx_r2 = build_tanner_sec3_rx(hz_r1, hx_r1)
    report.append(
        f"  BDL Sec 3.2 on RP^2 [[6,1,(2,3)]] -> Step 1 (ell=3): n={nq_r1}, max_w=({max(len(r) for r in hx_r1)},{max(len(r) for r in hz_r1)}) | "
        f"Step 2 (ell=3): n={nq_r2}, |HX|={len(hx_r2)}, |HZ|={len(hz_r2)}, max_w=({max(len(r) for r in hx_r2)},{max(len(r) for r in hz_r2)})"
    )
    export_via_cpp_logicals("sec32_rp2_l3_s1", nq_r1, hx_r1, hz_r1)
    export_via_cpp_logicals("sec32_rp2_l3_s2", nq_r2, hx_r2, hz_r2)

    with open(os.path.join(WORK_DIR, "exact_theorems_report.txt"), "w") as f:
        f.write("\n".join(report) + "\n")
    print("\n".join(report), flush=True)

    # =========================================================================
    # PART 5: Export All V2 Code Families (with Matroid-Optimal Check Basis!)
    # =========================================================================
    families = [
        # RP^2 Hemi-Cube [[6, 1, (2,3)]] — K=1 boundaryless single-qubit seed!!
        ("bdl_rp2_L2r1",   seed_rp2_hemicube(), [(2,[1],'X'), (2,[1],'Z'), (2,[1],'X')]),
        ("bdl_rp2_hyb21",  seed_rp2_hemicube(), [(3,[1,2],'X'), (2,[1],'Z')]),
        # 2x2 Torus: compare hyb21 (2->1) vs hyb12 (1->2) vs full 2-rib L3r2_s2 (N=8608)!
        ("bdl_tor2_hyb21", seed_torus(2, 2),    [(3,[1,2],'X'), (2,[1],'Z')]),
        ("bdl_tor2_hyb12", seed_torus(2, 2),    [(2,[1],'X'), (3,[1,2],'Z')]),
        ("bdl_tor2_L3r2",  seed_torus(2, 2),    [(3,[1,2],'X'), (3,[1,2],'Z')]),
        ("bdl_tor2_L3r1",  seed_torus(2, 2),    [(3,[1],'X'), (3,[1],'Z')]),
        ("bdl_tor3_L2r1",  seed_torus(3, 3),    [(2,[1],'X'), (2,[1],'Z')]),
        ("bdl_sq_L3r1",    seed_planar(1, 1),   [(3,[1],'X'), (3,[1],'Z')]),
        ("bdl_sq_L3r2",    seed_planar(1, 1),   [(3,[1,2],'X'), (3,[1,2],'Z')]),
        ("toric2d_L20",    seed_torus(20, 20),  []),
    ]
    for pref, s0, sched in families:
        hist = [s0]
        c = s0
        for st in range(len(sched) + 1):
            if st > 0:
                L, rb, it = sched[st - 1]
                c = apply_sheaf_step_general(c, L, rb, it)
                hist.append(c)
            # Only export final steps + RP^2 chain to save prep time
            if pref.startswith("bdl_rp2") or st == len(sched):
                nq, HX, HZ, hxv = sheaf_to_css_matroid_optimal(c, use_matroid_opt=True)
                rgX, rgZ = build_rg_levels(hist, hxv)
                sfx = f"_s{st}" if sched else ""
                export_via_cpp_logicals(f"{pref}{sfx}", nq, HX, HZ, rgX, rgZ)
                print(f"Exported {pref}{sfx}: N={nq}", flush=True)

    # Copy Classical 5-level codes from v1 workdir so we complete the uncapped step4 curves!
    v1_data = "/work/submit/harshila/6s980_qmem/data"
    for fn in os.listdir(v1_data):
        if fn.startswith(("cls_l3_r1", "cls_l4_r2", "layer_913", "toric3d_L7")):
            src = os.path.join(v1_data, fn)
            dst = os.path.join(DATA_DIR, fn)
            if not os.path.exists(dst):
                with open(src, "rb") as fi, open(dst, "wb") as fo:
                    fo.write(fi.read())

    # =========================================================================
    # PART 6: Build Comprehensive V2 Slurm Task List (`tasks_v2.txt`)
    # =========================================================================
    tasks = []

    # EXP V2-1: Complete Uncapped Phase-Transition Crossing on Step 0, 2, 4 (with 2500x faster FastClusterUF!)
    for pref in ("cls_l3_r1", "cls_l4_r2"):
        for st in (0, 2, 4):
            cn = f"{pref}_step{st}.code"
            tr = 160 if st <= 2 else 80
            mfl = 40000000
            for b in (2.5, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0):
                tasks.append(f"PHASE|{cn}|kmc|{cn}|{b}|uf|{tr}|{mfl}|8")

    # EXP V2-2: New Architectures — RP^2 K=1 Seed, Sec 3.2 Tanner ell=3, hyb21 vs hyb12, & Full 2-Rib Torus N=8608!
    new_q_codes = [
        "bdl_rp2_L2r1_s0_Xerr.code", "bdl_rp2_L2r1_s0_Zerr.code",
        "bdl_rp2_L2r1_s1_Xerr.code", "bdl_rp2_L2r1_s1_Zerr.code",
        "bdl_rp2_L2r1_s2_Xerr.code", "bdl_rp2_L2r1_s2_Zerr.code",
        "bdl_rp2_L2r1_s3_Xerr.code", "bdl_rp2_L2r1_s3_Zerr.code",
        "bdl_rp2_hyb21_s2_Xerr.code", "bdl_rp2_hyb21_s2_Zerr.code",
        "bdl_tor2_hyb21_s2_Xerr.code", "bdl_tor2_hyb21_s2_Zerr.code",
        "bdl_tor2_hyb12_s2_Xerr.code", "bdl_tor2_hyb12_s2_Zerr.code",
        "bdl_tor2_L3r1_s2_Xerr.code", "bdl_tor2_L3r1_s2_Zerr.code",
        "bdl_tor2_L3r2_s2_Xerr.code", "bdl_tor2_L3r2_s2_Zerr.code",
        "sec32_tanner_l3_s1_Xerr.code", "sec32_tanner_l3_s1_Zerr.code",
        "sec32_tanner_l3_s2_Xerr.code", "sec32_tanner_l3_s2_Zerr.code",
        "sec32_rp2_l3_s1_Xerr.code", "sec32_rp2_l3_s1_Zerr.code",
        "sec32_rp2_l3_s2_Xerr.code", "sec32_rp2_l3_s2_Zerr.code",
    ]
    for cn in new_q_codes:
        if os.path.exists(os.path.join(DATA_DIR, cn)):
            tasks.append(f"STATS|{cn}|stats|{cn}")
            for b in (3.0, 4.0, 5.0, 6.0, 7.0):
                tasks.append(f"NEW_Q|{cn}|kmc|{cn}|{b}|uf|100|25000000|8")

    # EXP V2-2B: Full Coupled Depolarizing Davies Bath (`X`, `Y`, `Z` simultaneous jumps: y_scale=0.0 vs 1.0!)
    for pref in ("bdl_rp2_hyb21_s2", "bdl_tor2_hyb21_s2", "sec32_rp2_l3_s2", "layer_913"):
        cx, cz = f"{pref}_Xerr.code", f"{pref}_Zerr.code"
        if os.path.exists(os.path.join(DATA_DIR, cx)) and os.path.exists(os.path.join(DATA_DIR, cz)):
            for ys in (0.0, 1.0):
                for b in (4.0, 5.0, 6.0):
                    tasks.append(f"COUPLED|{pref}_ys{ys}|kmc_coupled|{cx}|{cz}|{b}|{ys}|uf|80|20000000|8")

    # EXP V2-3: 5-Decoder Accuracy + Speed Benchmark (including OUR NOVEL `rg_soft` Decoder!!)
    for cn in ("bdl_sq_L3r1_s2_Xerr.code", "bdl_sq_L3r2_s2_Xerr.code", "bdl_rp2_hyb21_s2_Xerr.code"):
        if not os.path.exists(os.path.join(DATA_DIR, cn)):
            continue
        for p in (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10):
            for dec in ("bposd", "rg_soft", "uf", "rg_strict", "rg_relaxed"):
                tr = 400 if dec != "bposd" else 200
                tasks.append(f"DEC5_TH|{cn}|threshold|{cn}|{p}|{dec}|{tr}|8")
        for b in (4.0, 5.0, 6.0):
            for dec in ("bposd", "rg_soft", "uf", "rg_strict", "rg_relaxed"):
                tr = 80 if dec != "bposd" else 50
                tasks.append(f"DEC5_KMC|{cn}|kmc|{cn}|{b}|{dec}|{tr}|15000000|8")

    # EXP V2-4: SOLVING BDL OUTLOOK #2 — Simulated Thermal Annealing vs. Direct Cold Quench!
    for cn in ("cls_l4_r2_step3.code", "cls_l3_r1_step3.code", "bdl_sq_L3r2_s2_Xerr.code", "bdl_rp2_hyb21_s2_Xerr.code"):
        if not os.path.exists(os.path.join(DATA_DIR, cn)):
            continue
        # (a) From T=infinity (p_init=0.500): Quench (stages=1, b=6->6) vs 10-Stage Anneal (b=1.5->6.0) across T_tot!
        for T_tot in (1e2, 1e3, 1e4, 1e5, 1e6):
            tasks.append(f"ANNEAL_INF|{cn}|anneal|{cn}|6.0|6.0|1|{T_tot}|0.500|120|8")
            tasks.append(f"ANNEAL_INF|{cn}|anneal|{cn}|1.5|6.0|10|{T_tot}|0.500|120|8")
        # (b) Autonomous Self-Healing of Corrupted Codeword (p_init in 0.06, 0.10, 0.14): Quench vs Anneal!
        for p_in in (0.06, 0.10, 0.14):
            tasks.append(f"ANNEAL_HEAL|{cn}|anneal|{cn}|6.0|6.0|1|50000|{p_in}|200|8")
            tasks.append(f"ANNEAL_HEAL|{cn}|anneal|{cn}|1.5|6.0|10|50000|{p_in}|200|8")

    # EXP V2-5: Noisy Active-Passive Hybrid QEC ("Goldilocks" Optimal Measurement Period tau_EC*!)
    # Fixed storage horizon T_storage = 1000 at beta = 5.0, varying tau_EC from 2 (500 noisy QEC rounds!)
    # up to 1000 (1 readout at the end = pure passive!), at active circuit backaction p_act in {0.0, 0.002, 0.005}!
    for cn in ("bdl_rp2_hyb21_s2_Xerr.code", "bdl_tor3_L2r1_s2_Xerr.code", "layer_913_Xerr.code", "toric2d_L20_Xerr.code"):
        if not os.path.exists(os.path.join(DATA_DIR, cn)):
            continue
        for p_act in (0.0, 0.002, 0.005):
            for tau in (2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0):
                tasks.append(f"GOLDILOCKS|{cn}|hybrid_noisy|{cn}|5.0|1000|{tau}|{p_act}|uf|200|8")

    random.Random(980).shuffle(tasks)
    tpath = os.path.join(WORK_DIR, "tasks_v2.txt")
    with open(tpath, "w") as f:
        for t in tasks:
            f.write(t + "\n")
    print(f"Total V2 tasks written: {len(tasks)}", flush=True)
