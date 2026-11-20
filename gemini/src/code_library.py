#!/usr/bin/env python3
"""
Unified code family generator for evaluating hypothetical research questions on 3D Quantum Memories:
  1. BDL 3D Quantum Sheaf Codes Q_{(k)} (arbitrary seed, arbitrary L_i and rib count r_i, plus RG hierarchy)
  2. BDL 2D Classical Dual-Self-Correcting Codes C_{(k)} (ell=3 [1 rib] vs ell=4 [2 ribs])
  3. 3D Layer Codes L(C_in) (Williamson-Baspin / Gu-Kubica arXiv:2510.06659)
  4. 2D Toric / Planar Surface Codes & 3D Toric Codes (baseline benchmarks)

Exports compact text code files for our C++20 Kinetic Monte Carlo + Multi-Decoder simulator.
"""

from collections import defaultdict
import os


class OrientedCubical2D:
    """
    General 2D square cell complex supporting multi-edges (needed for e.g. 2x2 Torus [[8,2,2]]!):
      - num_vertices: int
      - edges: list of (tail, head) directed edges
      - edge_colors: list of 'none', 'blue', 'red'
      - faces: list of (v00, v10, v11, v01, e_bot, e_top, e_lft, e_rgt)
        where:
          e_bot has direction v00 -> v10
          e_top has direction v01 -> v11
          e_lft has direction v00 -> v01
          e_rgt has direction v10 -> v11
      - parent_face_of_face: optional list mapping each face f in X_{(i)} to its parent face F in X_{(i-1)}
        (used by the recursive Sheaf-RG decoder!)
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

        for f_idx, fdata in enumerate(faces):
            v00, v10, v11, v01, eb, et, el, er = fdata
            assert edges[eb] == (v00, v10), f"Face {f_idx} eb mismatch"
            assert edges[et] == (v01, v11), f"Face {f_idx} et mismatch"
            assert edges[el] == (v00, v01), f"Face {f_idx} el mismatch"
            assert edges[er] == (v10, v11), f"Face {f_idx} er mismatch"
            for e_idx in (eb, et, el, er):
                self.edge_faces[e_idx].append(f_idx)
            for v in {v00, v10, v11, v01}:
                self.vert_faces[v].append(f_idx)


def seed_planar_surface(dx: int, dz: int) -> OrientedCubical2D:
    """dx x dz planar surface code sheaf seed (dx=1, dz=1 is BDL's single square [[2,1,1]])."""
    def vid(ix, iz):
        return iz * (dx + 1) + ix

    num_v = (dx + 1) * (dz + 1)
    edges = []
    colors = []
    h_edge = {}
    v_edge = {}

    for iz in range(dz + 1):
        for ix in range(dx):
            e_idx = len(edges)
            edges.append((vid(ix, iz), vid(ix + 1, iz)))
            colors.append('blue' if (iz == 0 or iz == dz) else 'none')
            h_edge[(ix, iz)] = e_idx

    for iz in range(dz):
        for ix in range(dx + 1):
            e_idx = len(edges)
            edges.append((vid(ix, iz), vid(ix, iz + 1)))
            colors.append('red' if (ix == 0 or ix == dx) else 'none')
            v_edge[(ix, iz)] = e_idx

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
    """
    Boundaryless Lx x Lz Torus T^2 sheaf seed ([[2*Lx*Lz, 2, min(Lx,Lz)]], exact Delta=(2,2) for Lx,Lz>=2!).
    Note: embeds cleanly in Z^3 without self-intersections as a hollow tube!
    """
    assert Lx >= 2 and Lz >= 2

    def vid(ix, iz):
        return (iz % Lz) * Lx + (ix % Lx)

    num_v = Lx * Lz
    edges = []
    colors = []
    h_edge = {}
    v_edge = {}

    for iz in range(Lz):
        for ix in range(Lx):
            e_idx = len(edges)
            edges.append((vid(ix, iz), vid(ix + 1, iz)))
            colors.append('none')
            h_edge[(ix, iz)] = e_idx

    for iz in range(Lz):
        for ix in range(Lx):
            e_idx = len(edges)
            edges.append((vid(ix, iz), vid(ix, iz + 1)))
            colors.append('none')
            v_edge[(ix, iz)] = e_idx

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
    """Apply R_X ('X') or R_Z ('Z') with subdivision L and interior rib coordinates `rib_coords`."""
    assert iter_type in ('X', 'Z')
    new_col = 'blue' if iter_type == 'X' else 'red'
    rib_set = set(rib_coords)

    xp_num_v = 0
    def add_xp_v():
        nonlocal xp_num_v
        r = xp_num_v
        xp_num_v += 1
        return r

    orig_v = [add_xp_v() for _ in range(comp.num_vertices)]

    xp_edges = []       # (u, w)
    xp_colors = []
    xp_is_rib = []
    edge_sub_verts = []
    edge_sub_eids = []
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

    xp_faces = []  # (v00, v10, v11, v01, eb, et, el, er, orig_f_idx)
    xp_edge_parent_face = {}  # for interior edges of X', which parent face F they belong to

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

    # Doubling: Copy 0 and Copy 1 + Cylinder
    total_v = 2 * xp_num_v
    def vc(v, c):
        return v + c * xp_num_v

    out_edges = []
    out_colors = []
    out_faces = []
    parent_face = []

    n_xp_e = len(xp_edges)
    # Copy 0 edges: indices 0 .. n_xp_e - 1
    # Copy 1 edges: indices n_xp_e .. 2*n_xp_e - 1
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

    # Rib vertices and cylinder vertical edges
    rib_verts = set()
    for eid, isr in enumerate(xp_is_rib):
        if isr:
            u, w = xp_edges[eid]
            rib_verts.add(u)
            rib_verts.add(w)

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

    # Cylinder faces
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
    """Build exact CSS matrices (num_qubits, H_X, H_Z) from OrientedCubical2D."""
    num_faces = len(comp.faces)
    edge_qstart = []
    qubits = []

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
        else:
            raise ValueError(f"Bad edge {e_idx}: d={d}, col={col}")

    num_q = len(qubits)
    H_Z = [[] for _ in range(num_faces)]
    for q_idx, (_, vec) in enumerate(qubits):
        for f_idx in vec:
            H_Z[f_idx].append(q_idx)

    H_X = []
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

    return num_q, H_X, H_Z


def make_3d_toric_code(L: int):
    """
    3D Toric Code on L x L x L periodic cubic lattice (qubits on edges, n = 3 L^3, k = 3):
      - H_X: vertex star checks (weight 6, L^3 checks) -> detects Z-errors (string-like, NOT self-correcting, Delta_Z = 2!)
      - H_Z: face plaquette checks (weight 4, 3 L^3 checks) -> detects X-errors (membrane-like, SELF-CORRECTING, Delta_X = 2*L or L+2!)
    Wait: check exact convention:
      If qubits are on EDGES:
        - X-check at vertex v = product of X on 6 incident edges -> Z-errors are closed loops (d_Z = L, energy barrier = 2, O(1)!)
        - Z-check on plaquette p = product of Z on 4 boundary edges -> X-errors are dual membranes (d_X = L^2, energy barrier = Theta(L)!)
    """
    def eid_x(x, y, z): return 0 * (L**3) + ((z % L) * L + (y % L)) * L + (x % L)
    def eid_y(x, y, z): return 1 * (L**3) + ((z % L) * L + (y % L)) * L + (x % L)
    def eid_z(x, y, z): return 2 * (L**3) + ((z % L) * L + (y % L)) * L + (x % L)

    nq = 3 * (L ** 3)
    H_X = []
    for z in range(L):
        for y in range(L):
            for x in range(L):
                H_X.append(sorted([
                    eid_x(x, y, z), eid_x(x - 1, y, z),
                    eid_y(x, y, z), eid_y(x, y - 1, z),
                    eid_z(x, y, z), eid_z(x, y, z - 1),
                ]))

    H_Z = []
    for z in range(L):
        for y in range(L):
            for x in range(L):
                # xy plaquette
                H_Z.append(sorted([eid_x(x, y, z), eid_y(x + 1, y, z), eid_x(x, y + 1, z), eid_y(x, y, z)]))
                # xz plaquette
                H_Z.append(sorted([eid_x(x, y, z), eid_z(x + 1, y, z), eid_x(x, y, z + 1), eid_z(x, y, z)]))
                # yz plaquette
                H_Z.append(sorted([eid_y(x, y, z), eid_z(x, y + 1, z), eid_y(x, y, z + 1), eid_z(x, y, z)]))

    return nq, H_X, H_Z


def make_layer_code_from_css(n_in: int, HX_in: list, HZ_in: list):
    """
    Exact 3D Layer Code L(C_in) via the 2D Topological Defect Network / Sheaf Code definition
    (Williamson-Baspin 2023 / Gu-Kubica arXiv:2510.06659 Appendix A / BDL Section 2.4)!
    WAIT A SECOND — LOOK HOW CLEAN LAYER CODES ARE IN OUR EXACT `OrientedCubical2D` SHEAF FRAMEWORK!!
    Let's check Williamson-Baspin / Gu-Kubica Appendix A:
      Given an input CSS code C_in with:
        - n_in qubits (j = 0 .. n_in - 1) -> n_in vertical 'Q-layers' (say planes x = j)
        - m_X X-checks (a = 0 .. m_X - 1) -> m_X horizontal 'X-layers' (say planes y = a)
        - m_Z Z-checks (b = 0 .. m_Z - 1) -> m_Z horizontal 'Z-layers' (say planes y = m_X + b)
      Wait: in Layer Codes, X-layers sit in one coordinate (say y = a) and Z-layers sit in the OTHER
      transverse coordinate (say z = b)? NO! Look at Williamson-Baspin / Gu-Kubica Fig. 1 & Appendix A:
      Let's read lines 427-524 of `papers/layer_2510/arxiv_v2.tex` before implementing Layer Codes
      so we match `arXiv:2510.06659` 100% identically!!
    """
    pass


if __name__ == "__main__":
    for name, s in [
        ("2x2 Torus [[8,2,2]]", seed_torus(2, 2)),
        ("3x3 Torus [[18,2,3]]", seed_torus(3, 3)),
    ]:
        nq, HX, HZ = sheaf_to_css(s)
        print(f"{name}: n={nq}, |HX|={len(HX)}, |HZ|={len(HZ)}")
        c1 = apply_sheaf_step(s, 2, [1], 'X')
        nq1, HX1, HZ1 = sheaf_to_css(c1)
        c2 = apply_sheaf_step(c1, 2, [1], 'Z')
        nq2, HX2, HZ2 = sheaf_to_css(c2)
        print(f"  after R_X (L=2,r=1): n={nq1}, |HX|={len(HX1)}, |HZ|={len(HZ1)}")
        print(f"  after R_Z o R_X (L=2,r=1): n={nq2}, |HX|={len(HX2)}, |HZ|={len(HZ2)}")
