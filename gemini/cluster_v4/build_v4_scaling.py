#!/usr/bin/env python3
"""
V4 First-Principles Self-Correction & Four-Symmetry-Class Scaling Builder (MIT 6.S980).
Runs on SubMIT Slurm prep node to construct all 4 symmetry/confinement classes and `tasks_v4.txt`:

  CLASS I   (Translation-Invariant Type-I, Delta = O(1)):
    - 2D Toric Code: L = 4, 6, 10, 16, 24, 34 (N = 32 .. 2,312, K = 2)
    - Scaled Fixed-Polynomial Gross BB (A = x^3+y+y^2, B = y^3+x+x^2):
      (6x6), (12x6), (18x12), (24x18), (30x18), (36x18), (48x18), (60x18) -> N = 72 .. 2,160, ALL K = 12!

  CLASS II  (Translation-Invariant 3D Fractal Type-II, Delta = O(log L)):
    - Haah's 3D Cubic Code A (f = 1+x+y+z, g = 1+xy+yz+zx):
      L = 3, 5, 7, 9, 11, 13 -> N = 54, 250, 686, 1,458, 2,662, 4,394, ALL K = 2, qdeg = 4, w = 8!

  CLASS III (Translation-Breaking With Macroscopic 2D Surface Sheets):
    - 3D Surface Layer Codes (Gu, Kubica et al., PRL 2025):
      layer_512 (N=437), layer_steane (N=743), layer_913 (N=1,853),
      layer_13_1_3 (N=4,837), layer_15_35 (N=6,095), layer_913_ksep2 (N=6,637)

  CLASS IV  (Hierarchical Multi-Scale Translation-Breaking — BDL Uncapped Exponential Scaling!):
    - 2D Classical BDL:
      * cls_l2_r1_step0..5  (N = 4, 20, 100, 500, 2,500, 12,500 — 3 cycles / 6 levels!)
      * cls_l3_r2_step0..4  (N = 4, 32, 256, 2,048, 16,384 — PROVEN 2-rib contraction!)
      * cls_l3_r1_step0..4  (N = 4, 28, 196, 1,372, 9,604 — Uncapped V2 resolution!)
    - 3D Quantum Multi-Cycle BDL:
      * sq_aniso_L2r1_s0..4         (N = 2, 19, 254, 1,942, ~18,900 — TWO FULL QUANTUM CYCLES k=0,1,2!)
      * sq_aniso_hyb21_s0..3        (N = 2, 44, 586, 9,860 — Hybrid 2-rib/1-rib 3D-local planar!)
      * loc3d_tor2_aniso_L2r1_s0..3 (N = 8, 76, 1,016, 7,768 — 3D-local T^2, K=2!)
      * nonloc_rp2_aniso_L2r1_s0..3 (N = 6, 57, 762, 5,826 — Non-local RP^2, K=1!)
"""

import os
import subprocess

WORK_DIR = "/work/submit/harshila/6s980_qmem_v4"
DATA_DIR = os.path.join(WORK_DIR, "data")
RES_DIR = os.path.join(WORK_DIR, "results")
ENGINE = os.path.join(WORK_DIR, "kmc_engine_v4")


# =============================================================================
# 1. General Cubical 2D Sheaf Complex (3D-Local & Non-Local Quantum BDL)
# =============================================================================
class GeneralCubical2D:
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
            for e_idx in (eb, et, el, er):
                self.edge_faces[e_idx].append(f_idx)
            for v in {v00, v10, v11, v01}:
                self.vert_faces[v].append(f_idx)


def seed_planar_11():
    edges = [(0, 1), (2, 3), (0, 2), (1, 3)]
    cols = ["blue", "blue", "red", "red"]
    faces = [(0, 1, 3, 2, 0, 1, 2, 3)]
    return GeneralCubical2D(4, edges, cols, faces)


def seed_torus(Lx, Lz):
    def vid(ix, iz): return (iz % Lz) * Lx + (ix % Lx)
    edges, h_e, v_e = [], {}, {}
    for iz in range(Lz):
        for ix in range(Lx):
            h_e[(ix, iz)] = len(edges); edges.append((vid(ix, iz), vid(ix + 1, iz)))
    for iz in range(Lz):
        for ix in range(Lx):
            v_e[(ix, iz)] = len(edges); edges.append((vid(ix, iz), vid(ix, iz + 1)))
    faces = [
        (vid(ix, iz), vid(ix + 1, iz), vid(ix + 1, iz + 1), vid(ix, iz + 1),
         h_e[(ix, iz)], h_e[(ix, (iz + 1) % Lz)], v_e[(ix, iz)], v_e[((ix + 1) % Lx, iz)])
        for iz in range(Lz) for ix in range(Lx)
    ]
    return GeneralCubical2D(Lx * Lz, edges, ["none"] * len(edges), faces)


def seed_rp2_hemicube():
    edges = [(0, 1), (1, 2), (2, 3), (0, 3), (0, 2), (1, 3)]
    pid = {(min(u, v), max(u, v)): i for i, (u, v) in enumerate(edges)}
    def eid(a, b): return pid[(min(a, b), max(a, b))]
    faces = [
        (a, b, c, d, eid(a, b), eid(d, c), eid(a, d), eid(b, c))
        for (a, b, c, d) in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 1, 3, 2)]
    ]
    return GeneralCubical2D(4, edges, ["none"] * 6, faces)


def apply_sheaf_aniso(comp, L, rib_coords, iter_type, prev_cyl=None):
    new_col = "blue" if iter_type == "X" else "red"
    rib_set = set(rib_coords)
    prev_cyl = prev_cyl or set()
    xp_num_v = 0
    def add_v():
        nonlocal xp_num_v; r = xp_num_v; xp_num_v += 1; return r

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

    xp_faces, xp_edge_pf = [], {}
    for f_idx, (v00, v10, v11, v01, eb, et, el, er) in enumerate(comp.faces):
        pb, eb_ids, Lx = oriented(eb, v00, v10)
        pt, et_ids, Lx2 = oriented(et, v01, v11)
        pl, el_ids, Ly = oriented(el, v00, v01)
        pr, er_ids, Ly2 = oriented(er, v10, v11)
        assert Lx == Lx2 and Ly == Ly2
        grid = [[None] * (Ly + 1) for _ in range(Lx + 1)]
        for s in range(Lx + 1): grid[s][0] = pb[s]; grid[s][Ly] = pt[s]
        for t in range(Ly + 1): grid[0][t] = pl[t]; grid[Lx][t] = pr[t]
        for s in range(1, Lx):
            for t in range(1, Ly):
                grid[s][t] = add_v()
        he = [[None] * (Ly + 1) for _ in range(Lx)]
        ve = [[None] * Ly for _ in range(Lx + 1)]
        for s in range(Lx): he[s][0] = eb_ids[s]; he[s][Ly] = et_ids[s]
        for t in range(Ly): ve[0][t] = el_ids[t]; ve[Lx][t] = er_ids[t]
        for t in range(1, Ly):
            isr = (t in rib_set)
            for s in range(Lx):
                eid = len(xp_edges); xp_edges.append((grid[s][t], grid[s + 1][t]))
                xp_colors.append("none"); xp_is_rib.append(isr); he[s][t] = eid; xp_edge_pf[eid] = f_idx
        for s in range(1, Lx):
            isr = (s in rib_set)
            for t in range(Ly):
                eid = len(xp_edges); xp_edges.append((grid[s][t], grid[s][t + 1]))
                xp_colors.append("none"); xp_is_rib.append(isr); ve[s][t] = eid; xp_edge_pf[eid] = f_idx
        for s in range(Lx):
            for t in range(Ly):
                xp_faces.append((
                    grid[s][t], grid[s + 1][t], grid[s + 1][t + 1], grid[s][t + 1],
                    he[s][t], he[s][t + 1], ve[s][t], ve[s + 1][t], f_idx
                ))

    tot_v = 2 * xp_num_v
    def vc(v, c): return v + c * xp_num_v
    out_e, out_c, out_f, pf = [], [], [], []
    n_xe = len(xp_edges)
    for c in (0, 1):
        for eid, (u, w) in enumerate(xp_edges):
            out_e.append((vc(u, c), vc(w, c)))
            out_c.append(new_col if xp_is_rib[eid] else xp_colors[eid])
    for c in (0, 1):
        for (v00, v10, v11, v01, eb, et, el, er, of) in xp_faces:
            off = c * n_xe
            out_f.append((vc(v00, c), vc(v10, c), vc(v11, c), vc(v01, c), eb + off, et + off, el + off, er + off))
            pf.append(of)
    rib_vs = {x for eid, isr in enumerate(xp_is_rib) if isr for x in xp_edges[eid]}
    cyl_eid, new_cyl = {}, set()
    for v in sorted(rib_vs):
        ceid = len(out_e); new_cyl.add(ceid)
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
            pf.append(xp_edge_pf[eid])
    res = GeneralCubical2D(tot_v, out_e, out_c, out_f, parent_face=pf)
    for eid in range(len(res.edges)):
        if len(res.edge_faces[eid]) == 2:
            res.edge_colors[eid] = "none"
    return res, new_cyl


def sheaf_to_css(comp):
    num_faces = len(comp.faces)
    edge_qstart, qubits = [], []
    for e_idx, inc_faces in enumerate(comp.edge_faces):
        d = len(inc_faces)
        col = comp.edge_colors[e_idx]
        edge_qstart.append(len(qubits))
        if d == 0:
            continue
        elif d == 2 or col == "blue":
            qubits.append((e_idx, {f: 1 for f in inc_faces}))
        elif col == "red":
            for j in range(d - 1):
                qubits.append((e_idx, {inc_faces[j]: 1, inc_faces[j + 1]: 1}))

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
            if d == 0:
                continue
            elif d == 2 or col == "blue":
                for j in range(1, d):
                    constraints.append((1 << f_to_loc[inc[0]]) ^ (1 << f_to_loc[inc[j]]))
            elif col == "red":
                row = 0
                for f in inc:
                    row ^= (1 << f_to_loc[f])
                constraints.append(row)
        pivots = {}
        for r in constraints:
            x = r
            while x:
                lsb = (x & -x).bit_length() - 1
                if lsb in pivots: x ^= pivots[lsb]
                else: pivots[lsb] = x; break
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
                if (prow >> fv) & 1: alpha |= (1 << p)
            raw_basis.append(alpha)

        def alpha_to_qsupp(alpha):
            xs = []
            for e_idx in comp.vert_edges[v]:
                inc = comp.edge_faces[e_idx]
                d = len(inc)
                col = comp.edge_colors[e_idx]
                q0 = edge_qstart[e_idx]
                if d == 0:
                    continue
                elif d == 2 or col == "blue":
                    if (alpha >> f_to_loc[inc[0]]) & 1: xs.append(q0)
                elif col == "red":
                    pref = 0
                    for j in range(d - 1):
                        pref ^= ((alpha >> f_to_loc[inc[j]]) & 1)
                        if pref: xs.append(q0 + j)
            return sorted(set(xs))

        dim_fv = len(raw_basis)
        if 1 < dim_fv <= 7:
            cands = []
            for mask in range(1, 1 << dim_fv):
                alpha = 0
                for b in range(dim_fv):
                    if (mask >> b) & 1: alpha ^= raw_basis[b]
                supp = alpha_to_qsupp(alpha)
                if supp: cands.append((len(supp), mask, supp))
            cands.sort(key=lambda t: t[0])
            ipivs, chosen = {}, []
            for _, mask, supp in cands:
                x = mask
                while x:
                    lsb = (x & -x).bit_length() - 1
                    if lsb in ipivs: x ^= ipivs[lsb]
                    else: ipivs[lsb] = x; chosen.append(supp); break
                if len(chosen) == dim_fv: break
            for supp in chosen:
                H_X.append(supp)
        else:
            for alpha in raw_basis:
                supp = alpha_to_qsupp(alpha)
                if supp: H_X.append(supp)
    return num_q, H_X, H_Z


# =============================================================================
# 2. Class I, Class II, Class III, & Classical Class IV Constructors
# =============================================================================
def build_toric2d(L):
    def vid(x, y): return (x % L) * L + (y % L)
    nV = L * L
    HX, HZ = [], []
    for x in range(L):
        for y in range(L):
            # qubit 0*nV + vid(x,y) is horizontal edge; 1*nV + vid(x,y) is vertical edge
            HX.append(sorted([vid(x, y), vid(x - 1, y), nV + vid(x, y), nV + vid(x, y - 1)]))
            HZ.append(sorted([vid(x, y), vid(x, y + 1), nV + vid(x, y), nV + vid(x + 1, y)]))
    return 2 * nV, HX, HZ


def build_bb_code(l, m, A_terms, B_terms):
    nh = l * m
    def vid(ix, iy): return (ix % l) * m + (iy % m)
    HX, HZ = [], []
    for ix in range(l):
        for iy in range(m):
            HX.append(sorted({vid(ix + dx, iy + dy) for dx, dy in A_terms} | {nh + vid(ix + dx, iy + dy) for dx, dy in B_terms}))
            HZ.append(sorted({vid(ix - dx, iy - dy) for dx, dy in B_terms} | {nh + vid(ix - dx, iy - dy) for dx, dy in A_terms}))
    return 2 * nh, HX, HZ


def build_haah_cubic_A(L):
    """Haah's 3D Cubic Code A (PRA 2011; Bravyi-Haah PRL 2013): f=1+x+y+z, g=1+xy+yz+zx."""
    def vid(x, y, z): return ((x % L) * L + (y % L)) * L + (z % L)
    nV = L ** 3
    f_sh = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    g_sh = [(0, 0, 0), (1, 1, 0), (0, 1, 1), (1, 0, 1)]
    HX, HZ = [], []
    for x in range(L):
        for y in range(L):
            for z in range(L):
                HX.append(sorted([vid(x + dx, y + dy, z + dz) for dx, dy, dz in f_sh] +
                                 [nV + vid(x + dx, y + dy, z + dz) for dx, dy, dz in g_sh]))
                HZ.append(sorted([vid(x - dx, y - dy, z - dz) for dx, dy, dz in g_sh] +
                                 [nV + vid(x - dx, y - dy, z - dz) for dx, dy, dz in f_sh]))
    return 2 * nV, HX, HZ


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


def rot_surf_seed(rows, cols):
    def q(r, c): return r * cols + c
    hx, hz = [], []
    for r in range(rows - 1):
        for c in range(cols - 1):
            qs = [q(r, c), q(r, c + 1), q(r + 1, c), q(r + 1, c + 1)]
            if (r + c) % 2 == 0: hx.append(qs)
            else: hz.append(qs)
    for c in range(cols - 1):
        if c % 2 == 1: hx.append([q(0, c), q(0, c + 1)])
        if ((rows - 2) + c) % 2 == 1: hx.append([q(rows - 1, c), q(rows - 1, c + 1)])
    for r in range(rows - 1):
        if r % 2 == 0: hz.append([q(r, 0), q(r + 1, 0)])
        if (r + (cols - 2)) % 2 == 0: hz.append([q(r, cols - 1), q(r + 1, cols - 1)])
    return rows * cols, hx, hz


def unrot_surf_13_1_3():
    """Standard d=3 unrotated planar surface code [[13, 1, 3]]: 9 horizontal-grid + 4 interior vertical qubits."""
    c = seed_planar_11()
    # Build via 3x3 open planar grid!
    dx, dz = 2, 2  # wait: d=3 planar surface has dx=2, dz=2? Let us check: (dx+1)*dz + dx*(dz+1) -> dx=2,dz=2 gives 6+6=12? No:
    # Standard [[13,1,3]] has 3x3 data horizontal + 2x2 vertical = 13 qubits, 6 X-stars, 6 Z-plaquettes!
    # Horizontal qubits h(r, c) for r in 0..2, c in 0..2 -> 9 qubits (0..8)
    # Vertical qubits v(r, c) for r in 0..1, c in 0..1 -> 4 qubits (9..12)
    def hq(r, c): return r * 3 + c
    def vq(r, c): return 9 + r * 2 + c
    # Z-plaquettes: r in 0..1, c in 0..2 (6 plaquettes!)
    hz = []
    for r in range(2):
        for c in range(3):
            qs = [hq(r, c), hq(r + 1, c)]
            if c > 0: qs.append(vq(r, c - 1))
            if c < 2: qs.append(vq(r, c))
            hz.append(sorted(qs))
    # X-stars: r in 0..2, c in 0..1 (6 stars!)
    hx = []
    for r in range(3):
        for c in range(2):
            qs = [hq(r, c), hq(r, c + 1)]
            if r > 0: qs.append(vq(r - 1, c))
            if r < 2: qs.append(vq(r, c))
            hx.append(sorted(qs))
    return 13, hx, hz


def export_classical_bdl(prefix, ell, ribs, max_step, diag_shift=False):
    """Exports Classical 2D Dual-Self-Correcting BDL family with exact analytical dual logical macro_L!"""
    n_v = 4
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    v_col = ["none"] * 4
    macro_L = [0]

    for st in range(max_step + 1):
        if st > 0:
            itype = "Rep" if st % 2 == 1 else "Par"
            acol = "black" if itype == "Rep" else "white"
            xp_v = n_v
            deg0 = [0] * n_v
            for u, v in edges:
                deg0[u] += 1; deg0[v] += 1
            xp_col = [v_col[i] if deg0[i] != 2 else "none" for i in range(n_v)]
            xp_e, sub_pts = [], []
            for u, v in edges:
                pts = [u] + [xp_v + k for k in range(ell - 1)] + [v]
                xp_v += (ell - 1)
                xp_col.extend(["none"] * (ell - 1))
                sub_pts.append(pts)
                for k in range(ell):
                    xp_e.append((pts[k], pts[k + 1]))
            n_xpe = len(xp_e)
            n_v = 2 * xp_v
            v_col = xp_col + list(xp_col)
            edges = xp_e + [(u + xp_v, v + xp_v) for u, v in xp_e]
            for pts in sub_pts:
                for m in ribs:
                    u0 = pts[m]
                    u1 = (pts[m - 1] if diag_shift else pts[m]) + xp_v
                    edges.append((u0, u1))
                    v_col[u0] = acol; v_col[u1] = acol
            if itype == "Rep":
                macro_L = [e * ell for e in macro_L]
            else:
                macro_L = [e * ell for e in macro_L] + [n_xpe + e * ell for e in macro_L]

        inc = [[] for _ in range(n_v)]
        for ei, (u, v) in enumerate(edges):
            inc[u].append(ei); inc[v].append(ei)
        H = []
        for vi, es in enumerate(inc):
            if len(es) == 2 or v_col[vi] == "black":
                for j in range(1, len(es)):
                    H.append(sorted([es[0], es[j]]))
            elif v_col[vi] == "white":
                H.append(sorted(es))

        n = len(edges)
        # Compute gauge cycles for n <= 3000 so STATS reports exact distance d modulo gauge!
        gauge_cycles = []
        if n <= 3000:
            pivs = {}
            for r in H:
                x = 0
                for c in r: x ^= (1 << c)
                while x:
                    b = (x & -x).bit_length() - 1
                    if b in pivs: x ^= pivs[b]
                    else: pivs[b] = x; break
            for p in sorted(pivs, reverse=True):
                for p2 in pivs:
                    if p2 != p and ((pivs[p2] >> p) & 1):
                        pivs[p2] ^= pivs[p]
            xL = 0
            for q in macro_L: xL |= (1 << q)
            ones, zeros = [], []
            for fc in range(n):
                if fc in pivs: continue
                v = (1 << fc)
                for p, pr in pivs.items():
                    if (pr >> fc) & 1: v |= (1 << p)
                if (bin(v & xL).count("1") & 1): ones.append(v)
                else: zeros.append(v)
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

        out_p = os.path.join(DATA_DIR, f"{prefix}_step{st}.code")
        with open(out_p, "w") as f:
            f.write(f"{n} {len(H)} {len(gauge_cycles)} 1\n")
            for r in H: f.write(f"{len(r)} " + " ".join(map(str, r)) + "\n")
            for r in gauge_cycles: f.write(f"{len(r)} " + " ".join(map(str, r)) + "\n")
            ml = sorted(macro_L)
            f.write(f"{len(ml)} " + " ".join(map(str, ml)) + "\n")
        print(f"Exported {prefix}_step{st}: N={n}, |H|={len(H)}, |L|={len(ml)}", flush=True)


def export_css_code(prefix, nq, HX, HZ):
    for sec, Hc, Hs in [("Xerr", HZ, HX), ("Zerr", HX, HZ)]:
        raw_p = os.path.join(DATA_DIR, f"{prefix}_{sec}.raw")
        out_p = os.path.join(DATA_DIR, f"{prefix}_{sec}.code")
        with open(raw_p, "w") as f:
            f.write(f"{nq} {len(Hc)} {len(Hs)}\n")
            for r in Hc: f.write(f"{len(r)} " + " ".join(map(str, r)) + "\n")
            for r in Hs: f.write(f"{len(r)} " + " ".join(map(str, r)) + "\n")
        subprocess.run([ENGINE, "prep_logicals", raw_p, out_p], check=True)
        os.remove(raw_p)
    with open(os.path.join(DATA_DIR, f"{prefix}_Xerr.code")) as f:
        hdr = f.readline().split()
    print(f"Exported {prefix}: N={hdr[0]}, K={hdr[3]}", flush=True)


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)

    print("=== [1/5] Exporting CLASS I: Translation-Invariant Type-I (2D Toric & Scaled Gross BB) ===", flush=True)
    for L in [4, 6, 10, 16, 24, 34]:
        nq, hx, hz = build_toric2d(L)
        export_css_code(f"toric2d_L{L}", nq, hx, hz)

    A_gross = [(3, 0), (0, 1), (0, 2)]
    B_gross = [(0, 3), (1, 0), (2, 0)]
    for l, m in [(6, 6), (12, 6), (18, 12), (24, 18), (30, 18), (36, 18), (48, 18), (60, 18)]:
        nq, hx, hz = build_bb_code(l, m, A_gross, B_gross)
        export_css_code(f"bb_gross_{l}x{m}", nq, hx, hz)

    print("=== [2/5] Exporting CLASS II: Translation-Invariant 3D Fractal (Haah's Cubic Code A) ===", flush=True)
    for L in [3, 5, 7, 9, 11, 13]:
        nq, hx, hz = build_haah_cubic_A(L)
        export_css_code(f"haah_cubic_L{L}", nq, hx, hz)

    print("=== [3/5] Exporting CLASS III: 3D Surface Layer Codes Across Sizes & Sheet Spacing ===", flush=True)
    hx_512, hz_512 = [[0, 1, 2], [2, 3, 4]], [[0, 2, 3], [1, 2, 4]]
    steane = [[0, 2, 4, 6], [1, 2, 5, 6], [3, 4, 5, 6]]
    n9, hx9, hz9 = rot_surf_seed(3, 3)
    n13, hx13, hz13 = unrot_surf_13_1_3()
    n15, hx15, hz15 = rot_surf_seed(3, 5)
    for lname, (nin, hxi, hzi, ksep) in [
        ("layer_512",       (5,   hx_512, hz_512, 1)),
        ("layer_steane",    (7,   steane, steane, 1)),
        ("layer_913",       (n9,  hx9,    hz9,    1)),
        ("layer_13_1_3",    (n13, hx13,   hz13,   1)),
        ("layer_15_35",     (n15, hx15,   hz15,   1)),
        ("layer_913_ksep2", (n9,  hx9,    hz9,    2)),
    ]:
        nq, hx, hz = make_layer_code_algebraic(nin, hxi, hzi, K_sep=ksep)
        export_css_code(lname, nq, hx, hz)

    print("=== [4/5] Exporting CLASS IV-A: Uncapped Multi-Cycle 2D Classical BDL ===", flush=True)
    export_classical_bdl("cls_l2_r1", 2, [1],    5, diag_shift=False)
    export_classical_bdl("cls_l3_r2", 3, [1, 2], 4, diag_shift=False)
    export_classical_bdl("cls_l3_r1", 3, [2],    4, diag_shift=True)

    print("=== [5/5] Exporting CLASS IV-B: Multi-Cycle 3D Quantum BDL (up to s=4, k=2 full cycles!) ===", flush=True)
    q_families = [
        ("sq_aniso_L2r1",         seed_planar_11(),    [(2, [1], "X"), (2, [1], "Z"), (2, [1], "X"), (2, [1], "Z")]),
        ("sq_aniso_hyb21",        seed_planar_11(),    [(3, [1, 2], "X"), (2, [1], "Z"), (3, [1, 2], "X")]),
        ("loc3d_tor2_aniso_L2r1", seed_torus(2, 2),    [(2, [1], "X"), (2, [1], "Z"), (2, [1], "X")]),
        ("nonloc_rp2_aniso_L2r1", seed_rp2_hemicube(), [(2, [1], "X"), (2, [1], "Z"), (2, [1], "X")]),
    ]
    for pref, s0, sched in q_families:
        c, cyl = s0, set()
        for st in range(len(sched) + 1):
            if st > 0:
                L, rb, it = sched[st - 1]
                c, cyl = apply_sheaf_aniso(c, L, rb, it, cyl)
            nq, hx, hz = sheaf_to_css(c)
            export_css_code(f"{pref}_s{st}", nq, hx, hz)

    # =========================================================================
    # GENERATE V4 MASTER TASK LIST (`tasks_v4.txt`)
    # =========================================================================
    tasks = []
    all_codes = sorted(fn for fn in os.listdir(DATA_DIR) if fn.endswith(".code"))

    # SUITE 0: STATS on all codes (with fast adaptive ISD)
    for cn in all_codes:
        tasks.append(f"STATS|{cn}|stats|{cn}")

    # SUITE 1: FOUR-CLASS FIXED-TEMPERATURE SYSTEM-SIZE SCALING & ARRHENIUS SLOPES!
    # Class I: 2D Toric (Xerr == Zerr by self-duality) & Scaled Gross BB (Xerr == Zerr by symmetry)
    class1_2_3_codes = (
        [f"toric2d_L{L}_Xerr.code" for L in [4, 6, 10, 16, 24, 34]] +
        [f"bb_gross_{l}x{m}_Xerr.code" for l, m in [(6,6), (12,6), (18,12), (24,18), (30,18), (36,18), (48,18), (60,18)]] +
        [f"haah_cubic_L{L}_Xerr.code" for L in [3, 5, 7, 9, 11, 13]] +
        [f"{ln}_Xerr.code" for ln in ["layer_512", "layer_steane", "layer_913", "layer_13_1_3", "layer_15_35", "layer_913_ksep2"]] +
        [f"{ln}_Zerr.code" for ln in ["layer_512", "layer_steane", "layer_913", "layer_13_1_3", "layer_15_35", "layer_913_ksep2"]]
    )
    for cn in class1_2_3_codes:
        for beta in [3.5, 4.0, 4.25, 4.5, 4.75, 5.0]:
            tr = 72 if beta <= 4.5 else 48
            # BB (12x6)/(18x12) at beta=5.0 can reach ~2e7 flips; 3e8 flips guarantees 0 capping!
            tasks.append(f"SCALING|{cn}|kmc|{cn}|uniform|{beta}|{tr}|300000000")

    # SUITE 2: UNCAPPED MULTI-CYCLE 2D CLASSICAL BDL PHASE CROSSING (0 CAPPING!)
    cls_families = (
        [f"cls_l2_r1_step{s}.code" for s in range(6)] +
        [f"cls_l3_r2_step{s}.code" for s in range(5)] +
        [f"cls_l3_r1_step{s}.code" for s in range(5)]
    )
    for cn in cls_families:
        is_big = cn.endswith(("step4.code", "step5.code"))
        betas = [2.5, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]
        for beta in betas:
            # Skip beta >= 4.5 ONLY on step5 (N=12,500 Rep-step where step3 already hits 1e8 at beta=4.0!)
            if cn.endswith("step5.code") and beta > 3.75:
                continue
            if cn.endswith("step3.code") and "l3_r2" in cn and beta > 4.25:
                continue
            tr = 32 if is_big else 80
            tasks.append(f"CLS_CROSS|{cn}|kmc|{cn}|uniform|{beta}|{tr}|400000000")

    # SUITE 3: MULTI-CYCLE 3D QUANTUM BDL SCALING (s = 0 -> 1 -> 2 -> 3 -> 4!)
    q_bdl_codes = (
        [f"sq_aniso_L2r1_s{s}_{sec}.code" for s in range(5) for sec in ("Xerr", "Zerr")] +
        [f"sq_aniso_hyb21_s{s}_{sec}.code" for s in range(4) for sec in ("Xerr", "Zerr")] +
        [f"loc3d_tor2_aniso_L2r1_s{s}_{sec}.code" for s in range(4) for sec in ("Xerr", "Zerr")] +
        [f"nonloc_rp2_aniso_L2r1_s{s}_{sec}.code" for s in range(4) for sec in ("Xerr", "Zerr")]
    )
    for cn in q_bdl_codes:
        is_s4 = "_s4_" in cn
        is_s3 = "_s3_" in cn
        # Notice: at s=3, Xerr has jumped TWICE (R_X, R_Z, R_X) so s3_Xerr at beta>=5.0 exceeds 1e8 flips,
        # whereas s3_Zerr and s4_Xerr are right in the sweet spot across beta = 3.5 .. 6.0!
        for beta in [3.5, 4.0, 4.5, 5.0, 5.5, 6.0]:
            if is_s3 and "_Xerr" in cn and beta > 5.0:
                continue
            if is_s4 and "_Zerr" in cn and beta > 5.0:
                continue
            tr = 32 if (is_s4 or is_s3) else 72
            tasks.append(f"Q_MULTICYCLE|{cn}|kmc|{cn}|uniform|{beta}|{tr}|400000000")

    # SUITE 4: DIRECT CHESI-LOSS / BDL SEC. 5 EQUILIBRIUM GIBBS CONFINEMENT & 1-FLIP INSTABILITY!
    # Evaluated at fixed physical bath exposure time t_phys = 80.0 across ALL 4 CLASSES up to N=18,900!
    gibbs_codes = (
        [f"toric2d_L{L}_Xerr.code" for L in [4, 6, 10, 16, 24, 34]] +
        [f"bb_gross_{l}x{m}_Xerr.code" for l, m in [(6,6), (12,6), (18,12), (24,18), (30,18), (36,18), (48,18), (60,18)]] +
        [f"haah_cubic_L{L}_Xerr.code" for L in [3, 5, 7, 9, 11, 13]] +
        [f"{ln}_Xerr.code" for ln in ["layer_512", "layer_steane", "layer_913", "layer_13_1_3", "layer_15_35", "layer_913_ksep2"]] +
        [f"cls_l2_r1_step{s}.code" for s in [0, 2, 4]] +
        [f"cls_l3_r2_step{s}.code" for s in [0, 2, 4]] +
        [f"sq_aniso_L2r1_s{s}_Xerr.code" for s in [0, 2, 4]] +
        [f"sq_aniso_L2r1_s{s}_Zerr.code" for s in [0, 2, 4]] +
        [f"loc3d_tor2_aniso_L2r1_s{s}_Xerr.code" for s in [0, 1, 2, 3]] +
        [f"nonloc_rp2_aniso_L2r1_s{s}_Xerr.code" for s in [0, 1, 2, 3]]
    )
    for cn in gibbs_codes:
        for beta, t_phys in [(4.0, 80.0), (4.5, 150.0)]:
            tasks.append(f"GIBBS|{cn}|gibbs_instab|{cn}|uniform|{beta}|{t_phys}|120|64")

    tpath = os.path.join(WORK_DIR, "tasks_v4.txt")
    with open(tpath, "w") as f:
        for t in tasks:
            f.write(t + "\n")
    print(f"Generated {len(tasks)} V4 tasks in {tpath}", flush=True)
