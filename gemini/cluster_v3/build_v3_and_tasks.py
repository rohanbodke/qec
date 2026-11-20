#!/usr/bin/env python3
"""
V3 Master Code Builder & High-Value Simulation Task Generator (runs on SubMIT Slurm):
  Track A (Strictly 3D Euclidean-Local):
    - Sparse-Loop Anisotropic T^2 Sheaf (`loc3d_tor2_sparse_L2r1_s2`: N=974, K=2, w<=6!)
    - Anisotropic T^2 Sheaf (`loc3d_tor2_aniso_L2r1_s2`: N=1016, K=2; `loc3d_tor2_sparse_hyb21_s2`: N=2224)
    - Graded Hamiltonian Check Couplings J_rib = 1.8 vs Uniform J = 1.0 (zero qubit overhead!)
  Track B (Non-Local Bounded-Degree qLDPC for Neutral Atoms / Reconfigurable Hardware):
    - Ultra-Compact Anisotropic RP^2 Sheaf (`nonloc_rp2_aniso_L2r1_s2`: N=762, K=1; `hyb21`: N=1758)
    - High-Rate Hyperbolic {4,6} K_{2,2,2,2} Sheaf (`nonloc_hyp_g3_aniso_s2`: K=6, 508 qubits/logical!)
    - IBM Bivariate Bicycle Codes (`nonloc_ibm_gross_144` [[144,12,12]] & `nonloc_ibm_twogross_288` [[288,12,18]])
    - 4D (2,2) Homological Toric Codes (`nonloc_toric4d_L3` N=486, K=6 & `L4` N=1536, K=6)

  Three High-Value Literature Simulation Suites:
    1. Free-Energy Scaling (F = E - TS, beating the Michnicki Trap) & Graded Couplings J_rib=1.8
    2. Digital Gate-Based Measurement-Free QEC (Hong-Guo-Lucas Nat. Comm. 2025) under gate noise p_gate
    3. Multi-Round Single-Shot Active QEC under noisy syndrome measurements (p_data, q_meas = p_data)
"""

import os
import random
import subprocess

WORK_DIR = "/work/submit/harshila/6s980_qmem_v3"
DATA_DIR = os.path.join(WORK_DIR, "data")
RES_DIR = os.path.join(WORK_DIR, "results")
ENGINE = os.path.join(WORK_DIR, "kmc_engine_v3")


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
    edges = [(0,1), (1,2), (2,3), (0,3), (0,2), (1,3)]
    pid = {(min(u,v), max(u,v)): i for i, (u,v) in enumerate(edges)}
    def eid(a, b): return pid[(min(a,b), max(a,b))]
    faces = [
        (a, b, c, d, eid(a,b), eid(d,c), eid(a,d), eid(b,c))
        for (a, b, c, d) in [(0,1,2,3), (0,2,1,3), (0,1,3,2)]
    ]
    return GeneralCubical2D(4, edges, ["none"] * 6, faces)


def seed_hyperbolic_k2222():
    """Exact {4, 6} orientable genus-3 hyperbolic surface on K_{2,2,2,2} (V=8, E=24, F=12 -> K=6)!"""
    adj = [[v for v in range(8) if v != u and (u - v) % 4 != 0] for u in range(8)]
    edges = [(u, v) for u in range(8) for v in adj[u] if u < v]
    eid_of = {e: i for i, e in enumerate(edges)}
    def ge(a, b): return eid_of[(min(a,b), max(a,b))]
    faces_raw = [
        (0,1,2,3), (0,1,3,2), (0,2,4,5), (0,3,4,6),
        (0,5,2,7), (0,6,1,7), (1,2,7,4), (1,3,5,6),
        (1,4,6,7), (2,4,7,5), (3,4,5,6), (3,5,7,6),
    ]
    faces = [(a,b,c,d, ge(a,b), ge(d,c), ge(a,d), ge(b,c)) for (a,b,c,d) in faces_raw]
    return GeneralCubical2D(8, edges, ["none"] * 24, faces)


def apply_sheaf_aniso_sparse(comp, L, rib_coords, iter_type, prev_cyl=None, h_rib_faces=None, v_rib_faces=None):
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
        allow_h = (h_rib_faces is None) or (f_idx in h_rib_faces)
        allow_v = (v_rib_faces is None) or (f_idx in v_rib_faces)
        for t in range(1, Ly):
            isr = allow_h and (t in rib_set)
            for s in range(Lx):
                eid = len(xp_edges); xp_edges.append((grid[s][t], grid[s + 1][t]))
                xp_colors.append("none"); xp_is_rib.append(isr); he[s][t] = eid; xp_edge_pf[eid] = f_idx
        for s in range(1, Lx):
            isr = allow_v and (s in rib_set)
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
            out_f.append((vc(v00,c), vc(v10,c), vc(v11,c), vc(v01,c), eb+off, et+off, el+off, er+off))
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
            out_f.append((vc(u,0), vc(w,0), vc(w,1), vc(u,1), eid, eid+n_xe, cyl_eid[u], cyl_eid[w]))
            pf.append(xp_edge_pf[eid])
    res = GeneralCubical2D(tot_v, out_e, out_c, out_f, parent_face=pf)
    for eid in range(len(res.edges)):
        if len(res.edge_faces[eid]) == 2:
            res.edge_colors[eid] = "none"
    return res, new_cyl


def sheaf_to_css_with_jgrad(comp):
    num_faces = len(comp.faces)
    edge_qstart, qubits = [], []
    for e_idx, inc_faces in enumerate(comp.edge_faces):
        d = len(inc_faces)
        col = comp.edge_colors[e_idx]
        edge_qstart.append(len(qubits))
        if d == 2 or col == "blue":
            qubits.append((e_idx, {f: 1 for f in inc_faces}))
        elif col == "red":
            for j in range(d - 1):
                qubits.append((e_idx, {inc_faces[j]: 1, inc_faces[j + 1]: 1}))

    num_q = len(qubits)
    H_Z = [[] for _ in range(num_faces)]
    for q_idx, (_, vec) in enumerate(qubits):
        for f_idx in vec:
            H_Z[f_idx].append(q_idx)

    # Graded J_tenths for HZ (Xerr checks): 18 (J=1.8) if face touches any seam edge (fdeg > 2), else 10 (J=1.0)
    jz_grad = []
    for f_idx, (_, _, _, _, eb, et, el, er) in enumerate(comp.faces):
        is_seam = any(len(comp.edge_faces[e]) > 2 for e in (eb, et, el, er))
        jz_grad.append(18 if is_seam else 10)

    H_X, jx_grad = [], []
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
            if d == 2 or col == "blue":
                for j in range(1, d):
                    constraints.append((1 << f_to_loc[inc[0]]) ^ (1 << f_to_loc[inc[j]]))
            elif col == "red":
                row = 0
                for f in inc: row ^= (1 << f_to_loc[f])
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
                if d == 2 or col == "blue":
                    if (alpha >> f_to_loc[inc[0]]) & 1: xs.append(q0)
                elif col == "red":
                    pref = 0
                    for j in range(d - 1):
                        pref ^= ((alpha >> f_to_loc[inc[j]]) & 1)
                        if pref: xs.append(q0 + j)
            return sorted(set(xs))

        dim_fv = len(raw_basis)
        v_is_seam = any(len(comp.edge_faces[e]) > 2 for e in comp.vert_edges[v])
        jv = 18 if v_is_seam else 10
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
                H_X.append(supp); jx_grad.append(jv)
        else:
            for alpha in raw_basis:
                supp = alpha_to_qsupp(alpha)
                if supp: H_X.append(supp); jx_grad.append(jv)

    return num_q, H_X, H_Z, jx_grad, jz_grad


def build_bb_code(l, m, A_terms, B_terms):
    nh = l * m
    def vid(ix, iy): return (ix % l) * m + (iy % m)
    HX, HZ = [], []
    for ix in range(l):
        for iy in range(m):
            HX.append(sorted({vid(ix+dx, iy+dy) for dx,dy in A_terms} | {nh + vid(ix+dx, iy+dy) for dx,dy in B_terms}))
            HZ.append(sorted({vid(ix-dx, iy-dy) for dx,dy in B_terms} | {nh + vid(ix-dx, iy-dy) for dx,dy in A_terms}))
    return 2 * nh, HX, HZ


def build_toric4d_22(L):
    def v_idx(c): return ((c[0]%L)*L*L*L + (c[1]%L)*L*L + (c[2]%L)*L + (c[3]%L))
    nV = L**4
    planes = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    cubes  = [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]
    def e_id(c, d): return d * nV + v_idx(c)
    def f_id(c, p_i): return p_i * nV + v_idx(c)
    def c3_id(c, c_i): return c_i * nV + v_idx(c)
    HX = [[] for _ in range(4 * nV)]
    HZ = [[] for _ in range(4 * nV)]
    for x0 in range(L):
        for x1 in range(L):
            for x2 in range(L):
                for x3 in range(L):
                    base = [x0, x1, x2, x3]
                    for pi, (a, b) in enumerate(planes):
                        q = f_id(base, pi)
                        ca = list(base); ca[a] += 1
                        cb = list(base); cb[b] += 1
                        for eid in (e_id(base, a), e_id(ca, b), e_id(cb, a), e_id(base, b)):
                            HX[eid].append(q)
                    for ci, (a, b, c) in enumerate(cubes):
                        zc = c3_id(base, ci)
                        for (u, w, third) in ((a,b,c), (a,c,b), (b,c,a)):
                            pi = planes.index((u, w))
                            cs = list(base); cs[third] += 1
                            HZ[zc].append(f_id(base, pi)); HZ[zc].append(f_id(cs, pi))
    return 6 * nV, [sorted(r) for r in HX], [sorted(r) for r in HZ]


def export_code(prefix, nq, HX, HZ, jx=None, jz=None):
    for sec, Hc, Hs, jw in [("Xerr", HZ, HX, jz), ("Zerr", HX, HZ, jx)]:
        raw_p = os.path.join(DATA_DIR, f"{prefix}_{sec}.raw")
        out_p = os.path.join(DATA_DIR, f"{prefix}_{sec}.code")
        with open(raw_p, "w") as f:
            f.write(f"{nq} {len(Hc)} {len(Hs)}\n")
            for r in Hc: f.write(f"{len(r)} " + " ".join(map(str, r)) + "\n")
            for r in Hs: f.write(f"{len(r)} " + " ".join(map(str, r)) + "\n")
        subprocess.run([ENGINE, "prep_logicals", raw_p, out_p], check=True)
        os.remove(raw_p)
        if jw is not None:
            with open(out_p + ".jgrad", "w") as f:
                f.write(" ".join(map(str, jw)) + "\n")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)

    # 1. Export Track A (Strictly 3D-Local) & Track B (Non-Local Sheaf) Families
    sheaf_specs = [
        # Track A: Strictly 3D-Local Sparse-Loop & Anisotropic T^2
        ("loc3d_tor2_sparse_L2r1", seed_torus(2, 2), [(2,[1],"X",{0,1},{0,2}), (2,[1],"Z",None,None)], True),
        ("loc3d_tor2_aniso_L2r1",  seed_torus(2, 2), [(2,[1],"X",None,None),   (2,[1],"Z",None,None)], False),
        ("loc3d_tor2_sparse_hyb21",seed_torus(2, 2), [(3,[1,2],"X",{0,1},{0,2}),(2,[1],"Z",None,None)],False),
        ("loc3d_tor2_aniso_hyb21", seed_torus(2, 2), [(3,[1,2],"X",None,None), (2,[1],"Z",None,None)], False),
        # Track B: Non-Local Bounded-Degree Sheaf (RP^2 & Hyperbolic {4,6} K=6!)
        ("nonloc_rp2_aniso_L2r1",  seed_rp2_hemicube(), [(2,[1],"X",None,None),(2,[1],"Z",None,None)], True),
        ("nonloc_rp2_aniso_hyb21", seed_rp2_hemicube(), [(3,[1,2],"X",None,None),(2,[1],"Z",None,None)],False),
        ("nonloc_hyp_g3_aniso",    seed_hyperbolic_k2222(), [(2,[1],"X",None,None),(2,[1],"Z",None,None)], True),
    ]
    for pref, s0, sched, export_all_steps in sheaf_specs:
        c, cyl = s0, set()
        for st in range(len(sched) + 1):
            if st > 0:
                L, rb, it, hf, vf = sched[st - 1]
                c, cyl = apply_sheaf_aniso_sparse(c, L, rb, it, cyl, hf, vf)
            if export_all_steps or st == len(sched):
                nq, hx, hz, jx, jz = sheaf_to_css_with_jgrad(c)
                export_code(f"{pref}_s{st}", nq, hx, hz, jx, jz)
                print(f"Exported {pref}_s{st}: N={nq}", flush=True)

    # 2. Export Non-Local Modern qLDPC Competitors (IBM BB Gross/Two-Gross & 4D (2,2) Toric!)
    for pref, fn in [
        ("nonloc_ibm_gross_144",    lambda: build_bb_code(12, 6,  [(3,0),(0,1),(0,2)], [(0,3),(1,0),(2,0)])),
        ("nonloc_ibm_twogross_288", lambda: build_bb_code(12, 12, [(3,0),(0,2),(0,7)], [(0,3),(1,0),(2,0)])),
        ("nonloc_toric4d_L3",       lambda: build_toric4d_22(3)),
        ("nonloc_toric4d_L4",       lambda: build_toric4d_22(4)),
    ]:
        nq, hx, hz = fn()
        export_code(pref, nq, hx, hz)
        print(f"Exported {pref}: N={nq}", flush=True)

    # 3. Copy Baselines from V2 (`layer_913`, `toric3d_L7`, `toric2d_L20`, `bdl_tor2_hyb21_s2`, `bdl_rp2_hyb21_s2`)
    v2_data = "/work/submit/harshila/6s980_qmem_v2/data"
    for fn in os.listdir(v2_data):
        if fn.startswith(("layer_913_", "toric3d_L7_", "toric2d_L20_", "bdl_tor2_hyb21_s2_", "bdl_rp2_hyb21_s2_")) and fn.endswith(".code"):
            src = os.path.join(v2_data, fn)
            dst = os.path.join(DATA_DIR, fn)
            if not os.path.exists(dst):
                with open(src, "rb") as fi, open(dst, "wb") as fo:
                    fo.write(fi.read())

    # =========================================================================
    # BUILD V3 HIGH-VALUE SIMULATION TASK LIST (`tasks_v3.txt`)
    # =========================================================================
    tasks = []
    all_codes = sorted(fn for fn in os.listdir(DATA_DIR) if fn.endswith(".code"))

    # SUITE 0: Exact Code Stats (including Meta-Checks M_meta, d, Delta, Check/Qubit Degrees!)
    for cn in all_codes:
        tasks.append(f"STATS|{cn}|stats|{cn}")

    # SUITE 1: Free-Energy Scaling (Beating Michnicki Trap across sizes!) + Graded J_rib=1.8 Funnel!
    kmc_families = [
        # Track A 3D-Local multi-step scaling & compact showdown
        "loc3d_tor2_sparse_L2r1_s0_Xerr.code", "loc3d_tor2_sparse_L2r1_s0_Zerr.code",
        "loc3d_tor2_sparse_L2r1_s1_Xerr.code", "loc3d_tor2_sparse_L2r1_s1_Zerr.code",
        "loc3d_tor2_sparse_L2r1_s2_Xerr.code", "loc3d_tor2_sparse_L2r1_s2_Zerr.code",
        "loc3d_tor2_aniso_L2r1_s2_Xerr.code",  "loc3d_tor2_aniso_L2r1_s2_Zerr.code",
        "loc3d_tor2_sparse_hyb21_s2_Xerr.code","loc3d_tor2_sparse_hyb21_s2_Zerr.code",
        "loc3d_tor2_aniso_hyb21_s2_Xerr.code", "loc3d_tor2_aniso_hyb21_s2_Zerr.code",
        # Track B Non-Local multi-step scaling & modern qLDPC competitors
        "nonloc_rp2_aniso_L2r1_s0_Xerr.code",  "nonloc_rp2_aniso_L2r1_s0_Zerr.code",
        "nonloc_rp2_aniso_L2r1_s1_Xerr.code",  "nonloc_rp2_aniso_L2r1_s1_Zerr.code",
        "nonloc_rp2_aniso_L2r1_s2_Xerr.code",  "nonloc_rp2_aniso_L2r1_s2_Zerr.code",
        "nonloc_rp2_aniso_hyb21_s2_Xerr.code", "nonloc_rp2_aniso_hyb21_s2_Zerr.code",
        "nonloc_hyp_g3_aniso_s0_Xerr.code",    "nonloc_hyp_g3_aniso_s0_Zerr.code",
        "nonloc_hyp_g3_aniso_s1_Xerr.code",    "nonloc_hyp_g3_aniso_s1_Zerr.code",
        "nonloc_hyp_g3_aniso_s2_Xerr.code",    "nonloc_hyp_g3_aniso_s2_Zerr.code",
        "nonloc_ibm_gross_144_Xerr.code",      "nonloc_ibm_gross_144_Zerr.code",
        "nonloc_ibm_twogross_288_Xerr.code",   "nonloc_ibm_twogross_288_Zerr.code",
        "nonloc_toric4d_L3_Xerr.code",         "nonloc_toric4d_L3_Zerr.code",
        "nonloc_toric4d_L4_Xerr.code",         "nonloc_toric4d_L4_Zerr.code",
    ]
    for cn in kmc_families:
        for b in (3.0, 4.0, 5.0, 6.0, 7.0):
            tasks.append(f"KMC_UNI|{cn}|kmc|{cn}|uniform|{b}|100|25000000|8")

    # Graded J_rib = 1.8 vs Uniform J = 1.0 on ultra-compact sub-1000-qubit 3D-Local & Non-Local codes!
    for cn in (
        "loc3d_tor2_sparse_L2r1_s2_Xerr.code", "loc3d_tor2_sparse_L2r1_s2_Zerr.code",
        "loc3d_tor2_aniso_L2r1_s2_Xerr.code",  "loc3d_tor2_aniso_L2r1_s2_Zerr.code",
        "nonloc_rp2_aniso_L2r1_s2_Xerr.code",  "nonloc_rp2_aniso_L2r1_s2_Zerr.code",
        "nonloc_hyp_g3_aniso_s2_Xerr.code",    "nonloc_hyp_g3_aniso_s2_Zerr.code",
    ):
        for b in (3.0, 4.0, 5.0, 6.0):
            tasks.append(f"KMC_JGRAD|{cn}|kmc|{cn}|{cn}.jgrad|{b}|100|25000000|8")

    # SUITE 2: Gate-Based Digital Measurement-Free QEC (Hong-Guo-Lucas Nat. Comm. 2025!)
    # 150 discrete sweeps with idling noise p_env = 0.002/sweep and noisy cooling gates p_gate!
    mfqec_codes = [
        ("loc3d_tor2_sparse_L2r1_s2_Xerr.code", "uniform",                       "3D_Tor2_Sparse_974"),
        ("loc3d_tor2_sparse_L2r1_s2_Xerr.code", "loc3d_tor2_sparse_L2r1_s2_Xerr.code.jgrad", "3D_Tor2_Sparse_974_Jgrad"),
        ("loc3d_tor2_sparse_hyb21_s2_Xerr.code","uniform",                       "3D_Tor2_Hyb_2224"),
        ("nonloc_rp2_aniso_L2r1_s2_Xerr.code",  "uniform",                       "NL_RP2_Aniso_762"),
        ("nonloc_rp2_aniso_L2r1_s2_Xerr.code",  "nonloc_rp2_aniso_L2r1_s2_Xerr.code.jgrad",  "NL_RP2_Aniso_762_Jgrad"),
        ("nonloc_hyp_g3_aniso_s2_Xerr.code",    "uniform",                       "NL_HypK6_3048"),
        ("nonloc_ibm_gross_144_Xerr.code",      "uniform",                       "NL_IBM_Gross_144"),
        ("nonloc_ibm_twogross_288_Xerr.code",   "uniform",                       "NL_IBM_2Gross_288"),
        ("nonloc_toric4d_L3_Xerr.code",         "uniform",                       "NL_4DToric_486"),
        ("layer_913_Xerr.code",                 "uniform",                       "3D_Layer_1853"),
        ("toric2d_L20_Xerr.code",               "uniform",                       "2D_Toric_800"),
    ]
    for cn, jspec, tag in mfqec_codes:
        for p_gate in (0.0, 0.0005, 0.001, 0.002, 0.004, 0.008):
            tasks.append(f"MFQEC|{tag}|mfqec|{cn}|{jspec}|6.0|0.002|{p_gate}|150|200|8")

    # SUITE 3: 20-Round Single-Shot Active QEC Under Noisy Syndrome Extraction (p_data = q_meas)!
    ss_codes = [
        ("loc3d_tor2_sparse_L2r1_s2_Xerr.code", "3D_Tor2_Sparse_974"),
        ("loc3d_tor2_sparse_hyb21_s2_Xerr.code","3D_Tor2_Hyb_2224"),
        ("nonloc_rp2_aniso_L2r1_s2_Xerr.code",  "NL_RP2_Aniso_762"),
        ("nonloc_rp2_aniso_hyb21_s2_Xerr.code", "NL_RP2_Hyb_1758"),
        ("nonloc_hyp_g3_aniso_s2_Xerr.code",    "NL_HypK6_3048"),
        ("nonloc_ibm_gross_144_Xerr.code",      "NL_IBM_Gross_144"),
        ("nonloc_ibm_twogross_288_Xerr.code",   "NL_IBM_2Gross_288"),
        ("nonloc_toric4d_L3_Xerr.code",         "NL_4DToric_486"),
        ("nonloc_toric4d_L4_Xerr.code",         "NL_4DToric_1536"),
        ("layer_913_Xerr.code",                 "3D_Layer_1853"),
        ("toric3d_L7_Xerr.code",                "3D_Toric_Membrane_1029"),
        ("toric3d_L7_Zerr.code",                "3D_Toric_String_1029"),
        ("toric2d_L20_Xerr.code",               "2D_Toric_800"),
    ]
    for cn, tag in ss_codes:
        for pq in (0.001, 0.002, 0.004, 0.006, 0.008, 0.012, 0.016, 0.020):
            tasks.append(f"SINGLESHOT|{tag}|singleshot|{cn}|{pq}|{pq}|20|250|8")

    random.Random(2026).shuffle(tasks)
    with open(os.path.join(WORK_DIR, "tasks_v3.txt"), "w") as f:
        for t in tasks:
            f.write(t + "\n")
    print(f"Total V3 tasks written: {len(tasks)}", flush=True)
