#!/usr/bin/env python3
"""
build_v6_ambitious.py
Builds all codes, coupling profiles, and manifest tasks for the MIT 6.S980 V6 Ambitious Campaign on SubMIT:
  1. External Ground-Truth Validation & Baselines:
     - 3D Toric Code (`toric3d_L3..7`): both self-correcting membrane X-sector & point-defect Z-sector (Gu-Caha App. E.3)
     - Gu-Caha Published Random CSS Layer Codes (`cahalibor/random-layer-codes`, n=11..19): exact validation vs PRL CSVs + scaling
     - Haah's 3D Cubic Code A (`cubic_L5..11`)
  2. Exact Renewal Forward Flux Sampling (FFS) Validation + Deep Low-T Crossings (beta up to 9.0, lifetimes 10^8 -> 10^18)
     + Microscopic Transition-State "Failure Movies"
  3. Hayden S8.3 Leaner Welded Gadget Percolation Threshold:
     - 4-segment `+`-weld (`L2r1`) vs 3-segment Checkerboard `T`-weld (`cb3rib`, non-percolating!) vs 2-segment `L`-weld (`cb2rib`, percolating!)
     - Scaling `sq_aniso_L2r1` and `sq_cb3rib` up to s=5 (N ~ 145k - 190k!) and 2-bridge `L3r2` up to s=3 (N ~ 23k)
  4. Curing the 418x Z^3 Cubic-Grid Stretching Penalty with Budget-Neutral Local Wire-Hill Couplings (<J> == 1.000)
     up to BDL's Full 56,312-Qubit ell=8 Code (`sq_bdl_8_8_r2`)
  5. Soft-Junction Passive Cooling vs. Hard Quench & Standard Simulated Annealing (Hayden S8.3 #2 & BDL State Prep)
"""

import bisect
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import csv
import os
import subprocess
import sys
import types

WORK_DIR = "/work/submit/harshila/6s980_qmem_v6"
DATA_DIR = os.path.join(WORK_DIR, "data")
RES_DIR = os.path.join(WORK_DIR, "results")
MOV_DIR = os.path.join(WORK_DIR, "movies")
ENGINE = os.path.join(WORK_DIR, "kmc_engine_v6")
CAHA_REPO = os.path.join(WORK_DIR, "random-layer-codes")


# Pure-Python numpy/matplotlib shim so Gu-Caha's `LayerCode.py` runs on any stock Python3 node
class _ShimArr(list):
    def __init__(self, data):
        super().__init__([_ShimArr(r) if isinstance(r, list) else int(r) for r in data])

    @property
    def shape(self):
        return (len(self), len(self[0]) if self and isinstance(self[0], list) else 0)

    def __add__(self, other):
        return _ShimArr([a + b for a, b in zip(self, other)])

    def searchsorted(self, v, side="left"):
        return bisect.bisect_right(self, v) if side == "right" else bisect.bisect_left(self, v)


def _install_caha_shims():
    if "numpy" not in sys.modules:
        sys.modules["numpy"] = types.SimpleNamespace(
            asarray=lambda x, dtype=None: _ShimArr(x),
            nonzero=lambda r: (_ShimArr([i for i, v in enumerate(r) if v]),),
        )
    for m in ["matplotlib", "matplotlib.pyplot", "mpl_toolkits", "mpl_toolkits.mplot3d", "mpl_toolkits.mplot3d.art3d"]:
        sys.modules[m] = types.SimpleNamespace(pyplot=None, Line3DCollection=None, Poly3DCollection=None)
    if CAHA_REPO not in sys.path:
        sys.path.insert(0, CAHA_REPO)


class GeneralCubical2D:
    def __init__(self, num_vertices, edges, edge_colors, faces):
        self.num_vertices = num_vertices
        self.edges = edges
        self.edge_colors = edge_colors
        self.faces = faces
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
    return GeneralCubical2D(
        4, [(0, 1), (2, 3), (0, 2), (1, 3)], ["blue", "blue", "red", "red"], [(0, 1, 3, 2, 0, 1, 2, 3)]
    )


def seed_torus(Lx, Lz):
    def vid(ix, iz):
        return (iz % Lz) * Lx + (ix % Lx)

    edges, h_e, v_e = [], {}, {}
    for iz in range(Lz):
        for ix in range(Lx):
            h_e[(ix, iz)] = len(edges)
            edges.append((vid(ix, iz), vid(ix + 1, iz)))
    for iz in range(Lz):
        for ix in range(Lx):
            v_e[(ix, iz)] = len(edges)
            edges.append((vid(ix, iz), vid(ix, iz + 1)))
    faces = [
        (
            vid(ix, iz), vid(ix + 1, iz), vid(ix + 1, iz + 1), vid(ix, iz + 1),
            h_e[(ix, iz)], h_e[(ix, (iz + 1) % Lz)], v_e[(ix, iz)], v_e[((ix + 1) % Lx, iz)],
        )
        for iz in range(Lz)
        for ix in range(Lx)
    ]
    return GeneralCubical2D(Lx * Lz, edges, ["none"] * len(edges), faces)


def seed_rot_torus_422():
    edges = [(0, 1), (0, 1), (0, 1), (0, 1)]
    faces = [(0, 1, 0, 1, 0, 3, 2, 1), (1, 0, 1, 0, 1, 2, 3, 0)]
    return GeneralCubical2D(2, edges, ["none"] * 4, faces)


def apply_sheaf_step(comp, L, rib_coords, iter_type, prev_cyl=None, gadget_mode="full"):
    """
    gadget_mode:
      - 'full': standard BDL / compressed square cross ribs (4 interior rib segments for L=2, r=[1])
      - 'cb3rib': Hayden S8.3 leaner Checkerboard T-weld (3 interior rib segments per 2x2 square; non-percolating!)
      - 'cb2rib': 2-segment L-weld control (2 interior rib segments per 2x2 square; percolating trominoes!)
    """
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

    parent_bf = getattr(comp, "bridge_faces", set())
    xp_faces, xp_is_bf = [], []
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

        # Parity bit from face index + corner vertex ids gives deterministic checkerboard orientation!
        par = (f_idx + v00 + v11) & 1

        for t in range(1, Ly):
            for s in range(Lx):
                isr = (t in rib_set)
                if isr and Lx == 2 and Ly == 2:
                    if gadget_mode == "cb2rib" and s == 1:
                        isr = False
                eid = len(xp_edges)
                xp_edges.append((grid[s][t], grid[s + 1][t]))
                xp_colors.append("none")
                xp_is_rib.append(isr)
                he[s][t] = eid
        for s in range(1, Lx):
            for t in range(Ly):
                isr = (s in rib_set)
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
        for s in range(Lx):
            for t in range(Ly):
                xp_faces.append(
                    (grid[s][t], grid[s + 1][t], grid[s + 1][t + 1], grid[s][t + 1], he[s][t], he[s][t + 1], ve[s][t], ve[s + 1][t])
                )
                xp_is_bf.append(f_idx in parent_bf)

    tot_v = 2 * xp_num_v

    def vc(v, c):
        return v + c * xp_num_v

    out_e, out_c, out_f, out_bf = [], [], [], set()
    n_xe = len(xp_edges)
    for c in (0, 1):
        for eid, (u, w) in enumerate(xp_edges):
            out_e.append((vc(u, c), vc(w, c)))
            out_c.append(new_col if xp_is_rib[eid] else xp_colors[eid])
    for c in (0, 1):
        for xfi, (v00, v10, v11, v01, eb, et, el, er) in enumerate(xp_faces):
            off = c * n_xe
            if xp_is_bf[xfi]:
                out_bf.add(len(out_f))
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
            out_bf.add(len(out_f))
            out_f.append((vc(u, 0), vc(w, 0), vc(w, 1), vc(u, 1), eid, eid + n_xe, cyl_eid[u], cyl_eid[w]))
    res = GeneralCubical2D(tot_v, out_e, out_c, out_f)
    res.bridge_faces = out_bf
    res.cyl_edges = new_cyl
    for eid in range(len(res.edges)):
        if len(res.edge_faces[eid]) == 2:
            res.edge_colors[eid] = "none"
    return res, new_cyl


def sheaf_to_css_with_seams(comp):
    """Returns (num_q, H_X, H_Z, seam_qubits, bridge_faces, cyl_qubits)."""
    num_faces = len(comp.faces)
    cyl_edges = getattr(comp, "cyl_edges", set())
    bridge_faces = getattr(comp, "bridge_faces", set())
    edge_qstart, qubits = [], []
    seam_qubits, cyl_qubits = set(), set()
    for e_idx, inc_faces in enumerate(comp.edge_faces):
        d = len(inc_faces)
        col = comp.edge_colors[e_idx]
        edge_qstart.append(len(qubits))
        if d == 0:
            continue
        elif d == 2 or col == "blue":
            qi = len(qubits)
            qubits.append((e_idx, {f: 1 for f in inc_faces}))
            if d >= 3:
                seam_qubits.add(qi)
            if e_idx in cyl_edges:
                cyl_qubits.add(qi)
        elif col == "red":
            for j in range(d - 1):
                qi = len(qubits)
                qubits.append((e_idx, {inc_faces[j]: 1, inc_faces[j + 1]: 1}))
                if d >= 3:
                    seam_qubits.add(qi)
                if e_idx in cyl_edges:
                    cyl_qubits.add(qi)

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
            alpha = 1 << fv
            for p, prow in pivots.items():
                if (prow >> fv) & 1:
                    alpha |= 1 << p
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
                    if (alpha >> f_to_loc[inc[0]]) & 1:
                        xs.append(q0)
                elif col == "red":
                    pref = 0
                    for j in range(d - 1):
                        pref ^= (alpha >> f_to_loc[inc[j]]) & 1
                        if pref:
                            xs.append(q0 + j)
            return sorted(set(xs))

        dim_fv = len(raw_basis)
        if 1 < dim_fv <= 7:
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
            ipivs, chosen = {}, []
            for _, mask, supp in cands:
                x = mask
                while x:
                    lsb = (x & -x).bit_length() - 1
                    if lsb in ipivs:
                        x ^= ipivs[lsb]
                    else:
                        ipivs[lsb] = x
                        chosen.append(supp)
                        break
                if len(chosen) == dim_fv:
                    break
            H_X.extend(chosen)
        else:
            for alpha in raw_basis:
                supp = alpha_to_qsupp(alpha)
                if supp:
                    H_X.append(supp)
    return num_q, H_X, H_Z, seam_qubits, bridge_faces, cyl_qubits


def build_3d_toric(L):
    """Exact LxLxL 3D Toric Code: N = 3*L^3 edges, M_X = L^3 vertex stars (wt 6), M_Z = 3*L^3 plaquettes (wt 4), K = 3."""
    def eid(x, y, z, d):
        return (((x % L) * L + (y % L)) * L + (z % L)) * 3 + d

    N = 3 * (L ** 3)
    H_X, H_Z = [], []
    for x in range(L):
        for y in range(L):
            for z in range(L):
                H_X.append(sorted([
                    eid(x, y, z, 0), eid(x - 1, y, z, 0),
                    eid(x, y, z, 1), eid(x, y - 1, z, 1),
                    eid(x, y, z, 2), eid(x, y, z - 1, 2),
                ]))
                # xy plaquette
                H_Z.append(sorted([eid(x, y, z, 0), eid(x, y + 1, z, 0), eid(x, y, z, 1), eid(x + 1, y, z, 1)]))
                # yz plaquette
                H_Z.append(sorted([eid(x, y, z, 1), eid(x, y, z + 1, 1), eid(x, y, z, 2), eid(x, y + 1, z, 2)]))
                # xz plaquette
                H_Z.append(sorted([eid(x, y, z, 0), eid(x, y, z + 1, 0), eid(x, y, z, 2), eid(x + 1, y, z, 2)]))
    return N, H_X, H_Z


def build_haah_cubic_a(L):
    """Haah's 3D Cubic Code A on LxLxL torus: N = 2*L^3, weight-8 generators, K=2 for odd L not divisible by 3.."""
    def qid(x, y, z, c):
        return (((x % L) * L + (y % L)) * L + (z % L)) * 2 + c

    N = 2 * (L ** 3)
    H_X, H_Z = [], []
    for x in range(L):
        for y in range(L):
            for z in range(L):
                # G_X: (1+x+y+z, 1+xy+yz+zx)
                rx = [
                    qid(x, y, z, 0), qid(x + 1, y, z, 0), qid(x, y + 1, z, 0), qid(x, y, z + 1, 0),
                    qid(x, y, z, 1), qid(x + 1, y + 1, z, 1), qid(x, y + 1, z + 1, 1), qid(x + 1, y, z + 1, 1),
                ]
                # G_Z: (1+xy+yz+zx, 1+x^{-1}+y^{-1}+z^{-1}) -> wait: standard transposed partner so H_X H_Z^T = 0:
                # For (A, B)_X and (B^T, A^T)_Z where T inverts (x,y,z) -> (-x,-y,-z):
                rz = [
                    qid(x, y, z, 0), qid(x - 1, y - 1, z, 0), qid(x, y - 1, z - 1, 0), qid(x - 1, y, z - 1, 0),
                    qid(x, y, z, 1), qid(x - 1, y, z, 1), qid(x, y - 1, z, 1), qid(x, y, z - 1, 1),
                ]
                H_X.append(sorted(set(rx)))
                H_Z.append(sorted(set(rz)))
    return N, H_X, H_Z


def write_coupling_profiles(tag, sec, checks, N, seam_qubits, bridge_faces, cyl_qubits):
    """
    Writes local check-coupling files (`j_tenths`, where 10 = J=1.0):
      1. `<tag>_hill_norm.j`: strictly budget-neutral (<J> == 1.000, mean j_tenths == 10.0) wire-hill profile!
      2. `<tag>_hill_bnd.j`:  bounded local hill J in [1.0, 2.2] (j_tenths in [10, 22]) peaking away from junctions!
      3. `<tag>_soft_junc.j`: soft-junction cooling profile (junction checks softened to 4, sheet checks boosted so <J>==1.0)!
    """
    M = len(checks)
    if M == 0:
        return None, None, None
    q_to_c = [[] for _ in range(N)]
    for ci, r in enumerate(checks):
        for q in r:
            q_to_c[q].append(ci)

    # Identify junction checks: checks touching seam_qubits (or qubits with check-degree >= 3)
    dist = [-1] * M
    dq = deque()
    for ci, r in enumerate(checks):
        is_junc = any((q in seam_qubits) or (len(q_to_c[q]) >= 3) or (len(r) >= 5) for q in r)
        if is_junc:
            dist[ci] = 0
            dq.append(ci)
    if not dq:
        dist = [0] * M
    else:
        while dq:
            u = dq.popleft()
            if dist[u] >= 4:
                continue
            for q in checks[u]:
                for v in q_to_c[q]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        dq.append(v)
        for ci in range(M):
            if dist[ci] == -1:
                dist[ci] = 4

    # 1) Bounded hill J in [1.0, 2.2]: 10 + 4 * min(dist, 3) -> {10, 14, 18, 22}
    jbnd = [10 + 4 * min(d, 3) for d in dist]
    # 2) Budget-neutral hill: rescale jbnd so mean == 10.000!
    mean_b = sum(jbnd) / M
    jnorm = [max(4, int(round(10.0 * v / mean_b))) for v in jbnd]
    # Fine-tune exact sum == 10 * M:
    diff = 10 * M - sum(jnorm)
    idx = 0
    while diff != 0 and idx < 4 * M:
        ci = idx % M
        if diff > 0:
            jnorm[ci] += 1
            diff -= 1
        elif diff < 0 and jnorm[ci] > 5:
            jnorm[ci] -= 1
            diff += 1
        idx += 1

    # 3) Soft-junction (bridge-decoupled) cooling profile:
    #    On X_err (Z-face checks): perpendicular bridge faces get J=0.15 (raw 2), base sheet faces get raw 14!
    #    On Z_err (X-vertex checks): bridge-coupling checks (touching cyl_qubits or weight >= 5) get raw 2, sheet checks get raw 14!
    jsoft_raw = []
    for ci, r in enumerate(checks):
        is_br = (ci in bridge_faces) if sec == "X_err" else (len(r) >= 5 or any(q in cyl_qubits for q in r))
        jsoft_raw.append(2 if is_br else 14)
    ms = sum(jsoft_raw) / M
    jsoft = [max(2, int(round(10.0 * v / ms))) for v in jsoft_raw]

    p_norm = os.path.join(DATA_DIR, f"{tag}_hill_norm.j")
    p_bnd = os.path.join(DATA_DIR, f"{tag}_hill_bnd.j")
    p_soft = os.path.join(DATA_DIR, f"{tag}_soft_junc.j")
    for path, arr in [(p_norm, jnorm), (p_bnd, jbnd), (p_soft, jsoft)]:
        with open(path, "w") as f:
            f.write("\n".join(str(v) for v in arr) + "\n")
    return p_norm, p_bnd, p_soft


def export_raw_pair(spec):
    name, kind, params = spec
    seam_q, bridge_f, cyl_q = set(), set(), set()
    if kind == "toric3d":
        N, HX, HZ = build_3d_toric(params["L"])
    elif kind == "cubic":
        N, HX, HZ = build_haah_cubic_a(params["L"])
    elif kind == "caha":
        _install_caha_shims()
        from LayerCode import LayerCode
        lc = LayerCode(params["Hx"], params["Hz"], c_x=1, c_y=1, c_z=1)
        HX = [sorted(s) for s in lc.build_XStabilizers()]
        HZ = [sorted(s) for s in lc.build_ZStabilizers()]
        N = lc.num_total_qubits
    elif kind == "sheaf":
        seed_name, steps, aniso, gmode = params["seed"], params["steps"], params["aniso"], params.get("gmode", "full")
        if seed_name == "planar":
            comp = seed_planar_11()
        elif seed_name == "tor2":
            comp = seed_torus(2, 2)
        elif seed_name == "rottor":
            comp = seed_rot_torus_422()
        else:
            raise ValueError(seed_name)
        cyl = set()
        for L_step, ribs, itype in steps:
            comp, cyl = apply_sheaf_step(comp, L_step, ribs, itype, cyl if aniso else None, gadget_mode=gmode)
        N, HX, HZ, seam_q, bridge_f, cyl_q = sheaf_to_css_with_seams(comp)
    else:
        raise ValueError(kind)

    out_files = {}
    sectors = params.get("sectors", ["X_err", "Z_err"])
    for sec in sectors:
        checks, stabs = (HZ, HX) if sec == "X_err" else (HX, HZ)
        tag = f"{name}_{sec}"
        raw_p = os.path.join(DATA_DIR, f"{tag}.raw")
        fin_p = os.path.join(DATA_DIR, f"{tag}.code")
        if not os.path.exists(fin_p):
            with open(raw_p, "w") as f:
                f.write(f"{N} {len(checks)} {len(stabs)}\n")
                for r in checks:
                    f.write(str(len(r)) + " " + " ".join(map(str, r)) + "\n")
                for r in stabs:
                    f.write(str(len(r)) + " " + " ".join(map(str, r)) + "\n")
            subprocess.run([ENGINE, "prep_logicals", raw_p, fin_p], check=True)
            os.remove(raw_p)
        if params.get("make_j", False):
            write_coupling_profiles(tag, sec, checks, N, seam_q, bridge_f, cyl_q)
        out_files[sec] = fin_p
    return name, N, out_files


def load_caha_specs():
    specs = []
    # 1) Exact validation on n=11 (first 6 codes including code_100.txt!)
    csv11 = os.path.join(CAHA_REPO, "data/memory-simulations/11/n_11_beta_6.0.csv")
    seen = {}
    with open(csv11) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            cname = os.path.splitext(os.path.basename(row["code"]))[0]
            if cname not in seen:
                Hx = [[int(x) for x in r.split(",")] for r in row["Hx"].strip("[]").split(";")]
                Hz = [[int(x) for x in r.split(",")] for r in row["Hz"].strip("[]").split(";")]
                seen[cname] = (Hx, Hz)
            if len(seen) >= 6:
                break
    for cname, (Hx, Hz) in seen.items():
        specs.append((f"caha_n11_{cname}", "caha", {"Hx": Hx, "Hz": Hz, "sectors": ["X_err"]}))

    # 2) Fixed-T scaling across n = 13, 15, 17, 19 (3 random codes per n from published CSVs!)
    for n_val in [13, 15, 17, 19]:
        dpath = os.path.join(CAHA_REPO, f"data/memory-simulations/{n_val}")
        csv_files = sorted(f for f in os.listdir(dpath) if f.endswith(".csv"))
        s_n = {}
        with open(os.path.join(dpath, csv_files[0])) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                cname = os.path.splitext(os.path.basename(row["code"]))[0]
                if cname not in s_n:
                    Hx = [[int(x) for x in r.split(",")] for r in row["Hx"].strip("[]").split(";")]
                    Hz = [[int(x) for x in r.split(",")] for r in row["Hz"].strip("[]").split(";")]
                    s_n[cname] = (Hx, Hz)
                if len(s_n) >= 3:
                    break
        for cname, (Hx, Hz) in s_n.items():
            specs.append((f"caha_n{n_val}_{cname}", "caha", {"Hx": Hx, "Hz": Hz, "sectors": ["X_err"]}))
    return specs


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)
    os.makedirs(MOV_DIR, exist_ok=True)

    specs = []
    # A. 3D Toric Code One-Sector Benchmark (L=3..7)
    for L in [3, 4, 5, 6, 7]:
        specs.append((f"toric3d_L{L}", "toric3d", {"L": L, "sectors": ["X_err", "Z_err"]}))

    # B. Haah's 3D Cubic Code A (L=5, 7, 9, 11)
    for L in [5, 7, 9, 11]:
        specs.append((f"cubic_L{L}", "cubic", {"L": L, "sectors": ["X_err"]}))

    # C. Gu-Caha Random CSS Layer Codes (n=11..19)
    specs.extend(load_caha_specs())

    # D. Hayden S8.3 Leaner Gadget Percolation Series (`full` 4-rib vs `cb3rib` 3-rib vs `cb2rib` 2-rib)
    #    PLUS s=5 (N ~ 145k - 190k!) scaling!
    def steps_L2(n_steps):
        return [(2, [1], "X" if i % 2 == 0 else "Z") for i in range(n_steps)]

    for gmode, prefix in [("full", "sq_L2r1"), ("cb3rib", "sq_cb3rib"), ("cb2rib", "sq_cb2rib")]:
        max_s = 5 if gmode in ("full", "cb3rib") else 4
        for s in range(1, max_s + 1):
            specs.append((
                f"{prefix}_s{s}",
                "sheaf",
                {"seed": "planar", "steps": steps_L2(s), "aniso": True, "gmode": gmode, "make_j": (s >= 2)},
            ))

    for gmode, prefix in [("full", "rottor_L2r1"), ("cb3rib", "rottor_cb3rib"), ("cb2rib", "rottor_cb2rib")]:
        for s in [2, 3, 4]:
            specs.append((
                f"{prefix}_s{s}",
                "sheaf",
                {"seed": "rottor", "steps": steps_L2(s), "aniso": True, "gmode": gmode, "make_j": True},
            ))

    # Proven 2-Bridge (`L=3, r=[1,2]`) at s=2 (N=1,480) and s=3 (N ~ 22,000, 1.5 cycles!)
    for s in [2, 3]:
        st3 = [(3, [1, 2], "X" if i % 2 == 0 else "Z") for i in range(s)]
        specs.append((f"sq_L3r2_s{s}", "sheaf", {"seed": "planar", "steps": st3, "aniso": True, "make_j": True}))

    # E. Z^3 Cubic-Grid Wire-Hill Rescue Codes (Up to BDL's Full 56,312-Qubit ell=8 Code!)
    specs.extend([
        ("sq_z3_3_5_r1", "sheaf", {"seed": "planar", "steps": [(3, [2], "X"), (5, [4], "Z")], "aniso": False, "make_j": True}),
        ("sq_z3_4_6_r2", "sheaf", {"seed": "planar", "steps": [(4, [2, 3], "X"), (6, [4, 5], "Z")], "aniso": False, "make_j": True}),
        ("tor2_z3_4_6_r2", "sheaf", {"seed": "tor2", "steps": [(4, [2, 3], "X"), (6, [4, 5], "Z")], "aniso": False, "make_j": True}),
        ("sq_bdl_8_8_r2", "sheaf", {"seed": "planar", "steps": [(8, [4, 5], "X"), (8, [4, 5], "Z")], "aniso": False, "make_j": True}),
    ])

    # Build smaller codes in parallel first, then huge codes (N > 40,000) sequentially so 16 threads handle RREF cleanly!
    small_specs = [sp for sp in specs if not any(x in sp[0] for x in ["_s5", "bdl_8_8", "tor2_z3_4_6", "_s4"])]
    large_specs = [sp for sp in specs if any(x in sp[0] for x in ["_s5", "bdl_8_8", "tor2_z3_4_6", "_s4"])]

    print(f"Building {len(small_specs)} small/medium code pairs in parallel...")
    with ProcessPoolExecutor(max_workers=8) as ex:
        for name, N, _ in ex.map(export_raw_pair, small_specs):
            print(f"  Built {name} (N={N})", flush=True)

    print(f"Building {len(large_specs)} large code pairs sequentially with multi-threaded C++ RREF...")
    for sp in large_specs:
        name, N, _ = export_raw_pair(sp)
        print(f"  Built LARGE {name} (N={N})", flush=True)

    # =========================================================================
    # Build Manifest of Simulation Tasks (`manifest_v6.txt`)
    # =========================================================================
    tasks = []

    def add_kmc(tag, j_mode, dec, beta, trials, max_flips):
        j_arg = "uniform" if j_mode == "uniform" else os.path.join(DATA_DIR, f"{tag}_{j_mode}.j")
        tid = f"kmc__{tag}__{j_mode}__{dec}__b{beta:.2f}"
        tasks.append(f"{tid}|kmc|{os.path.join(DATA_DIR, tag + '.code')}|{j_arg}|{dec}|{beta:.2f}|{trials}|{max_flips}")

    def add_ffs(tag, j_mode, dec, beta, shots, reps, save_movie=False):
        j_arg = "uniform" if j_mode == "uniform" else os.path.join(DATA_DIR, f"{tag}_{j_mode}.j")
        tid = f"ffs__{tag}__{j_mode}__{dec}__b{beta:.2f}"
        mov = os.path.join(MOV_DIR, f"{tid}.txt") if save_movie else ""
        tasks.append(f"{tid}|ffs|{os.path.join(DATA_DIR, tag + '.code')}|{j_arg}|{dec}|{beta:.2f}|{shots}|{reps}|{mov}")

    def add_cool(tag, dec, proto, init_m, bi, bf, ttot, trials):
        j_soft = os.path.join(DATA_DIR, f"{tag}_soft_junc.j")
        tid = f"cool__{tag}__{proto}__{init_m}__t{int(ttot)}"
        tasks.append(f"{tid}|cool|{os.path.join(DATA_DIR, tag + '.code')}|{j_soft}|{dec}|{proto}|{init_m}|{bi}|{bf}|{ttot}|{trials}")

    # 1A. 3D Toric Code One-Sector Loop Benchmark (`sweep_uf` on X_err vs `uf` on Z_err)
    for L in [3, 4, 5, 6, 7]:
        for b in [1.50, 1.75, 2.00, 2.25]:
            add_kmc(f"toric3d_L{L}_X_err", "uniform", "sweep_uf", b, 64, 50_000_000)
        for b in [2.00, 3.00, 4.00]:
            add_kmc(f"toric3d_L{L}_Z_err", "uniform", "uf", b, 64, 20_000_000)

    # 1B. Gu-Caha Random CSS Layer Codes: Exact Validation (n=11) + Scaling (n=11..19)
    for sp_name, kind, _ in specs:
        if kind == "caha":
            for b in [6.00, 7.00, 8.00]:
                add_kmc(f"{sp_name}_X_err", "uniform", "uf", b, 48, 80_000_000)
            if "n11_code_100" in sp_name or sp_name.startswith("caha_n15") or sp_name.startswith("caha_n19"):
                add_ffs(f"{sp_name}_X_err", "uniform", "uf", 8.00, 1600, 4)
                add_ffs(f"{sp_name}_X_err", "uniform", "uf", 9.00, 1600, 4)

    # 2A & 2B. FFS Validation + Deep Low-T Scaling (beta = 5.5 .. 9.0) & Failure Movies!
    # Head-to-head KMC vs FFS on s=2,4 and Cubic Code:
    for L in [5, 7, 9, 11]:
        for b in [4.25, 5.00]:
            add_kmc(f"cubic_L{L}_X_err", "uniform", "uf", b, 32, 120_000_000)
        for b in [5.00, 6.00, 7.00, 8.00]:
            add_ffs(f"cubic_L{L}_X_err", "uniform", "uf", b, 1500, 4, save_movie=(L == 7 and b == 6.00))

    # Boundaryless 3D-local [[4,2,2]] Rotated Torus (`rottor_L2r1` & `rottor_cb3rib` & `rottor_cb2rib`)
    for gprefix in ["rottor_L2r1", "rottor_cb3rib", "rottor_cb2rib"]:
        for s in [2, 3, 4]:
            for sec in ["X_err", "Z_err"]:
                tag = f"{gprefix}_s{s}_{sec}"
                for b in [5.50, 6.00, 6.50]:
                    add_kmc(tag, "uniform", "uf", b, 32, 120_000_000)
                for b in [6.00, 6.50, 7.00, 7.50, 8.00, 8.50]:
                    if gprefix == "rottor_cb2rib" and b > 7.50:
                        continue
                    save_m = (gprefix == "rottor_L2r1" and s in (2, 4) and b == 7.50 and sec == "X_err")
                    shots = 1200 if s == 4 else 2000
                    add_ffs(tag, "uniform", "uf", b, shots, 4, save_movie=save_m)

    # Planar Hierarchy Up to s=5 (N ~ 145k - 190k!) + Lean Gadget Percolation Series
    for gprefix in ["sq_L2r1", "sq_cb3rib", "sq_cb2rib"]:
        s_list = [2, 3, 4, 5] if gprefix in ("sq_L2r1", "sq_cb3rib") else [2, 3, 4]
        for s in s_list:
            for sec in ["X_err", "Z_err"]:
                tag = f"{gprefix}_s{s}_{sec}"
                for b in [5.50, 6.00, 6.50]:
                    if s <= 4:
                        add_kmc(tag, "uniform", "uf", b, 32, 120_000_000)
                    else:
                        add_kmc(tag, "uniform", "uf", b, 16, 80_000_000)
                for b in [6.50, 7.00, 7.50, 8.00]:
                    if gprefix == "sq_cb2rib" and b > 7.00:
                        continue
                    shots = 900 if s == 5 else (1200 if s == 4 else 2000)
                    save_m = (gprefix == "sq_L2r1" and s in (2, 4) and b == 7.50 and sec == "Z_err")
                    add_ffs(tag, "uniform", "uf", b, shots, 4, save_movie=save_m)

    # Proven 2-Bridge (`sq_L3r2_s2` & `sq_L3r2_s3`)
    for s in [2, 3]:
        for sec in ["X_err", "Z_err"]:
            tag = f"sq_L3r2_s{s}_{sec}"
            for b in [5.50, 6.00, 6.50]:
                add_kmc(tag, "uniform", "uf", b, 32, 100_000_000)
            for b in [6.50, 7.00, 7.50, 8.00]:
                add_ffs(tag, "uniform", "uf", b, 1500, 4)

    # 4. Z^3 Cubic-Grid Wire-Hill Rescue (Uniform vs Budget-Neutral Hill <J>==1 vs Bounded Hill J in [1, 2.2])!
    for zname in ["sq_z3_3_5_r1", "sq_z3_4_6_r2", "tor2_z3_4_6_r2", "sq_bdl_8_8_r2"]:
        for sec in ["X_err", "Z_err"]:
            tag = f"{zname}_{sec}"
            trials = 20 if ("bdl_8_8" in zname or "tor2_z3" in zname) else 32
            for j_mode in ["uniform", "hill_norm", "hill_bnd"]:
                for b in [5.50, 6.00, 6.50, 7.00]:
                    add_kmc(tag, j_mode, "uf", b, trials, 120_000_000)

    # 5. Soft-Junction Passive Cooling vs Hard Quench & Standard Annealing!
    for cname in ["rottor_L2r1_s2", "rottor_L2r1_s3", "sq_L2r1_s3", "sq_L2r1_s4"]:
        for sec in ["X_err", "Z_err"]:
            tag = f"{cname}_{sec}"
            trials = 24 if "s4" in cname else 64
            for init_m in ["hot", "noisy_0.12"]:
                for ttot in [50.0, 250.0, 1000.0]:
                    if "s4" in cname and ttot > 250.0:
                        continue
                    for proto in ["quench", "anneal", "soft_junc"]:
                        add_cool(tag, "uf", proto, init_m, 2.5, 7.5, ttot, trials)

    mpath = os.path.join(WORK_DIR, "manifest_v6.txt")
    with open(mpath, "w") as f:
        for line in tasks:
            f.write(line + "\n")
    print(f"Wrote {len(tasks)} total V6 tasks to {mpath}!")


if __name__ == "__main__":
    main()
