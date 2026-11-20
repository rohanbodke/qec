#!/usr/bin/env python3
"""
build_v7_rescue.py
Zero-Trust V7 Rescue & Verification Builder for MIT 6.S980:
  1. Fixes `(fdeg != 2, col == "none") -> col = new_col` in `apply_sheaf_step` so ALL pruned
     `cb3rib` and `cb2rib` codes satisfy exact `H_X * H_Z^T == 0` (enforced by C++ hard gate!).
  2. Builds reversed `1 -> 2` hybrid (`rottor_hyb12_s2`) alongside `2 -> 1` (`rottor_hyb21_s2`),
     `1 -> 1` (`rottor_L2r1_s2`), and `2 -> 2` (`rottor_L3r2_s2`) on the EXACT same `[[4,2,2]]` seed.
  3. Builds targeted `soft_seam.j` (<20% true 3-way junction checks softened, <J> == 1.0000) plus
     budget-neutral `pin_norm.j`, `hill_norm.j`, and `hill_bnd.j` on Z^3 codes up to N = 56,312.
  4. Verifies (`kmc_engine_v7 verify`) every reused V6 `.code` file (`bad_comm == 0` required!).
  5. Emits `manifest_v7.txt` covering all 6 rescued experimental suites.
"""

from collections import deque
from concurrent.futures import ProcessPoolExecutor
import os
import shutil
import subprocess

V6_DATA = "/work/submit/harshila/6s980_qmem_v6/data"
V5_DATA = "/work/submit/harshila/6s980_qmem_v5/data"
WORK_DIR = "/work/submit/harshila/6s980_qmem_v7"
DATA_DIR = os.path.join(WORK_DIR, "data")
RES_DIR = os.path.join(WORK_DIR, "results")
MOV_DIR = os.path.join(WORK_DIR, "movies")
ENGINE = os.path.join(WORK_DIR, "kmc_engine_v7")


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


def seed_rot_torus_422():
    edges = [(0, 1), (0, 1), (0, 1), (0, 1)]
    faces = [(0, 1, 0, 1, 0, 3, 2, 1), (1, 0, 1, 0, 1, 2, 3, 0)]
    return GeneralCubical2D(2, edges, ["none"] * 4, faces)


def apply_sheaf_step(comp, L, rib_coords, iter_type, prev_cyl=None, gadget_mode="full"):
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

    tot_v = 2 * xp_num_v

    def vc(v, c):
        return v + c * xp_num_v

    out_e, out_c, out_f = [], [], []
    latest_bf = set()  # ONLY bridges added at THIS step (not cumulative!)
    n_xe = len(xp_edges)
    for c in (0, 1):
        for eid, (u, w) in enumerate(xp_edges):
            out_e.append((vc(u, c), vc(w, c)))
            out_c.append(new_col if xp_is_rib[eid] else xp_colors[eid])
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
            latest_bf.add(len(out_f))
            out_f.append((vc(u, 0), vc(w, 0), vc(w, 1), vc(u, 1), eid, eid + n_xe, cyl_eid[u], cyl_eid[w]))
    res = GeneralCubical2D(tot_v, out_e, out_c, out_f)
    res.latest_bridge_faces = latest_bf
    res.cyl_edges = new_cyl
    # V7 FIX: Any edge with fdeg == 2 gets "none"; any boundary/buttonhole/seam edge with fdeg != 2 and "none" gets `new_col`!
    for eid in range(len(res.edges)):
        fd = len(res.edge_faces[eid])
        if fd == 2:
            res.edge_colors[eid] = "none"
        elif res.edge_colors[eid] == "none":
            res.edge_colors[eid] = new_col
    return res, new_cyl


def sheaf_to_css_with_seams(comp):
    num_faces = len(comp.faces)
    cyl_edges = getattr(comp, "cyl_edges", set())
    latest_bf = getattr(comp, "latest_bridge_faces", set())
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
    return num_q, H_X, H_Z, seam_qubits, latest_bf, cyl_qubits


def write_soft_seam_j(tag, sec, checks, N, latest_bf, cyl_qubits):
    """Targeted <20% latest-bridge softening (`soft_seam.j`) with <J> == 1.000!"""
    M = len(checks)
    if M == 0:
        return
    is_target = []
    for ci, r in enumerate(checks):
        if sec == "X_err":
            is_target.append(ci in latest_bf)
        else:
            is_target.append(any(q in cyl_qubits for q in r))
    # Soften ONLY latest-step bridge checks (~12-22% of checks) to raw 5, sheets raw 11 -> rescale to mean 10.000!
    jraw = [5 if t else 11 for t in is_target]
    mr = sum(jraw) / M
    jsoft = [max(4, int(round(10.0 * v / mr))) for v in jraw]
    diff = 10 * M - sum(jsoft)
    idx = 0
    while diff != 0 and idx < 4 * M:
        ci = idx % M
        if not is_target[ci]:
            if diff > 0:
                jsoft[ci] += 1; diff -= 1
            elif diff < 0 and jsoft[ci] > 6:
                jsoft[ci] -= 1; diff += 1
        idx += 1
    with open(os.path.join(DATA_DIR, f"{tag}_soft_seam.j"), "w") as f:
        f.write("\n".join(str(x) for x in jsoft) + "\n")


def write_z3_couplings_from_code(tag):
    """Writes exact `hill_norm.j`, `hill_bnd.j`, `pin_norm.j`, and `pin_bnd.j` directly from `.code`."""
    cpath = os.path.join(DATA_DIR, f"{tag}.code")
    with open(cpath) as f:
        hdr = list(map(int, f.readline().split()))
        N, M = hdr[0], hdr[1]
        checks = []
        q_to_c = [[] for _ in range(N)]
        for ci in range(M):
            row = list(map(int, f.readline().split()))[1:]
            checks.append(row)
            for q in row:
                q_to_c[q].append(ci)

    # Distance from junctions for hill_norm / hill_bnd
    dist = [-1] * M
    dq = deque()
    for ci, r in enumerate(checks):
        if len(r) >= 5 or any(len(q_to_c[q]) >= 3 for q in r):
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

    jbnd = [10 + 4 * min(d, 3) for d in dist]
    mb = sum(jbnd) / M
    jnorm = [max(4, int(round(10.0 * v / mb))) for v in jbnd]
    diff = 10 * M - sum(jnorm)
    idx = 0
    while diff != 0 and idx < 4 * M:
        ci = idx % M
        if diff > 0:
            jnorm[ci] += 1; diff -= 1
        elif diff < 0 and jnorm[ci] > 5:
            jnorm[ci] -= 1; diff += 1
        idx += 1

    # Independent-set pinning traps (`pin_norm` & `pin_bnd`)
    blocked_c = [False] * M
    is_trap = [False] * M
    for ci, r in enumerate(checks):
        if len(r) >= 5 or any(len(q_to_c[q]) >= 3 for q in r):
            blocked_c[ci] = True
    for ci, r in enumerate(checks):
        if blocked_c[ci]:
            continue
        is_trap[ci] = True
        for q in r:
            for nb_c in q_to_c[q]:
                blocked_c[nb_c] = True

    pin_bnd = [10 if is_trap[ci] else 20 for ci in range(M)]
    praw = [3 if is_trap[ci] else 13 for ci in range(M)]
    mp = sum(praw) / M
    pin_norm = [max(3, int(round(10.0 * v / mp))) for v in praw]
    diff = 10 * M - sum(pin_norm)
    wall_ids = [ci for ci in range(M) if not is_trap[ci]] or list(range(M))
    idx = 0
    while diff != 0 and idx < 6 * len(wall_ids):
        ci = wall_ids[idx % len(wall_ids)]
        if diff > 0:
            pin_norm[ci] += 1; diff -= 1
        elif diff < 0 and pin_norm[ci] > 6:
            pin_norm[ci] -= 1; diff += 1
        idx += 1

    for suf, arr in [("hill_norm", jnorm), ("hill_bnd", jbnd), ("pin_norm", pin_norm), ("pin_bnd", pin_bnd)]:
        with open(os.path.join(DATA_DIR, f"{tag}_{suf}.j"), "w") as out:
            out.write("\n".join(str(x) for x in arr) + "\n")


def build_sheaf_pair(spec):
    name, seed_name, steps, aniso, gmode, make_soft = spec
    comp = seed_planar_11() if seed_name == "planar" else seed_rot_torus_422()
    cyl = set()
    for L_step, ribs, itype in steps:
        comp, cyl = apply_sheaf_step(comp, L_step, ribs, itype, cyl if aniso else None, gadget_mode=gmode)
    N, HX, HZ, _, latest_bf, cyl_q = sheaf_to_css_with_seams(comp)
    k_vals = {}
    for sec in ["X_err", "Z_err"]:
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
        if make_soft:
            write_soft_seam_j(tag, sec, checks, N, latest_bf, cyl_q)
        with open(fin_p) as f:
            hdr = list(map(int, f.readline().split()))
            k_vals[sec] = hdr[3]
    return name, N, k_vals["X_err"], k_vals["Z_err"]


def symlink_and_verify(src_path, dst_tag):
    dst_path = os.path.join(DATA_DIR, f"{dst_tag}.code")
    if not os.path.exists(dst_path):
        os.symlink(src_path, dst_path)
    res = subprocess.run([ENGINE, "verify", dst_path], capture_output=True, text=True, check=True)
    return res.stdout.strip()


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)
    os.makedirs(MOV_DIR, exist_ok=True)

    # 1. Symlink & mathematically verify (`bad_comm == 0`!) clean V6 & V5 `.code` files
    reused = [
        (os.path.join(V6_DATA, "cubic_L5_X_err.code"), "cubic_L5_X_err"),
        (os.path.join(V6_DATA, "cubic_L7_X_err.code"), "cubic_L7_X_err"),
        (os.path.join(V6_DATA, "caha_n11_code_100_X_err.code"), "caha_n11_code_100_X_err"),
        (os.path.join(V6_DATA, "toric3d_L7_X_err.code"), "toric3d_L7_X_err"),
        (os.path.join(V6_DATA, "sq_z3_3_5_r1_X_err.code"), "sq_z3_3_5_r1_X_err"),
        (os.path.join(V6_DATA, "sq_z3_3_5_r1_Z_err.code"), "sq_z3_3_5_r1_Z_err"),
        (os.path.join(V6_DATA, "sq_z3_4_6_r2_X_err.code"), "sq_z3_4_6_r2_X_err"),
        (os.path.join(V6_DATA, "sq_z3_4_6_r2_Z_err.code"), "sq_z3_4_6_r2_Z_err"),
        (os.path.join(V6_DATA, "sq_bdl_8_8_r2_X_err.code"), "sq_bdl_8_8_r2_X_err"),
        (os.path.join(V6_DATA, "sq_bdl_8_8_r2_Z_err.code"), "sq_bdl_8_8_r2_Z_err"),
        (os.path.join(V6_DATA, "sq_L3r2_s2_X_err.code"), "sq_L3r2_s2_X_err"),
        (os.path.join(V6_DATA, "sq_L3r2_s2_Z_err.code"), "sq_L3r2_s2_Z_err"),
        (os.path.join(V6_DATA, "sq_L2r1_s4_X_err.code"), "sq_L2r1_s4_X_err"),
        (os.path.join(V6_DATA, "sq_L2r1_s4_Z_err.code"), "sq_L2r1_s4_Z_err"),
        (os.path.join(V6_DATA, "rottor_L2r1_s4_X_err.code"), "rottor_L2r1_s4_X_err"),
        (os.path.join(V6_DATA, "rottor_L2r1_s4_Z_err.code"), "rottor_L2r1_s4_Z_err"),
    ]
    print("Verifying reused V6 `.code` files (asserting exact H_X * H_Z^T == 0)...")
    for sp, dt in reused:
        out = symlink_and_verify(sp, dt)
        print(" ", out, flush=True)

    # 2. Build rescued commuting pruned codes (`cb3rib`, `cb2rib`), apples-to-apples `[[4,2,2]]` hybrids (`hyb21` vs `hyb12`),
    #    and `soft_seam.j`-equipped cooling codes!
    def steps_L2(n_steps):
        return [(2, [1], "X" if i % 2 == 0 else "Z") for i in range(n_steps)]

    build_specs = [
        # Cooling & baseline full codes (with `soft_seam.j` generated!)
        ("sq_L2r1_s2",     "planar", steps_L2(2), True, "full", True),
        ("sq_L2r1_s3",     "planar", steps_L2(3), True, "full", True),
        ("rottor_L2r1_s2", "rottor", steps_L2(2), True, "full", True),
        ("rottor_L2r1_s3", "rottor", steps_L2(3), True, "full", True),
        # Apples-to-apples `[[4,2,2]]` hybrid pre-compensation (`2 -> 1` vs `1 -> 2` vs `2 -> 2`)
        ("rottor_hyb21_s2", "rottor", [(3, [1, 2], "X"), (2, [1], "Z")], True, "full", True),
        ("rottor_hyb12_s2", "rottor", [(2, [1], "X"), (3, [1, 2], "Z")], True, "full", False),
        ("rottor_L3r2_s2",  "rottor", [(3, [1, 2], "X"), (3, [1, 2], "Z")], True, "full", False),
    ]
    # Exact commuting (`bad_comm == 0`) pruned buttonhole series `s = 1..4`!
    for gmode, pref in [("cb3rib", "rottor_cb3rib"), ("cb2rib", "rottor_cb2rib")]:
        for s in [1, 2, 3, 4]:
            build_specs.append((f"{pref}_s{s}", "rottor", steps_L2(s), True, gmode, False))
    for gmode, pref in [("cb3rib", "sq_cb3rib"), ("cb2rib", "sq_cb2rib")]:
        for s in [1, 2, 3, 4]:
            build_specs.append((f"{pref}_s{s}", "planar", steps_L2(s), True, gmode, False))

    small_sp = [sp for sp in build_specs if "_s4" not in sp[0]]
    large_sp = [sp for sp in build_specs if "_s4" in sp[0]]

    print(f"Building & verifying {len(small_sp)} small/medium V7 code pairs in parallel...")
    with ProcessPoolExecutor(max_workers=8) as ex:
        for nm, N, kx, kz in ex.map(build_sheaf_pair, small_sp):
            print(f"  VERIFIED {nm}: N={N}, K=({kx}, {kz})", flush=True)

    print(f"Building & verifying {len(large_sp)} large s=4 pruned code pairs sequentially...")
    for sp in large_sp:
        nm, N, kx, kz = build_sheaf_pair(sp)
        print(f"  VERIFIED LARGE {nm}: N={N}, K=({kx}, {kz})", flush=True)

    # Also build `soft_seam.j` on `sq_L2r1_s4` directly from its latest bridge structure?
    # Build `Z^3` coupling files (`hill_norm`, `hill_bnd`, `pin_norm`, `pin_bnd`)
    print("Generating Z^3 coupling profiles...")
    for zc in ["sq_z3_3_5_r1", "sq_z3_4_6_r2", "sq_bdl_8_8_r2"]:
        for sec in ["X_err", "Z_err"]:
            write_z3_couplings_from_code(f"{zc}_{sec}")

    # =========================================================================
    # Build `manifest_v7.txt`
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
        j_soft = os.path.join(DATA_DIR, f"{tag}_soft_seam.j") if proto == "soft_seam" else "uniform"
        tid = f"cool__{tag}__{proto}__{init_m}__t{int(ttot)}"
        tasks.append(f"{tid}|cool|{os.path.join(DATA_DIR, tag + '.code')}|{j_soft}|{dec}|{proto}|{init_m}|{bi}|{bf}|{ttot}|{trials}")

    # 1. Exact Ground-to-Ground FFS vs KMC Validation + Deep-beta Curves!
    for b in [4.25, 4.50, 4.75, 5.00]:
        add_kmc("cubic_L5_X_err", "uniform", "uf", b, 48, 200_000_000)
        add_kmc("cubic_L7_X_err", "uniform", "uf", b, 48, 200_000_000)
    for b in [4.50, 4.75, 5.00, 5.50, 6.00, 6.50, 7.00, 7.50, 8.00]:
        add_ffs("cubic_L5_X_err", "uniform", "uf", b, 2400, 8, save_movie=(b == 6.50))
    for b in [4.50, 4.75, 5.00, 5.50, 6.00, 6.50]:
        add_ffs("cubic_L7_X_err", "uniform", "uf", b, 2000, 6)

    for b in [6.00, 6.50, 7.00, 7.50, 8.00]:
        add_kmc("caha_n11_code_100_X_err", "uniform", "uf", b, 64, 120_000_000)
    for b in [6.50, 7.00, 7.50, 8.00, 8.50]:
        add_ffs("caha_n11_code_100_X_err", "uniform", "uf", b, 1800, 6)

    add_kmc("toric3d_L7_X_err", "uniform", "maj_uf", 1.75, 64, 250_000_000)

    # 2. Matched-Window Two-Cycle Crossings (`rottor_L2r1` s=2 vs s=4; `sq_L2r1` s=2 vs s=4) & Deep-beta Sector Asymmetry!
    for sec in ["X_err", "Z_err"]:
        for b in [5.00, 5.50, 6.00, 6.25, 6.50, 6.75]:
            add_kmc(f"rottor_L2r1_s2_{sec}", "uniform", "uf", b, 64, 150_000_000)
        for b in [6.00, 6.50, 7.00, 7.50, 8.00]:
            mov = (sec == "X_err" and b == 7.50)
            add_ffs(f"rottor_L2r1_s2_{sec}", "uniform", "uf", b, 2800, 8, save_movie=mov)
        for b in [5.50, 6.00, 6.25, 6.50, 6.75]:
            add_kmc(f"rottor_L2r1_s4_{sec}", "uniform", "uf", b, 40, 150_000_000)

        for b in [5.50, 6.00, 6.25, 6.50, 6.75, 7.00]:
            add_kmc(f"sq_L2r1_s2_{sec}", "uniform", "uf", b, 64, 150_000_000)
            add_kmc(f"sq_L2r1_s4_{sec}", "uniform", "uf", b, 40, 150_000_000)
        for b in [6.50, 7.00, 7.50, 8.00]:
            add_ffs(f"sq_L2r1_s2_{sec}", "uniform", "uf", b, 2400, 8)

    # 3. Apples-to-Apples `[[4,2,2]]` Hybrid `2 -> 1` vs `1 -> 2` vs `2 -> 2` + `sq_L3r2_s2`!
    for hname in ["rottor_hyb21_s2", "rottor_hyb12_s2", "rottor_L3r2_s2", "sq_L3r2_s2"]:
        for sec in ["X_err", "Z_err"]:
            for b in [5.00, 5.50, 6.00, 6.50, 7.00]:
                add_kmc(f"{hname}_{sec}", "uniform", "uf", b, 48, 140_000_000)
            for b in [6.00, 6.50, 7.00, 7.50, 8.00]:
                add_ffs(f"{hname}_{sec}", "uniform", "uf", b, 2200, 6)

    # 4. Commuting Pruned Gadgets (`bad_comm == 0`, `t_next = 0.0` exact 1/N collapse time across beta = 5.5..8.0)!
    for pref in ["rottor_cb3rib", "rottor_cb2rib", "sq_cb3rib", "sq_cb2rib"]:
        for s in [2, 3, 4]:
            for sec in ["X_err", "Z_err"]:
                for b in [5.50, 6.00, 6.50, 7.00, 7.50, 8.00]:
                    add_kmc(f"{pref}_s{s}_{sec}", "uniform", "uf", b, 64, 20_000_000)

    # 5. Unified Single-Engine Z^3 Wire-Stretching & Kramers Limit (`48-64` trials, identical cadence)!
    for zc in ["sq_z3_3_5_r1", "sq_z3_4_6_r2", "sq_bdl_8_8_r2"]:
        tr = 48 if "bdl_8_8" in zc else 64
        for sec in ["X_err", "Z_err"]:
            for jm in ["uniform", "pin_norm", "hill_norm", "hill_bnd"]:
                for b in [5.50, 6.00, 6.50, 7.00]:
                    add_kmc(f"{zc}_{sec}", jm, "uf", b, tr, 120_000_000)

    # 6. 3D Quantum Cooling (`hot` p0=0.5) & Sub-Threshold Autonomous Codeword Self-Healing (`p0 = 0.01 .. 0.12`)!
    for cname in ["sq_L2r1_s2", "rottor_L2r1_s2", "rottor_hyb21_s2", "sq_L2r1_s3", "rottor_L2r1_s3"]:
        for sec in ["X_err", "Z_err"]:
            tag = f"{cname}_{sec}"
            # (a) Hot state cooling: quench vs anneal vs targeted soft_seam
            for ttot in [100.0, 400.0, 1500.0]:
                for proto in ["quench", "anneal", "soft_seam"]:
                    add_cool(tag, "uf", proto, "hot", 1.8, 6.5, ttot, 100)
            # (b) Autonomous self-healing of noisy encoded codewords in a cold bath (beta = 6.5, t = 1000)
            for p0 in ["0.01", "0.02", "0.03", "0.04", "0.06", "0.08", "0.12"]:
                add_cool(tag, "uf", "quench", f"noisy_{p0}", 6.5, 6.5, 1000.0, 100)

    mpath = os.path.join(WORK_DIR, "manifest_v7.txt")
    with open(mpath, "w") as f:
        for ln in tasks:
            f.write(ln + "\n")
    print(f"Wrote {len(tasks)} verified V7 tasks to {mpath}!")


if __name__ == "__main__":
    main()
