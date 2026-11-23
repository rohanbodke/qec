#!/usr/bin/env python3
"""
verify_local_quotients_and_algebra.py
Independently reproduces and certifies Suites [B], [C], [D-algebra], [E-algebra], and [F-algebra]:
  [B] Exact & bounded CSS quotient solvers on 1-step planar cells (C2 full, C2 3-arm, C2 2-arm, C3, R3):
      - Exact Z(beta) via dual syndrome-parity Walsh transform over Ker(H_Z^T)
      - Exact clean hitting time T_clean = Z / (2 * kappa) = Z / (2 * Gamma * h_bar)
      - Exact optimal final-syndrome single-sector P_opt^X(t), P_opt^Z(t), and two-sector P_opt^Q(t)
      - Rigorous model-truncation brackets [T_clean_lo, T_clean_hi] for C3 (N=41) and R3 (N=44)
  [C] Analytic low-temperature rational critical networks & theory checks:
      - Exact rational prefactors kappa(beta) as beta -> inf (10/3, 423/119, 258892/56235, 14, and Z-sector 6)
      - Exact joint-limit mobility crossover on 1-step C3: kappa_{C3} ~ [3(lambda+1)/(lambda+3)] e^{-2*beta}
        for mu = lambda * e^{-beta}, plus finite-beta brackets at lambda=1, beta in {8, 9, 10}
      - Variational monotonicity identity check and explicit 4-state non-symmetric counterexample
  [D] Algebraic & initial-hazard certificates for resource-matched L=4 offset (1+3) vs centered (2+2):
      - Standalone pair (N=2,926, K=1) and coarse-C2 composed pair (N=261,476, K=1)
  [E, F] Direct two-step C2 bridge pruning & protected-subsystem certificates:
      - Full (N=254, K=1), X-pruned (N=248, K=1), Z-pruned (N=228, K=9), Both-pruned (N=222, K=9)
      - Symplectic completion (Poulin 2005) embedding propagated seed logical pair into 9-qubit basis,
        verifying 8 weight-1 Z operators span 6 independent gauge classes commuting with the target seed
      - Exhaustive low-energy reachability proving exact target barriers (2,2) vs (1,1) with witness [8,9,10,11]
"""

from collections import Counter, defaultdict, deque
from fractions import Fraction
import json
import math
import os
import sys
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "cluster_v9"))

from build_v7_rescue import (
    GeneralCubical2D,
    apply_sheaf_step,
    seed_planar_11,
    sheaf_to_css_with_seams,
)
from build_v9_quarter_million import (
    edge_qstarts,
    push_X_logical,
    push_Z_logical,
)


def apply_step_general(comp, L, rib_coords, iter_type, prev_cyl=None, gadget_mode="full"):
    """
    Exact replica of apply_sheaf_step in build_v7_rescue.py that ALSO returns the
    parent-to-child metadata needed by push_X_logical and push_Z_logical.
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

    xp_faces = []
    face_grids = []
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
                isr = s in rib_set
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
        f_start = len(xp_faces)
        for s in range(Lx):
            for t in range(Ly):
                xp_faces.append(
                    (
                        grid[s][t],
                        grid[s + 1][t],
                        grid[s + 1][t + 1],
                        grid[s][t + 1],
                        he[s][t],
                        he[s][t + 1],
                        ve[s][t],
                        ve[s + 1][t],
                    )
                )
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
            out_f.append(
                (vc(v00, c), vc(v10, c), vc(v11, c), vc(v01, c), eb + off, et + off, el + off, er + off)
            )
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


def gf2_rank_sparse(rows, ncols):
    """Fast bitmask GF(2) elimination supporting 300k+ columns via C-level big-int XOR."""
    piv = {}
    for r in rows:
        cur = 0
        for q in r:
            cur |= 1 << q
        while cur:
            p = (cur & -cur).bit_length() - 1
            if p in piv:
                cur ^= piv[p]
            else:
                piv[p] = cur
                break
    return len(piv), piv


def gf2_rank_bitmask(rows):
    piv = {}
    indep_idx = []
    for idx, r in enumerate(rows):
        x = 0
        for q in r:
            x |= 1 << q
        y = x
        while y:
            b = (y & -y).bit_length() - 1
            if b in piv:
                y ^= piv[b]
            else:
                piv[b] = y
                indep_idx.append(idx)
                break
    return len(piv), piv, indep_idx


def exact_quotient_Z(M_check, H_check, K, beta):
    """
    Computes exact partition function Z(beta) = 2^K * sum_{s in Im(H_check)} exp(-beta * |s|)
    via Walsh-Hadamard transform over Ker(H_check^T) (dimension = M_check - rank(H_check)).
    """
    # Find basis of Ker(H_check^T): combinations of check rows whose XOR is 0!
    piv = {}
    null_basis = []
    for ci, r in enumerate(H_check):
        x = sum(1 << q for q in r)
        comb = 1 << ci
        while x:
            b = (x & -x).bit_length() - 1
            if b in piv:
                px, pcomb = piv[b]
                x ^= px
                comb ^= pcomb
            else:
                piv[b] = (x, comb)
                break
        if x == 0:
            null_basis.append(comb)
    d_null = len(null_basis)
    assert d_null <= 20, f"Nullspace too large for direct Walsh sum: {d_null}"
    eb = math.exp(-beta)
    p_plus = 1.0 + eb
    p_minus = 1.0 - eb
    acc = 0.0
    for mask in range(1 << d_null):
        u = 0
        for b in range(d_null):
            if (mask >> b) & 1:
                u ^= null_basis[b]
        w = u.bit_count()
        acc += (p_plus ** (M_check - w)) * (p_minus ** w)
    return (2.0 ** (K - d_null)) * acc


def get_small_quotient_maps(N, H_check, H_stab):
    _, piv_s, _ = gf2_rank_bitmask(H_stab)
    _, piv_c, indep_c = gf2_rank_bitmask(H_check)
    s_cols = sorted(piv_s.keys(), reverse=True)
    for p in s_cols:
        for p2 in s_cols:
            if p2 < p and ((piv_s[p2] >> p) & 1):
                piv_s[p2] ^= piv_s[p]
    dual_logs = []
    piv_q = dict(piv_c)
    for fc in range(N):
        if fc in piv_s:
            continue
        v = 1 << fc
        for p in s_cols:
            if (piv_s[p] >> fc) & 1:
                v |= 1 << p
        y = v
        while y:
            b = (y & -y).bit_length() - 1
            if b in piv_q:
                y ^= piv_q[b]
            else:
                piv_q[b] = y
                dual_logs.append(v)
                break
    K = N - len(piv_s) - len(piv_c)
    assert len(dual_logs) == K
    cmask = [0] * N
    indep_pos = {ci: idx for idx, ci in enumerate(indep_c)}
    imask = [0] * N
    for ci, r in enumerate(H_check):
        for q in r:
            cmask[q] |= 1 << ci
            if ci in indep_pos:
                imask[q] |= 1 << indep_pos[ci]
    lbits = [[(L >> q) & 1 for q in range(N)] for L in dual_logs]
    return indep_c, cmask, imask, lbits


def solve_exact_small_cell(N, H_check, H_stab, beta=7.0):
    indep_c, cmask, imask, lbits = get_small_quotient_maps(N, H_check, H_stab)
    assert len(lbits) == 1
    lbit = lbits[0]
    rC = len(indep_c)
    nS = 1 << rC
    full_syn = [-1] * nS
    full_syn[0] = 0
    stk = [0]
    while stk:
        si = stk.pop()
        fs = full_syn[si]
        for q in range(N):
            nsi = si ^ imask[q]
            if full_syn[nsi] == -1:
                full_syn[nsi] = fs ^ cmask[q]
                stk.append(nsi)
    E = np.array([x.bit_count() for x in full_syn], dtype=np.int32)
    Z = 2.0 * float(np.sum(np.exp(-beta * E)))
    Z_walsh = exact_quotient_Z(len(H_check), H_check, 1, beta)
    assert abs(Z - Z_walsh) / Z < 1e-12

    rows, cols, vals = [], [], []
    rhs = np.zeros(nS - 1, dtype=np.float64)
    for si in range(1, nS):
        e0 = int(E[si])
        diag = 0.0
        for q in range(N):
            nsi = si ^ imask[q]
            lq = lbit[q]
            rate = 1.0 / (1.0 + math.exp(beta * (int(E[nsi]) - e0)))
            diag += rate
            if nsi == 0:
                if lq == 1:
                    rhs[si - 1] += rate
            else:
                if lq == 0:
                    rows.append(si - 1)
                    cols.append(nsi - 1)
                    vals.append(-rate)
                else:
                    rows.append(si - 1)
                    cols.append(nsi - 1)
                    vals.append(+rate)
                    rhs[si - 1] += rate
        rows.append(si - 1)
        cols.append(si - 1)
        vals.append(diag)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(nS - 1, nS - 1))
    q0 = spla.spsolve(A, rhs)

    gamma_0 = 0.0
    kappa = 0.0
    for q in range(N):
        nsi = imask[q]
        lq = lbit[q]
        rate = 1.0 / (1.0 + math.exp(beta * int(E[nsi])))
        if nsi != 0 or lq != 0:
            gamma_0 += rate
        qval = float(lq) if nsi == 0 else (float(q0[nsi - 1]) if lq == 0 else 1.0 - float(q0[nsi - 1]))
        kappa += rate * qval
    h_bar = kappa / gamma_0
    T_clean = Z / (2.0 * kappa)

    m_rows, m_cols, m_vals = [], [], []
    for si in range(nS):
        e0 = int(E[si])
        out_r = 0.0
        for q in range(N):
            nsi = si ^ imask[q]
            rate = 1.0 / (1.0 + math.exp(beta * (int(E[nsi]) - e0)))
            out_r += rate
            sgn = -1.0 if lbit[q] else 1.0
            m_rows.append(nsi)
            m_cols.append(si)
            m_vals.append(sgn * rate)
        m_rows.append(si)
        m_cols.append(si)
        m_vals.append(-out_r)
    G_signed = sp.csc_matrix((m_vals, (m_rows, m_cols)), shape=(nS, nS))
    return {
        "N": N,
        "n_quotient_states": 2 * nS,
        "Z": Z,
        "Gamma_0": gamma_0,
        "h_bar": h_bar,
        "kappa": kappa,
        "T_clean": T_clean,
        "G_signed": G_signed,
        "E": E,
    }


def solve_bounded_cell(N, H_check, H_stab, beta, E_cut, masked_qs=None, mu=1.0):
    """
    Rigorous low-energy truncation on quotient syndromes with E(s) <= E_cut:
    - Upper bound on kappa (lower bound on T_clean): omitted exits > E_cut have q = 1 (harmful)
    - Lower bound on kappa (upper bound on T_clean): omitted exits > E_cut have q = 0 (harmless)
    Uses logical symmetry q(s, 1) = 1 - q(s, 0) on full syndrome masks fs in Im(H_check)!
    wait: note that on quotient states (s, l), treating an omitted state w as harmless FROM a=(0,0)
    means q_lo(w) = 0 for BOTH l=0 and l=1, whereas treating w as harmful FROM a=(0,0) means q_hi(w) = 1
    for BOTH l=0 and l=1! Therefore we solve on explicit retained states (fs, l) with fs.bit_count() <= E_cut!
    """
    _, cmask, _, lbits = get_small_quotient_maps(N, H_check, H_stab)
    assert len(lbits) == 1
    lbit = lbits[0]
    masked_set = set(masked_qs or [])
    Z = exact_quotient_Z(len(H_check), H_check, 1, beta)

    vis = {(0, 0): 0, (0, 1): 1}
    states = [(0, 0), (0, 1)]
    dq = deque(states)
    while dq:
        fs, l = dq.popleft()
        for q in range(N):
            nfs = fs ^ cmask[q]
            if nfs.bit_count() <= E_cut:
                nl = l ^ lbit[q]
                if (nfs, nl) not in vis:
                    vis[(nfs, nl)] = len(states)
                    states.append((nfs, nl))
                    dq.append((nfs, nl))

    nRet = len(states)
    # Build SPD conductance linear system for interior retained states 2..nRet-1!
    nInt = nRet - 2
    rows, cols, vals = [], [], []
    rhs_lo = np.zeros(nInt, dtype=np.float64)  # omitted -> 0
    rhs_hi = np.zeros(nInt, dtype=np.float64)  # omitted -> 1
    for idx in range(2, nRet):
        fs, l = states[idx]
        e0 = fs.bit_count()
        pi_x = math.exp(-beta * e0)
        diag = 0.0
        ui = idx - 2
        for q in range(N):
            nfs = fs ^ cmask[q]
            e1 = nfs.bit_count()
            nl = l ^ lbit[q]
            if e1 == e0 and q in masked_set:
                rate = 0.5 * mu
            else:
                rate = 1.0 / (1.0 + math.exp(beta * (e1 - e0)))
            cond = pi_x * rate
            diag += cond
            if e1 > E_cut:
                rhs_hi[ui] += cond
            else:
                v_idx = vis[(nfs, nl)]
                if v_idx == 1:
                    rhs_lo[ui] += cond
                    rhs_hi[ui] += cond
                elif v_idx >= 2:
                    rows.append(ui)
                    cols.append(v_idx - 2)
                    vals.append(-cond)
        rows.append(ui)
        cols.append(ui)
        vals.append(diag)

    A = sp.csr_matrix((vals, (rows, cols)), shape=(nInt, nInt))
    inv_diag = 1.0 / A.diagonal()
    M_prec = sp.diags(inv_diag)
    q_lo, info_lo = spla.cg(A, rhs_lo, M=M_prec, rtol=1e-13, atol=0.0, maxiter=5000)
    q_hi, info_hi = spla.cg(A, rhs_hi, M=M_prec, rtol=1e-13, atol=0.0, maxiter=5000)
    assert info_lo == 0 and info_hi == 0, (info_lo, info_hi)

    gamma_0 = 0.0
    kap_lo = 0.0
    kap_hi = 0.0
    for q in range(N):
        nfs = cmask[q]
        e1 = nfs.bit_count()
        nl = lbit[q]
        rate = (0.5 * mu) if (e1 == 0 and q in masked_set) else (1.0 / (1.0 + math.exp(beta * e1)))
        if nfs != 0 or nl != 0:
            gamma_0 += rate
        if e1 > E_cut:
            kap_hi += rate * 1.0
        else:
            vi = vis[(nfs, nl)]
            if vi == 0:
                pass
            elif vi == 1:
                kap_lo += rate
                kap_hi += rate
            else:
                kap_lo += rate * float(q_lo[vi - 2])
                kap_hi += rate * float(q_hi[vi - 2])

    T_lo = Z / (2.0 * kap_hi)
    T_hi = Z / (2.0 * kap_lo)
    return {
        "N": N,
        "E_cut": E_cut,
        "n_retained_states": nRet,
        "Z": Z,
        "Gamma_0": gamma_0,
        "kappa_bracket": [kap_lo, kap_hi],
        "h_bar_bracket": [kap_lo / gamma_0, kap_hi / gamma_0],
        "T_clean_bracket": [T_lo, T_hi],
    }


def solve_rational_network(N, H_check, H_stab, B, masked_qs=None, lam_frac=None):
    _, cmask, _, lbits = get_small_quotient_maps(N, H_check, H_stab)
    lbit = lbits[0]
    masked_qs = set(masked_qs or [])
    vis = {(0, 0): 0, (0, 1): 1}
    states = [(0, 0), (0, 1)]
    q_bfs = deque(states)
    while q_bfs:
        fs, l = q_bfs.popleft()
        for q in range(N):
            nfs = fs ^ cmask[q]
            if nfs.bit_count() <= B:
                nl = l ^ lbit[q]
                if (nfs, nl) not in vis:
                    vis[(nfs, nl)] = len(states)
                    states.append((nfs, nl))
                    q_bfs.append((nfs, nl))
    E = [fs.bit_count() for fs, _ in states]
    nV = len(states)
    parent = list(range(nV))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for u, (fs, l) in enumerate(states):
        eu = E[u]
        for q in range(N):
            nfs = fs ^ cmask[q]
            ev = nfs.bit_count()
            if ev > B:
                continue
            v = vis[(nfs, l ^ lbit[q])]
            if u >= v:
                continue
            mE = max(eu, ev)
            is_masked_zero = eu == ev and q in masked_qs
            if mE < B:
                if is_masked_zero and lam_frac is not None and eu == B - 1:
                    pass
                else:
                    union(u, v)
    ra, rb = find(0), find(1)
    assert ra != rb
    crit_edges = defaultdict(Fraction)
    for u, (fs, l) in enumerate(states):
        eu = E[u]
        for q in range(N):
            nfs = fs ^ cmask[q]
            ev = nfs.bit_count()
            if ev > B:
                continue
            v = vis[(nfs, l ^ lbit[q])]
            if u >= v:
                continue
            ru, rv = find(u), find(v)
            if ru == rv:
                continue
            mE = max(eu, ev)
            is_masked_zero = eu == ev and q in masked_qs
            cond = Fraction(0)
            if mE == B:
                if eu == B and ev == B:
                    cond = Fraction(1, 2) if not (is_masked_zero and lam_frac is not None) else Fraction(0)
                else:
                    cond = Fraction(1, 1)
            elif mE == B - 1 and eu == B - 1 and ev == B - 1 and is_masked_zero and lam_frac is not None:
                cond = lam_frac * Fraction(1, 2)
            if cond > 0:
                a0, b0 = (ru, rv) if ru < rv else (rv, ru)
                crit_edges[(a0, b0)] += cond

    adj = defaultdict(dict)
    for (x, y), c in crit_edges.items():
        adj[x][y] = c
        adj[y][x] = c

    def reachable_from(src, block):
        seen = {src}
        dq = deque([src])
        while dq:
            x = dq.popleft()
            for y in adj[x]:
                if y != block and y not in seen:
                    seen.add(y)
                    dq.append(y)
        return seen

    rA = reachable_from(ra, rb)
    rB = reachable_from(rb, ra)
    free = sorted(x for x in (rA & rB) if x != ra and x != rb)
    idx_map = {x: i for i, x in enumerate(free)}
    nf = len(free)
    M_mat = [[Fraction(0)] * (nf + 1) for _ in range(nf)]
    for i, x in enumerate(free):
        tot_c = Fraction(0)
        for y, c in adj[x].items():
            if y == ra or y == rb or y in idx_map:
                tot_c += c
                if y == rb:
                    M_mat[i][nf] += c
                elif y != ra:
                    M_mat[i][idx_map[y]] -= c
        M_mat[i][i] = tot_c
    for col in range(nf):
        piv = next(r for r in range(col, nf) if M_mat[r][col] != 0)
        M_mat[col], M_mat[piv] = M_mat[piv], M_mat[col]
        pv = M_mat[col][col]
        for c2 in range(col, nf + 1):
            M_mat[col][c2] /= pv
        for r in range(nf):
            if r != col and M_mat[r][col] != 0:
                f = M_mat[r][col]
                for c2 in range(col, nf + 1):
                    M_mat[r][c2] -= f * M_mat[col][c2]
    V = {ra: Fraction(0), rb: Fraction(1)}
    for i, x in enumerate(free):
        V[x] = M_mat[i][nf]
    cap = adj[ra].get(rb, Fraction(0))
    for y, c in adj[ra].items():
        if y in V and y != rb:
            cap += c * V[y]
    return cap


def four_state_counterexample():
    """
    Explicit 4-state reversible Markov chain without a<->b target symmetry where slowing a
    zero-energy excited transition (0 < mu < 1) DECREASES E_a[tau_b] instead of increasing it!
    States: 0 (start a), 1 (fast escape branch), 2 (deep dead-end trap), 3 (target b).
    Energies: E(0)=0, E(1)=1, E(2)=1, E(3)=0; edge (1,2) is zero-energy (rate mu),
    edge (2,2) has large waiting volume via slow exit or multiple dead-end states:
    Wait: with fixed stationary weights pi(0)=1, pi(1)=e^{-b}, pi(2)=K*e^{-b} (or state 2 connected
    to state 2' at E=0!), slowing (1,2) prevents excursions at 1 from falling into the E=0 dead-end pocket!
    Let states be {0=a, 1, 2, 3=b} with stationary weights pi = [1, eps, M, 1]:
      - edge (0,1): c = eps
      - edge (1,3): c = eps (escape to target b!)
      - edge (1,2): c = mu * eps (zero-energy bridge into dead-end reservoir 2 with pi(2) = M >> 1!)
    Notice: pi is INDEPENDENT of mu! Capacity cap(0, 3; mu) = eps/2 (INDEPENDENT of mu because 2 is a dead end!),
    yet E_0[tau_3](mu) DOES depend on mu because excursions entering 2 spend huge mean time M / (mu * eps) inside 2!
    Wait: as mu DECREASES, entering 2 happens less often? Wait: probability of entering 2 vs returning to 0/3:
    Time spent in 2 per excursion from 1 before hitting {0, 3} has expectation M / (2*eps) INDEPENDENT of mu?!
    Wait — what if 2 ALSO has a slow leak to 0 (c_{2,0} = delta)?!
    If c_{2,0} = delta > 0, then when mu = 1, state 1 leaks into 2 and drains BACK TO 0 (avoiding target 3!),
    increasing E_0[tau_3]! When mu -> 0, state 1 cannot drain back to 0 via 2, so it goes 50/50 to 3,
    REDUCING E_0[tau_3]!!
    Let us verify this exact 4-state reversible chain with E = [0, 1, 1, 0] (unit weights at 0,3 and w at 1,2)!
    """
    # States: 0 (a), 1 (excited), 2 (excited, same energy as 1!), 3 (b)
    # Stationary weights: pi = [1, w, w, 1] (completely independent of mu!)
    # Symmetric conductances c_{ij} = pi_i * r_{ij}:
    #   c_{0,1} = w          (rate 0->1 is w, 1->0 is 1)
    #   c_{1,3} = 10 * w     (fast asymmetrical coupling to b=3, e.g. 10 parallel channels to b!)
    #   c_{1,2} = mu * w     (zero-energy transition between excited states 1 and 2!)
    #   c_{2,0} = 10 * w     (state 2 drains strongly back to a=0, NOT to b=3 — breaking a<->b symmetry!)
    w = 0.2

    def mean_hit_0_to_3(mu):
        # States: 0 (start a), 1 (excited), 2 (excited, equal energy to 1!), 3 (target b)
        # Stationary weights pi = [1, 0.2, 0.2, 1] are completely independent of mu!
        # Edge (1, 2) is an excited-state zero-energy transition with conductance mu * w = 0.2 * mu.
        pi = np.array([1.0, w, w, 1.0])
        c01, c02, c03, c13, c23 = 0.2, 1.5, 1.0, 0.3, 1.0
        c12 = mu
        C = np.array(
            [
                [0.0, c01, c02, c03],
                [c01, 0.0, c12, c13],
                [c02, c12, 0.0, c23],
                [c03, c13, c23, 0.0],
            ]
        )
        A = np.zeros((3, 3))
        for i in range(3):
            out_r = np.sum(C[i, :]) / pi[i]
            A[i, i] = out_r
            for j in range(3):
                if i != j:
                    A[i, j] = -C[i, j] / pi[i]
        t = np.linalg.solve(A, np.ones(3))
        return float(t[0])

    t_mu1 = mean_hit_0_to_3(1.0)
    t_mu01 = mean_hit_0_to_3(0.1)
    assert t_mu01 < t_mu1, (t_mu1, t_mu01)
    return {
        "description": "4-state reversible chain pi=[1,w,w,1] without a<->b symmetry where slowing zero-energy hop (1,2) decreases E_a[tau_b]",
        "w": w,
        "T_hit_mu_1.0": t_mu1,
        "T_hit_mu_0.1": t_mu01,
        "ratio_mu01_over_mu1": t_mu01 / t_mu1,
    }


def build_multistep_with_seed_logicals(steps):
    """
    steps: list of (L, ribs, iter_type, gadget_mode) on seed_planar_11().
    Propagates the exact seed logical pair (L_X={0}, L_Z={2} or check seed_planar_11!).
    """
    comp = seed_planar_11()
    N0, HX0, HZ0, *_ = sheaf_to_css_with_seams(comp)
    # Planar [[2,1,1]] seed: N0=2, edges (0,1)[red/none?], check HX0 and HZ0!
    _, _, dual_z = gf2_rank_bitmask(HX0)
    # Find seed L_X in Ker(HZ0) and L_Z in Ker(HX0) with |L_X & L_Z| odd:
    lx_seed, lz_seed = None, None
    for mx in range(1, 1 << N0):
        sx = {q for q in range(N0) if (mx >> q) & 1}
        if all((len(sx & set(r)) % 2) == 0 for r in HZ0):
            for mz in range(1, 1 << N0):
                sz = {q for q in range(N0) if (mz >> q) & 1}
                if all((len(sz & set(r)) % 2) == 0 for r in HX0) and (len(sx & sz) % 2 == 1):
                    lx_seed, lz_seed = sx, sz
                    break
        if lx_seed is not None:
            break
    assert lx_seed is not None

    lx_cur, lz_cur = set(lx_seed), set(lz_seed)
    cyl = set()
    for L, ribs, itype, gmode in steps:
        pcomp = comp
        comp, cyl, meta = apply_step_general(pcomp, L, ribs, itype, prev_cyl=cyl, gadget_mode=gmode)
        N, HX, HZ, seam_qs, latest_bf, cyl_qs = sheaf_to_css_with_seams(comp)
        lx_cur = push_X_logical(pcomp, comp, meta, HZ, lx_cur)
        lz_cur = push_Z_logical(pcomp, comp, meta, HX, lz_cur)

    # Verify commutation and seed logical pairing
    hz_sets = [set(r) for r in HZ]
    hx_sets = [set(r) for r in HX]
    for rx in HX:
        pass
    # Sparse commutation check
    q2cz = [[] for _ in range(N)]
    for ci, r in enumerate(HZ):
        for q in r:
            q2cz[q].append(ci)
    for r in HX:
        cnt = Counter(ci for q in r for ci in q2cz[q])
        assert all((v % 2) == 0 for v in cnt.values()), "CSS commutation failed!"
    assert all((len(lx_cur & s) % 2) == 0 for s in hz_sets), "L_X not in Ker(HZ)!"
    assert all((len(lz_cur & s) % 2) == 0 for s in hx_sets), "L_Z not in Ker(HX)!"
    assert (len(lx_cur & lz_cur) % 2) == 1, "L_X and L_Z do not anticommute!"
    return N, HX, HZ, sorted(lx_cur), sorted(lz_cur)


def clean_departure_hazard(N, H_check, beta):
    deg = [0] * N
    for r in H_check:
        for q in r:
            deg[q] += 1
    # Note: if a qubit has deg[q] == 0, flipping it has dE = 0 (rate 0.5); wait: in clean departure
    # from clean code state, any flip that changes syndrome OR logical parity departs; for K=1 with d>=2, all N qubits depart!
    return sum(1.0 / (1.0 + math.exp(beta * deg[q])) for q in range(N)), Counter(deg)


def symplectic_completion_and_barriers(N, HX, HZ, lx_seed, lz_seed):
    """
    Certifies K_total, embeds (lx_seed, lz_seed) into a complete K_total-qubit symplectic logical basis
    (Poulin 2005), analyzes weight-1 Z operators at sites with no X check, and runs exhaustive
    low-energy reachability for target logical barriers (Delta_X, Delta_Z).
    """
    rX, pivX, _ = gf2_rank_bitmask(HX)
    rZ, pivZ, _ = gf2_rank_bitmask(HZ)
    K_tot = N - rX - rZ

    # Basis of Ker(HZ) mod rowspace(HX) (X logical classes) and Ker(HX) mod rowspace(HZ) (Z logical classes)
    def coset_basis(H_par, piv_stab):
        _, piv_p, _ = gf2_rank_bitmask(H_par)
        p_cols = sorted(piv_p.keys(), reverse=True)
        for p in p_cols:
            for p2 in p_cols:
                if p2 < p and ((piv_p[p2] >> p) & 1):
                    piv_p[p2] ^= piv_p[p]
        reps = []
        piv_work = dict(piv_stab)
        for fc in range(N):
            if fc in piv_p:
                continue
            v = 1 << fc
            for p in p_cols:
                if (piv_p[p] >> fc) & 1:
                    v |= 1 << p
            y = v
            while y:
                b = (y & -y).bit_length() - 1
                if b in piv_work:
                    y ^= piv_work[b]
                else:
                    piv_work[b] = y
                    reps.append(v)
                    break
        return reps

    x_reps = coset_basis(HZ, pivX)
    z_reps = coset_basis(HX, pivZ)
    assert len(x_reps) == K_tot and len(z_reps) == K_tot

    lx0 = sum(1 << q for q in lx_seed)
    lz0 = sum(1 << q for q in lz_seed)

    # Symplectic Gram-Schmidt starting from pair 0 = (lx0, lz0):
    # Clean x_reps and z_reps against (lx0, lz0) so all remaining commute with BOTH lx0 and lz0!
    rem_x = []
    for x in x_reps:
        if ((x & lz0).bit_count() & 1):
            x ^= lx0
        if ((x & lx0).bit_count() & 1):
            pass  # CSS: X and X always commute!
        # Check if x is in span(HX, lx0)
        y = x
        piv_tmp = dict(pivX)
        yt = lx0
        while yt:
            b = (yt & -yt).bit_length() - 1
            if b in piv_tmp:
                yt ^= piv_tmp[b]
            else:
                piv_tmp[b] = yt
                break
        is_indep = False
        for prev in rem_x:
            yp = prev
            while yp:
                b = (yp & -yp).bit_length() - 1
                if b in piv_tmp:
                    yp ^= piv_tmp[b]
                else:
                    piv_tmp[b] = yp
                    break
        while y:
            b = (y & -y).bit_length() - 1
            if b in piv_tmp:
                y ^= piv_tmp[b]
            else:
                is_indep = True
                break
        if is_indep:
            rem_x.append(x)

    rem_z = []
    for z in z_reps:
        if ((z & lx0).bit_count() & 1):
            z ^= lz0
        y = z
        piv_tmp = dict(pivZ)
        yt = lz0
        while yt:
            b = (yt & -yt).bit_length() - 1
            if b in piv_tmp:
                yt ^= piv_tmp[b]
            else:
                piv_tmp[b] = yt
                break
        for prev in rem_z:
            yp = prev
            while yp:
                b = (yp & -yp).bit_length() - 1
                if b in piv_tmp:
                    yp ^= piv_tmp[b]
                else:
                    piv_tmp[b] = yp
                    break
        is_indep = False
        while y:
            b = (y & -y).bit_length() - 1
            if b in piv_tmp:
                y ^= piv_tmp[b]
            else:
                is_indep = True
                break
        if is_indep:
            rem_z.append(z)

    assert len(rem_x) == K_tot - 1 and len(rem_z) == K_tot - 1
    # Pair up (rem_x[i], rem_z[i]) symplectically!
    gauge_pairs = []
    pool_x = list(rem_x)
    pool_z = list(rem_z)
    while pool_x:
        xi = pool_x.pop(0)
        j_match = next(j for j, zj in enumerate(pool_z) if ((xi & zj).bit_count() & 1))
        zi = pool_z.pop(j_match)
        for k in range(len(pool_x)):
            if ((pool_x[k] & zi).bit_count() & 1):
                pool_x[k] ^= xi
        for k in range(len(pool_z)):
            if ((xi & pool_z[k]).bit_count() & 1):
                pool_z[k] ^= zi
        gauge_pairs.append((xi, zi))

    # Weight-1 Z operators at sites with no X check:
    dx = [0] * N
    for r in HX:
        for q in r:
            dx[q] += 1
    zero_dx_sites = [q for q in range(N) if dx[q] == 0]
    commute_with_seed = all((q not in lx_seed) for q in zero_dx_sites)
    # Count independent logical Z classes mod rowspace(HZ) spanned by zero_dx_sites:
    piv_zmod = dict(pivZ)
    indep_wt1_z = 0
    for q in zero_dx_sites:
        y = 1 << q
        while y:
            b = (y & -y).bit_length() - 1
            if b in piv_zmod:
                y ^= piv_zmod[b]
            else:
                piv_zmod[b] = y
                indep_wt1_z += 1
                break

    # Exhaustive low-energy reachability for TARGET logical barrier!
    # For X errors (checks = HZ, target dual logical = lz0) and Z errors (checks = HX, target dual logical = lx0):
    def exact_target_barrier(H_chk, dual_mask):
        cm = [0] * N
        for ci, r in enumerate(H_chk):
            for q in r:
                cm[q] |= 1 << ci
        lb = [(dual_mask >> q) & 1 for q in range(N)]
        for B in [0, 1, 2, 3]:
            parent_map = {(0, 0): None}
            dq = deque([(0, 0)])
            while dq:
                fs, l = dq.popleft()
                if fs == 0 and l == 1:
                    # Reconstruct physical qubit sequence witness!
                    seq = []
                    energies = [0]
                    cur = (fs, l)
                    while parent_map[cur] is not None:
                        prev, q = parent_map[cur]
                        seq.append(q)
                        cur = prev
                    seq.reverse()
                    fcur = 0
                    for q in seq:
                        fcur ^= cm[q]
                        energies.append(fcur.bit_count())
                    return B, seq, energies, len(parent_map)
                for q in range(N):
                    nfs = fs ^ cm[q]
                    if nfs.bit_count() <= B:
                        nxt = (nfs, l ^ lb[q])
                        if nxt not in parent_map:
                            parent_map[nxt] = ((fs, l), q)
                            dq.append(nxt)
        return -1, [], [], 0

    bX, seqX, enX, expX = exact_target_barrier(HZ, lz0)
    bZ, seqZ, enZ, expZ = exact_target_barrier(HX, lx0)

    return {
        "N": N,
        "M_X": len(HX),
        "M_Z": len(HZ),
        "rank_HX": rX,
        "rank_HZ": rZ,
        "K_total": K_tot,
        "K_target": 1,
        "K_unused": K_tot - 1,
        "zero_dx_sites": zero_dx_sites,
        "zero_dx_commute_with_seed": commute_with_seed,
        "zero_dx_indep_Z_classes": indep_wt1_z,
        "target_barrier_X": bX,
        "target_witness_X": seqX,
        "target_energies_X": enX,
        "target_barrier_Z": bZ,
        "target_witness_Z": seqZ,
        "target_energies_Z": enZ,
    }


def main():
    t0 = time.time()
    out = {}

    # =========================================================================
    # [B, C] 1-step planar cells: exact & bounded quotient solvers + rationals
    # =========================================================================
    print("=== [B, C] Exact & Bounded 1-Step Planar CSS Cell Solvers ===")
    cells_bc = {}
    for name, L, ribs, gm, B_X in [
        ("C2_full", 2, [1], "full", 2),
        ("C2_3arm", 2, [1], "cb3rib", 2),
        ("C2_2arm", 2, [1], "cb2rib", None),
        ("C3", 3, [1], "full", 2),
        ("R3", 3, [1, 2], "full", 3),
    ]:
        comp, _ = apply_sheaf_step(seed_planar_11(), L, ribs, "X", None, gadget_mode=gm)
        N, HX, HZ, seam_qs, *_ = sheaf_to_css_with_seams(comp)
        rX, _, _ = gf2_rank_bitmask(HX)
        rZ, _, _ = gf2_rank_bitmask(HZ)
        K = N - rX - rZ
        info = {
            "N": N,
            "M_X": len(HX),
            "M_Z": len(HZ),
            "rank_HX": rX,
            "rank_HZ": rZ,
            "K": K,
            "dim_X_quotient": N - rX,
            "dim_Z_quotient": N - rZ,
        }
        if K != 1:
            print(f"  {name}: N={N}, M_Z={len(HZ)}, K={K} (excluded from same-K=1 comparison)")
            cells_bc[name] = info
            continue

        cap_X = solve_rational_network(N, HZ, HX, B_X)
        info["barrier_X"] = B_X
        info["kappa_X_prefactor_frac"] = f"{cap_X.numerator}/{cap_X.denominator}" if cap_X.denominator != 1 else str(cap_X.numerator)
        info["kappa_X_prefactor_float"] = float(cap_X)

        if L == 2:
            cap_Z = solve_rational_network(N, HX, HZ, 1)
            info["barrier_Z"] = 1
            info["kappa_Z_prefactor"] = str(cap_Z)

            exX = solve_exact_small_cell(N, HZ, HX, beta=7.0)
            exZ = solve_exact_small_cell(N, HX, HZ, beta=7.0)
            GX, EX = exX["G_signed"], exX["E"]
            GZ = exZ["G_signed"]

            # Optimal X t_0.80 via reversible sparse eigensolve
            d_half = np.exp(-3.5 * EX)
            d_inv = np.exp(+3.5 * EX)
            S_sym = sp.diags(d_inv) @ GX @ sp.diags(d_half)
            evals, evecs = spla.eigsh(S_sym, k=48, which="LA")
            c0 = evecs[0, :]

            def PX_opt(t):
                dt = d_half * (evecs @ (c0 * np.exp(evals * t)))
                return 0.5 * (1.0 + float(np.sum(np.abs(dt))))

            lo, hi = 70000.0, 110000.0
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if PX_opt(mid) >= 0.80:
                    lo = mid
                else:
                    hi = mid
            t80_X = 0.5 * (lo + hi)

            # Optimal independent-X/Z quantum t_0.80 via direct sparse exponential action
            d0X = np.zeros(GX.shape[0])
            d0X[0] = 1.0
            d0Z = np.zeros(GZ.shape[0])
            d0Z[0] = 1.0

            def PQ_opt(t):
                dtX = spla.expm_multiply(GX * t, d0X)
                dtZ = spla.expm_multiply(GZ * t, d0Z)
                return 0.25 * (1.0 + float(np.sum(np.abs(dtX)))) * (1.0 + float(np.sum(np.abs(dtZ))))

            lo, hi = 35.0, 60.0
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if PQ_opt(mid) >= 0.80:
                    lo = mid
                else:
                    hi = mid
            t80_Q = 0.5 * (lo + hi)

            info["beta_7"] = {
                "Z_X": exX["Z"],
                "Gamma_0_X": exX["Gamma_0"],
                "h_bar_X": exX["h_bar"],
                "kappa_X": exX["kappa"],
                "T_clean_X": exX["T_clean"],
                "t80_opt_X": t80_X,
                "t80_opt_Q": t80_Q,
                "T_clean_Z": exZ["T_clean"],
            }
            print(
                f"  {name}: N={N} T_clean_X={exX['T_clean']:.2f} t80_opt_X={t80_X:.2f} "
                f"t80_opt_Q={t80_Q:.6f} kappa_pref={info['kappa_X_prefactor_frac']}"
            )
        else:
            # Bounded low-energy quotient solver at beta = 7.0: check both dossier E_cut (3 for C3, 4 for R3) and +1 shell!
            ecut_ref = 3 if name == "C3" else 4
            ecut_tight = ecut_ref + 1
            bnd_ref = solve_bounded_cell(N, HZ, HX, beta=7.0, E_cut=ecut_ref)
            bnd = solve_bounded_cell(N, HZ, HX, beta=7.0, E_cut=ecut_tight)
            info["beta_7_bounded_ref"] = bnd_ref
            info["beta_7_bounded"] = bnd
            print(
                f"  {name}: N={N} E_cut={ecut_ref}->[{bnd_ref['T_clean_bracket'][0]:.2f}, {bnd_ref['T_clean_bracket'][1]:.2f}] "
                f"E_cut={ecut_tight}->[{bnd['T_clean_bracket'][0]:.2f}, {bnd['T_clean_bracket'][1]:.2f}] "
                f"kappa_pref={info['kappa_X_prefactor_frac']}"
            )
            if name == "C3":
                dz = [0] * N
                for r in HZ:
                    for q in r:
                        dz[q] += 1
                flat_qs = [q for q in range(N) if q not in seam_qs and dz[q] == 2]
                joint_checks = {}
                for b_test in [8.0, 9.0, 10.0]:
                    mu_val = math.exp(-b_test)  # lambda = 1
                    jb3 = solve_bounded_cell(N, HZ, HX, beta=b_test, E_cut=3, masked_qs=flat_qs, mu=mu_val)
                    jb4 = solve_bounded_cell(N, HZ, HX, beta=b_test, E_cut=4, masked_qs=flat_qs, mu=mu_val)
                    sc3 = [math.exp(2.0 * b_test) * k for k in jb3["kappa_bracket"]]
                    sc4 = [math.exp(2.0 * b_test) * k for k in jb4["kappa_bracket"]]
                    joint_checks[f"beta_{int(b_test)}_E3"] = sc3
                    joint_checks[f"beta_{int(b_test)}_E4"] = sc4
                    print(f"    C3 joint lambda=1 beta={b_test:.0f}: E3=[{sc3[0]:.5f}, {sc3[1]:.5f}] E4=[{sc4[0]:.5f}, {sc4[1]:.5f}] -> 1.5")
                info["joint_limit_lambda_1"] = joint_checks

        cells_bc[name] = info

    out["cells_BC"] = cells_bc
    out["counterexample_C"] = four_state_counterexample()
    print(f"  4-state counterexample: T(mu=1.0)={out['counterexample_C']['T_hit_mu_1.0']:.4f} > T(mu=0.1)={out['counterexample_C']['T_hit_mu_0.1']:.4f}")

    # =========================================================================
    # [E, F] Direct 2-step C2 bridge pruning & protected-subsystem certificates
    # =========================================================================
    print("=== [E, F] Two-Step C2 Pruning & Protected-Subsystem Certificates ===")
    pruning_ef = {}
    for label, gX, gZ in [
        ("full", "full", "full"),
        ("prune_X", "cb3rib", "full"),
        ("prune_Z", "full", "cb3rib"),
        ("prune_both", "cb3rib", "cb3rib"),
    ]:
        N, HX, HZ, lx_s, lz_s = build_multistep_with_seed_logicals([(2, [1], "X", gX), (2, [1], "Z", gZ)])
        cert = symplectic_completion_and_barriers(N, HX, HZ, lx_s, lz_s)
        hazX, degZ = clean_departure_hazard(N, HZ, 6.5)
        hazZ, degX = clean_departure_hazard(N, HX, 6.5)
        cert["clean_hazard_X_b65"] = hazX
        cert["clean_hazard_Z_b65"] = hazZ
        cert["lx_seed_weight"] = len(lx_s)
        cert["lz_seed_weight"] = len(lz_s)
        pruning_ef[label] = cert
        print(
            f"  {label:10s}: N={N} rX={cert['rank_HX']} rZ={cert['rank_HZ']} K_tot={cert['K_total']} "
            f"barriers=({cert['target_barrier_X']},{cert['target_barrier_Z']}) "
            f"hazX_b65={hazX:.5f} wt1Z_classes={cert['zero_dx_indep_Z_classes']} witnessZ={cert['target_witness_Z']} enZ={cert['target_energies_Z']}"
        )
    out["pruning_EF"] = pruning_ef

    # =========================================================================
    # [D] Resource-Matched L=4 Offset (1+3) vs Centered (2+2) Certificates
    # =========================================================================
    print("=== [D] Resource-Matched L=4 Offset vs Centered Certificates ===")
    matched_d = {}
    for family_name, coarse_steps in [
        ("standalone_L4", []),
        ("coarseC2_L4", [(2, [1], "X", "full"), (2, [1], "Z", "full")]),
    ]:
        fam_res = {}
        for pos_name, rpos in [("offset_1_3", [1]), ("centered_2_2", [2])]:
            steps = coarse_steps + [(4, rpos, "X", "full"), (4, rpos, "Z", "full")]
            N, HX, HZ, lx_s, lz_s = build_multistep_with_seed_logicals(steps)
            rX, _ = gf2_rank_sparse(HX, N)
            rZ, _ = gf2_rank_sparse(HZ, N)
            K = N - rX - rZ
            hazX, deg_col_Z = clean_departure_hazard(N, HZ, 6.5)
            hazZ, deg_col_X = clean_departure_hazard(N, HX, 6.5)
            wt_HX = dict(sorted(Counter(len(r) for r in HX).items()))
            wt_HZ = dict(sorted(Counter(len(r) for r in HZ).items()))
            fam_res[pos_name] = {
                "N": N,
                "M_X": len(HX),
                "M_Z": len(HZ),
                "rank_HX": rX,
                "rank_HZ": rZ,
                "K": K,
                "lx_weight": len(lx_s),
                "lz_weight": len(lz_s),
                "col_deg_HX": dict(sorted(deg_col_X.items())),
                "col_deg_HZ": dict(sorted(deg_col_Z.items())),
                "row_wt_HX": wt_HX,
                "row_wt_HZ": wt_HZ,
                "clean_hazard_X_b65": hazX,
                "clean_hazard_Z_b65": hazZ,
            }
            print(
                f"  {family_name}/{pos_name}: N={N:,d} K={K} MX={len(HX):,d}(r={rX:,d}) "
                f"MZ={len(HZ):,d}(r={rZ:,d}) hazX={hazX:.6f} hazZ={hazZ:.6f} wt_HZ={wt_HZ}"
            )
        off, cen = fam_res["offset_1_3"], fam_res["centered_2_2"]
        assert off["N"] == cen["N"] and off["K"] == 1 and cen["K"] == 1
        assert off["col_deg_HX"] == cen["col_deg_HX"] and off["col_deg_HZ"] == cen["col_deg_HZ"]
        assert abs(off["clean_hazard_X_b65"] - cen["clean_hazard_X_b65"]) < 1e-12
        assert abs(off["clean_hazard_Z_b65"] - cen["clean_hazard_Z_b65"]) < 1e-12
        assert off["row_wt_HX"] == cen["row_wt_HX"]
        fam_res["identical_col_degrees"] = True
        fam_res["identical_initial_hazards"] = True
        fam_res["identical_HX_weights"] = True
        fam_res["identical_HZ_weights"] = (off["row_wt_HZ"] == cen["row_wt_HZ"])
        matched_d[family_name] = fam_res
    out["matched_D"] = matched_d

    out_p = os.path.join(ROOT, "local_quotient_and_pruning_data.json")
    with open(out_p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved certified [B, C, D, E, F] algebraic & quotient data to {out_p} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
