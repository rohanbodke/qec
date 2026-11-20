#!/usr/bin/env python3
"""
Zero-Trust Independent Verification & Analytical Markov Master-Equation Audit (`independent_audit.py`):
  1. Exact 64x64 Continuous-Time Markov Generator (-Q_SS * t = 1) analytical first-passage time
     on the 6-qubit RP^2 Hemi-Cube code (`bdl_rp2_L2r1_s0`, Xerr & Zerr) vs. C++ BKL-KMC output!
  2. Independent GF(2) CSS commutativity, rank, logical symplectic pairing, and check-weight audit
     across all V3 3D-Local & Non-Local constructors.
"""

import math
import os
import sys

WS = "/Users/harshilavlani/Downloads/6.S980_gemini"
sys.path.insert(0, os.path.join(WS, "cluster_v3"))
from build_v3_and_tasks import (
    seed_torus, seed_rp2_hemicube, seed_hyperbolic_k2222,
    apply_sheaf_aniso_sparse, sheaf_to_css_with_jgrad,
    build_bb_code, build_toric4d_22
)


def solve_linear_real(A, b):
    """Exact dense Gaussian elimination with partial pivoting over R."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        div = M[col][col]
        for j in range(col, n + 1):
            M[col][j] /= div
        for r in range(n):
            if r != col and M[r][col] != 0.0:
                fac = M[r][col]
                for j in range(col, n + 1):
                    M[r][j] -= fac * M[col][j]
    return [M[i][n] for i in range(n)]


def exact_markov_mfpt(N, checks, stabs, log_op, beta):
    """
    Build exact 2^N state space, decode every syndrome by exact minimum-weight coset leader (ML/UF),
    partition into Surviving states S vs Absorbing logically-failed states F, and solve -Q_SS t = 1!
    Wait: in continuous-time KMC with logarithmic/periodic ideal readout, failure occurs upon entering
    any state e whose minimum-weight syndrome coset leader differs from e by the nontrivial logical!
    """
    n_states = 1 << N
    c_masks = [sum(1 << q for q in r) for r in checks]
    l_mask = sum(1 << q for q in log_op)

    def syn_of(e):
        s = 0
        for i, cm in enumerate(c_masks):
            if bin(e & cm).count("1") & 1:
                s |= (1 << i)
        return s

    # Group states by syndrome -> pick minimum-weight decoder correction for each syndrome!
    syn_min_corr = {}
    for e in range(n_states):
        s = syn_of(e)
        w = bin(e).count("1")
        if s not in syn_min_corr or w < syn_min_corr[s][0]:
            syn_min_corr[s] = (w, e)

    surv = []
    surv_idx = {}
    for e in range(n_states):
        corr = syn_min_corr[syn_of(e)][1]
        failed = (bin((e ^ corr) & l_mask).count("1") & 1) == 1
        if not failed:
            surv_idx[e] = len(surv)
            surv.append(e)

    ns = len(surv)
    A = [[0.0] * ns for _ in range(ns)]
    for i, e in enumerate(surv):
        s_cur = syn_of(e)
        w_cur = bin(s_cur).count("1")
        diag = 0.0
        for q in range(N):
            e2 = e ^ (1 << q)
            w_nxt = bin(syn_of(e2)).count("1")
            dE = w_nxt - w_cur
            rate = 1.0 / (1.0 + math.exp(beta * dE))
            diag += rate
            if e2 in surv_idx:
                A[i][surv_idx[e2]] -= rate
        A[i][i] = diag

    t_vec = solve_linear_real(A, [1.0] * ns)
    return t_vec[surv_idx[0]]


def gf2_rank(rows, n):
    piv = {}
    for r in rows:
        x = sum(1 << c for c in r)
        while x:
            lsb = (x & -x).bit_length() - 1
            if lsb in piv: x ^= piv[lsb]
            else: piv[lsb] = x; break
    return len(piv)


if __name__ == "__main__":
    print("=== AUDIT 1: Exact 64x64 Markov Master-Equation MFPT vs C++ KMC Engine ===")
    s0 = seed_rp2_hemicube()
    nq, hx, hz, _, _ = sheaf_to_css_with_jgrad(s0)
    # Find logicals of RP2 s0 (N=6):
    # Xerr: checks=hz (3 rows), stabs=hx (4 rows)
    # Zerr: checks=hx (4 rows), stabs=hz (3 rows)
    # Check non-stabilizer kernel element:
    def find_log(checks, stabs, n):
        for m in range(1, 1 << n):
            if all(bin(m & sum(1<<q for q in c)).count("1")%2 == 0 for c in checks):
                if gf2_rank(stabs + [[q for q in range(n) if (m>>q)&1]], n) > gf2_rank(stabs, n):
                    return [q for q in range(n) if (m>>q)&1]
        return []

    lx = find_log(hz, hx, nq)
    lz = find_log(hx, hz, nq)
    # Empirical KMC numbers from parsed_v2_summary.txt (lines 68-69, 100 trials, logarithmic readout 0.15*t):
    kmc_x = {3: 1.469e2, 4: 1.136e3, 5: 8.282e3, 6: 6.867e4, 7: 3.840e5}
    kmc_z = {3: 2.875e2, 4: 2.128e3, 5: 1.612e4, 6: 1.254e5, 7: 7.076e5}
    for sec, chks, stbs, lg, emp in [("RP2_s0_Xerr", hz, hx, lx, kmc_x), ("RP2_s0_Zerr", hx, hz, lz, kmc_z)]:
        print(f"  [{sec}]")
        for b in (3, 4, 5, 6, 7):
            t_ex = exact_markov_mfpt(nq, chks, stbs, lg, float(b))
            # Note: KMC checks readout every ~1.075x on average (+7.5% discrete readout lag vs continuous hitting)
            ratio = emp[b] / t_ex
            print(f"    beta={b}: Exact_Master_Eq = {t_ex:10.3e} | C++_KMC = {emp[b]:10.3e} | ratio = {ratio:.3f}")

    print("\n=== AUDIT 2: Zero-Trust Algebraic CSS Verification of All V3 Track-A & Track-B Codes ===")
    cases = []
    for lbl, seed_fn, sched in [
        ("TrackA_3D_Tor2_Sparse_974", seed_torus(2,2),       [(2,[1],"X",{0,1},{0,2}), (2,[1],"Z",None,None)]),
        ("TrackA_3D_Tor2_Aniso_1016", seed_torus(2,2),       [(2,[1],"X",None,None),   (2,[1],"Z",None,None)]),
        ("TrackB_NL_RP2_Aniso_762",   seed_rp2_hemicube(),   [(2,[1],"X",None,None),   (2,[1],"Z",None,None)]),
        ("TrackB_NL_HypK6_3048",      seed_hyperbolic_k2222(),[(2,[1],"X",None,None),  (2,[1],"Z",None,None)]),
    ]:
        c, cyl = seed_fn, set()
        for L, rb, it, hf, vf in sched:
            c, cyl = apply_sheaf_aniso_sparse(c, L, rb, it, cyl, hf, vf)
        nq_c, hx_c, hz_c, _, _ = sheaf_to_css_with_jgrad(c)
        cases.append((lbl, nq_c, hx_c, hz_c))
    cases.append(("TrackB_NL_IBM_Gross_144", *build_bb_code(12, 6, [(3,0),(0,1),(0,2)], [(0,3),(1,0),(2,0)])))
    cases.append(("TrackB_NL_IBM_2Gross_288",*build_bb_code(12,12, [(3,0),(0,2),(0,7)], [(0,3),(1,0),(2,0)])))
    cases.append(("TrackB_NL_4DToric_486",   *build_toric4d_22(3)))

    for lbl, n_c, hx_c, hz_c in cases:
        q2z = [set() for _ in range(n_c)]
        for zi, r in enumerate(hz_c):
            for q in r: q2z[q].add(zi)
        bad = 0
        for rx in hx_c:
            acc = set()
            for q in rx: acc ^= q2z[q]
            if acc: bad += 1
        rk_x = gf2_rank(hx_c, n_c)
        rk_z = gf2_rank(hz_c, n_c)
        k_c = n_c - rk_x - rk_z
        meta_x = len(hx_c) - rk_x
        meta_z = len(hz_c) - rk_z
        print(
            f"  {lbl:27s} | N={n_c:4d} K={k_c:2d} N/K={n_c/k_c:6.1f} | "
            f"bad_comm={bad} | meta=({meta_x:3d},{meta_z:3d}) | "
            f"max_w=({max(len(r) for r in hx_c)},{max(len(r) for r in hz_c)})"
        )
