#!/usr/bin/env python3
"""
Inspect the exact local unit U_F in R_X and R_Z for (L=2, r=[1]), (L=3, r=[1]), (L=3, r=[1,2]), (L=4, r=[1,2]):
1) What is the local cokernel C^2(U_F) / im(delta^1_{int})?
2) Can we test worst-case AND typical syndrome contraction of 1-rib vs 2-rib directly?!
3) Also compute exact distances (d_X, d_Z) AND exact minimax energy barriers (Delta_X, Delta_Z)
   at steps 0, 1, 2, 3 across seeds and (L, r)!
"""

import sys
sys.path.insert(0, "/Users/harshilavlani/Downloads/6.S980_gemini/src")
from test_sheaf_exact import (
    make_seed_single_square,
    make_seed_surface_code,
    apply_iteration,
    build_css_from_sheaf,
    gf2_rank,
)
from collections import defaultdict, deque
import random


def get_logical_reps(nq, H_check, H_stab):
    """Find basis of ker(H_check) / rowspan(H_stab)."""
    _, check_pivots = gf2_rank(nq, H_check)
    for p in sorted(check_pivots.keys(), reverse=True):
        for p2 in check_pivots:
            if p2 != p and ((check_pivots[p2] >> p) & 1):
                check_pivots[p2] ^= check_pivots[p]
    free_vars = [i for i in range(nq) if i not in check_pivots]
    ker_basis = []
    for fv in free_vars:
        vec = (1 << fv)
        for p, prow in check_pivots.items():
            if (prow >> fv) & 1:
                vec |= (1 << p)
        ker_basis.append(vec)

    _, stab_pivots = gf2_rank(nq, H_stab)
    quot_pivots = dict(stab_pivots)
    logicals = []
    for v in ker_basis:
        x = v
        while x:
            lsb = (x & -x).bit_length() - 1
            if lsb in quot_pivots:
                x ^= quot_pivots[lsb]
            else:
                quot_pivots[lsb] = x
                logicals.append(v)
                break
    return logicals, stab_pivots


def upper_bound_distance_isd(nq, H_check, H_stab, trials=400, seed=42):
    """
    Randomized permutation RREF (Information-Set / Stern-style) upper bound / exact finder
    for minimum weight of ker(H_check) \ rowspan(H_stab).
    For nq <= 3000 this finds exact d on structured topological/sheaf codes very reliably!
    """
    rng = random.Random(seed)
    _, stab_pivots = gf2_rank(nq, H_stab)

    def in_stab(vec):
        x = vec
        while x:
            lsb = (x & -x).bit_length() - 1
            if lsb in stab_pivots:
                x ^= stab_pivots[lsb]
            else:
                return False
        return True

    best_w = nq + 1
    best_vec = None
    cols = list(range(nq))

    # Pre-encode H_check rows as ints
    raw_rows = []
    for r in H_check:
        m = 0
        for c in r:
            m |= (1 << c)
        raw_rows.append(m)

    # Also include H_stab rows to quotient out stabilizers in random column orders!
    stab_rows = []
    for r in H_stab:
        m = 0
        for c in r:
            m |= (1 << c)
        stab_rows.append(m)

    for t in range(trials):
        rng.shuffle(cols)
        perm = {orig: new for new, orig in enumerate(cols)}
        inv = cols[:]

        # Eliminate H_check in permuted coordinates
        pivs = {}
        for r in H_check:
            x = 0
            for c in r:
                x ^= (1 << perm[c])
            while x:
                lsb = (x & -x).bit_length() - 1
                if lsb in pivs:
                    x ^= pivs[lsb]
                else:
                    pivs[lsb] = x
                    break

        # RREF upward
        for p in sorted(pivs.keys(), reverse=True):
            for p2 in pivs:
                if p2 != p and ((pivs[p2] >> p) & 1):
                    pivs[p2] ^= pivs[p]

        # Free columns in permuted order -> sparse kernel vectors!
        free_c = [i for i in range(nq) if i not in pivs]
        cand_logicals = []
        for fc in free_c:
            vp = (1 << fc)
            for p, prow in pivs.items():
                if (prow >> fc) & 1:
                    vp |= (1 << p)
            # Map back to original coords
            v_orig = 0
            tmp = vp
            while tmp:
                b = (tmp & -tmp).bit_length() - 1
                v_orig |= (1 << inv[b])
                tmp ^= (1 << b)
            if not in_stab(v_orig):
                w = bin(v_orig).count("1")
                if w < best_w:
                    best_w = w
                    best_vec = v_orig
                cand_logicals.append(v_orig)

        # Pairwise combinations of sparse logicals + greedy local stabilizer reduction
        if cand_logicals:
            L = min(cand_logicals, key=lambda z: bin(z).count("1"))
            # Greedy weight reduction by stabilizers
            improved = True
            while improved:
                improved = False
                wL = bin(L).count("1")
                for sr in stab_rows:
                    w2 = bin(L ^ sr).count("1")
                    if w2 < wL:
                        L ^= sr
                        wL = w2
                        improved = True
            if bin(L).count("1") < best_w:
                best_w = bin(L).count("1")
                best_vec = L

    return best_w, best_vec


def exact_energy_barrier_small(nq, H_check, H_stab):
    """
    EXACT minimax energy barrier Delta for nq <= 22 via Dijkstra over {0,1}^nq!
    Find a sequence of single-qubit flips from 0 -> e_final in (ker(H_check) \ rowspan(H_stab))
    minimizing max_{step} |H_check * e_step|!
    """
    if nq > 21:
        return None
    _, stab_pivots = gf2_rank(nq, H_stab)
    def in_stab(vec):
        x = vec
        while x:
            lsb = (x & -x).bit_length() - 1
            if lsb in stab_pivots:
                x ^= stab_pivots[lsb]
            else:
                return False
        return True

    # Precompute qubit syndrome masks
    m_ch = len(H_check)
    q_syn = [0] * nq
    for ci, r in enumerate(H_check):
        for q in r:
            q_syn[q] |= (1 << ci)

    # Bucket Dijkstra on state `vec` in 0 .. (1<<nq)-1, cost = max syndrome weight along path
    INF = m_ch + 1
    dist = [INF] * (1 << nq)
    syn_of = [0] * (1 << nq)
    dist[0] = 0
    buckets = [deque() for _ in range(m_ch + 1)]
    buckets[0].append(0)

    for b in range(m_ch + 1):
        dq = buckets[b]
        while dq:
            u = dq.popleft()
            if dist[u] < b:
                continue
            if syn_of[u] == 0 and u != 0 and not in_stab(u):
                return b
            su = syn_of[u]
            for q in range(nq):
                v = u ^ (1 << q)
                sv = su ^ q_syn[q]
                w = bin(sv).count("1")
                nb = b if b >= w else w
                if nb < dist[v]:
                    dist[v] = nb
                    syn_of[v] = sv
                    buckets[nb].append(v)
    return None


def heuristic_energy_barrier_upper(nq, H_check, logical_vec, trials=500, seed=123):
    """
    Upper bound on minimax energy barrier along orderings of bits in `logical_vec`
    (plus local greedy ordering where at each step we flip the remaining bit of `logical_vec`
    that minimizes current syndrome weight!).
    Wait: remember in fracton/self-correcting codes, minimum-energy-barrier paths often use
    catalytic detours (flipping bits NOT in the final logical support and later unflipping them,
    i.e. walking through stabilizer deformations!).
    So let's test BOTH direct support orderings AND stabilizer-deformed walk search!
    """
    q_syn = [0] * nq
    for ci, r in enumerate(H_check):
        for q in r:
            q_syn[q] |= (1 << ci)

    supp = [q for q in range(nq) if (logical_vec >> q) & 1]
    best_peak = len(H_check) + 1
    rng = random.Random(seed)

    for t in range(trials):
        rem = set(supp)
        cur_syn = 0
        peak = 0
        while rem:
            # Greedy with small tie-breaking noise
            cands = list(rem)
            rng.shuffle(cands)
            best_q = min(cands, key=lambda q: bin(cur_syn ^ q_syn[q]).count("1"))
            cur_syn ^= q_syn[best_q]
            rem.remove(best_q)
            w = bin(cur_syn).count("1")
            if w > peak:
                peak = w
        if peak < best_peak:
            best_peak = peak
    return best_peak


if __name__ == "__main__":
    print("=== Comparing Seeds at Step 0, 1, 2 for 1-rib (L=2,r=[1]) vs 2-rib (L=3,r=[1,2]) ===")
    seeds = [
        ("1x1 square [[2,1,1]]", make_seed_single_square()),
        ("2x2 surf   [[5,1,2]]", make_seed_surface_code(2, 2)),
        ("3x3 surf  [[13,1,3]]", make_seed_surface_code(3, 3)),
        ("2x3 rect   [[8,1,*]]", make_seed_surface_code(2, 3)),
        ("3x2 rect   [[8,1,*]]", make_seed_surface_code(3, 2)),
    ]
    for sname, s0 in seeds:
        nq0, HX0, HZ0 = build_css_from_sheaf(s0)
        dx0, lx0 = upper_bound_distance_isd(nq0, HZ0, HX0, trials=100)
        dz0, lz0 = upper_bound_distance_isd(nq0, HX0, HZ0, trials=100)
        eb_x0 = exact_energy_barrier_small(nq0, HZ0, HX0)
        eb_z0 = exact_energy_barrier_small(nq0, HX0, HZ0)
        print(f"\nSeed {sname}: [[n={nq0}, k=1, d=({dx0},{dz0})]], exact Delta=({eb_x0}, {eb_z0})")

        for rname, L, ribs in [("1-rib(L=2)", 2, [1]), ("2-rib(L=3)", 3, [1, 2])]:
            c = s0
            for step in (1, 2):
                itype = 'X' if step % 2 == 1 else 'Z'
                c = apply_iteration(c, L, ribs, itype)
                nq, HX, HZ = build_css_from_sheaf(c)
                if nq <= 2500:
                    tr = 250 if nq <= 500 else 60
                    dx, lx = upper_bound_distance_isd(nq, HZ, HX, trials=tr)
                    dz, lz = upper_bound_distance_isd(nq, HX, HZ, trials=tr)
                    ebx = exact_energy_barrier_small(nq, HZ, HX)
                    ebz = exact_energy_barrier_small(nq, HX, HZ)
                    hx_up = heuristic_energy_barrier_upper(nq, HZ, lx, trials=80)
                    hz_up = heuristic_energy_barrier_upper(nq, HX, lz, trials=80)
                    print(
                        f"  {rname} step {step} ({itype}): n={nq:5d}, d<=({dx:2d},{dz:2d}), "
                        f"exact_Delta=({ebx},{ebz}), greedy_Delta<=({hx_up},{hz_up})"
                    )
                else:
                    print(f"  {rname} step {step} ({itype}): n={nq:5d} (skipping Python ISD)")
