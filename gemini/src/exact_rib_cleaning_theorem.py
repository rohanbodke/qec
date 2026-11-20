#!/usr/bin/env python3
"""
EXACT Local Unit Syndrome-Cleaning & Contraction Analysis for 1-Rib vs. 2-Rib BDL Units!

Tests BDL's analytical claim (Section 6.2.5, line 5384 & Section 6.1.5, line 4183):
  "Having two parallel ribs allows us to clean the syndrome from the interior of the unit
   without increasing its weight, whereas with only one rib in each direction such a cleanup
   could increase the syndrome weight."

Let's test BOTH:
  (A) 2D Classical Ladder Unit (1 rung [ell=3] vs 2 rungs [ell=4] vs 3 rungs [ell=5])
  (B) 3D Quantum Doubled Square Unit U_F inside a bulk patch (periodic / open neighbor context)
      for 1-rib (L=2, r=[1] and L=3, r=[1]) vs 2-rib (L=3, r=[1,2] and L=4, r=[1,2])!
"""

import itertools
import math
from collections import defaultdict


def analyze_classical_ladder(ell: int):
    """
    Consider an R_Rep ladder unit replacing a single edge u -- v in X_{(i-1)}:
      - Top rail (Copy 0): u0 -- a_1 -- a_2 -- ... -- a_r -- v0  (where r = ell - 2 rungs!)
        Wait: in X', the edge has ell segments, with attaching vertices at m = 2 .. ell-1 (r = ell-2 vertices),
        plus degree-2 vertex at m = 1!
        Topologically, subdividing a degree-2 segment doesn't change syndrome weight if we merge degree-2 checks,
        OR keeping exact lattice segments of BDL Sec 6.1:
          Copy 0 vertices: p_0 (=u0), p_1 (deg 2), p_2 (black), ..., p_{ell-1} (black), p_ell (=v0)
          Copy 1 vertices: q_0 (=u1), q_1 (black), ..., q_{ell-2} (black), q_{ell-1} (deg 2), q_ell (=v1)
          Rungs: p_m -- q_{m-1} for m = 2 .. ell-1!
      - Boundary checks (shared with neighboring edges of X_{(i-1)}): checks at u0, v0, u1, v1 (4 boundary bits: du0, dv0, du1, dv1).
      - Interior checks:
          - 1 check at p_1 (deg 2): e(p0,p1) + e(p1,p2) = 0
          - 1 check at q_{ell-1} (deg 2): e(q_{ell-2}, q_{ell-1}) + e(q_{ell-1}, q_ell) = 0
          - 2 checks at each black vertex p_m (m=2..ell-1): e_left + e_right = 0, e_left + e_rung = 0
          - 2 checks at each black vertex q_s (s=1..ell-2): e_left + e_right = 0, e_left + e_rung = 0
      - Given ANY error `e` on the (2*ell + (ell-2)) = 3*ell - 2 edges of the ladder:
          - Interior syndrome s_int = H_int * e
          - Boundary flips induced on (u0, v0, u1, v1) by any local correction `f` supported on the ladder edges
            that cleans `s_int` (i.e. H_int * f = s_int)!
          - Wait: at boundary vertex u0, flipping the incident edge e(p0,p1) toggles u0's check by 1!
            So the syndrome pushed onto the 4 boundary vertices (u0, v0, u1, v1) by `f` has weight AT MOST
            |f(p0,p1)| + |f(p_{ell-1},p_ell)| + |f(q0,q1)| + |f(q_{ell-1},q_ell)|!
          - Let's find, for EVERY achievable interior syndrome `s_int` (all 2^{rk(H_int)} of them!):
            min_{f : H_int * f = s_int} (number of boundary rail endpoints flipped by f)
            and compare it to |s_int|!!
    """
    r = ell - 2
    # Edges:
    #   0 .. ell-1       : Copy 0 segments (p_k -- p_{k+1})
    #   ell .. 2*ell-1   : Copy 1 segments (q_k -- q_{k+1})
    #   2*ell .. 3*ell-3 : Rungs m=2..ell-1 connecting p_m to q_{m-1} (index 2*ell + (m-2))
    n_edges = 3 * ell - 2
    bdy_edges = [0, ell - 1, ell, 2 * ell - 1]  # incident to u0, v0, u1, v1

    checks = []
    # Copy 0 interior vertices k = 1 .. ell-1
    for k in range(1, ell):
        eL, eR = k - 1, k
        if k == 1:
            checks.append([eL, eR])
        else:
            e_rung = 2 * ell + (k - 2)
            # Wait: at a black repetition vertex {000, 111}, there are 3 non-zero syndromes of {000,111};
            # in a parity-check matrix representation, 2 generators specify {000,111}.
            # Notice also in sheaf norm (or check-matrix norm), let's check BOTH check-matrix Hamming weight
            # AND vertex-violation count (1 if local codeword violated at vertex, 0 otherwise)!
            checks.append([eL, eR])
            checks.append([eL, e_rung])

    # Copy 1 interior vertices s = 1 .. ell-1
    for s in range(1, ell):
        eL, eR = ell + s - 1, ell + s
        if s == ell - 1:
            checks.append([eL, eR])
        else:
            e_rung = 2 * ell + (s - 1)
            checks.append([eL, eR])
            checks.append([eL, e_rung])

    m_int = len(checks)
    # Enumerate all 2^n_edges local configurations (for ell<=5, n_edges <= 13 -> 8192 configs, instant!)
    syn_to_min_bdy = {}
    syn_to_min_err_w = {}

    # Also compute vertex-sheaf violation weight (number of interior vertices not in local code F_v!)
    syn_to_vweight = {}

    for mask in range(1 << n_edges):
        # Compute interior syndrome mask
        sm = 0
        for ci, ch in enumerate(checks):
            p = 0
            for q in ch:
                p ^= ((mask >> q) & 1)
            if p:
                sm |= (1 << ci)
        bdy_cost = sum((mask >> be) & 1 for be in bdy_edges)
        err_w = bin(mask).count("1")
        if sm not in syn_to_min_bdy or bdy_cost < syn_to_min_bdy[sm]:
            syn_to_min_bdy[sm] = bdy_cost
        if sm not in syn_to_min_err_w or err_w < syn_to_min_err_w[sm]:
            syn_to_min_err_w[sm] = err_w

        # Vertex-sheaf weight (each interior vertex counts 1 if ANY check at that vertex is violated)
        vw = 0
        # Copy 0 k=1
        if ((mask >> 0) ^ (mask >> 1)) & 1: vw += 1
        for k in range(2, ell):
            b1, b2, b3 = (mask >> (k - 1)) & 1, (mask >> k) & 1, (mask >> (2 * ell + k - 2)) & 1
            if not (b1 == b2 == b3): vw += 1
        for s in range(1, ell - 1):
            b1, b2, b3 = (mask >> (ell + s - 1)) & 1, (mask >> (ell + s)) & 1, (mask >> (2 * ell + s - 1)) & 1
            if not (b1 == b2 == b3): vw += 1
        if ((mask >> (2 * ell - 2)) ^ (mask >> (2 * ell - 1))) & 1: vw += 1
        syn_to_vweight[sm] = vw

    # Check how many interior syndromes have min_bdy > |s_int| (both matrix weight and vertex-sheaf weight!)
    viol_mat = 0
    viol_vtx = 0
    max_ratio_mat = 0.0
    max_ratio_vtx = 0.0
    examples_vtx = []

    for sm, min_bdy in syn_to_min_bdy.items():
        if sm == 0:
            continue
        w_mat = bin(sm).count("1")
        w_vtx = syn_to_vweight[sm]
        if min_bdy > w_mat:
            viol_mat += 1
        if min_bdy > w_vtx:
            viol_vtx += 1
            if len(examples_vtx) < 3:
                examples_vtx.append((w_vtx, w_mat, min_bdy, syn_to_min_err_w[sm]))
        max_ratio_mat = max(max_ratio_mat, min_bdy / w_mat)
        max_ratio_vtx = max(max_ratio_vtx, min_bdy / w_vtx)

    total_nz = len(syn_to_min_bdy) - 1
    print(f"Classical Ladder ell={ell} (r={r} ribs), n_edges={n_edges}, total nonzero syndromes={total_nz}:")
    print(f"  [Check-Matrix Energy |s|] violations (bdy_pushed > |s_int|): {viol_mat}/{total_nz}, max(bdy/|s|) = {max_ratio_mat:.3f}")
    print(f"  [Vertex-Sheaf Energy |v|] violations (bdy_pushed > |v_int|): {viol_vtx}/{total_nz}, max(bdy/|v|) = {max_ratio_vtx:.3f}")
    if examples_vtx:
        print(f"    Counterexamples (|v_int|, |s_mat|, min_bdy_pushed, min_err_weight): {examples_vtx}")

    for beta in (1.5, 3.0, 5.0):
        num = 0.0
        den = 0.0
        viol_prob = 0.0
        for sm, min_bdy in syn_to_min_bdy.items():
            if sm == 0:
                continue
            w_mat = bin(sm).count("1")
            boltz = math.exp(-beta * w_mat)
            num += boltz * (min_bdy / w_mat)
            den += boltz
            if min_bdy > syn_to_vweight[sm]:
                viol_prob += boltz
        print(f"    Thermal beta={beta:.1f}: E_boltz[bdy / |s_mat|] = {num/den:.4f}, P_boltz(sheaf_viol | nonzero) = {viol_prob/den:.5f}")


def analyze_quantum_rx_unit(L: int, rib_coords: list):
    """
    Analyze a SINGLE bulk square unit U_F during an R_X iteration (blue ribs {000,111})
    for Z-error cleaning (using X-syndromes on vertices of U_F, which is dual to classical ladder in 2D!)
    AND X-error cleaning (using Z-syndromes on faces of U_F)!

    Look at X-error cleaning on U_F:
      Wait: in R_X, Z-checks correspond to FACES of U_F (all inside U_F!).
      Wait: why did BDL say on line 5384:
        "Here the basic unit is the doubling of a square illustrated in Fig. 10.
         This is the reason for choosing ell = 8: the resulting unit contains two parallel horizontal ribs
         and two parallel vertical ribs. Having two parallel ribs allows us to clean the syndrome
         from the interior of the unit without increasing its weight, whereas with only one rib in each direction
         such a cleanup could increase the syndrome weight.
         After this cleaning, the syndrome is supported on the boundaries of unit,
         precisely on the edges corresponding to those in X_{i-1}."
      AHA!! READ LINES 5389-5390 AGAIN!!
      "After this cleaning, the syndrome is supported on the BOUNDARIES OF THE UNIT,
       precisely on the EDGES corresponding to those in X_{i-1}!!"
      IN SHEAF COHOMOLOGY C^0(X, F) -> C^1(X, F) -> C^2(X, F):
        - X-errors (1-cochains in C^1) have syndromes in C^2(X, F) = FACES!
        - Z-errors (1-chains in C_1, OR equivalently 1-cochains on dual sheaf, wait: in C_2 -> C_1 -> C_0,
          Z-errors on faces C_2 have syndromes on EDGES C_1, OR in C^0 -> C^1 -> C^2, Z-errors in C_1
          have syndromes in C_0 = VERTICES)!!
        Specifically, in the sheaf picture where data lives on faces X(2) (so a Z-error flips face values
        and violates edge-sheaf constraints F_e on EDGES X(1)!!):
        LOOK AT THAT:
          - An error in C^2 / dual 0-chain flips FACES of U_F, and its syndrome lives on EDGES of U_F!
          - Cleaning the syndrome from the INTERIOR EDGES of U_F pushes the syndrome onto the
            BOUNDARY EDGES of U_F ("precisely on the edges corresponding to those in X_{i-1}")!!
      BINGO!! WE JUST DECODED BDL'S EXACT GEOMETRIC CLEANING PROOF IN SECTION 6.2.5!!
      Let's test THAT EXACT 2D-SQUARE-WITH-RIBS EDGE-SYNDROME CLEANING FOR:
        - L=2, ribs=[1]     (1 rib in each direction!)
        - L=3, ribs=[1]     (1 rib in each direction!)
        - L=3, ribs=[1, 2]  (2 ribs in each direction!)
    """
    rib_set = set(rib_coords)
    # Build the faces and interior/boundary edges of ONE doubled square U_F:
    # Variables = flips on the (2*L^2 + 2*len(rib_coords)*L) faces of U_F!
    # Wait: for L=2, ribs=[1]: 2*4 + 4 = 12 faces -> 2^12 = 4,096 face-flip patterns!
    # For L=3, ribs=[1]: 2*9 + 6 = 24 faces! Wait: 2^24 = 16M, or solve via GF(2)/MIP!
    # Wait: let's check what flipping a face does in C_2 -> C_1 (dual of C^1 -> C^2):
    # Wait: in our CSS code (C^0 -> C^1 -> C^2), qubits are in C^1 (edges) and X-checks are in C^0 (vertices)!
    # So Z-errors live on QUBITS (edges of U_F) and their syndrome lives on VERTICES of U_F!
    # Let's check Z-error cleaning on VERTICES of U_F directly on the actual CSS matrix H_X of a 3x3 torus!
    pass


if __name__ == "__main__":
    print("=== PART A: Exact 2D Classical Ladder Unit Cleaning (1-rib vs 2-rib vs 3-rib) ===")
    for ell in (3, 4, 5):
        analyze_classical_ladder(ell)
